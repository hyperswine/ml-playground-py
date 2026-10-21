import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from qos_ai import QosAi, StudentConfig
from qos_ai_data import digest
from qos_ai_resume import load_progress, save_progress
from qos_ai_stage4_curriculum import record, TRAIN_WORDING, VALID_WORDING
from qos_ai_stage4_data import mix, targets, token_sample
from qos_ai_stage4_eval import Unsupported, fingerprint, parse_expression, score, sha
from qos_ai_stage4_recovery import require_space, verify_backups, REQUIRED_ROLES
from qos_ai_stage4_train import plan_for, train_updates

ROOT = Path(__file__).parent
FPR = Path(os.environ.get('QOS_FPR', '/nonexistent/fpr'))
SUITE = ROOT / 'qos_ai_benchmark/stage4_20261021/multi_input.json'


class AdmissionTests(unittest.TestCase):
  @unittest.skipUnless((ROOT / 'runpod/557m/bundle/vocabulary.pt').is_file(), 'restore the fixed vocabulary for prompt encoding audit')
  def test_frozen_prompts_fit_known_vocabulary_and_generation_budget(self):
    import qos_ai as ai
    from qos_ai_runpod_data import FrozenTokenizer
    vocab = torch.load(ROOT / 'runpod/557m/bundle/vocabulary.pt', weights_only=True)
    tokenizer = ai.StudentTokenizer(vocab['qwen_ids'], FrozenTokenizer(ROOT / 'runpod/557m/bundle/tokenizer'))
    suite = json.loads(SUITE.read_text())
    for row in suite['rows']:
      ids = tokenizer.encode(ai.chat_text(row['question']))
      self.assertNotIn(0, ids, row['id'])
      self.assertLessEqual(len(ids) + 200, 512, row['id'])

  def test_effects_free_functions_and_hidden_list_growth_rejected(self):
    for text in ('> print 1.', 'File.read "secrets"', 'loop xs',
                 'List.map (fn x -> if x > 0 then [x] else [0]) xs',
                 'List.map (fn x -> if x > 0 then List.range 0 10 else xs) xs',
                 'List.range 0 999', 'xs = [1]. xs', '```sol\n1\n```\n```sol\n2\n```'):
      with self.subTest(text=text), self.assertRaises((Unsupported, RecursionError)):
        parse_expression(text, ('xs', 'k', 'limit'))

  def test_tampered_suite_and_incomplete_answers_refused(self):
    with tempfile.TemporaryDirectory() as d:
      compiler = Path(d) / 'fpr'; compiler.write_text('test identity')
      suite = {'compiler_sha256': sha(compiler), 'rows': [{'id': 'a'}, {'id': 'b'}]}
      suite['hash'] = fingerprint(suite)
      for answers in ([], [{'id': 'a', 'answer': '0'}] * 2):
        with self.assertRaisesRegex(ValueError, 'exactly once'): score(suite, answers, compiler)
      suite['rows'][0]['id'] = 'changed'
      with self.assertRaisesRegex(ValueError, 'suite hash'): score(suite, [], compiler)
      suite['hash'] = fingerprint({k: v for k, v in suite.items() if k != 'hash'})
      compiler.write_text('changed binary')
      with self.assertRaisesRegex(ValueError, 'compiler identity'): score(suite, [], compiler)

  @unittest.skipUnless(FPR.is_file(), 'set QOS_FPR to the pinned compiler for executable checks')
  def test_references_pass_and_adversarial_answers_fail(self):
    suite = json.loads(SUITE.read_text())
    answers = [{'id': r['id'], 'answer': r['reference']} for r in suite['rows']]
    self.assertEqual(score(suite, answers, FPR)['correct'], 18)
    replacements = {'map_sum': '65',
                    'signed_branch': 'if k > 0 then List.sum (List.map (fn x -> x * k) xs) else 0',
                    'filter_sum': 'List.sum (List.filter (fn x -> x >= limit) xs)'}
    for answer, task in zip(answers, suite['rows']):
      answer['answer'] = replacements.get(task['family'], answer['answer'])
    result = score(suite, answers, FPR)
    self.assertEqual(result['correct'], 9)
    self.assertTrue(all(not r['correct'] for r in result['rows'] if r['family'] in replacements))


class DataTests(unittest.TestCase):
  def test_arithmetic_groups_and_intermediate_calculations(self):
    root = ROOT / 'qos_ai_benchmark/stage4_20261021/arithmetic'
    rows = {s: [json.loads(l) for l in (root / f'{s}.jsonl').read_text().splitlines()] for s in ('train', 'validation')}
    self.assertFalse({r['group'] for r in rows['train']} & {r['group'] for r in rows['validation']})
    self.assertEqual((len(rows['train']), len(rows['validation'])), (899, 125))
    self.assertFalse(set(TRAIN_WORDING) & set(VALID_WORDING))
    for split, records in rows.items():
      for r in records:
        self.assertEqual(r['split'], split)
        values = r['inputs']; calculated = record(r['family'], **values)
        self.assertEqual(json.loads(json.dumps(calculated['stages'])), r['stages'])
        self.assertEqual(calculated['expected'], r['expected'])
    self.assertEqual(record('map_sum', [-3, 0, 8], -2, 0)['expected'], '-10')
    self.assertEqual(record('filter_sum', [4, 4, 5], 0, 4)['expected'], '5')

  def test_replay_accounting_uses_targets_and_preserves_examples(self):
    pool = [([1, 2, 3], [False, False, True]), ([1, 2, 3, 4], [False, True, True, True])]
    selected, stats = token_sample(pool, 17, 7)
    self.assertEqual(sum(targets(e) for e in selected), stats['target_tokens'])
    self.assertGreaterEqual(stats['target_tokens'], 17)
    self.assertLess(stats['overshoot'], 3)
    self.assertTrue(all(e in pool for e in selected))
    self.assertEqual((selected, stats), token_sample(pool, 17, 7))
    with self.assertRaises(ValueError): targets(([1, 0], [False, True]))
    with self.assertRaises(ValueError): targets(([1, 2], [False, False]))
    with self.assertRaises(ValueError): token_sample([], 10, 7)

  def test_mixer_uses_only_hash_checked_training_stream(self):
    with tempfile.TemporaryDirectory() as d:
      root = Path(d); stream = root / 'train.bin'
      np.asarray([1, 2, 3, 4] * 256, dtype=np.uint32).tofile(stream)
      example = ([1, 2, 3, 4], [False, False, True, True])
      pools = {'verified': [example] * 1000, 'qa_pool': [example], 'docs': [example]}
      pools['hash'] = digest(pools)
      pools.update(pad_id=1, splits_hash='split', vocabulary_sha256='vocab')
      torch.save(pools, root / 'pools.pt')
      manifest = {'vocabulary_sha256': 'vocab', 'results': {'english': {'files': {
        'train': {'path': 'train.bin', 'sha256': sha(stream)},
        'validation': {'path': 'deliberately_missing.bin', 'sha256': 'never used'}}}}}
      manifest['hash'] = digest(manifest)
      (root / 'dataset.json').write_text(json.dumps(manifest))
      args = SimpleNamespace(pools=root / 'pools.pt', pretraining_data=root,
             pretraining_manifest_sha256=sha(root / 'dataset.json'), output=root / 'out')
      result = mix(args)
      payload = torch.load(args.output / 'sft.pt', weights_only=True)
      self.assertEqual(payload['qa_pool'], [])
      self.assertEqual(result['target_tokens'], sum(targets(e) for e in payload['verified']))
      self.assertEqual(result['english_train_sha256'], sha(stream))
      args.output = root / 'bad'; stream.write_bytes(b'tampered')
      with self.assertRaisesRegex(ValueError, 'training stream'): mix(args)
      self.assertFalse(args.output.exists())


class RecoveryTests(unittest.TestCase):
  def test_disk_guard_requires_full_replacement_alongside_old_file(self):
    with patch('qos_ai_stage4_recovery.shutil.disk_usage', return_value=SimpleNamespace(free=100)):
      with self.assertRaises(OSError): require_space('.', 90, 20)
      self.assertEqual(require_space('.', 80, 20)['free_bytes'], 100)

  def test_missing_or_bad_backup_cannot_report_complete(self):
    with tempfile.TemporaryDirectory() as d:
      path = Path(d) / 'backup'; path.write_bytes(b'checked artifact')
      artifacts = [{'role': role, 'backup_path': str(path), 'sha256': sha(path)} for role in REQUIRED_ROLES]
      self.assertTrue(verify_backups({'artifacts': artifacts})['complete'])
      artifacts[0]['sha256'] = None
      self.assertFalse(verify_backups({'artifacts': artifacts})['complete'])
      with self.assertRaises(ValueError): verify_backups({'artifacts': artifacts[:-1]})

  def test_short_training_resume_matches_uninterrupted(self):
    torch.manual_seed(7)
    base = QosAi(StudentConfig(vocabulary_size=16, context_length=8, embedding_size=16, head_count=2, layer_count=1))
    full, segmented = copy.deepcopy(base), copy.deepcopy(base)
    make_opt = lambda model: torch.optim.AdamW(model.parameters(), lr=.01, betas=(.9, .95), weight_decay=.1)
    oa, ob = make_opt(full), make_opt(segmented)
    examples = [([1, 2, 3, i], [False, False, True, True]) for i in range(4, 11)]
    plan = plan_for(len(examples), 'one_2e5', 2, 2)
    train_updates(full, oa, examples, plan, 1, .01, 1)
    first = train_updates(segmented, ob, examples, plan, 1, .01, 1, after_step=lambda row: True)
    with tempfile.TemporaryDirectory() as d:
      path = Path(d) / 'progress.pt'
      save_progress(path, segmented, ob, {'case': 'tiny'}, {'step': 1, 'history': first})
      resumed = copy.deepcopy(base); oc = make_opt(resumed)
      progress = load_progress(path, resumed, oc, {'case': 'tiny'})
      train_updates(resumed, oc, examples, plan, 1, .01, 1, start=progress['step'])
      for name, value in full.state_dict().items(): self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
      for a, b in zip(oa.state.values(), oc.state.values()):
        for key in a: self.assertTrue(torch.equal(a[key], b[key]), key)
    self.assertEqual(len(plan_for(100, 'half_2e5')), 2)


if __name__ == '__main__': unittest.main()
