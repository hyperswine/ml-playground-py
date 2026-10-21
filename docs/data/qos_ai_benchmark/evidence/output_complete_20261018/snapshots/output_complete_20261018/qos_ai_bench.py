"""Reviewable qos_ai benchmark: source validation, reference execution, model reports.

Rubric matching is a diagnostic proxy. Factual accuracy requires reviewed judgments;
reference-program success does not imply generated code works.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from qos_ai_data import digest, file_hash, lexical_score, read_rows, rows_fingerprint, source_check

DEFAULT_DATA = Path(__file__).parent / 'qos_ai_benchmark'


def roots_from_args(args):
  return dict(item.split('=', 1) for item in args.root)


def validate_rows(rows):
  ids = set()
  questions = set()
  groups = {}
  for row in rows:
    if row['id'] in ids or row['question'] in questions:
      raise ValueError(f'Duplicate benchmark id/question: {row["id"]}')
    if row['group'] in groups and groups[row['group']] != row['split']:
      raise ValueError('Benchmark concept group crosses dev/test')
    groups[row['group']] = row['split']
    ids.add(row['id'])
    questions.add(row['question'])
    if row['split'] not in ('dev', 'test') or row['track'] not in ('familiar', 'generalization'):
      raise ValueError('Invalid benchmark partition or track')
    if not row['sources'] or not row['answer'].strip() or not row['rubric']['required']:
      raise ValueError('Benchmark needs sources, reference answer and a rubric')
    if not lexical_score(row, row['answer'])['rubric_proxy_pass']:
      raise ValueError(f'Reference answer fails its own rubric: {row["id"]}')

  from qos_ai_data import similar_pairs
  for i, j in similar_pairs([r['question'] for r in rows]):
    if rows[i]['split'] != rows[j]['split']:
      raise ValueError('Near-duplicate benchmark questions cross dev/test')


def validate(args):
  rows = read_rows(args.data / 'benchmark.jsonl')
  train = read_rows(args.data / 'verified_train.jsonl')
  if not rows or not train:
    raise ValueError('Benchmark and verified teaching examples must both exist')
  validate_rows(rows)
  for row in rows + train:
    source_check(row, roots_from_args(args))
  print(json.dumps({'benchmark': len(rows), 'verified_examples': len(train),
                    'benchmark_splits': dict(Counter(r['split'] for r in rows)),
                    'source_files': len({s['path'] for r in rows + train for s in r['sources']}),
                    'status': 'exact source hashes and evidence snippets verified'}, indent=2))


def references(args):
  """Execute only authored, source-reviewed reference fixtures, never model output."""
  rows = read_rows(args.data / 'benchmark.jsonl') + read_rows(args.data / 'verified_train.jsonl')
  results = []
  fpr = args.fpr.resolve()
  for row in rows:
    source_check(row, roots_from_args(args))
    program = row.get('program')
    if not program:
      continue
    with tempfile.TemporaryDirectory(prefix='qos-ai-reference-') as directory:
      path = Path(directory) / 'reference.sol'
      path.write_text(program['code'])
      env = dict(os.environ, SOL_JIT='0', SOL_GPU='0', SOL_CACHE='0', FPR_HOME=roots_from_args(args)['fprisc'])
      try:
        run = subprocess.run([str(fpr), 'sol', str(path)], capture_output=True, text=True,
                             cwd=directory, env=env, timeout=30)
        ok = run.returncode == program.get('exit_code', 0) and run.stdout.strip() == program['stdout'].strip()
        if 'stderr_contains' in program:
          ok = ok and program['stderr_contains'] in run.stderr
        results.append({'id': row['id'], 'passed': ok, 'returncode': run.returncode,
                        'stdout': run.stdout, 'stderr': run.stderr})
      except subprocess.TimeoutExpired:
        results.append({'id': row['id'], 'passed': False, 'error': 'timeout'})
  output = {'verified_data_hash': rows_fingerprint(read_rows(args.data / 'verified_train.jsonl')),
            'benchmark_data_hash': rows_fingerprint(read_rows(args.data / 'benchmark.jsonl')),
            'fpr_sha256': file_hash(fpr), 'host': platform.node(), 'machine': platform.machine(),
            'passed': sum(r['passed'] for r in results), 'total': len(results), 'results': results}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({k: v for k, v in output.items() if k != 'results'}, indent=2))
  if not results or output['passed'] != output['total']:
    raise SystemExit(1)


def run(args):
  import torch
  import qos_ai as ai
  rows = read_rows(args.data / 'benchmark.jsonl')
  validate_rows(rows)
  roots = roots_from_args(args)
  if not roots:
    snapshot = Path(__file__).parent / 'source_snapshot'
    roots = {name: str(snapshot / name) for name in ('fprisc', 'qos')}
  for row in rows:
    source_check(row, roots)
  proof = json.loads((args.data / 'verification.json').read_text())
  passed = {r['id'] for r in proof['results'] if r['passed']}
  if proof.get('benchmark_data_hash') != rows_fingerprint(rows) or any(r.get('program') and r['id'] not in passed for r in rows):
    raise ValueError('Benchmark reference execution is missing or stale')
  rows = [r for r in rows if r['split'] == args.split]
  if args.limit:
    import random
    candidates = list(rows)
    random.Random(args.seed).shuffle(candidates)
    chosen, seen_groups, categories = [], set(), set()
    for row in candidates:
      if row['category'] not in categories:
        chosen.append(row); seen_groups.add(row['group']); categories.add(row['category'])
    for row in candidates:
      if row['group'] not in seen_groups:
        chosen.append(row); seen_groups.add(row['group'])
    chosen.extend(row for row in candidates if row not in chosen)
    rows = chosen[:args.limit]
  if not rows:
    raise ValueError('No benchmark questions selected')
  if args.output.exists():
    raise ValueError('Output exists; choose a fresh report directory')
  args.output.mkdir(parents=True)
  torch.manual_seed(args.seed)
  device = ai.choose_device(args.device)
  model, checkpoint = ai.load_student(args.load, device)
  tokenizer = ai.StudentTokenizer(checkpoint['qwen_ids'])
  manifests = args.splits or (ai.WORK_DIR / 'splits.json')
  manifest = json.loads(manifests.read_text()) if manifests.exists() else None
  if manifest:
    from qos_ai_data import validate_manifest
    validate_manifest(manifest, benchmark=read_rows(args.data / 'benchmark.jsonl'))
  settings = {'checkpoint': str(args.load), 'checkpoint_sha256': file_hash(args.load),
              'benchmark_sha256': file_hash(args.data / 'benchmark.jsonl'),
              'split_manifest_hash': manifest['hash'] if manifest else None,
              'checkpoint_split_hash': checkpoint.get('split_hash'),
              'ancestry': checkpoint.get('benchmark_ancestry', 'legacy_or_unknown'),
              'split': args.split, 'seed': args.seed, 'temperature': args.temperature,
              'top_k': args.top_k, 'no_repeat': args.no_repeat, 'requested_max_new_tokens': args.max_new_tokens,
              'device': str(device), 'torch': torch.__version__, 'host': platform.node(),
              'machine': platform.machine(), 'code_sha256': file_hash(Path(ai.__file__)),
              'evaluator_sha256': file_hash(Path(__file__)),
              'split_policy_code_sha256': file_hash(Path(__file__).with_name('qos_ai_data.py')),
              'clean_test_claim': False,
              'exposure_note': 'Legacy checkpoints are not made clean by new splits. Ancestry and source overlap require review.'}
  results = []
  for row in rows:  # one prompt at a time: invariant token budget across batch compositions
    prompt = tokenizer.encode(ai.chat_text(row['question']))
    budget = min(args.max_new_tokens, model.config.context_length - len(prompt))
    if budget <= 0:
      raise ValueError(f'Prompt exceeds model context: {row["id"]}')
    started = time.perf_counter()
    with torch.inference_mode():
      ids, finished = model.generate_batch([prompt], budget, checkpoint['im_end_id'], args.temperature, args.top_k, args.no_repeat)[0]
    text = tokenizer.decode(ids).strip()
    record = {'id': row['id'], 'group': row['group'], 'question': row['question'], 'category': row['category'], 'track': row['track'],
              'answer': text, 'reference_answer': row['answer'], 'rubric': row['rubric'],
              'finished': finished, 'tokens': len(ids), 'effective_token_budget': budget,
              'context_limited': budget < args.max_new_tokens,
              'severe_loop': ai.loops(ids, phrase=4, times=4), 'seconds': time.perf_counter() - started,
              'generated_code_execution': None, **lexical_score(row, text)}
    results.append(record)
    with (args.output / 'answers.jsonl').open('a') as f:
      f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'{len(results)}/{len(rows)} {row["id"]}: proxy={record["rubric_proxy_pass"]} finished={finished}', flush=True)
  summary = {**settings, 'count': len(results), 'concept_groups': len({r['group'] for r in results}),
             'rubric_proxy_pass_rate': sum(r['rubric_proxy_pass'] for r in results) / len(results),
             'finished_rate': sum(r['finished'] for r in results) / len(results),
             'severe_loop_rate': sum(r['severe_loop'] for r in results) / len(results),
             'mean_tokens': sum(r['tokens'] for r in results) / len(results),
             'context_limited_count': sum(r['context_limited'] for r in results),
             'factual_accuracy': None, 'human_review_required': True}
  (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  template = [{'id': r['id'], 'answer_hash': digest(r['answer']), 'reviewer': '', 'factual_correct': None,
               'generated_code_compiles': None, 'generated_code_behavior_correct': None, 'evidence': ''} for r in results]
  (args.output / 'review_template.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in template))
  print(json.dumps(summary, indent=2))


def prepare(args):
  from qos_ai_data import row_split, validate_manifest
  rows = read_rows(args.data / 'verified_train.jsonl')
  proof = json.loads((args.data / 'verification.json').read_text())
  manifest = json.loads(args.splits.read_text())
  validate_manifest(manifest, verified=rows, benchmark=read_rows(args.data / 'benchmark.jsonl'))
  if proof.get('verified_data_hash') != rows_fingerprint(rows):
    raise ValueError('Reference execution is stale')
  passed = {r['id'] for r in proof['results'] if r['passed']}
  selected = []
  for row in rows:
    source_check(row, roots_from_args(args))
    if row.get('program') and row['id'] not in passed:
      raise ValueError('Teaching example lacks passing reference execution')
    if row_split(row, manifest) == 'train':
      selected.append(row)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(''.join(json.dumps(r) + '\n' for r in selected))
  print(json.dumps({'prepared_training_examples':len(selected), 'output':str(args.output),
                    'split_hash':manifest['hash'], 'output_sha256':file_hash(args.output)}, indent=2))


def reviewed_scores(answers, judgments):
  by_id = {r['id']: r for r in answers}
  seen = set()
  metrics = ('factual_correct', 'generated_code_compiles', 'generated_code_behavior_correct')
  counts = {key: [] for key in metrics}
  for row in judgments:
    if row['id'] in seen or row['id'] not in by_id:
      raise ValueError('Duplicate or unknown review id')
    seen.add(row['id'])
    if row.get('answer_hash') != digest(by_id[row['id']]['answer']):
      raise ValueError('Review does not match the generated answer')
    values = [row.get(key) for key in metrics]
    if any(value is not None and type(value) is not bool for value in values):
      raise ValueError('Judgments must be true, false or null')
    if any(value is not None for value in values) and (not row.get('reviewer') or not row.get('evidence')):
      raise ValueError('A judgment requires a reviewer and evidence')
    for key, value in zip(metrics, values):
      if value is not None:
        counts[key].append(value)
  return {key: {'reviewed': len(values), 'correct': sum(values),
                'rate_on_reviewed': sum(values) / len(values) if values else None,
                'coverage': len(values) / len(answers) if answers else 0}
          for key, values in counts.items()}


def score(args):
  answers = read_rows(args.report / 'answers.jsonl')
  judgments = read_rows(args.judgments)
  result = {'answers_sha256': file_hash(args.report / 'answers.jsonl'),
            'judgments_sha256': file_hash(args.judgments), 'metrics': reviewed_scores(answers, judgments)}
  (args.report / 'reviewed_scores.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='command', required=True)
  for command, fn in [('validate', validate), ('references', references), ('run', run)]:
    parser = sub.add_parser(command)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--root', action='append', default=[], help='Source repository mapping, e.g. fprisc=/path/fprisc')
    parser.set_defaults(fn=fn)
    if command == 'references':
      parser.add_argument('--fpr', required=True, type=Path)
      parser.add_argument('--output', required=True, type=Path)
    if command == 'run':
      parser.add_argument('--load', required=True, type=Path)
      parser.add_argument('--output', required=True, type=Path)
      parser.add_argument('--splits', type=Path)
      parser.add_argument('--split', choices=('dev', 'test'), default='dev')
      parser.add_argument('--limit', type=int, default=0)
      parser.add_argument('--seed', type=int, default=7)
      parser.add_argument('--temperature', type=float, default=0.0)
      parser.add_argument('--top-k', type=int, default=20)
      parser.add_argument('--no-repeat', type=int, default=0)
      parser.add_argument('--max-new-tokens', type=int, default=200)
      parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
  prep = sub.add_parser('prepare', help='export source-checked, execution-verified training-only examples')
  prep.add_argument('--data', type=Path, default=DEFAULT_DATA)
  prep.add_argument('--root', action='append', default=[])
  prep.add_argument('--splits', required=True, type=Path)
  prep.add_argument('--output', required=True, type=Path)
  prep.set_defaults(fn=prepare)
  scoring = sub.add_parser('score', help='score explicit answer reviews, preserving coverage and provenance')
  scoring.add_argument('--report', required=True, type=Path)
  scoring.add_argument('--judgments', required=True, type=Path)
  scoring.set_defaults(fn=score)
  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
