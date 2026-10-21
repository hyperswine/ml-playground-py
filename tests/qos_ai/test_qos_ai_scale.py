import copy
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from qos_ai import QosAi, StudentConfig
from qos_ai_scale import (advance, describe, draw, forward, optimizer_for,
                          partition_documents, restore, save_boundary, train,
                          write_json, load_dataset)
from qos_ai_data import digest, file_hash


class ScalingTests(unittest.TestCase):
  def setUp(self):
    torch.set_num_threads(1)

  def config(self):
    return StudentConfig(vocabulary_size=24, context_length=8, embedding_size=16, head_count=2, layer_count=2)

  def settings(self):
    return {'config': asdict(self.config()), 'seed': 7, 'batch_size': 2, 'sequence_length': 8,
            'gradient_accumulation': 2, 'optimizer': 'adamw', 'learning_rate': .003,
            'total_steps': 4, 'warmup_steps': 1, 'checkpoint_blocks': True, 'bf16': False}

  def streams(self):
    return {'train': np.asarray(([1, 2, 0, 3, 4, 5, 6] * 30), dtype=np.uint32)}

  def test_actual_557m_shape_and_tied_parameters(self):
    result = describe(StudentConfig(65536, 512, 1280, 20, 24))
    self.assertEqual(result['unique_parameters'], 556526080)
    model = QosAi(self.config())
    self.assertIs(model.token_embedding.weight, model.language_model_head.weight)

  def test_activation_checkpointing_preserves_logits_and_gradients(self):
    torch.manual_seed(1)
    a = QosAi(self.config()); b = copy.deepcopy(a)
    tokens = torch.tensor([[1, 2, 3, 4]])
    plain = forward(a, tokens, False); checked = forward(b, tokens, True)
    self.assertTrue(torch.equal(plain, checked))
    plain.square().sum().backward(); checked.square().sum().backward()
    for left, right in zip(a.parameters(), b.parameters()):
      self.assertTrue(torch.equal(left.grad, right.grad))

  def test_resume_matches_weights_moments_schedule_and_data(self):
    torch.manual_seed(7)
    a = QosAi(self.config()); b = copy.deepcopy(a)
    settings = self.settings(); streams = self.streams()
    oa = optimizer_for(a, settings); ob = optimizer_for(b, settings)
    full = [advance(a, oa, streams, settings, s) for s in range(1, 5)]
    partial = [advance(b, ob, streams, settings, s) for s in range(1, 3)]
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'progress.pt'
      save_boundary(path, b, ob, settings, 2, partial)
      expected_rng = torch.rand(4)
      c = QosAi(self.config()); oc = optimizer_for(c, settings)
      step, history = restore(path, c, oc, settings)
      self.assertEqual(step, 2); self.assertEqual(history, partial)
      self.assertTrue(torch.equal(expected_rng, torch.rand(4)))
      continued = [advance(c, oc, streams, settings, s) for s in range(step + 1, 5)]
      self.assertEqual(partial + continued, full)
      for key, value in a.state_dict().items():
        self.assertTrue(torch.equal(value, c.state_dict()[key]), key)
      for left, right in zip(oa.state.values(), oc.state.values()):
        for key in left:
          self.assertTrue(torch.equal(left[key], right[key]), key)
      saved = torch.load(path, weights_only=True, mmap=True)
      self.assertEqual(saved['model_state']['token_embedding.weight'].data_ptr(),
                       saved['model_state']['language_model_head.weight'].data_ptr())
      for name in ('total_steps', 'seed', 'sequence_length', 'optimizer'):
        changed = {**settings, name: 'changed'}
        original = copy.deepcopy(c.state_dict())
        with self.assertRaisesRegex(ValueError, 'differ'):
          restore(path, c, oc, changed)
        for key, value in original.items():
          self.assertTrue(torch.equal(value, c.state_dict()[key]))

  def test_accumulation_is_weighted_by_known_targets(self):
    torch.manual_seed(7)
    a = QosAi(self.config()); b = copy.deepcopy(a)
    settings = self.settings(); settings['checkpoint_blocks'] = False
    oa = optimizer_for(a, settings); ob = optimizer_for(b, settings)
    record = advance(a, oa, self.streams(), settings, 1)
    batches = [draw(self.streams()['train'], 7, 1, m, 2, 8) for m in range(2)]
    batch = torch.cat(batches)
    from qos_ai_scale import loss_sum, rate
    count = int((batch[:, 1:] != 0).sum())
    loss = loss_sum(b, batch) / count
    loss.backward()
    torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
    ob.param_groups[0]['lr'] = rate(.003, 1, 4, 1); ob.step()
    self.assertEqual(record['known_target_tokens'], count)
    for left, right in zip(a.parameters(), b.parameters()):
      self.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=1e-5))

  def test_adafactor_segment_restores_factored_moments(self):
    torch.manual_seed(7)
    a = QosAi(self.config()); b = copy.deepcopy(a)
    settings = {**self.settings(), 'optimizer': 'adafactor', 'learning_rate': .01}
    oa = optimizer_for(a, settings); ob = optimizer_for(b, settings)
    for step in (1, 2):
      advance(a, oa, self.streams(), settings, step)
    history = [advance(b, ob, self.streams(), settings, 1)]
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'progress.pt'
      save_boundary(path, b, ob, settings, 1, history)
      c = QosAi(self.config()); oc = optimizer_for(c, settings)
      step, _ = restore(path, c, oc, settings)
      advance(c, oc, self.streams(), settings, step + 1)
      for left, right in zip(a.parameters(), c.parameters()):
        self.assertTrue(torch.equal(left, right))
      for left, right in zip(oa.state.values(), oc.state.values()):
        for key in left:
          self.assertTrue(torch.equal(left[key], right[key]))

  def test_case_and_source_groups_are_disjoint_and_exact_copies_removed(self):
    docs = [{'id': str(i), 'group': 'case:' + str(i // 2), 'kind': 'verified_chat', 'text': str(i)} for i in range(200)]
    initial, _ = partition_documents(docs)
    train_group = initial['train'][0]['group']; validation_group = initial['validation'][0]['group']
    docs += [{'id': 'copy-a', 'group': train_group, 'text': 'copied'},
             {'id': 'copy-b', 'group': validation_group, 'text': 'copied'}]
    parts, removed = partition_documents(docs)
    self.assertEqual(removed, 1)
    self.assertFalse({d['group'] for d in parts['train']} & {d['group'] for d in parts['validation']})
    self.assertFalse({d['text'] for d in parts['train']} & {d['text'] for d in parts['validation']})

  def test_complete_runner_exports_loadable_model_and_resumes_without_schedule_reset(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp); data = root / 'data'; data.mkdir()
      config = root / 'config.json'; write_json(config, asdict(self.config()))
      for split in ('train', 'validation'):
        np.save(data / (split + '.npy'), self.streams()['train'])
      torch.save({'qwen_ids': torch.arange(-1, 23), 'pad_id': 1, 'im_end_id': 2}, data / 'vocabulary.pt')
      write_json(data / 'document_partition.json', {})
      metadata = {'split_hash': 'test-split', 'vocabulary_sha256': file_hash(data / 'vocabulary.pt'),
                  'partition_sha256': file_hash(data / 'document_partition.json'),
                  'stats': {s: {'tokens': len(self.streams()['train']), 'sha256': file_hash(data / (s + '.npy'))}
                            for s in ('train', 'validation')}}
      metadata['hash'] = digest(metadata); write_json(data / 'dataset.json', metadata)
      args = argparse.Namespace(config=config, data=data, output=root / 'full', device='cpu', optimizer='adamw',
                                sequence_length=8, steps=4, batch_size=2, grad_accumulation=2, learning_rate=.003,
                                max_minutes=1, save_every=2, segment_steps=0, warmup_steps=1, seed=7,
                                bf16=False, checkpoint_blocks=True, resume=False)
      with contextlib.redirect_stdout(io.StringIO()):
        train(args)
        args.output = root / 'segments'; args.segment_steps = 2
        train(args)
        self.assertFalse((args.output / 'student.pt').exists())
        args.resume = True; train(args)
      a = torch.load(root / 'full/progress.pt', weights_only=True, mmap=True)
      b = torch.load(root / 'segments/progress.pt', weights_only=True, mmap=True)
      self.assertEqual(a['step'], 4); self.assertEqual(b['step'], 4)
      for key, value in a['model_state'].items():
        self.assertTrue(torch.equal(value, b['model_state'][key]))
      from qos_ai import load_student
      model, exported = load_student(root / 'segments/student.pt', torch.device('cpu'))
      self.assertEqual(exported['step'], 4)
      self.assertIs(model.token_embedding.weight, model.language_model_head.weight)
      np.save(data / 'train.npy', np.ones(210, dtype=np.uint32))
      with self.assertRaisesRegex(ValueError, 'stream changed'):
        load_dataset(data, self.config())


if __name__ == '__main__':
  unittest.main()
