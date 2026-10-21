"""Audit a finished scaling pilot and sample a few explicitly non-benchmark prompts."""
import argparse
import json
import math
from pathlib import Path

import torch

import qos_ai as ai
from qos_ai_data import digest, file_hash
from qos_ai_scale import load_dataset, read_config, rate, write_json


def audit(args):
  config = read_config(args.config)
  dataset, _, vocabulary = load_dataset(args.data, config)
  progress_path = args.experiment / 'progress.pt'
  progress = torch.load(progress_path, map_location='cpu', weights_only=True, mmap=True)
  settings = progress['settings']; history = progress['history']
  if progress['settings_hash'] != digest(settings) or settings['dataset_hash'] != dataset['hash']:
    raise ValueError('Progress provenance mismatch')
  if progress['step'] != settings['total_steps'] or [r['step'] for r in history] != list(range(1, progress['step'] + 1)):
    raise ValueError('Missing, repeated or incomplete accepted updates')
  for record in history:
    if not all(math.isfinite(record[k]) for k in ('loss', 'gradient_norm', 'learning_rate')):
      raise ValueError('Nonfinite accepted update')
    expected = rate(settings['learning_rate'], record['step'], settings['total_steps'], settings['warmup_steps'])
    if record['learning_rate'] != expected:
      raise ValueError('Schedule changed')
    if record['tokens'] != settings['sequence_length'] * settings['batch_size'] * settings['gradient_accumulation']:
      raise ValueError('Token accounting changed')
  state_bytes = sum(v.numel() * v.element_size() for state in progress['optimizer_state']['state'].values()
                    for v in state.values() if isinstance(v, torch.Tensor))
  checkpoint_path = args.experiment / 'student.pt'
  exported = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
  if exported['config'] != settings['config'] or exported['step'] != progress['step']:
    raise ValueError('Published model config/step differs')
  for key in ('qwen_ids', 'pad_id', 'im_end_id'):
    if isinstance(vocabulary[key], torch.Tensor):
      if not torch.equal(exported[key], vocabulary[key]):
        raise ValueError('Vocabulary mapping differs')
    elif exported[key] != vocabulary[key]:
      raise ValueError('Vocabulary special token differs')
  for key, value in exported['model_state'].items():
    if not torch.isfinite(value).all().item() or not torch.equal(value, progress['model_state'][key]):
      raise ValueError('Invalid or different published weight: ' + key)
  if exported['model_state']['token_embedding.weight'].data_ptr() != exported['model_state']['language_model_head.weight'].data_ptr():
    raise ValueError('Published embedding storage is not tied')
  benchmarks = {}
  for path, sha in dataset['frozen_benchmarks'].items():
    if file_hash(Path(path)) != sha:
      raise ValueError('Benchmark changed: ' + path)
    benchmarks[path] = sha
  step = progress['step']
  del value, progress, exported
  # This also proves that existing inference loading accepts the final artifact.
  model, loaded = ai.load_student(checkpoint_path, torch.device(args.device))
  if ai.parameter_count(model) != 556526080:
    raise ValueError('Unexpected full-model parameter count')
  tokenizer = ai.StudentTokenizer(loaded['qwen_ids'])
  model.eval()
  prompts = ['The purpose of a compiler is', 'What is Sol?', 'Write a Sol function that adds two numbers.']
  samples = []
  for prompt in prompts:
    ids = tokenizer.encode(ai.chat_text(prompt))
    tokens = torch.tensor([ids], device=args.device)
    output = model.generate(tokens, 32, loaded['im_end_id'], temperature=0, top_k=20, no_repeat=0)
    answer = output[0, len(ids):].tolist()
    samples.append({'prompt': prompt, 'answer': tokenizer.decode(answer),
                    'finished': loaded['im_end_id'] in answer, 'generated_tokens': len(answer)})
  args.output.mkdir(parents=True, exist_ok=False)
  with (args.output / 'accepted_updates.jsonl').open('w') as out:
    for record in history:
      out.write(json.dumps(record) + '\n')
  result = {'parameter_count': ai.parameter_count(model), 'optimizer': settings['optimizer'],
            'optimizer_state_tensor_bytes': state_bytes, 'updates': step,
            'training_tokens': sum(r['tokens'] for r in history),
            'known_target_tokens': sum(r['known_target_tokens'] for r in history),
            'progress_sha256': file_hash(progress_path), 'student_sha256': file_hash(checkpoint_path),
            'settings_hash': digest(settings), 'dataset_hash': dataset['hash'],
            'frozen_benchmarks_verified': benchmarks, 'all_exported_weights_finite_and_match_progress': True,
            'samples': samples, 'sample_scope': 'three short smoke prompts, greedy decoding, no repetition guard; no capability score'}
  write_json(args.output / 'audit.json', result)
  print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', type=Path, default=Path('qos_ai_557m.json'))
  parser.add_argument('--experiment', type=Path, required=True)
  parser.add_argument('--data', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
  audit(parser.parse_args())
