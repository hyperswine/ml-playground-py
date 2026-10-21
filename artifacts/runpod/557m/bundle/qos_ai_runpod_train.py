"""Budgeted CUDA pretraining and verified SFT for our own 557M architecture.

No pretrained model weights are downloaded. The token-budget schedule is frozen
after a disposable throughput profile. Published model files retain the native
qos_ai format. The external controller releases the GPU at its wall deadline.
"""
import argparse
import gc
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import qos_ai as ai
from qos_ai_data import digest, file_hash, lexical_score, read_rows, rows_fingerprint
from qos_ai_finetune import answer_loss, make_batch, pad_examples
from qos_ai_resume import optimizer_plan
from qos_ai_runpod_data import FrozenTokenizer
from qos_ai_scale import read_config, rate, write_json


def status(root, phase, **fields):
  row = {'phase': phase, 'unix_time': time.time(), **fields}
  temporary = root / 'status.partial.json'; write_json(temporary, row)
  temporary.replace(root / 'status.json')
  print(json.dumps(row), flush=True)


def corpus(data, config):
  manifest = json.loads((data / 'dataset.json').read_text())
  if manifest['hash'] != digest({k: v for k, v in manifest.items() if k != 'hash'}):
    raise ValueError('Corrupt dataset manifest')
  streams = {}
  for kind, result in manifest['results'].items():
    streams[kind] = {}
    for split, info in result['files'].items():
      path = data / info['path']
      if file_hash(path) != info['sha256']:
        raise ValueError('Changed token stream')
      tokens = np.memmap(path, mode='r', dtype=np.uint32)
      if len(tokens) != result['tokens'][split] or int(tokens.max()) >= config.vocabulary_size:
        raise ValueError('Invalid token count or vocabulary')
      streams[kind][split] = tokens
  vocabulary = torch.load(data / 'vocabulary.pt', map_location='cpu', weights_only=True)
  if file_hash(data / 'vocabulary.pt') != manifest['vocabulary_sha256'] or len(vocabulary['qwen_ids']) != config.vocabulary_size:
    raise ValueError('Changed vocabulary mapping')
  return manifest, streams, vocabulary


def draw(streams, seed, step, micro, batch, length, split='train'):
  rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro]))
  kind = rng.choice(['english', 'python'], p=[.8, .2])
  stream = streams[kind][split]
  starts = rng.integers(0, len(stream) - length, size=batch)
  return torch.from_numpy(np.stack([stream[s:s + length + 1].astype(np.int64) for s in starts])), kind


def language_loss(model, batch, bf16):
  with torch.autocast(batch.device.type, dtype=torch.bfloat16, enabled=bf16):
    logits = model(batch[:, :-1])
    return F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten(), ignore_index=0, reduction='sum')


def pretrain_step(model, optimizer, streams, settings, step):
  batches = [draw(streams, settings['seed'], step, micro, settings['batch_size'], settings['sequence_length'])
             for micro in range(settings['accumulation'])]
  known = sum(int((b[:, 1:] != 0).sum()) for b, _ in batches)
  if not known:
    raise ValueError('No known targets')
  optimizer.zero_grad(set_to_none=True)
  lr = rate(settings['learning_rate'], step, settings['total_steps'], settings['warmup_steps'])
  for group in optimizer.param_groups:
    group['lr'] = lr
  numerator = 0.
  for batch, _ in batches:
    loss = language_loss(model, batch.to(next(model.parameters()).device), settings['bf16'])
    if not torch.isfinite(loss).item():
      raise ValueError('Nonfinite training loss')
    numerator += loss.item(); (loss / known).backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
  optimizer.step(); optimizer.zero_grad(set_to_none=True)
  tokens = sum(b[:, 1:].numel() for b, _ in batches)
  return {'step': step, 'loss': numerator / known, 'known_tokens': known, 'tokens': tokens,
          'source_tokens': {k: sum(b[:, 1:].numel() for b, kind in batches if kind == k) for k in streams},
          'learning_rate': lr, 'gradient_norm': norm}


@torch.no_grad()
def validate(model, streams):
  model.eval(); result = {}
  for kind in ('english', 'python'):
    stream = streams[kind]['validation']; rng = np.random.default_rng(1234)
    numerator = denominator = 0
    for _ in range(16):
      start = int(rng.integers(0, len(stream) - 513))
      batch = torch.from_numpy(stream[start:start + 513].astype(np.int64)[None, :])
      denominator += int((batch[:, 1:] != 0).sum())
      numerator += language_loss(model, batch.to(next(model.parameters()).device), True).item()
    value = numerator / denominator
    if not math.isfinite(value):
      raise ValueError('Nonfinite validation loss')
    result[kind] = {'cross_entropy': value, 'known_target_tokens': denominator, 'windows': 16}
  model.train()
  return result


def optimizer(model, lr):
  return torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.1,
                           fused=next(model.parameters()).is_cuda)


def atomic_save(path, payload):
  temporary = path.with_name(path.name + '.partial')
  torch.save(payload, temporary)
  probe = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
  if probe.get('step') != payload.get('step'):
    raise ValueError('Saved boundary verification failed')
  del probe
  temporary.replace(path)


def publish(path, model, vocabulary, step, lineage):
  for p in model.parameters():
    if not torch.isfinite(p).all().item():
      raise ValueError('Nonfinite weights; refusing publication')
  atomic_save(path, {'model_state': model.state_dict(), 'config': asdict(model.config), **vocabulary,
                     'step': step, 'split_hash': lineage.get('sft_split_hash', ''),
                     'benchmark_ancestry': 'new random lineage; public-corpus semantic exposure unknown',
                     'runpod_557m': lineage})


def save_progress(path, model, opt, settings, step, history):
  atomic_save(path, {'schema': 1, 'settings': settings, 'settings_hash': digest(settings), 'step': step,
                     'history': history, 'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(),
                     'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()})


def profile(config, streams, output):
  results = []
  for batch_size in (4, 8):
    model = opt = None
    try:
      torch.manual_seed(7); model = ai.QosAi(config).cuda(); opt = optimizer(model, 3e-4)
      settings = {'seed': 17, 'batch_size': batch_size, 'sequence_length': 512,
                  'accumulation': 1, 'learning_rate': 3e-4, 'total_steps': 10, 'warmup_steps': 2, 'bf16': True}
      pretrain_step(model, opt, streams, settings, 1)
      torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
      for step in (2, 3, 4, 5):
        pretrain_step(model, opt, streams, settings, step)
      torch.cuda.synchronize(); seconds = time.perf_counter() - started
      results.append({'batch_size': batch_size, 'tokens_per_second': batch_size * 512 * 4 / seconds,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'success': True})
    except torch.OutOfMemoryError:
      results.append({'batch_size': batch_size, 'success': False, 'reason': 'CUDA out of memory'})
    finally:
      del model, opt; gc.collect(); torch.cuda.empty_cache()
  write_json(output / 'profile.json', {'gpu': torch.cuda.get_device_name(), 'results': results})
  passed = [r for r in results if r['success']]
  if not passed:
    raise ValueError('No profiled batch size fits; stopping instead of renting a larger GPU')
  return max(passed, key=lambda r: r['tokens_per_second'])


def choose_steps(speed, seconds, tokens_per_step):
  if speed <= 0 or seconds < 60 or tokens_per_step <= 0:
    raise ValueError('Insufficient usable training budget')
  return max(1, min(500_000_000, int(speed * seconds * .85)) // tokens_per_step)


def pretrain(args, root, manifest, streams, vocabulary, config, budget):
  deadline = budget['created_unix'] + 6.5 * 3600
  plan_path = root / 'training_plan.json'; progress = root / 'progress_pretrain.pt'
  code = {p: file_hash(Path(p)) for p in ('qos_ai.py', 'qos_ai_runpod_train.py', 'qos_ai_runpod_data.py')}
  if plan_path.exists():
    settings = json.loads(plan_path.read_text())
    if settings['code_sha256'] != code or settings['dataset_hash'] != manifest['hash'] or settings['config'] != asdict(config):
      raise ValueError('Resume code/data/architecture differs')
  else:
    status(root, 'profile')
    best = profile(config, streams, root)
    batch_size = best['batch_size']; accumulation = 16 // batch_size
    steps = choose_steps(best['tokens_per_second'], deadline - time.time(), 8192)
    settings = {'config': asdict(config), 'seed': 7, 'batch_size': batch_size, 'sequence_length': 512,
                'accumulation': accumulation, 'learning_rate': 3e-4, 'warmup_steps': min(128, max(1, steps // 20)),
                'total_steps': steps, 'bf16': True, 'optimizer': 'fused AdamW', 'weight_decay': .1,
                'dataset_hash': manifest['hash'], 'code_sha256': code, 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name(), 'estimated_tokens_per_second': best['tokens_per_second'],
                'budget_created_unix': budget['created_unix'], 'pretrain_deadline_unix': deadline,
                'target_tokens': steps * 8192, 'scope': 'pretraining from random weights; no teacher embedding initialization'}
    write_json(plan_path, settings)
  torch.manual_seed(settings['seed']); model = ai.QosAi(config).cuda(); opt = optimizer(model, settings['learning_rate'])
  step = 0; history = []
  if progress.exists():
    saved = torch.load(progress, map_location='cpu', weights_only=True, mmap=True)
    if saved['settings_hash'] != digest(settings) or saved['settings'] != settings:
      raise ValueError('Resume optimizer schedule differs')
    model.load_state_dict(saved['model_state']); opt.load_state_dict(saved['optimizer_state'])
    torch.set_rng_state(saved['cpu_rng_state']); torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    step = saved['step']; history = saved['history']; del saved
    write_json(root / f'resume_{int(time.time())}.json', {'saved_step': step, 'checkpoint_sha256': file_hash(progress)})
  if not (root / 'initial_validation.json').exists():
    write_json(root / 'initial_validation.json', validate(model, streams))
  start = time.perf_counter(); first = step; last_save = time.monotonic(); last_validate_tokens = step * 8192
  milestones = (25_000_000, 100_000_000)
  while step < settings['total_steps'] and time.time() < deadline - 60:
    row = pretrain_step(model, opt, streams, settings, step + 1); step += 1
    row['elapsed_this_process_seconds'] = time.perf_counter() - start
    history.append(row)
    if step <= first + 3 or step % 100 == 0:
      status(root, 'pretrain', **row, total_steps=settings['total_steps'], target_tokens=settings['target_tokens'],
             processed_tokens=step * 8192, tokens_per_second=(step - first) * 8192 / max(1e-6, time.perf_counter() - start))
    if step * 8192 - last_validate_tokens >= 10_000_000:
      result = {'step': step, 'tokens': step * 8192, 'validation': validate(model, streams)}
      with (root / 'validation_curve.jsonl').open('a') as out:
        out.write(json.dumps(result) + '\n')
      last_validate_tokens = step * 8192
    milestone = next((n for n in milestones if (step - 1) * 8192 < n <= step * 8192), None)
    if milestone:
      publish(root / f'pretrain_{milestone // 1_000_000}m.pt', model, vocabulary, step, settings)
    if step == first + 20 or time.monotonic() - last_save >= 900 or milestone:
      status(root, 'save_pretrain', step=step, processed_tokens=step * 8192)
      save_progress(progress, model, opt, settings, step, history); last_save = time.monotonic()
  save_progress(progress, model, opt, settings, step, history)
  lineage = {**settings, 'accepted_steps': step, 'processed_tokens': step * 8192,
             'budget_stop': step != settings['total_steps']}
  publish(root / 'pretrained.pt', model, vocabulary, step, lineage)
  result = {**lineage, 'validation': validate(model, streams), 'checkpoint_sha256': file_hash(root / 'pretrained.pt'),
            'progress_sha256': file_hash(progress), 'known_tokens': sum(r['known_tokens'] for r in history),
            'source_tokens': {k: sum(r['source_tokens'][k] for r in history) for k in streams}}
  write_json(root / 'pretraining_result.json', result)
  with (root / 'accepted_pretraining_updates.jsonl').open('w') as out:
    for row in history:
      out.write(json.dumps(row) + '\n')
  del opt; gc.collect(); torch.cuda.empty_cache()
  return model, result


def sft(model, vocabulary, root, args, lineage):
  data = torch.load(args.sft, map_location='cpu', weights_only=True)
  examples = data['examples']
  if digest(examples) != data['encoded_hash'] or data['stats']['unknown_answer_tokens_skipped']:
    raise ValueError('Verified SFT encoding changed or contains unknown answers')
  examples = pad_examples(examples, vocabulary['pad_id'], 192)
  plan = optimizer_plan(len(examples), 8, 4, 2, 7)
  settings = {'parent_sha256': file_hash(root / 'pretrained.pt'), 'sft_data_sha256': file_hash(args.sft),
              'encoded_hash': data['encoded_hash'], 'plan_hash': digest(plan), 'steps': len(plan),
              'epochs': 2, 'learning_rate': 5e-5, 'warmup_steps': 8, 'batch_size': 8, 'accumulation': 4,
              'fixed_padding': 192, 'bf16': True, 'loss': 'answer-only known-token weighted cross entropy',
              'split_hash': data['split_hash']}
  if (root / 'sft_result.json').exists() and (root / 'student.pt').exists():
    prior = json.loads((root / 'sft_result.json').read_text())
    if any(prior[k] != v for k, v in settings.items()) or prior['checkpoint_sha256'] != file_hash(root / 'student.pt'):
      raise ValueError('Completed SFT identity differs')
    saved = torch.load(root / 'student.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model_state'])
    return
  write_json(root / 'sft_settings.json', settings)
  opt = optimizer(model, 5e-5)
  # SFT gets its own optimizer by design. A failed SFT starts again from the
  # immutable published pretraining artifact; it never partially repeats updates.
  model.train(); history = []
  for index, group in enumerate(plan):
    if time.time() > args.created_unix + 7.25 * 3600:
      raise TimeoutError('SFT exceeded reserved wall budget; pretrained checkpoint is preserved')
    step = index + 1; opt.zero_grad(set_to_none=True)
    known = sum(sum(examples[i][1][1:]) for batch in group['batches'] for i in batch)
    lr = rate(5e-5, step, len(plan), 8)
    for parameter_group in opt.param_groups:
      parameter_group['lr'] = lr
    numerator = 0.
    for indices in group['batches']:
      batch, mask = make_batch([examples[i] for i in indices], vocabulary['pad_id'], 'cuda')
      with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, count = answer_loss(model, batch, mask)
      if int(count) != sum(sum(examples[i][1][1:]) for i in indices) or not torch.isfinite(loss).item():
        raise ValueError('SFT target count or finite loss check failed')
      numerator += loss.item(); (loss / known).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    history.append({'step': step, 'known_answer_tokens': known, 'loss': numerator / known,
                    'learning_rate': lr, 'gradient_norm': norm})
    if step == 1 or step % 25 == 0:
      status(root, 'verified_sft', **history[-1], total_steps=len(plan))
  final = {**lineage, 'verified_sft': settings, 'sft_split_hash': data['split_hash'],
           'sft_answer_tokens': sum(r['known_answer_tokens'] for r in history)}
  publish(root / 'student.pt', model, vocabulary, lineage['accepted_steps'] + len(plan), final)
  write_json(root / 'sft_result.json', {**settings, 'answer_tokens': final['sft_answer_tokens'],
                                       'checkpoint_sha256': file_hash(root / 'student.pt'), 'history': history})
  del opt; gc.collect(); torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(model, vocabulary, root, args, label):
  model.eval(); tok = ai.StudentTokenizer(vocabulary['qwen_ids'], FrozenTokenizer(args.tokenizer))
  for name, filename, budget in [('development', 'benchmark.jsonl', 200),
                                 ('coding', 'coding_diagnostic.jsonl', 128),
                                 ('generalization', 'generalization_diagnostic.jsonl', 128)]:
    path = args.benchmarks / filename
    rows = read_rows(path)
    if name == 'development':
      rows = [r for r in rows if r['split'] == 'dev']
    output = root / 'evaluation' / label / name
    settings = {'checkpoint_sha256': file_hash(root / ('pretrained.pt' if label == 'pretrained' else 'student.pt')),
                'benchmark_sha256': file_hash(path), 'seed': 7, 'temperature': 0, 'top_k': 20, 'no_repeat': 3,
                'diagnostic_hash': rows_fingerprint(rows), 'diagnostic_sha256': file_hash(path),
                'max_new_tokens': budget, 'device': 'cuda', 'inference_precision': 'fp32',
                'generation_model_code_sha256': file_hash(Path(ai.__file__)),
                'evaluator_sha256': file_hash(Path(__file__)), 'count': len(rows),
                'scope': 'same frozen development prompts and decoding; CUDA generations graded on M1'}
    if (output / 'summary.json').exists():
      prior = json.loads((output / 'summary.json').read_text())
      if all(prior[k] == v for k, v in settings.items()) and len(read_rows(output / 'answers.jsonl')) == len(rows):
        continue
      raise ValueError('Completed evaluation identity differs')
    if output.exists():
      output.rename(output.with_name(output.name + '.incomplete_' + str(int(time.time()))))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'settings.json', settings); answers = []
    for start in range(0, len(rows), 8):
      part = rows[start:start + 8]; prompts = [tok.encode(ai.chat_text(r['question'])) for r in part]
      if max(map(len, prompts)) + budget > model.config.context_length or any(0 in p for p in prompts):
        raise ValueError('Evaluation prompt truncation or unknown token')
      generated = model.generate_batch(prompts, budget, vocabulary['im_end_id'], 0, 20, 3)
      for row, (ids, finished) in zip(part, generated):
        answer = tok.decode(ids).strip()
        record = {'id': row['id'], 'question': row['question'], 'answer': answer, 'finished': finished,
                  'tokens': len(ids), 'severe_loop': ai.loops(ids, phrase=4, times=4), 'effective_token_budget': budget}
        record.update({k: row[k] for k in ('category', 'family', 'mode', 'stratum', 'track', 'group', 'split') if k in row})
        if name == 'development':
          record.update(lexical_score(row, answer))
        answers.append(record)
        with (output / 'answers.jsonl').open('a') as out:
          out.write(json.dumps(record) + '\n')
      status(root, 'evaluate', model=label, track=name, completed=len(answers), total=len(rows))
    summary = {**settings, 'finished': sum(a['finished'] for a in answers),
               'severe_loops': sum(a['severe_loop'] for a in answers),
               'lexical_proxy_pass': sum(a.get('rubric_proxy_pass', False) for a in answers)}
    write_json(output / 'summary.json', summary)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('config', 'data', 'sft', 'tokenizer', 'benchmarks', 'output', 'budget'):
    p.add_argument('--' + name, type=Path, required=True)
  args = p.parse_args(); root = args.output; root.mkdir(parents=True, exist_ok=True)
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise ValueError('A working bf16 CUDA device is required')
  torch.backends.cuda.matmul.allow_tf32 = True
  budget = json.loads(args.budget.read_text()); args.created_unix = budget['created_unix']
  config = read_config(args.config)
  with torch.device('meta'):
    if ai.parameter_count(ai.QosAi(config)) != 556526080:
      raise ValueError('Wrong architecture')
  manifest, streams, vocabulary = corpus(args.data, config)
  status(root, 'start', gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), host=platform.node())
  model, result = pretrain(args, root, manifest, streams, vocabulary, config, budget)
  evaluate(model, vocabulary, root, args, 'pretrained')
  sft(model, vocabulary, root, args, result)
  evaluate(model, vocabulary, root, args, 'sft')
  status(root, 'complete', pretrained_sha256=file_hash(root / 'pretrained.pt'), student_sha256=file_hash(root / 'student.pt'),
         processed_pretraining_tokens=result['processed_tokens'], budget_stop=result['budget_stop'],
         message='Weights and raw development outputs ready; GPU release controlled externally')


if __name__ == '__main__':
  try:
    main()
  except Exception as error:
    import sys
    if '--output' in sys.argv:
      directory = Path(sys.argv[sys.argv.index('--output') + 1])
      if directory.exists():
        status(directory, 'failed', error=type(error).__name__ + ': ' + str(error)[:1000])
    raise
