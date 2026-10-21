"""Stage 2 on one GPU: continued 557M pretraining, and a gentler mixed SFT.

  pretrain  Resume the pilot's pretraining state (weights + AdamW moments) and continue on the same
            English/Python streams (a second pass, new random windows) with a fresh warmup and cosine
            schedule. The step count is frozen after a 50-step speed check so the run ends by
            --deadline-unix. Resumable from progress_stage2.pt. Publishes pretrained.pt.
  sft       From a domain checkpoint: all verified examples plus an equal number of training-split
            4B Q&A pairs (re-sampled each epoch), answer-only loss, dynamic padding. Publishes student.pt.

Both write result.json with the stage-1 loss probes before and after.
"""
import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import qos_ai as ai
from qos_ai_finetune import answer_loss, make_batch
from qos_ai_stage1_train import all_probes, file_hash, pilot_windows, publish, rate, status, write_json

WINDOW = 513


def streams_of(pretraining_data):
  manifest = json.loads((pretraining_data / 'dataset.json').read_text())
  return {kind: {split: np.memmap(pretraining_data / info['path'], mode='r', dtype=np.uint32)
                 for split, info in result['files'].items()} for kind, result in manifest['results'].items()}


def build_probes(args, streams):
  load = lambda name: np.load(args.stage1_data / f'{name}.npy', mmap_mode='r')
  return {'pilot_english_validation': (pilot_windows(streams['english']['validation']), None),
          'pilot_python_validation': (pilot_windows(streams['python']['validation']), None),
          'fineweb013_english': (np.load(args.english_probe), None),
          'dev_docs': (load('probe_docs'), None),
          'dev_qa_answers': (load('probe_qa_rows'), load('probe_qa_mask'))}


def draw(streams, seed, step, micro, batch):
  """The pilot's sampler (80% English / 20% Python windows), with a new seed for the second pass."""
  rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro]))
  stream = streams[str(rng.choice(['english', 'python'], p=[.8, .2]))]['train']
  starts = rng.integers(0, len(stream) - WINDOW, size=batch)
  return torch.from_numpy(np.stack([stream[s:s + WINDOW].astype(np.int64) for s in starts]))


def save_progress(path, model, opt, step, settings, history):
  tmp = path.with_name(path.name + '.partial')
  torch.save({'step': step, 'settings': settings, 'history': history, 'model_state': model.state_dict(),
              'optimizer_state': opt.state_dict()}, tmp)
  tmp.replace(path)


def pretrain(args):
  root = args.output; root.mkdir(parents=True, exist_ok=True)
  streams = streams_of(args.pretraining_data)
  vocabulary = torch.load(args.pretraining_data / 'vocabulary.pt', map_location='cpu', weights_only=True)
  progress = root / 'progress_stage2.pt'
  source = progress if progress.exists() else args.resume_from
  status(root, 'loading', source=str(source))
  saved = torch.load(source, map_location='cpu', weights_only=True)
  config = ai.StudentConfig(**(saved['settings']['config']))
  device = torch.device(args.device)
  model = ai.QosAi(config).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(.9, .95), weight_decay=.1,
                          fused=device.type == 'cuda')
  model.load_state_dict(saved['model_state']); opt.load_state_dict(saved['optimizer_state'])
  per_update = args.batch_size * args.accumulation * (WINDOW - 1)
  if source == progress:
    settings, step, history = saved['settings'], saved['step'], saved['history']
    if args.total_steps and args.total_steps != settings['total_steps']:
      # Shorten (or lengthen) a resumed schedule; the cosine continues from the current step.
      settings.setdefault('total_steps_overrides', []).append(
        {'at_step': step, 'from': settings['total_steps'], 'to': args.total_steps})
      settings['total_steps'] = args.total_steps
  else:
    settings = {'config': saved['settings']['config'], 'parent_progress': str(args.resume_from),
                'parent_step': saved['step'], 'seed': 20261020, 'batch_size': args.batch_size,
                'accumulation': args.accumulation, 'learning_rate': args.learning_rate,
                'warmup_steps': args.warmup_steps, 'target_tokens': args.target_tokens,
                'total_steps': math.ceil(args.target_tokens / per_update), 'frozen': False,
                'data': 'second pass over the pilot English/Python training streams, new random windows'}
    step, history = 0, []
  del saved; gc.collect()
  result_path = root / 'result.json'
  result = json.loads(result_path.read_text()) if result_path.exists() else {}
  probes = build_probes(args, streams)
  if 'parent_probes' not in result:
    status(root, 'probe_parent'); result['parent_probes'] = all_probes(model, probes); write_json(result_path, result)

  model.train(); started = time.perf_counter(); first = step; last_save = time.monotonic()
  while step < settings['total_steps']:
    if time.time() > args.deadline_unix:
      raise TimeoutError(f'Deadline reached before the frozen schedule finished (step {step})')
    step += 1
    lr = rate(settings['learning_rate'], step, settings['total_steps'], settings['warmup_steps'])
    for group in opt.param_groups:
      group['lr'] = lr
    batches = [draw(streams, settings['seed'], step, k, settings['batch_size']).to(device) for k in range(settings['accumulation'])]
    known = sum(int((b[:, 1:] != 0).sum()) for b in batches)
    numerator = 0.
    for b in batches:
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = model(b[:, :-1])
      loss = F.cross_entropy(logits.float().flatten(0, 1), b[:, 1:].flatten(), ignore_index=0, reduction='sum')
      if not torch.isfinite(loss).item():
        raise ValueError('Nonfinite training loss')
      numerator += loss.item(); (loss / known).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    speed = (step - first) * per_update / max(1e-6, time.perf_counter() - started)
    if not settings['frozen'] and step - first == 50:
      # Freeze the schedule: finish by the deadline with a 5% margin; never exceed the token target.
      fits = first + 50 + int(speed * max(0, args.deadline_unix - time.time()) * .95 / per_update)
      settings['total_steps'] = max(step + 1, min(settings['total_steps'], fits))
      settings['frozen'] = True; settings['measured_tokens_per_second'] = speed
      write_json(root / 'schedule.json', settings)
    if step % 100 == 0 or step <= first + 3:
      row = {'step': step, 'total_steps': settings['total_steps'], 'loss': numerator / known, 'learning_rate': lr,
             'gradient_norm': norm, 'tokens_per_second': speed}
      history.append(row); status(root, 'pretrain', **row)
    if step % 5000 == 0:
      with (root / 'validation_curve.jsonl').open('a') as f:
        f.write(json.dumps({'step': step, **{k: v for k, v in all_probes(model, {
          k: probes[k] for k in ('pilot_english_validation', 'fineweb013_english', 'dev_docs')}).items()}}) + '\n')
      model.train()
    if time.monotonic() - last_save > args.save_minutes * 60:
      status(root, 'save_progress', step=step); save_progress(progress, model, opt, step, settings, history)
      last_save = time.monotonic()
  parent = {'config': settings['config'], **vocabulary, 'step': settings['parent_step'] + step,
            'benchmark_ancestry': 'new random lineage; public-corpus semantic exposure unknown'}
  publish(root / 'pretrained.pt', model, parent, {'stage': 'continued_pretraining', **settings, 'accepted_steps': step,
                                                  'processed_tokens': step * per_update})
  result.update(settings=settings, accepted_steps=step, processed_tokens=step * per_update,
                pretrained_sha256=file_hash(root / 'pretrained.pt'), history=history,
                final_probes=all_probes(model, probes))
  write_json(result_path, result)
  status(root, 'save_final_progress'); save_progress(progress, model, opt, step, settings, history)
  status(root, 'complete', pretrained_sha256=result['pretrained_sha256'])
  del opt; gc.collect(); torch.cuda.empty_cache()


def sft(args):
  root = args.output; root.mkdir(parents=True, exist_ok=True)
  if (root / 'student.pt').exists():
    status(root, 'complete', note='already done'); return
  streams = streams_of(args.pretraining_data)
  probes = build_probes(args, streams)
  data = torch.load(args.data, map_location='cpu', weights_only=True)
  verified, pool, pad_id = data['verified'], data['qa_pool'], data['pad_id']
  device = torch.device(args.device)
  parent_sha = file_hash(args.load)
  model, parent = ai.load_student(args.load, device)
  result = {'parent': str(args.load), 'parent_sha256': parent_sha, 'data_hash': data['hash'],
            'parameters': ai.parameter_count(model)}
  result['parent_probes'] = all_probes(model, probes); write_json(root / 'result.json', result)
  replay = int(len(verified) * args.replay_ratio)
  plan = []
  for epoch in range(args.epochs):
    rng = random.Random(7 + epoch)
    examples = verified + rng.sample(pool, replay)
    rng.shuffle(examples)
    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]
    plan += [batches[i:i + args.accumulation] for i in range(0, len(batches), args.accumulation)]
  if args.max_steps:
    plan = plan[:args.max_steps]
  settings = {'epochs': args.epochs, 'learning_rate': args.learning_rate, 'warmup_steps': args.warmup_steps,
              'batch_size': args.batch_size, 'accumulation': args.accumulation, 'verified_examples': len(verified),
              'replay_per_epoch': replay, 'steps': len(plan), 'loss': 'answer-only; dynamic padding'}
  opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(.9, .95), weight_decay=.1,
                          fused=device.type == 'cuda')
  model.train(); history = []
  for index, group in enumerate(plan):
    step = index + 1; lr = rate(args.learning_rate, step, len(plan), args.warmup_steps)
    for g in opt.param_groups:
      g['lr'] = lr
    known = sum(sum(learn[1:]) for batch in group for _, learn in batch)
    numerator = 0.
    for batch in group:
      tokens, mask = make_batch(batch, pad_id, device)
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        loss, count = answer_loss(model, tokens, mask)
      if int(count) != sum(sum(learn[1:]) for _, learn in batch) or not torch.isfinite(loss).item():
        raise ValueError('SFT target count or finite loss check failed')
      numerator += loss.item(); (loss / known).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    history.append({'step': step, 'loss': numerator / known, 'learning_rate': lr, 'gradient_norm': norm})
    if step % 50 == 0 or step == len(plan):
      status(root, 'sft', **history[-1], total_steps=len(plan))
  publish(root / 'student.pt', model, parent, {'stage': 'mixed_sft', 'parent_sha256': parent_sha, **settings})
  result.update(settings=settings, history=history, student_sha256=file_hash(root / 'student.pt'),
                sft_probes=all_probes(model, probes))
  write_json(root / 'result.json', result)
  status(root, 'complete', student_sha256=result['student_sha256'])


def main():
  p = argparse.ArgumentParser(description=__doc__)
  s = p.add_subparsers(dest='command', required=True)
  for name in ('pretrain', 'sft'):
    c = s.add_parser(name)
    c.add_argument('--pretraining-data', type=Path, required=True)
    c.add_argument('--stage1-data', type=Path, required=True)
    c.add_argument('--english-probe', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--batch-size', type=int, default=8)
    c.add_argument('--device', default='cuda')
  pre = s.choices['pretrain']
  pre.add_argument('--resume-from', type=Path, required=True)
  pre.add_argument('--target-tokens', type=int, default=541_000_000)
  pre.add_argument('--learning-rate', type=float, default=1.5e-4)
  pre.add_argument('--warmup-steps', type=int, default=500)
  pre.add_argument('--accumulation', type=int, default=2)
  pre.add_argument('--save-minutes', type=float, default=45)
  pre.add_argument('--total-steps', type=int, default=0, help='override the frozen total when resuming')
  pre.add_argument('--deadline-unix', type=float, required=True)
  ft = s.choices['sft']
  ft.add_argument('--load', type=Path, required=True)
  ft.add_argument('--data', type=Path, required=True)
  ft.add_argument('--epochs', type=int, default=2)
  ft.add_argument('--learning-rate', type=float, default=2e-5)
  ft.add_argument('--warmup-steps', type=int, default=20)
  ft.add_argument('--accumulation', type=int, default=4)
  ft.add_argument('--replay-ratio', type=float, default=1.0)
  ft.add_argument('--max-steps', type=int, default=0)
  args = p.parse_args()
  torch.backends.cuda.matmul.allow_tf32 = True
  pretrain(args) if args.command == 'pretrain' else sft(args)


if __name__ == '__main__':
  main()
