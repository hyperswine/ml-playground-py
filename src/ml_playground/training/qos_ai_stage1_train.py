"""Stage 1 on one GPU: domain stage, then the pilot's verified SFT, with loss probes before and after.

For one parent checkpoint (the 557M pretraining parent, or the 168M control):
  1. probes on the parent
  2. domain stage: packed training-partition docs + 4B Q&A rows (answer-only loss on Q&A),
     with a fixed share of each batch replayed from the parent's English/Python streams
  3. probes, publish domain.pt
  4. the pilot's SFT treatment unchanged (2 epochs over verified_sft.pt, lr 5e-5, batch 8 x 4)
  5. probes, publish student.pt

Probes: the 557M pilot's fixed English/Python validation windows, the FineWeb file-013 English
probe, dev-partition doc windows and dev-partition Q&A answers. Generation benchmarks run later
on the M1 with qos_ai_noguard.py. Resumable from progress_domain.pt.
"""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import qos_ai as ai
from qos_ai_finetune import answer_loss, make_batch, pad_examples
from qos_ai_resume import optimizer_plan

WINDOW = 513


def file_hash(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 24), b''):
      h.update(block)
  return h.hexdigest()


def write_json(path, value):
  tmp = path.with_name(path.name + '.partial'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def status(root, phase, **fields):
  row = {'phase': phase, 'unix_time': time.time(), **fields}
  write_json(root / 'status.json', row); print(json.dumps(row), flush=True)


def rate(base, step, total, warmup):
  if step <= warmup:
    return base * step / max(1, warmup)
  progress = (step - warmup) / max(1, total - warmup)
  return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))


def masked_loss(model, rows, mask):
  """Summed cross entropy over targets where mask is set (mask aligned to tokens; target flag = mask[:, 1:])."""
  with torch.autocast(rows.device.type, dtype=torch.bfloat16, enabled=rows.device.type == 'cuda'):
    logits = model(rows[:, :-1])
  losses = F.cross_entropy(logits.float().flatten(0, 1), rows[:, 1:].flatten(), reduction='none').view(rows.shape[0], -1)
  target = mask[:, 1:].float()
  return (losses * target).sum(), target.sum()


@torch.no_grad()
def probe(model, rows, mask=None, batch=8):
  model.eval(); device = next(model.parameters()).device
  total = count = 0.
  for start in range(0, len(rows), batch):
    r = torch.as_tensor(rows[start:start + batch].astype(np.int64), device=device)
    m = (r != 0) if mask is None else torch.as_tensor(mask[start:start + batch].astype(bool), device=device)
    loss, n = masked_loss(model, r, m)
    total += loss.item(); count += n.item()
  model.train()
  return {'cross_entropy': total / count, 'perplexity': math.exp(total / count), 'target_tokens': int(count)}


def pilot_windows(stream, seed=1234):
  """The 557M pilot's validate(): 16 windows drawn with rng 1234 (pad separators are real targets there)."""
  rng = np.random.default_rng(seed)
  return np.stack([stream[(s := int(rng.integers(0, len(stream) - 513))):s + 513] for _ in range(16)])


def all_probes(model, probes):
  return {name: probe(model, rows, mask) for name, (rows, mask) in probes.items()}


def save_progress(path, model, opt, step, settings):
  tmp = path.with_name(path.name + '.partial')
  torch.save({'step': step, 'settings': settings, 'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(),
              'cpu_rng_state': torch.get_rng_state()}, tmp)
  tmp.replace(path)


def publish(path, model, parent, extra):
  for p in model.parameters():
    if not torch.isfinite(p).all().item():
      raise ValueError('Nonfinite weights; refusing to publish')
  tmp = path.with_name(path.name + '.partial')
  torch.save({'model_state': model.state_dict(), 'config': parent['config'], 'qwen_ids': parent['qwen_ids'],
              'pad_id': parent['pad_id'], 'im_end_id': parent['im_end_id'], 'step': parent.get('step', 0),
              'benchmark_ancestry': parent.get('benchmark_ancestry', 'legacy_or_unknown'), 'stage1': extra}, tmp)
  tmp.replace(path)


def domain_stage(model, args, data, replay, root, parent_sha):
  rows, masks = data['rows'], data['masks']
  per_update = args.batch_size * args.accumulation
  domain_per_micro = args.batch_size - round(args.batch_size * args.replay_fraction)
  replay_per_micro = args.batch_size - domain_per_micro
  total_steps = math.ceil(len(rows) / (domain_per_micro * args.accumulation))
  if args.max_steps:
    total_steps = min(total_steps, args.max_steps)
  settings = {'parent_sha256': parent_sha, 'data_hash': data['meta']['hash'], 'learning_rate': args.learning_rate,
              'warmup_steps': args.warmup_steps, 'batch_size': args.batch_size, 'accumulation': args.accumulation,
              'replay_fraction': replay_per_micro / args.batch_size, 'replay_mix': {'english': .8, 'python': .2},
              'total_steps': total_steps, 'tokens_per_update': per_update * (WINDOW - 1), 'weight_decay': .1,
              'optimizer': 'AdamW betas (0.9, 0.95), fresh state', 'bf16': True}
  opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(.9, .95), weight_decay=.1,
                          fused=next(model.parameters()).is_cuda)
  progress = root / 'progress_domain.pt'; step = 0
  if progress.exists():
    saved = torch.load(progress, map_location='cpu', weights_only=True)
    if saved['settings'] != settings:
      raise ValueError('Resume settings differ')
    model.load_state_dict(saved['model_state']); opt.load_state_dict(saved['optimizer_state'])
    torch.set_rng_state(saved['cpu_rng_state']); step = saved['step']; del saved
  write_json(root / 'domain_settings.json', settings)
  device = next(model.parameters()).device
  history_path = root / 'domain_updates.jsonl'
  started = time.perf_counter(); first = step; last_save = time.monotonic()
  model.train()
  while step < total_steps:
    if time.time() > args.deadline_unix:
      save_progress(progress, model, opt, step, settings)
      raise TimeoutError(f'Domain stage deadline reached at step {step}; progress saved')
    step += 1
    lr = rate(args.learning_rate, step, total_steps, args.warmup_steps)
    for group in opt.param_groups:
      group['lr'] = lr
    micro = []
    for k in range(args.accumulation):
      start = ((step - 1) * args.accumulation + k) * domain_per_micro
      idx = np.arange(start, start + domain_per_micro) % len(rows)
      r, m = rows[idx].astype(np.int64), masks[idx].astype(bool)
      if replay_per_micro:
        rng = np.random.default_rng(np.random.SeedSequence([20261019, step, k]))
        extra = []
        for _ in range(replay_per_micro):
          stream = replay['english' if rng.random() < .8 else 'python']
          s = int(rng.integers(0, len(stream) - WINDOW)); extra.append(stream[s:s + WINDOW].astype(np.int64))
        extra = np.stack(extra)
        r = np.concatenate([r, extra]); m = np.concatenate([m, extra != 0])
      micro.append((torch.from_numpy(r).to(device), torch.from_numpy(m).to(device)))
    known = sum(int(m[:, 1:].sum()) for _, m in micro)
    numerator = 0.
    for r, m in micro:
      loss, _ = masked_loss(model, r, m)
      if not torch.isfinite(loss).item():
        raise ValueError('Nonfinite training loss')
      numerator += loss.item(); (loss / known).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    if step <= first + 3 or step % 100 == 0 or step == total_steps:
      elapsed = time.perf_counter() - started
      row = {'step': step, 'total_steps': total_steps, 'loss': numerator / known, 'learning_rate': lr, 'gradient_norm': norm,
             'tokens_per_second': (step - first) * per_update * (WINDOW - 1) / max(elapsed, 1e-6)}
      with history_path.open('a') as f:
        f.write(json.dumps(row) + '\n')
      status(root, 'domain', **row)
    if time.monotonic() - last_save > args.save_minutes * 60:
      save_progress(progress, model, opt, step, settings); last_save = time.monotonic()
  del opt; gc.collect(); torch.cuda.empty_cache()
  progress.unlink(missing_ok=True)
  return settings


def sft(model, args, pad_id, root):
  data = torch.load(args.sft, map_location='cpu', weights_only=True)
  examples = pad_examples(data['examples'], pad_id, 192)
  plan = optimizer_plan(len(examples), 8, 4, 2, 7)
  if args.max_steps:
    plan = plan[:max(1, args.max_steps // 10)]
  settings = {'sft_data_sha256': file_hash(args.sft), 'encoded_hash': data['encoded_hash'], 'steps': len(plan),
              'epochs': 2, 'learning_rate': 5e-5, 'warmup_steps': 8, 'batch_size': 8, 'accumulation': 4,
              'fixed_padding': 192, 'loss': 'answer-only, identical to the 557M pilot SFT'}
  device = next(model.parameters()).device
  opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9, .95), weight_decay=.1, fused=device.type == 'cuda')
  model.train(); history = []
  for index, group in enumerate(plan):
    step = index + 1; lr = rate(5e-5, step, len(plan), 8)
    for g in opt.param_groups:
      g['lr'] = lr
    known = sum(sum(examples[i][1][1:]) for batch in group['batches'] for i in batch)
    numerator = 0.
    for indices in group['batches']:
      batch, mask = make_batch([examples[i] for i in indices], pad_id, device)
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        loss, count = answer_loss(model, batch, mask)
      if int(count) != sum(sum(examples[i][1][1:]) for i in indices) or not torch.isfinite(loss).item():
        raise ValueError('SFT target count or finite loss check failed')
      numerator += loss.item(); (loss / known).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    history.append({'step': step, 'loss': numerator / known, 'learning_rate': lr, 'gradient_norm': norm})
  status(root, 'sft_done', steps=len(plan), final_loss=history[-1]['loss'])
  del opt; gc.collect(); torch.cuda.empty_cache()
  return {**settings, 'history': history}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--load', type=Path, required=True)
  p.add_argument('--data', type=Path, required=True, help='qos_ai_stage1_data.py output')
  p.add_argument('--pretraining-data', type=Path, required=True, help="the 557M pilot's pretraining_data directory")
  p.add_argument('--english-probe', type=Path, required=True)
  p.add_argument('--sft', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--learning-rate', type=float, default=1e-4)
  p.add_argument('--warmup-steps', type=int, default=100)
  p.add_argument('--batch-size', type=int, default=8)
  p.add_argument('--accumulation', type=int, default=2)
  p.add_argument('--replay-fraction', type=float, default=.25)
  p.add_argument('--save-minutes', type=float, default=20)
  p.add_argument('--deadline-unix', type=float, default=float('inf'))
  p.add_argument('--max-steps', type=int, default=0, help='smoke test only')
  p.add_argument('--device', default='cuda')
  args = p.parse_args()
  root = args.output; root.mkdir(parents=True, exist_ok=True)
  if args.device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
  device = torch.device(args.device)

  meta = json.loads((args.data / 'stage1_data.json').read_text())
  load = lambda name: np.load(args.data / f'{name}.npy', mmap_mode='r')
  data = {'rows': load('domain_rows'), 'masks': load('domain_mask'), 'meta': meta}
  manifest = json.loads((args.pretraining_data / 'dataset.json').read_text())
  streams = {kind: {split: np.memmap(args.pretraining_data / info['path'], mode='r', dtype=np.uint32)
                    for split, info in result['files'].items()} for kind, result in manifest['results'].items()}
  replay = {kind: streams[kind]['train'] for kind in ('english', 'python')}
  english_probe = np.load(args.english_probe)
  probes = {'pilot_english_validation': (pilot_windows(streams['english']['validation']), None),
            'pilot_python_validation': (pilot_windows(streams['python']['validation']), None),
            'fineweb013_english': (english_probe, None),
            'dev_docs': (load('probe_docs'), None),
            'dev_qa_answers': (load('probe_qa_rows'), load('probe_qa_mask'))}

  parent_sha = file_hash(args.load)
  model, parent = ai.load_student(args.load, device)
  if parent['qwen_ids'].shape[0] != 65536 or parent['pad_id'] != meta['pad_id']:
    raise ValueError('Vocabulary mapping differs from the stage-1 data')
  result_path = root / 'result.json'
  result = json.loads(result_path.read_text()) if result_path.exists() else {
    'parent': str(args.load), 'parent_sha256': parent_sha, 'parameters': ai.parameter_count(model), 'data': meta['hash']}
  if 'parent_probes' not in result:
    status(root, 'probe_parent'); result['parent_probes'] = all_probes(model, probes); write_json(result_path, result)

  if not (root / 'domain.pt').exists():
    result['domain_settings'] = domain_stage(model, args, data, replay, root, parent_sha)
    publish(root / 'domain.pt', model, parent, {'stage': 'domain', 'parent_sha256': parent_sha, **result['domain_settings']})
    result['domain_sha256'] = file_hash(root / 'domain.pt')
    result['domain_probes'] = all_probes(model, probes); write_json(result_path, result)
  else:
    model.load_state_dict(torch.load(root / 'domain.pt', map_location='cpu', weights_only=True)['model_state'])

  if not (root / 'student.pt').exists():
    result['sft'] = sft(model, args, parent['pad_id'], root)
    publish(root / 'student.pt', model, parent, {'stage': 'sft', 'parent_sha256': result['domain_sha256'],
                                                  **{k: v for k, v in result['sft'].items() if k != 'history'}})
    result['student_sha256'] = file_hash(root / 'student.pt')
    result['sft_probes'] = all_probes(model, probes); write_json(result_path, result)
  status(root, 'complete', domain_sha256=result['domain_sha256'], student_sha256=result['student_sha256'])


if __name__ == '__main__':
  main()
