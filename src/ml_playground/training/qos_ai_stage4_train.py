"""Resumable short Stage 4 SFT on a prepared token-balanced mixture.

An absolute deadline saves progress and exits; it does not stop a paid pod.
The operator must supply an independently tested external shutdown controller.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import qos_ai as ai
from qos_ai_finetune import answer_loss, make_batch
from qos_ai_resume import optimizer_plan, save_progress, load_progress
from qos_ai_stage1_train import publish, rate, write_json
from qos_ai_stage4_data import checked_load, targets
from qos_ai_stage4_eval import sha
from qos_ai_stage4_recovery import GIB, require_space

PARENT = '5879f9c385158e1b679e8fda9c6fc53f8f15919e109d25c05c7a332f5d6df730'
VOCABULARY = '74138be221701ca868b413cf711642fc2abde19687ff7f3af37de475f817dcf7'
VARIANTS = {'half_2e5': (.5, 2e-5), 'one_2e5': (1., 2e-5), 'one_1e5': (1., 1e-5)}


def plan_for(count, variant, batch_size=8, accumulation=4):
  if count <= 0 or batch_size <= 0 or accumulation <= 0: raise ValueError('positive batch sizes/count required')
  fraction, _ = VARIANTS[variant]
  plan = optimizer_plan(count, batch_size, accumulation, 1, 7)
  return plan[:math.ceil(len(plan) * fraction)]


def train_updates(model, optimizer, examples, plan, pad_id, lr, warmup, start=0, after_step=None):
  """Token-weight accumulated gradients, including a short final microbatch."""
  device = next(model.parameters()).device; history = []
  model.train(); optimizer.zero_grad(set_to_none=True)
  for index in range(start, len(plan)):
    step = index + 1; group = plan[index]['batches']
    learning_rate = rate(lr, step, len(plan), warmup)
    for g in optimizer.param_groups: g['lr'] = learning_rate
    known = sum(targets(examples[i]) for batch in group for i in batch)
    numerator = 0.
    for indexes in group:
      batch = [examples[i] for i in indexes]
      tokens, mask = make_batch(batch, pad_id, device)
      with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        loss, count = answer_loss(model, tokens, mask)
      if int(count) != sum(targets(e) for e in batch) or not torch.isfinite(loss).item():
        raise ValueError('nonfinite loss or target count mismatch')
      numerator += loss.item(); (loss / known).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    row = {'step': step, 'loss': numerator / known, 'learning_rate': learning_rate,
           'gradient_norm': norm, 'target_tokens': known}
    history.append(row)
    if after_step is not None and after_step(row): break
  return history


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--load', type=Path, required=True)
  p.add_argument('--data', type=Path, required=True)
  p.add_argument('--data-sha256', required=True)
  p.add_argument('--variant', choices=VARIANTS, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--deadline-unix', type=float, required=True)
  p.add_argument('--device', default='cuda', choices=['cpu', 'mps', 'cuda'])
  p.add_argument('--batch-size', type=int, default=8)
  p.add_argument('--accumulation', type=int, default=4)
  p.add_argument('--save-every', type=int, default=100)
  args = p.parse_args()
  if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() + 300 or args.save_every <= 0:
    p.error('provide at least five minutes before the deadline and a positive save interval')
  if sha(args.load) != PARENT or sha(args.data) != args.data_sha256:
    raise ValueError('parent or prepared data identity mismatch')
  data = checked_load(args.data)
  if data['qa_pool']: raise ValueError('use the fully prepared mixture, with no unsampled QA pool')
  if data['vocabulary_sha256'] != VOCABULARY:
    raise ValueError('prepared mixture vocabulary differs from the fixed parent')
  examples = data['verified']; counts = [targets(e) for e in examples]
  plan = plan_for(len(examples), args.variant, args.batch_size, args.accumulation)
  args.output.mkdir(parents=True, exist_ok=True)
  if (args.output / 'student.pt').exists(): raise ValueError('completed output exists; do not overwrite')
  # 557M FP32 weights plus Adam moments need about 6.7 GB; allow 9 GiB for a replacement.
  require_space(args.output, 9 * GIB)
  _, lr = VARIANTS[args.variant]
  source_files = [Path(__file__), Path(ai.__file__), Path('qos_ai_resume.py'), Path('qos_ai_finetune.py'),
                  Path('qos_ai_stage1_train.py'), Path('qos_ai_stage4_data.py'), Path('qos_ai_stage4_recovery.py')]
  settings = {'parent_sha256': PARENT, 'data_hash': data['hash'], 'data_sha256': args.data_sha256,
              'variant': args.variant, 'learning_rate': lr, 'warmup_steps': 20, 'seed': 7,
              'batch_size': args.batch_size, 'accumulation': args.accumulation, 'device': args.device,
              'steps': len(plan), 'planned_target_tokens': sum(counts[i] for s in plan for b in s['batches'] for i in b),
              'source_hashes': {p.name: sha(p) for p in source_files}}
  settings_path = args.output / 'settings.json'
  if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
    raise ValueError('existing output settings differ; use a new experiment directory')
  torch.manual_seed(7)
  device = torch.device(args.device); model, parent = ai.load_student(args.load, device)
  if data['pad_id'] != parent['pad_id']:
    raise ValueError('prepared mixture padding identity differs from parent')
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.1,
                               fused=device.type == 'cuda')
  progress_path = args.output / 'progress.pt'; history = []; start = 0
  if progress_path.exists():
    progress = load_progress(progress_path, model, optimizer, settings)
    start, history = progress['step'], progress['history']
    if not 0 <= start <= len(plan) or len(history) != start or any(r['step'] != i + 1 for i, r in enumerate(history)):
      raise ValueError('invalid saved progress step/history')
  elif any(args.output.glob('*.partial')):
    raise ValueError('partial artifact without valid progress; inspect before restarting')
  write_json(settings_path, settings)
  def after(row):
    history.append(row)
    # Leave five minutes for saving and controller shutdown. Also check after model loading.
    stopping = time.time() >= args.deadline_unix - 300
    if row['step'] % args.save_every == 0 or row['step'] == len(plan) or stopping:
      require_space(args.output, 9 * GIB)
      save_progress(progress_path, model, optimizer, settings, {'step': row['step'], 'history': history})
      write_json(args.output / 'status.json', {'phase': 'paused' if stopping else 'training', **row})
      print(json.dumps(row), flush=True)
    return stopping
  if time.time() >= args.deadline_unix - 300:
    raise TimeoutError('deadline too close after loading; no updates started')
  train_updates(model, optimizer, examples, plan, data['pad_id'], lr, 20, start, after)
  if history and history[-1]['step'] == len(plan):
    require_space(args.output, 3 * GIB)
    publish(args.output / 'student.pt', model, parent, {'stage': 'stage4_sft', **settings})
    write_json(args.output / 'result.json', {'settings': settings, 'history': history,
               'student_sha256': sha(args.output / 'student.pt'), 'complete': True,
               'evaluation': 'pending separate frozen loss probes and generation benchmarks'})
    write_json(args.output / 'status.json', {'phase': 'complete', 'step': len(plan)})
  else:
    print('Saved an incomplete run. Resume with the same settings and a new deadline.', flush=True)


if __name__ == '__main__': main()
