"""Short supervised fine-tuning on verified, training-partition examples only."""
import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import qos_ai as ai
from qos_ai_data import digest, file_hash, read_rows, row_split, rows_fingerprint, source_check, validate_manifest


def verified_rows(data, manifest, roots, canonical_file=None):
  canonical_file = canonical_file or ai.VERIFIED_FILE
  canonical = read_rows(canonical_file)
  benchmark = read_rows(ai.BENCHMARK_FILE)
  validate_manifest(manifest, verified=canonical, benchmark=benchmark)
  proof = json.loads((canonical_file.parent / 'verification.json').read_text())
  if proof.get('verified_data_hash') != rows_fingerprint(canonical):
    raise ValueError('Verified teaching evidence is stale')
  expected = [r for r in canonical if row_split(r, manifest) == 'train']
  rows = read_rows(data)
  if not rows or rows_fingerprint(rows) != rows_fingerprint(expected):
    raise ValueError('Training input must be exactly the prepared verified training partition')
  passed = {r['id'] for r in proof['results'] if r['passed']}
  for row in rows:
    source_check(row, roots)
    if row.get('program') and row['id'] not in passed:
      raise ValueError(f'No passing reference execution: {row["id"]}')
    if row.get('debug_program') and row['id'] + ':broken' not in passed:
      raise ValueError(f'No passing broken-program verification: {row["id"]}')
  return rows


def encode_examples(rows, tokenizer, context_length):
  examples = []
  unknown, total = 0, 0
  for row in rows:
    prompt = tokenizer.encode(ai.chat_text(row['question']))
    tokens = tokenizer.encode(ai.chat_text(row['question'], row['answer']))
    if len(tokens) > context_length:
      raise ValueError(f'Complete training example exceeds model context: {row["id"]}')
    mask = [i >= len(prompt) and token != 0 for i, token in enumerate(tokens)]
    unknown += sum(token == 0 for token in tokens[len(prompt):])
    total += len(tokens) - len(prompt)
    if not any(mask[1:]):
      raise ValueError('Training example has no known answer tokens')
    examples.append((tokens, mask))
  return examples, {'answer_tokens': total, 'unknown_answer_tokens_skipped': unknown,
                    'longest_example': max(len(e[0]) for e in examples)}


def make_batch(examples, pad_id, device):
  width = max(len(tokens) for tokens, _ in examples)
  tokens = torch.full((len(examples), width), pad_id, dtype=torch.long)
  mask = torch.zeros((len(examples), width), dtype=torch.bool)
  for i, (ids, learn) in enumerate(examples):
    tokens[i, :len(ids)] = torch.tensor(ids)
    mask[i, :len(ids)] = torch.tensor(learn)
  # Right padding cannot affect earlier real tokens under causal attention.
  return tokens.to(device), mask[:, 1:].to(device)


def answer_loss(model, tokens, mask):
  logits = model(tokens[:, :-1]).float()
  losses = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction='none').view_as(mask)
  return (losses * mask).sum(), mask.sum()


@torch.no_grad()
def training_loss(model, examples, batch_size, pad_id, device):
  model.eval()
  numerator, denominator = 0.0, 0
  for start in range(0, len(examples), batch_size):
    loss, count = answer_loss(model, *make_batch(examples[start:start + batch_size], pad_id, device))
    numerator += loss.item()
    denominator += count.item()
  return numerator / denominator


def checkpoint_payload(parent, model, result, split_hash, step):
  # Preserve the tied embedding/head storage and use plain metadata types so
  # the ordinary weights_only=True reader can load the artifact.
  state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != 'language_model_head.weight'}
  state['language_model_head.weight'] = state['token_embedding.weight']
  metadata = json.loads(json.dumps(result))
  return {**parent, 'model_state': state, 'step': parent['step'] + step, 'split_hash': split_hash,
          'benchmark_ancestry': parent.get('benchmark_ancestry', 'legacy_or_unknown'),
          'parent_sha256': result['parent_sha256'], 'verified_sft': metadata}


def fine_tune(args):
  if args.epochs < 1 or min(args.batch_size, args.grad_accumulation, args.learning_rate, args.max_minutes) <= 0:
    raise ValueError('epochs, batch sizes, learning rate and time budget must be positive')
  if args.output.exists() or args.output.resolve() == args.load.resolve():
    raise ValueError('Choose a new output directory; the parent checkpoint is never overwritten')
  roots = dict(item.split('=', 1) for item in args.root)
  manifest = json.loads(args.splits.read_text())
  rows = verified_rows(args.data, manifest, roots, args.curriculum / 'verified_train.jsonl' if args.curriculum else None)
  torch.manual_seed(args.seed)
  device = ai.choose_device(args.device)
  model, parent = ai.load_student(args.load, device)
  tokenizer = ai.StudentTokenizer(parent['qwen_ids'])
  examples, stats = encode_examples(rows, tokenizer, model.config.context_length)
  # Checkpoint format and token IDs stay compatible with chat/benchmark.
  settings = {'method': 'answer-only supervised cross-entropy', 'parent': str(args.load.resolve()),
              'parent_sha256': file_hash(args.load), 'data_sha256': file_hash(args.data),
              'split_hash': manifest['hash'], 'benchmark_sha256': file_hash(ai.BENCHMARK_FILE),
              'epochs_requested': args.epochs, 'batch_size': args.batch_size,
              'grad_accumulation': args.grad_accumulation, 'learning_rate': args.learning_rate,
              'weight_decay': 0.01, 'warmup_steps': 5, 'gradient_clip': 1.0,
              'max_minutes': args.max_minutes, 'seed': args.seed, 'device': str(device),
              'host': platform.node(), 'torch': str(torch.__version__), 'examples': len(rows),
              'training_code_sha256': file_hash(Path(__file__)), **stats}
  print(json.dumps(settings, indent=2), flush=True)
  if args.dry_run:
    return
  args.output.mkdir(parents=True)
  (args.output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
  started = time.perf_counter()
  initial_loss = training_loss(model, examples, args.batch_size, parent['pad_id'], device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
  microbatches = math.ceil(len(examples) / args.batch_size)
  total_steps = args.epochs * math.ceil(microbatches / args.grad_accumulation)
  step, completed_epochs, seen_examples, supervised_tokens = 0, 0, 0, 0
  stopped_early = False
  for epoch in range(args.epochs):
    order = list(range(len(examples)))
    random.Random(args.seed + epoch).shuffle(order)
    batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    for start in range(0, len(batches), args.grad_accumulation):
      model.train()
      accumulated = [make_batch([examples[i] for i in ids], parent['pad_id'], device)
                     for ids in batches[start:start + args.grad_accumulation]]
      count = sum(int(mask.sum()) for _, mask in accumulated)
      optimizer.zero_grad(set_to_none=True)
      step += 1
      progress = min(1.0, max(0.0, (step - 5) / max(1, total_steps - 5)))
      lr = args.learning_rate * min(1.0, step / 5) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
      for group in optimizer.param_groups:
        group['lr'] = lr
      numerator = 0.0
      for tokens, mask in accumulated:
        loss, _ = answer_loss(model, tokens, mask)
        (loss / count).backward()
        numerator += loss.item()
        seen_examples += tokens.shape[0]
      grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
      if not math.isfinite(numerator) or not math.isfinite(grad_norm):
        raise ValueError('Nonfinite training loss/gradient; no checkpoint saved')
      optimizer.step()
      supervised_tokens += count
      record = {'step': step, 'epoch': epoch + 1, 'loss': numerator / count, 'learning_rate': lr,
                'seconds': time.perf_counter() - started, 'supervised_tokens': supervised_tokens}
      with (args.output / 'training.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
      if step == 1 or step % 5 == 0:
        print(json.dumps(record), flush=True)
      if time.perf_counter() - started > args.max_minutes * 60:
        stopped_early = True
        break
    if stopped_early:
      break
    completed_epochs += 1
  final_loss = training_loss(model, examples, args.batch_size, parent['pad_id'], device)
  result = {**settings, 'optimizer_steps': step, 'epochs_completed': completed_epochs,
            'examples_seen': seen_examples, 'supervised_tokens': supervised_tokens,
            'initial_training_loss': initial_loss, 'final_training_loss': final_loss,
            'elapsed_seconds': time.perf_counter() - started, 'budget_stop': stopped_early}
  checkpoint = checkpoint_payload(parent, model, result, manifest['hash'], step)
  temporary = args.output / 'student.partial.pt'
  torch.save(checkpoint, temporary)
  torch.load(temporary, map_location='cpu', weights_only=True)  # verify normal-reader compatibility before publishing
  temporary.replace(args.output / 'student.pt')
  result['checkpoint_sha256'] = file_hash(args.output / 'student.pt')
  if file_hash(args.load) != settings['parent_sha256']:
    raise ValueError('Parent checkpoint changed during training')
  (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2), flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--load', required=True, type=Path)
  p.add_argument('--data', required=True, type=Path)
  p.add_argument('--curriculum', type=Path, help='Directory containing canonical expanded examples and execution proof')
  p.add_argument('--splits', required=True, type=Path)
  p.add_argument('--root', action='append', default=[])
  p.add_argument('--output', required=True, type=Path)
  p.add_argument('--epochs', type=int, default=5)
  p.add_argument('--batch-size', type=int, default=2)
  p.add_argument('--grad-accumulation', type=int, default=2)
  p.add_argument('--learning-rate', type=float, default=5e-5)
  p.add_argument('--max-minutes', type=float, default=10)
  p.add_argument('--seed', type=int, default=7)
  p.add_argument('--device', choices=('mps','cpu','cuda','auto'), default='mps')
  p.add_argument('--dry-run', action='store_true')
  fine_tune(p.parse_args())


if __name__ == '__main__':
  main()
