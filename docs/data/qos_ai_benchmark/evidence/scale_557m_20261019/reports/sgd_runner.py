"""Isolated from-scratch scaling pilots; no teacher weights and no default replacement.

The existing verified/source corpus is a pipeline pilot, not general pretraining.
Training streams contain student token IDs. Random windows are keyed by optimizer
step and microbatch, so a saved optimizer boundary reproduces the data sequence.
"""
import argparse
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import qos_ai as ai
from qos_ai_data import digest, file_hash, read_rows, validate_manifest
from qos_ai_finetune import verified_rows


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_config(path):
  config = ai.StudentConfig(**json.loads(path.read_text()))
  if min(asdict(config).values()) <= 0 or config.embedding_size % config.head_count:
    raise ValueError('Invalid architecture')
  return config


def describe(config):
  with torch.device('meta'):
    model = ai.QosAi(config)
  count = ai.parameter_count(model)
  return {'config': asdict(config), 'unique_parameters': count,
          'fp32_weight_bytes': count * 4, 'fp32_sgd_weights_gradients_bytes': count * 8,
          'fp32_adam_weights_gradients_moments_bytes': count * 16,
          'memory_estimate_excludes': 'activations, logits, kernel workspaces, allocator and other processes',
          'initialization': 'random weights; existing tokenizer mapping only; no teacher or 168M weights'}


def validation_group(row):
  # All variants of one generated case stay together. Replay stays with its source.
  return 'case:' + row['case_id'] if row.get('case_id') else 'source:' + row['path']


def partition_documents(documents):
  """Internal pilot validation only; never imports benchmark questions as examples."""
  grouped = {}
  for doc in documents:
    grouped.setdefault(doc['group'], []).append(doc)
  validation = {g for g in grouped if int(digest(g)[:8], 16) % 10 == 0}
  if not validation or len(validation) == len(grouped):
    raise ValueError('Need distinct training and internal validation groups')
  splits = {'train': [], 'validation': []}
  for doc in documents:
    splits['validation' if doc['group'] in validation else 'train'].append(doc)
  # Exact text copies can arise across source and Q&A groups. Fail closed.
  overlap = {digest(d['text']) for d in splits['train']} & {digest(d['text']) for d in splits['validation']}
  for split in splits:
    splits[split] = [d for d in splits[split] if digest(d['text']) not in overlap]
  return splits, len(overlap)


def prepare(args):
  if args.output.exists():
    raise ValueError('Choose a new dataset directory')
  config = read_config(args.config)
  manifest = json.loads((args.curriculum / 'splits.json').read_text())
  corpus = read_rows(args.corpus)
  validate_manifest(manifest, corpus=corpus)
  roots = dict(item.split('=', 1) for item in args.root)
  rows = verified_rows(args.curriculum / 'balanced_train_ready.jsonl', manifest, roots,
                       args.curriculum / 'verified_train.jsonl',
                       json.loads((args.curriculum / 'training_selection.json').read_text()))
  documents = [{'id': r['id'], 'group': validation_group(r), 'kind': 'verified_chat',
                'text': ai.chat_text(r['question'], r['answer'])} for r in rows]
  for r in corpus:
    if manifest['files'].get(r['path']) != 'train':
      continue
    if digest([r['path'], r['chunk']]) in manifest['excluded_chunks']:
      continue
    documents.append({'id': digest([r['path'], r['chunk']]), 'group': 'source:' + r['path'],
                      'kind': 'source', 'text': f"# File: {r['path']}\n{r['text']}\n"})
  splits, removed = partition_documents(documents)
  # Read only vocabulary metadata. mmap avoids materializing the old model weights.
  parent = torch.load(args.vocabulary_from, map_location='cpu', weights_only=True, mmap=True)
  vocabulary = {k: parent[k] for k in ('qwen_ids', 'pad_id', 'im_end_id')}
  if len(vocabulary['qwen_ids']) != config.vocabulary_size:
    raise ValueError('Tokenizer mapping and configured vocabulary disagree')
  tokenizer = ai.StudentTokenizer(vocabulary['qwen_ids'])
  args.output.mkdir(parents=True)
  stats = {}
  for split, docs in splits.items():
    stream = []
    for doc in docs:
      ids = tokenizer.encode(doc['text']) + [vocabulary['pad_id']]
      stream.extend(ids)
    if len(stream) < config.context_length + 1:
      raise ValueError('Too few tokens in ' + split)
    np.save(args.output / (split + '.npy'), np.asarray(stream, dtype=np.uint32))
    stats[split] = {'documents': len(docs), 'groups': len({d['group'] for d in docs}),
                    'kinds': dict(Counter(d['kind'] for d in docs)), 'tokens': len(stream),
                    'unknown_tokens': stream.count(0), 'sha256': file_hash(args.output / (split + '.npy'))}
  torch.save(vocabulary, args.output / 'vocabulary.pt')
  write_json(args.output / 'document_partition.json',
             {s: [{k: d[k] for k in ('id', 'group', 'kind')} for d in docs] for s, docs in splits.items()})
  frozen = [args.corpus, args.config, args.curriculum / 'splits.json',
            args.curriculum / 'verified_train.jsonl', args.curriculum / 'verification.json',
            args.curriculum / 'training_selection.json', args.curriculum / 'balanced_train_ready.jsonl']
  benchmarks = [args.curriculum / n for n in ('benchmark.jsonl', 'coding_diagnostic.jsonl', 'generalization_diagnostic.jsonl')]
  dataset = {'schema': 1, 'scope': 'pipeline pilot from verified chats and training-partition source; no general-English pretraining',
             'split_hash': manifest['hash'], 'partition_policy': 'training partition only; 10 percent hashed whole source/case groups reserved internally',
             'validation_limit': 'internal validation is template-related; use frozen development diagnostics for capability claims',
             'cross_partition_exact_text_hashes_removed': removed, 'stats': stats,
             'frozen_inputs': {str(p): file_hash(p) for p in frozen},
             'frozen_benchmarks': {str(p): file_hash(p) for p in benchmarks},
             'tokenizer_source_sha256': file_hash(args.vocabulary_from),
             'vocabulary_sha256': file_hash(args.output / 'vocabulary.pt'),
             'partition_sha256': file_hash(args.output / 'document_partition.json'),
             'preparation_code_sha256': file_hash(Path(__file__))}
  dataset['hash'] = digest(dataset)
  write_json(args.output / 'dataset.json', dataset)
  print(json.dumps(dataset, indent=2), flush=True)


def load_dataset(path, config):
  metadata = json.loads((path / 'dataset.json').read_text())
  if metadata['hash'] != digest({k: v for k, v in metadata.items() if k != 'hash'}):
    raise ValueError('Dataset manifest corrupt')
  for name, sha in [('vocabulary.pt', metadata['vocabulary_sha256']), ('document_partition.json', metadata['partition_sha256'])]:
    if file_hash(path / name) != sha:
      raise ValueError('Dataset identity changed: ' + name)
  streams = {}
  for split, stat in metadata['stats'].items():
    if file_hash(path / (split + '.npy')) != stat['sha256']:
      raise ValueError('Dataset token stream changed')
    stream = np.load(path / (split + '.npy'), mmap_mode='r')
    if stream.dtype != np.uint32 or stream.ndim != 1 or len(stream) != stat['tokens'] or int(stream.max()) >= config.vocabulary_size:
      raise ValueError('Invalid token stream')
    streams[split] = stream
  vocabulary = torch.load(path / 'vocabulary.pt', map_location='cpu', weights_only=True)
  if len(vocabulary['qwen_ids']) != config.vocabulary_size:
    raise ValueError('Wrong vocabulary size')
  return metadata, streams, vocabulary


def draw(stream, seed, step, micro, batch_size, length):
  generator = np.random.default_rng(np.random.SeedSequence([seed, step, micro]))
  starts = generator.integers(0, len(stream) - length, size=batch_size)
  return torch.from_numpy(np.stack([stream[s:s + length + 1].astype(np.int64) for s in starts]))


def rate(base, step, total, warmup):
  if step <= warmup:
    return base * step / max(1, warmup)
  progress = (step - warmup) / max(1, total - warmup)
  return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))


def forward(model, tokens, checkpoint_blocks):
  if not checkpoint_blocks or not model.training:
    return model(tokens)
  positions = torch.arange(tokens.size(1), device=tokens.device)
  hidden = model.token_embedding(tokens) + model.position_embedding(positions)
  for block in model.blocks:
    hidden = activation_checkpoint(block, hidden, use_reentrant=False)
  return model.language_model_head(model.final_norm(hidden))


def loss_sum(model, batch, checkpoint_blocks=False, bf16=False):
  with torch.autocast(batch.device.type, dtype=torch.bfloat16, enabled=bf16):
    logits = forward(model, batch[:, :-1], checkpoint_blocks)
    return F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten(),
                           ignore_index=0, reduction='sum')


def optimizer_for(model, settings):
  if settings['optimizer'] == 'adamw':
    return torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                             betas=(.9, .95), weight_decay=.1, foreach=False)
  return torch.optim.SGD(model.parameters(), lr=settings['learning_rate'])


def advance(model, optimizer, streams, settings, step):
  device = next(model.parameters()).device
  batches = [draw(streams['train'], settings['seed'], step, micro,
                  settings['batch_size'], settings['sequence_length'])
             for micro in range(settings['gradient_accumulation'])]
  count = sum(int((b[:, 1:] != 0).sum()) for b in batches)
  if not count:
    raise ValueError('No supervised targets')
  optimizer.zero_grad(set_to_none=True)
  lr = rate(settings['learning_rate'], step, settings['total_steps'], settings['warmup_steps'])
  for group in optimizer.param_groups:
    group['lr'] = lr
  total = 0.
  for batch in batches:
    loss = loss_sum(model, batch.to(device), settings['checkpoint_blocks'], settings['bf16'])
    if not torch.isfinite(loss).item():
      raise ValueError('Nonfinite loss')
    total += loss.item()
    (loss / count).backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
  return {'step': step, 'loss': total / count, 'known_target_tokens': count,
          'tokens': sum(b[:, 1:].numel() for b in batches), 'learning_rate': lr, 'gradient_norm': norm}


@torch.no_grad()
def validation(model, streams, settings):
  model.eval()
  numerator = denominator = 0
  for micro in range(4):
    batch = draw(streams['validation'], 1234, 0, micro, 1, settings['sequence_length'])
    denominator += int((batch[:, 1:] != 0).sum())
    numerator += loss_sum(model, batch.to(next(model.parameters()).device), bf16=settings['bf16']).item()
  model.train()
  if not denominator:
    raise ValueError('No validation targets')
  return {'loss': numerator / denominator, 'known_target_tokens': denominator,
          'scope': 'four fixed internal-validation windows; infrastructure diagnostic only'}


def save_boundary(path, model, optimizer, settings, step, history):
  # Serialize device storages directly, without retaining a second full CPU model
  # and optimizer. The restricted mmap read validates metadata before publication.
  for parameter in model.parameters():
    if not torch.isfinite(parameter).all().item():
      raise ValueError('Nonfinite weights; refusing checkpoint')
  payload = {'schema': 1, 'settings': settings, 'settings_hash': digest(settings),
             'step': step, 'history': history, 'model_state': model.state_dict(),
             'optimizer_state': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}
  temporary = path.with_name(path.name + '.partial')
  torch.save(payload, temporary)
  check = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
  if check['step'] != step or check['settings_hash'] != digest(settings):
    raise ValueError('Checkpoint verification failed')
  del check
  temporary.replace(path)


def restore(path, model, optimizer, settings):
  saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
  if saved.get('schema') != 1 or saved.get('settings_hash') != digest(settings) or saved['settings'] != settings:
    raise ValueError('Resume architecture/data/code/optimizer/schedule differ')
  model.load_state_dict(saved['model_state'])
  optimizer.load_state_dict(saved['optimizer_state'])
  torch.set_rng_state(saved['cpu_rng'])
  if saved['cuda_rng']:
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
  return saved['step'], saved['history']


def train(args):
  config = read_config(args.config)
  if not 0 < args.sequence_length <= config.context_length or min(args.steps, args.batch_size, args.grad_accumulation, args.learning_rate, args.max_minutes, args.save_every) <= 0:
    raise ValueError('Invalid training limits')
  if args.segment_steps < 0 or args.warmup_steps < 0 or args.seed < 0:
    raise ValueError('Invalid segment/warmup/seed')
  metadata, streams, vocabulary = load_dataset(args.data, config)
  device = ai.choose_device(args.device)
  if args.bf16 and device.type != 'cuda':
    raise ValueError('This runner enables bf16 only on CUDA')
  if args.bf16 and not torch.cuda.is_bf16_supported():
    raise ValueError('CUDA device does not support bf16')
  description = describe(config)
  floor = description['fp32_adam_weights_gradients_moments_bytes' if args.optimizer == 'adamw' else 'fp32_sgd_weights_gradients_bytes']
  if device.type == 'mps' and floor > args.mps_memory_budget_gib * 1024**3:
    raise ValueError('Parameter/gradient/optimizer floor exceeds M1 pilot budget; use CUDA for AdamW')
  settings = {'config': asdict(config), 'dataset_hash': metadata['hash'], 'optimizer': args.optimizer,
              'learning_rate': args.learning_rate, 'total_steps': args.steps, 'warmup_steps': args.warmup_steps,
              'batch_size': args.batch_size, 'gradient_accumulation': args.grad_accumulation,
              'sequence_length': args.sequence_length, 'seed': args.seed, 'bf16': args.bf16,
              'checkpoint_blocks': args.checkpoint_blocks, 'device': device.type,
              'torch': str(torch.__version__), 'code': {str(Path(p).name): file_hash(Path(p)) for p in (__file__, ai.__file__)},
              'initialization': 'random; tokenizer mapping only', 'benchmark_ancestry': 'pilot train partition only; no inherited model weights'}
  path = args.output / 'progress.pt'
  if args.output.exists() and (not args.resume or not path.exists()):
    raise ValueError('Output exists; resume a valid saved boundary or choose a new directory')
  args.output.mkdir(parents=True, exist_ok=args.resume)
  # Set allocator budget before constructing the full model. Never raise the MPS
  # high-watermark to force an oversized optimizer into shared system memory.
  if device.type == 'mps':
    torch.mps.set_per_process_memory_fraction(min(1., args.mps_memory_budget_gib * 1024**3 / torch.mps.recommended_max_memory()))
  torch.manual_seed(args.seed)
  model = ai.QosAi(config).to(device)
  optimizer = optimizer_for(model, settings)
  step, history = restore(path, model, optimizer, settings) if args.resume else (0, [])
  start_step = step
  write_json(args.output / 'settings.json', settings)
  initial = validation(model, streams, settings)
  print(json.dumps({'phase': 'initial', 'step': step, 'validation': initial, 'host': platform.node(), **description}), flush=True)
  started = time.perf_counter()
  while step < args.steps:
    if step > start_step and (time.perf_counter() - started > args.max_minutes * 60 or
                              args.segment_steps and step - start_step >= args.segment_steps):
      break
    record = advance(model, optimizer, streams, settings, step + 1)
    step += 1
    record['elapsed_segment_seconds'] = time.perf_counter() - started
    history.append(record)
    print(json.dumps(record), flush=True)
    if step % args.save_every == 0 or step == args.steps:
      save_boundary(path, model, optimizer, settings, step, history)
    if device.type == 'mps':
      torch.mps.empty_cache()
  if not path.exists() or history and step % args.save_every and step != args.steps:
    save_boundary(path, model, optimizer, settings, step, history)
  final = validation(model, streams, settings)
  result = {'step': step, 'complete': step == args.steps, 'segment_start_step': start_step,
            'parameter_count': ai.parameter_count(model), 'initial_validation': initial, 'final_validation': final,
            'total_tokens': sum(r['tokens'] for r in history), 'known_target_tokens': sum(r['known_target_tokens'] for r in history),
            'segment_seconds': time.perf_counter() - started, 'checkpoint_sha256': file_hash(path),
            'settings_hash': digest(settings), 'scope': 'infrastructure pilot; not a capability comparison'}
  if device.type == 'mps':
    result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
  write_json(args.output / 'result.json', result)
  if step == args.steps:
    # Existing inference/evaluation tools can load this standard weight artifact.
    temporary = args.output / 'student.partial.pt'
    torch.save({'config': asdict(config), 'model_state': model.state_dict(), **vocabulary,
                'step': step, 'split_hash': metadata['split_hash'], 'benchmark_ancestry': settings['benchmark_ancestry'],
                'scaling_pilot': settings}, temporary)
    temporary.replace(args.output / 'student.pt')
  print(json.dumps(result, indent=2), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  inspect = commands.add_parser('inspect')
  inspect.add_argument('--config', type=Path, default=Path('qos_ai_557m.json'))
  prep = commands.add_parser('prepare')
  prep.add_argument('--config', type=Path, default=Path('qos_ai_557m.json'))
  prep.add_argument('--curriculum', type=Path, required=True)
  prep.add_argument('--corpus', type=Path, required=True)
  prep.add_argument('--vocabulary-from', type=Path, required=True)
  prep.add_argument('--root', action='append', required=True)
  prep.add_argument('--output', type=Path, required=True)
  run = commands.add_parser('train')
  run.add_argument('--config', type=Path, default=Path('qos_ai_557m.json'))
  run.add_argument('--data', type=Path, required=True)
  run.add_argument('--output', type=Path, required=True)
  run.add_argument('--device', choices=['cpu', 'cuda', 'mps'], required=True)
  run.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
  run.add_argument('--steps', type=int, default=100)
  run.add_argument('--sequence-length', type=int, default=512)
  run.add_argument('--batch-size', type=int, default=1)
  run.add_argument('--grad-accumulation', type=int, default=1)
  run.add_argument('--learning-rate', type=float, default=3e-4)
  run.add_argument('--warmup-steps', type=int, default=10)
  run.add_argument('--save-every', type=int, default=10)
  run.add_argument('--max-minutes', type=float, default=10)
  run.add_argument('--segment-steps', type=int, default=0)
  run.add_argument('--seed', type=int, default=7)
  run.add_argument('--mps-memory-budget-gib', type=float, default=6)
  run.add_argument('--bf16', action='store_true')
  run.add_argument('--checkpoint-blocks', action='store_true')
  run.add_argument('--resume', action='store_true')
  args = parser.parse_args()
  if args.command == 'inspect':
    print(json.dumps(describe(read_config(args.config)), indent=2))
  elif args.command == 'prepare':
    prepare(args)
  else:
    train(args)


if __name__ == '__main__':
  main()
