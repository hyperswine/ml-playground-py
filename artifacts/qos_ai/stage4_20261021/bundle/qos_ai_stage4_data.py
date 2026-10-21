"""Prepare Stage 4 pools, then mix replay by supervised tokens, never held-out probes."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import qos_ai as ai
from qos_ai_data import digest, read_rows, validate_manifest
from qos_ai_stage4_eval import sha


def targets(example):
  ids, mask = example
  if not 2 <= len(ids) <= 512 or len(ids) != len(mask):
    raise ValueError('invalid or incomplete example length')
  if any(type(t) is not int or not 0 < t < 65536 for t in ids):
    raise ValueError('unknown/out-of-vocabulary token')
  if any(type(m) is not bool for m in mask) or not any(mask[1:]):
    raise ValueError('invalid or empty target mask')
  return sum(mask[1:])


def token_sample(pool, budget, seed):
  """Whole examples, deterministic cycling; overshoot is less than one example."""
  if not pool or budget <= 0: raise ValueError('nonempty pool and positive budget required')
  counts = [targets(e) for e in pool]
  selected, total, cycle = [], 0, 0
  while total < budget:
    order = list(range(len(pool))); random.Random(seed + cycle).shuffle(order)
    for index in order:
      selected.append(pool[index]); total += counts[index]
      if total >= budget: break
    cycle += 1
  return selected, {'target_tokens': total, 'requested_tokens': budget,
                    'examples': len(selected), 'full_cycles': len(selected) // len(pool),
                    'overshoot': total - budget}


def checked_load(path):
  payload = torch.load(path, map_location='cpu', weights_only=True)
  if payload['hash'] != digest({'verified': payload['verified'], 'qa_pool': payload['qa_pool']}):
    raise ValueError('stale SFT payload hash')
  return payload


def prepare(args):
  if args.output.exists(): raise ValueError('choose a new output directory')
  base = checked_load(args.stage3)
  manifest = json.loads(args.splits.read_text()); corpus = read_rows(args.corpus)
  validate_manifest(manifest, corpus=corpus)
  vocab_sha = sha(args.vocabulary)
  if base['splits_hash'] != manifest['hash'] or base['vocabulary_sha256'] != vocab_sha:
    raise ValueError('base/split/vocabulary mismatch')
  verification = json.loads((args.arithmetic / 'verification.json').read_text())
  if verification['hash'] != digest({k: v for k, v in verification.items() if k != 'hash'}):
    raise ValueError('stale arithmetic verification')
  for split in ('train', 'validation'):
    if sha(args.arithmetic / f'{split}.jsonl') != verification['files'][split]:
      raise ValueError('arithmetic file differs from verified input')
  rows = {s: read_rows(args.arithmetic / f'{s}.jsonl') for s in ('train', 'validation')}
  if {r['group'] for r in rows['train']} & {r['group'] for r in rows['validation']}:
    raise ValueError('arithmetic group crosses partitions')
  vocab = torch.load(args.vocabulary, weights_only=True)
  tokenizer = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
  lookup = np.zeros(max(tokenizer.get_vocab_size(), int(vocab['qwen_ids'].max()) + 1), dtype=np.int64)
  lookup[vocab['qwen_ids'][1:].numpy()] = np.arange(1, len(vocab['qwen_ids']))
  def encode(text):
    return lookup[np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int64)].tolist()
  arithmetic = {}
  for split, records in rows.items():
    arithmetic[split] = []
    for r in records:
      if r['split'] != split: raise ValueError('wrong arithmetic partition')
      prompt = encode(ai.chat_text(r['question']))
      ids = encode(ai.chat_text(r['question'], r['answer']))
      example = (ids, [i >= len(prompt) for i in range(len(ids))])
      targets(example)  # Fail rather than silently truncate or drop a complete answer.
      arithmetic[split].append(example)
  docs, dropped = [], 0
  excluded = set(manifest['excluded_chunks'])
  for r in corpus:
    if manifest['files'][r['path']] != 'train' or digest([r['path'], r['chunk']]) in excluded: continue
    ids = encode(f"# File: {r['path']}\n{r['text']}\n") + [vocab['pad_id']]
    for start in range(0, len(ids), 512):
      piece = ids[start:start + 512]
      if len(piece) < 2 or 0 in piece: dropped += 1; continue
      docs.append((piece, [True] * len(piece)))
  pools = {'verified': base['verified'] + arithmetic['train'], 'qa_pool': base['qa_pool'], 'docs': docs}
  counts = {name: {'examples': len(pool), 'target_tokens': sum(targets(e) for e in pool)} for name, pool in pools.items()}
  pools.update(pad_id=base['pad_id'], vocabulary_sha256=vocab_sha, splits_hash=manifest['hash'])
  pools['hash'] = digest({name: pools[name] for name in ('verified', 'qa_pool', 'docs')})
  report = {'schema': 1, 'pools_hash': pools['hash'], 'counts': counts,
            'arithmetic_training': len(arithmetic['train']), 'arithmetic_validation': len(arithmetic['validation']),
            'arithmetic_validation_target_tokens': sum(targets(e) for e in arithmetic['validation']),
            'docs_rejected_short_or_unknown': dropped, 'unknown_tokens_in_accepted_examples': 0,
            'max_length': max(len(e[0]) for name in counts for e in pools[name]),
            'sources': {str(p): sha(p) for p in (args.stage3, args.splits, args.corpus, args.vocabulary,
                        args.tokenizer / 'tokenizer.json', args.arithmetic / 'verification.json')},
            'scope': 'training pools only; arithmetic validation encoded separately; English replay still required'}
  args.output.mkdir(parents=True)
  torch.save(pools, args.output / 'pools.pt')
  torch.save({'examples': arithmetic['validation'], 'pad_id': base['pad_id']}, args.output / 'arithmetic_validation.pt')
  report['pools_file_sha256'] = sha(args.output / 'pools.pt')
  report['arithmetic_validation_sha256'] = sha(args.output / 'arithmetic_validation.pt')
  (args.output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
  return report


def mix(args):
  if args.output.exists(): raise ValueError('choose a new output directory')
  pools = torch.load(args.pools, weights_only=True)
  if pools['hash'] != digest({n: pools[n] for n in ('verified', 'qa_pool', 'docs')}):
    raise ValueError('stale pool contents')
  manifest_path = args.pretraining_data / 'dataset.json'
  if sha(manifest_path) != args.pretraining_manifest_sha256:
    raise ValueError('pretraining manifest differs from the independently recorded identity')
  manifest = json.loads(manifest_path.read_text())
  if manifest['hash'] != digest({k: v for k, v in manifest.items() if k != 'hash'}):
    raise ValueError('pretraining manifest internal hash mismatch')
  if manifest['vocabulary_sha256'] != pools['vocabulary_sha256']:
    raise ValueError('English replay vocabulary mismatch')
  info = manifest['results']['english']['files']['train']
  stream_path = (args.pretraining_data / info['path']).resolve()
  if stream_path.parent != args.pretraining_data.resolve() or sha(stream_path) != info['sha256']:
    raise ValueError('invalid English training stream path/hash')
  stream = np.memmap(stream_path, mode='r', dtype=np.uint32)
  if len(stream) < 512: raise ValueError('English training stream too small')
  # Small fixed pool of windows, independent of all validation probes. No text is downloaded.
  rng = np.random.default_rng(20261021); english = []
  for start in rng.integers(0, len(stream) - 511, size=20000):
    ids = stream[start:start + 512].astype(np.int64).tolist()
    if 0 in ids: continue
    example = (ids, [True] * len(ids)); targets(example); english.append(example)
  base_tokens = sum(targets(e) for e in pools['verified'])
  # 65% verified, 14% English, 14% docs, 7% QA by supervised target tokens.
  all_examples = list(pools['verified']); report = {}
  for index, (name, pool, share) in enumerate((('english', english, .14), ('docs', pools['docs'], .14), ('qa', pools['qa_pool'], .07))):
    selected, stats = token_sample(pool, math.ceil(base_tokens * share / .65), 7 + index)
    all_examples.extend(selected); report[name] = stats
  total = sum(targets(e) for e in all_examples)
  payload = {'verified': all_examples, 'qa_pool': [], 'pad_id': pools['pad_id'],
             'splits_hash': pools['splits_hash'], 'vocabulary_sha256': pools['vocabulary_sha256']}
  payload['hash'] = digest({'verified': all_examples, 'qa_pool': []})
  report = {'schema': 1, 'data_hash': payload['hash'], 'replay': report, 'examples': len(all_examples),
            'target_tokens': total, 'verified_tokens': base_tokens, 'verified_share': base_tokens / total,
            'replay_shares': {n: s['target_tokens'] / total for n, s in report.items()},
            'pools_file_sha256': sha(args.pools), 'pretraining_manifest_sha256': sha(manifest_path),
            'english_train_sha256': info['sha256'], 'seed': 7}
  args.output.mkdir(parents=True)
  torch.save(payload, args.output / 'sft.pt')
  report['sft_file_sha256'] = sha(args.output / 'sft.pt')
  (args.output / 'mixture.json').write_text(json.dumps(report, indent=2) + '\n')
  return report


def main():
  p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
  prep = sub.add_parser('prepare')
  prep.add_argument('--stage3', type=Path, default=Path('qos_ai/stage3_20261020/sft_sol.pt'))
  prep.add_argument('--splits', type=Path, default=Path('qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/splits.json'))
  prep.add_argument('--corpus', type=Path, default=Path('qos_ai/corpus.jsonl'))
  prep.add_argument('--vocabulary', type=Path, default=Path('runpod/557m/bundle/vocabulary.pt'))
  prep.add_argument('--tokenizer', type=Path, default=Path('runpod/557m/bundle/tokenizer'))
  prep.add_argument('--arithmetic', type=Path, default=Path('qos_ai_benchmark/stage4_20261021/arithmetic'))
  mixer = sub.add_parser('mix')
  mixer.add_argument('--pools', type=Path, required=True)
  mixer.add_argument('--pretraining-data', type=Path, required=True)
  mixer.add_argument('--pretraining-manifest-sha256', required=True)
  for cmd in (prep, mixer): cmd.add_argument('--output', type=Path, required=True)
  args = p.parse_args(); print(json.dumps(prepare(args) if args.command == 'prepare' else mix(args), indent=2))


if __name__ == '__main__': main()
