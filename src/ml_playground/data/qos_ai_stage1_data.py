"""Stage-1 domain data for the 557M (and a matched 168M control): training-partition only.

Uses the frozen split manifest the M1 experiments use. Training rows are fprisc/qos
source chunks from training-partition files (formatted as `qos_ai.py`'s lessons:
"# File: path\\n...") and the historical Qwen3.5-4B Q&A pairs assigned to train
(chat format, loss on answer tokens only). Benchmark questions and their near
duplicates are already marked `exclude` in the manifest.

Outputs, in student token ids (same fixed 65,536 mapping as both lineages):
  domain_rows.npy / domain_mask.npy   [rows, 513] packed complete examples, `--epochs` shuffled passes
  probe_docs.npy                      dev-partition source chunks cut into 513-token windows
  probe_qa_rows.npy / probe_qa_mask.npy  first 512 dev-partition Q&A pairs, packed
  stage1_data.json                    counts, input fingerprints and output hashes
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import qos_ai as ai
from qos_ai_data import digest, read_rows, row_split, rows_fingerprint, validate_manifest

WINDOW = 513


def pack(examples, pad_id):
  """Greedily pack complete (ids, learn) examples into WINDOW-token rows; leftover space is id 0 (ignored)."""
  rows, masks, ids, learn = [], [], [], []
  def flush():
    rows.append(ids + [0] * (WINDOW - len(ids))); masks.append(learn + [False] * (WINDOW - len(learn)))
  for example_ids, example_learn in examples:
    if ids and len(ids) + len(example_ids) > WINDOW:
      flush(); ids, learn = [], []
    ids = ids + example_ids; learn = learn + example_learn
  if ids:
    flush()
  return np.asarray(rows, dtype=np.uint32), np.asarray(masks, dtype=np.uint8)


def sha(array):
  return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--corpus', type=Path, default=Path('qos_ai/corpus.jsonl'))
  p.add_argument('--qa', type=Path, default=Path('qos_ai/pod_pt/qa.jsonl'))
  p.add_argument('--splits', type=Path, required=True)
  p.add_argument('--tokenizer', type=Path, default=Path('runpod/557m/bundle/tokenizer'))
  p.add_argument('--vocabulary', type=Path, default=Path('runpod/557m/bundle/vocabulary.pt'))
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--epochs', type=int, default=2)
  args = p.parse_args()
  if args.output.exists():
    raise ValueError('Choose a new output directory')

  manifest = json.loads(args.splits.read_text())
  corpus, qa = read_rows(args.corpus), read_rows(args.qa)
  validate_manifest(manifest, corpus=corpus, qa=qa)
  vocabulary = torch.load(args.vocabulary, weights_only=True)
  qwen_ids, pad_id = vocabulary['qwen_ids'], vocabulary['pad_id']
  qwen = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
  lookup = np.zeros(max(qwen.get_vocab_size(), int(qwen_ids.max()) + 1), dtype=np.int64)
  lookup[qwen_ids[1:].numpy()] = np.arange(1, len(qwen_ids))
  encode = lambda text: lookup[np.asarray(qwen.encode(text, add_special_tokens=False).ids, dtype=np.int64)].tolist()
  newline = encode('\n')

  stats = {'unknown_tokens': 0, 'tokens': 0}
  def counted(ids):
    stats['unknown_tokens'] += sum(t == 0 for t in ids); stats['tokens'] += len(ids)
    return ids

  # Source chunks: whole documents split into WINDOW-token pieces; every token is a target.
  docs = {'train': [], 'dev': []}
  for row in corpus:
    split = manifest['files'][row['path']]
    if split in docs and digest([row['path'], row['chunk']]) not in manifest.get('excluded_chunks', []):
      ids = counted(encode(f"# File: {row['path']}\n{row['text']}\n")) + [pad_id]
      docs[split].extend((ids[i:i + WINDOW], [True] * len(ids[i:i + WINDOW])) for i in range(0, len(ids), WINDOW))

  # Q&A: complete examples, loss on answer tokens (as in lesson_sources).
  pairs = {'train': [], 'dev': []}; oversized = 0; split_counts = {}
  for row in qa:
    split = row_split(row, manifest); split_counts[split] = split_counts.get(split, 0) + 1
    if split not in pairs:
      continue
    prompt = encode(ai.chat_text(row['question']))
    ids = counted(encode(ai.chat_text(row['question'], row['answer']))) + newline
    if len(ids) > WINDOW:
      oversized += 1; continue
    pairs[split].append((ids, [i >= len(prompt) for i in range(len(ids))]))

  epochs_rows, epochs_masks = [], []
  for epoch in range(args.epochs):
    examples = docs['train'] + pairs['train']
    random.Random(20261019 + epoch).shuffle(examples)
    rows, masks = pack(examples, pad_id)
    epochs_rows.append(rows); epochs_masks.append(masks)
  rows, masks = np.concatenate(epochs_rows), np.concatenate(epochs_masks)

  dev_stream = [t for ids, _ in docs['dev'] for t in ids]
  probe_docs = np.asarray([dev_stream[i:i + WINDOW] for i in range(0, len(dev_stream) - WINDOW + 1, WINDOW)], dtype=np.uint32)
  probe_qa_rows, probe_qa_mask = pack(pairs['dev'][:512], pad_id)

  args.output.mkdir(parents=True)
  outputs = {'domain_rows': rows, 'domain_mask': masks, 'probe_docs': probe_docs,
             'probe_qa_rows': probe_qa_rows, 'probe_qa_mask': probe_qa_mask}
  for name, array in outputs.items():
    np.save(args.output / f'{name}.npy', array)
  meta = {
    'splits_hash': manifest['hash'], 'corpus_fingerprint': rows_fingerprint(corpus), 'qa_fingerprint': rows_fingerprint(qa),
    'vocabulary_sha256': hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(), 'pad_id': pad_id,
    'epochs': args.epochs, 'window': WINDOW,
    'train_doc_pieces': len(docs['train']), 'train_qa_pairs': len(pairs['train']), 'qa_split_counts': split_counts,
    'oversized_qa_dropped': oversized, 'unknown_token_rate': stats['unknown_tokens'] / max(1, stats['tokens']),
    'domain_rows': int(len(rows)), 'domain_tokens': int(len(rows) * WINDOW),
    'domain_target_tokens': int(masks[:, 1:].sum()), 'domain_nonpad_tokens': int((rows != 0).sum()),
    'probe_doc_windows': int(len(probe_docs)), 'probe_qa_pairs': min(512, len(pairs['dev'])),
    'probe_qa_target_tokens': int(probe_qa_mask[:, 1:].sum()),
    'sha256': {name: sha(array) for name, array in outputs.items()},
    'scope': 'training-partition sources and 4B Q&A only; dev partition used only as loss probes; test partition unused',
  }
  meta['hash'] = digest({k: v for k, v in meta.items()})
  (args.output / 'stage1_data.json').write_text(json.dumps(meta, indent=2) + '\n')
  print(json.dumps(meta, indent=2))


if __name__ == '__main__':
  main()
