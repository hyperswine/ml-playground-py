"""Stage-2 SFT data: every verified teaching example plus a pool of training-split 4B Q&A for replay.

Verified examples are the union (by exact question/answer) of the M1 curricula: 65 + expanded +
input-fidelity + balanced-repair + output-complete canonical rows + the generalization selection,
4,712 in all (the split manifest's verified training count). The replay pool is training-split 4B Q&A
only, sampled with a fixed seed. Both are encoded in the student vocabulary as (ids, learn) with
loss on answer tokens, in the chat format used everywhere else.
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
from qos_ai_data import digest, read_rows, row_split, validate_manifest

E = Path('qos_ai_benchmark/evidence')
VERIFIED = [Path('qos_ai_benchmark/verified_train_ready.jsonl'),
            E / 'verified_expanded/curricula/verified_expanded/verified_train_ready.jsonl',
            E / 'input_fidelity/curricula/input_fidelity_v2/focused_train_ready.jsonl',
            E / 'balanced_repair_20261018/curricula/balanced_repair_20261018/verified_train_ready.jsonl',
            E / 'output_complete_20261018/curricula/output_complete_20261018/verified_train_ready.jsonl',
            E / 'generalization_20261019/curricula/generalization_20261019/balanced_train_ready.jsonl']


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--splits', type=Path, required=True)
  p.add_argument('--qa', type=Path, default=Path('qos_ai/pod_pt/qa.jsonl'))
  p.add_argument('--tokenizer', type=Path, default=Path('runpod/557m/bundle/tokenizer'))
  p.add_argument('--vocabulary', type=Path, default=Path('runpod/557m/bundle/vocabulary.pt'))
  p.add_argument('--pool', type=int, default=20000)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  if args.output.exists():
    raise ValueError('Choose a new output file')
  manifest = json.loads(args.splits.read_text())
  qa = read_rows(args.qa)
  validate_manifest(manifest, qa=qa)
  vocabulary = torch.load(args.vocabulary, weights_only=True)
  qwen = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
  qwen_ids = vocabulary['qwen_ids']
  lookup = np.zeros(max(qwen.get_vocab_size(), int(qwen_ids.max()) + 1), dtype=np.int64)
  lookup[qwen_ids[1:].numpy()] = np.arange(1, len(qwen_ids))
  encode = lambda text: lookup[np.asarray(qwen.encode(text, add_special_tokens=False).ids, dtype=np.int64)].tolist()

  def example(question, answer):
    prompt = encode(ai.chat_text(question))
    ids = encode(ai.chat_text(question, answer))
    return ids, [i >= len(prompt) for i in range(len(ids))]

  seen, verified, sources, dropped = set(), [], {}, []
  for path in VERIFIED:
    for row in read_rows(path):
      key = (row['question'], row['answer'])
      if key in seen:
        continue
      seen.add(key)
      ids, learn = example(row['question'], row['answer'])
      if 0 in ids or len(ids) > 512:  # tokens outside the fixed student vocabulary, or too long
        dropped.append(row.get('id')); continue
      sources[str(path)] = sources.get(str(path), 0) + 1
      verified.append((ids, learn))

  train_qa = [r for r in qa if row_split(r, manifest) == 'train']
  random.Random(20261020).shuffle(train_qa)
  pool = []
  for row in train_qa:
    ids, learn = example(row['question'], row['answer'])
    if len(ids) <= 512 and not any(t == 0 for t in ids):
      pool.append((ids, learn))
    if len(pool) == args.pool:
      break

  payload = {'verified': verified, 'qa_pool': pool, 'dropped_verified': dropped, 'pad_id': vocabulary['pad_id'],
             'sources': sources, 'splits_hash': manifest['hash'],
             'vocabulary_sha256': hashlib.sha256(args.vocabulary.read_bytes()).hexdigest()}
  payload['hash'] = digest({'verified': verified, 'qa_pool': pool})
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, args.output)
  print(json.dumps({'verified': len(verified), 'verified_answer_tokens': sum(sum(l[1:]) for _, l in verified),
                    'longest_verified': max(len(i) for i, _ in verified), 'qa_pool': len(pool), 'dropped_verified': dropped,
                    'sources': sources, 'hash': payload['hash']}, indent=2))


if __name__ == '__main__':
  main()
