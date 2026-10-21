"""Same-token English loss probe for comparing 168M and 557M checkpoints.

`build` takes 64 documents from FineWeb-Edu sample/10BT file 013, which no
checkpoint trained on (the 168M runs used files 000-005, the 557M run 000-001),
and keeps the first 513 student tokens of each. `eval` reports next-token
cross-entropy over those windows, ignoring unknown (id 0) targets, so every
checkpoint sees exactly the same token ids.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = 'HuggingFaceFW/fineweb-edu'
REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'  # same pin as qos_ai_runpod_data.py
FILE = 'sample/10BT/013_00000.parquet'
WINDOW = 513
DOMAIN = ('fp-risc', 'qosportable', 'qosnative', 'jasonqin/fprisc')


def build(args):
  import pyarrow.parquet as pq
  from huggingface_hub import HfFileSystem
  from tokenizers import Tokenizer
  qwen = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
  qwen_ids = torch.load(args.vocabulary, weights_only=True)['qwen_ids']
  lookup = np.zeros(max(qwen.get_vocab_size(), int(qwen_ids.max()) + 1), dtype=np.int64)
  lookup[qwen_ids[1:].numpy()] = np.arange(1, len(qwen_ids))
  fs = HfFileSystem()
  windows, sources = [], []
  with fs.open(f'datasets/{REPO}@{REVISION}/{FILE}', 'rb') as handle:
    parquet = pq.ParquetFile(handle)
    for group in range(parquet.num_row_groups):
      for row in parquet.read_row_group(group, columns=['text', 'url']).to_pylist():
        text = row['text']
        if any(s in text.casefold() for s in DOMAIN):
          continue
        ids = lookup[np.asarray(qwen.encode(text, add_special_tokens=False).ids)]
        if len(ids) < WINDOW or (ids == 0).mean() > .01:
          continue
        windows.append(ids[:WINDOW]); sources.append(hashlib.sha256(text.encode()).hexdigest())
        if len(windows) == args.windows:
          break
      if len(windows) == args.windows:
        break
  tokens = np.stack(windows).astype(np.int64)
  np.save(args.output, tokens)
  meta = {'repo': REPO, 'revision': REVISION, 'file': FILE, 'windows': len(windows), 'window_tokens': WINDOW,
          'unknown_targets': int((tokens[:, 1:] == 0).sum()), 'document_sha256': sources,
          'tokens_sha256': hashlib.sha256(tokens.tobytes()).hexdigest()}
  args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
  print(json.dumps({k: v for k, v in meta.items() if k != 'document_sha256'}, indent=2))


@torch.no_grad()
def evaluate(args):
  import qos_ai as ai
  tokens = torch.from_numpy(np.load(args.probe))
  device = ai.choose_device(args.device)
  results = {}
  for path in args.load:
    model, checkpoint = ai.load_student(path, device)
    model.eval()
    total = count = 0.
    for start in range(0, len(tokens), 8):
      batch = tokens[start:start + 8].to(device)
      logits = model(batch[:, :-1]).float()
      total += F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), ignore_index=0, reduction='sum').item()
      count += int((batch[:, 1:] != 0).sum())
    results[str(path)] = {'cross_entropy': total / count, 'perplexity': float(np.exp(total / count)),
                          'parameters': ai.parameter_count(model), 'target_tokens': int(count)}
    print(path, json.dumps(results[str(path)]), flush=True)
    del model, checkpoint
  if args.output:
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  s = p.add_subparsers(dest='command', required=True)
  b = s.add_parser('build')
  b.add_argument('--tokenizer', type=Path, required=True)
  b.add_argument('--vocabulary', type=Path, required=True)
  b.add_argument('--output', type=Path, required=True)
  b.add_argument('--windows', type=int, default=64)
  e = s.add_parser('eval')
  e.add_argument('--probe', type=Path, required=True)
  e.add_argument('--load', type=Path, nargs='+', required=True)
  e.add_argument('--device', default='mps')
  e.add_argument('--output', type=Path)
  a = p.parse_args()
  build(a) if a.command == 'build' else evaluate(a)
