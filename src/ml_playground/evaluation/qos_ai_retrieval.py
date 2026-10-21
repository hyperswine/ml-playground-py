"""Retrieval-augmented answers for QOS/Sol questions: BM25 over documentation chunks, measured two ways.

  retrieval  For held-out dev-partition questions (4B Q&A), how often the question's own source file is among
             the top-k retrieved chunks. The index covers all corpus chunks: dev files were never trained on,
             but at answer time the model may read them.
  loss       Answer-only cross-entropy on those questions with and without the top retrieved chunk placed before
             the question, within the 512-token context (the chunk is trimmed to fit).

  uv run --locked qos retrieval retrieval [--k 1 3 5]
  uv run --locked qos retrieval loss --load CKPT --n 300 --device cpu
"""
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*|\d+')
STOP = set('the a an of to in and or is are be for on with as by it this that what how does do which from at when '
           'you your can its into if not use used using sol qos fp risc'.split())


def tokens(text):
  out = []
  for w in WORD.findall(text.lower()):
    if w in STOP: continue
    out.append(w)
    if '.' in w: out += [p for p in w.split('.') if p and p not in STOP]
  return out


class BM25:
  def __init__(self, docs, k1=1.4, b=.75):
    self.docs = docs; self.k1, self.b = k1, b
    self.tf = [Counter(tokens(d['text'])) for d in docs]
    self.len = [sum(c.values()) for c in self.tf]; self.avg = sum(self.len) / len(self.len)
    df = Counter(w for c in self.tf for w in c)
    self.idf = {w: math.log(1 + (len(docs) - n + .5) / (n + .5)) for w, n in df.items()}

  def search(self, query, k=3):
    q = tokens(query); scores = []
    for i, c in enumerate(self.tf):
      s = 0.
      for w in q:
        if w in c:
          f = c[w]; s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
      scores.append((s, i))
    return [self.docs[i] for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def load(args):
  from qos_ai_data import read_rows, row_split
  corpus = read_rows(args.corpus)
  docs = [{'path': r['path'], 'text': f"# File: {r['path']}\n{r['text']}"} for r in corpus]
  manifest = json.loads(Path(args.splits).read_text())
  qa = [r for r in read_rows(args.qa) if row_split(r, manifest) == 'dev']
  random.Random(20261021).shuffle(qa)
  return docs, qa[:args.n]


def retrieval(args):
  docs, qa = load(args)
  index = BM25(docs)
  hits = {k: 0 for k in args.k}
  for r in qa:
    found = [d['path'] for d in index.search(r['question'], max(args.k))]
    for k in args.k:
      hits[k] += r['path'] in found[:k]
  report = {'questions': len(qa), 'chunks': len(docs), **{f'top{k}_source_file': hits[k] / len(qa) for k in args.k}}
  print(json.dumps(report, indent=2))
  if args.output: Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


def loss(args):
  import torch
  import qos_ai as ai
  from qos_ai_finetune import answer_loss, make_batch
  docs, qa = load(args)
  index = BM25(docs)
  device = ai.choose_device(args.device)
  model, ck = ai.load_student(args.load, device); tok = ai.StudentTokenizer(ck['qwen_ids'])
  totals = {'plain': [0., 0], 'retrieved': [0., 0], 'oracle_file': [0., 0]}
  hit = 0
  for r in qa:
    full = tok.encode(ai.chat_text(r['question'], r['answer'])); prompt = tok.encode(ai.chat_text(r['question']))
    if len(full) > 400: continue
    room = 511 - len(full)
    top = index.search(r['question'], 1)
    same = [d for d in docs if d['path'] == r['path']]
    variants = {'plain': '', 'retrieved': top[0]['text'] if top else '',
                'oracle_file': random.Random(len(r['question'])).choice(same)['text'] if same else ''}
    hit += bool(top) and top[0]['path'] == r['path']
    for name, context in variants.items():
      ctx = tok.encode(f'Reference:\n{context}\n\n')[:max(0, room)] if context else []
      ids = ctx + full; learn = [False] * (len(ctx) + len(prompt)) + [True] * (len(full) - len(prompt))
      batch, mask = make_batch([(ids, learn)], ck['pad_id'], device)
      with torch.no_grad():
        l, n = answer_loss(model, batch, mask)
      totals[name][0] += l.item(); totals[name][1] += int(n)
  report = {'questions': len(qa), 'top1_source_file': hit / len(qa),
            **{f'{k}_answer_perplexity': math.exp(v[0] / v[1]) for k, v in totals.items()}}
  print(json.dumps(report, indent=2))
  if args.output: Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  for name in ('retrieval', 'loss'):
    c = s.add_parser(name)
    c.add_argument('--corpus', default='qos_ai/corpus.jsonl'); c.add_argument('--qa', default='qos_ai/pod_pt/qa.jsonl')
    c.add_argument('--splits', default='qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/splits.json')
    c.add_argument('--n', type=int, default=500); c.add_argument('--output')
  s.choices['retrieval'].add_argument('--k', type=int, nargs='+', default=[1, 3, 5])
  s.choices['loss'].add_argument('--load', required=True); s.choices['loss'].add_argument('--device', default='cpu')
  a = p.parse_args()
  retrieval(a) if a.command == 'retrieval' else loss(a)
