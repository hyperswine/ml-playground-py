"""Retrieval-augmented answers for QOS/Sol questions: BM25 over documentation chunks, measured two ways.

  build      Training examples with retrieved context: training-split 4B questions, chunks retrieved only from
             training-partition files, the reference placed in the user turn (`with_reference`), answer-only loss.
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


def with_reference(question, chunk):
  """The user-turn format used for both retrieval training and retrieval-augmented answering."""
  return f'Reference:\n{chunk}\n\nQuestion: {question}'


def build(args):
  import qos_ai as ai
  from tokenizers import Tokenizer
  import numpy as np
  import torch
  from qos_ai_data import read_rows, row_split
  manifest = json.loads(Path(args.splits).read_text())
  corpus = [r for r in read_rows(args.corpus) if manifest['files'].get(r['path']) == 'train']
  docs = [{'path': r['path'], 'text': f"# File: {r['path']}\n{r['text']}"} for r in corpus]
  index = BM25(docs)
  qa = [r for r in read_rows(args.qa) if row_split(r, manifest) == 'train']
  random.Random(20261022).shuffle(qa)
  v = torch.load(args.vocabulary, weights_only=True); qwen = Tokenizer.from_file(str(Path(args.tokenizer) / 'tokenizer.json'))
  lookup = np.zeros(max(qwen.get_vocab_size(), int(v['qwen_ids'].max()) + 1), dtype=np.int64)
  lookup[v['qwen_ids'][1:].numpy()] = np.arange(1, len(v['qwen_ids']))
  enc = lambda t: lookup[np.asarray(qwen.encode(t, add_special_tokens=False).ids, dtype=np.int64)].tolist()
  out, same_file = [], 0
  for r in qa:
    top = index.search(r['question'], 1)
    if not top: continue
    bare = enc(ai.chat_text(with_reference(r['question'], ''), r['answer']))
    room = 500 - len(bare)
    if room < 80: continue
    chunk_ids = enc(top[0]['text'])[:room]
    chunk = qwen.decode([int(v['qwen_ids'][i]) for i in chunk_ids if i], skip_special_tokens=False)
    q = with_reference(r['question'], chunk)
    if len(enc(ai.chat_text(q, r['answer']))) > 512 or 0 in enc(ai.chat_text(q, r['answer'])): continue
    same_file += top[0]['path'] == r['path']
    out.append({'question': q, 'answer': r['answer'], 'source_path': r['path'], 'retrieved_path': top[0]['path']})
    if len(out) == args.count: break
  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, 'w') as f:
    for i, r in enumerate(out): f.write(json.dumps({'id': f'rag-{i:05d}', **r}) + '\n')
  report = {'examples': len(out), 'retrieved_from_source_file': same_file / max(1, len(out)),
            'index': 'training-partition chunks only', 'chunks': len(docs)}
  Path(str(args.output) + '.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


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
      if args.format == 'user' and context:
        q_ids = tok.encode(ai.chat_text(with_reference(r['question'], '')))
        ctx = tok.encode(context)[:max(0, 511 - len(full) - (len(q_ids) - len(prompt)))]
        ids_q = tok.encode(ai.chat_text(with_reference(r['question'], tok.decode(ctx))))
        ids = ids_q + full[len(prompt):]; learn = [False] * len(ids_q) + [True] * (len(full) - len(prompt))
        ids, learn = ids[:512], learn[:512]
      else:
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
  for name in ('retrieval', 'loss', 'build'):
    c = s.add_parser(name)
    c.add_argument('--corpus', default='qos_ai/corpus.jsonl'); c.add_argument('--qa', default='qos_ai/pod_pt/qa.jsonl')
    c.add_argument('--splits', default='qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/splits.json')
    c.add_argument('--n', type=int, default=500); c.add_argument('--output')
  s.choices['retrieval'].add_argument('--k', type=int, nargs='+', default=[1, 3, 5])
  s.choices['loss'].add_argument('--load', required=True); s.choices['loss'].add_argument('--device', default='cpu')
  s.choices['loss'].add_argument('--format', choices=['prefix', 'user'], default='prefix')
  b = s.choices['build']; b.add_argument('--count', type=int, default=4000)
  b.add_argument('--vocabulary', default='runpod/557m/bundle/vocabulary.pt'); b.add_argument('--tokenizer', default='runpod/557m/bundle/tokenizer')
  a = p.parse_args()
  {'retrieval': retrieval, 'loss': loss, 'build': build}[a.command](a)
