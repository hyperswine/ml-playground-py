"""Hybrid retrieval for documentation questions: BM25 plus Qwen3-Embedding-0.6B vectors, fused by reciprocal rank.

BM25 alone finds the question's source file first for 56% of held-out dev questions. Dense vectors catch paraphrases
that share few words with the source; reciprocal-rank fusion keeps BM25's exact-name matches.

  embed     embed every corpus chunk (document side, no instruction)                               [Mac, MPS]
  evaluate  source-file hit rates at k = 1, 3, 5 for BM25, dense and hybrid on the held-out dev questions

  uv run --locked qos hybrid embed --output qos_ai/retrieval_20261022/embeddings.pt
  uv run --locked qos hybrid evaluate --embeddings qos_ai/retrieval_20261022/embeddings.pt --output report.json
"""
import argparse
import hashlib
import json
from pathlib import Path

EMBEDDER = 'Qwen/Qwen3-Embedding-0.6B'
TASK = ('Given a question about the FP-RISC compiler, the Sol language or the QOS operating system, '
        'retrieve the documentation or source passage that answers it')


class Embedder:
  def __init__(self, device='auto'):
    import torch
    from transformers import AutoModel, AutoTokenizer
    self.torch = torch
    self.device = torch.device(device if device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    self.tok = AutoTokenizer.from_pretrained(EMBEDDER, padding_side='left')
    self.model = AutoModel.from_pretrained(EMBEDDER, dtype=torch.float32).to(self.device).eval()

  def __call__(self, texts, query=False, batch_size=8, max_length=1024):
    torch = self.torch
    if query: texts = [f'Instruct: {TASK}\nQuery:{t}' for t in texts]
    out = []
    for s in range(0, len(texts), batch_size):
      enc = self.tok(texts[s:s + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(self.device)
      with torch.inference_mode():
        h = self.model(**enc).last_hidden_state[:, -1]  # left padding: the last position is each text's final token
      out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out)


def corpus_sha(rows):
  return hashlib.sha256(json.dumps([[r['path'], r['text']] for r in rows]).encode()).hexdigest()


class Hybrid:
  """search(query, k) over `docs` (each with 'path', 'text' and optionally 'index' into the embedded corpus)."""
  def __init__(self, docs, embeddings, corpus_rows=None, rrf=60, device='auto'):
    import torch
    from qos_ai_retrieval import BM25
    data = torch.load(embeddings, weights_only=True)
    if corpus_rows is not None and data['corpus_sha256'] != corpus_sha(corpus_rows):
      raise ValueError('embeddings were computed for a different corpus')
    self.docs, self.rrf = docs, rrf
    self.vectors = data['vectors'][[d.get('index', i) for i, d in enumerate(docs)]]
    self.bm25 = BM25(docs); self.embedder = None; self.device = device

  def rankings(self, query):
    import torch
    if self.embedder is None: self.embedder = Embedder(self.device)
    lexical = {id(d): r for r, d in enumerate(self.bm25.search(query, len(self.docs)))}
    scores = (self.vectors @ self.embedder([query], query=True)[0]).tolist()
    dense = {i: r for r, i in enumerate(sorted(range(len(self.docs)), key=lambda i: -scores[i]))}
    return lexical, dense

  def search(self, query, k=3, mode='hybrid'):
    lexical, dense = self.rankings(query)
    def score(i):
      d = self.docs[i]; s = 0.
      if mode in ('hybrid', 'bm25') and id(d) in lexical: s += 1 / (self.rrf + lexical[id(d)])
      if mode in ('hybrid', 'dense'): s += 1 / (self.rrf + dense[i])
      return s
    order = sorted(range(len(self.docs)), key=lambda i: -score(i))
    return [self.docs[i] for i in order[:k]]


def embed(args):
  import torch
  from qos_ai_data import read_rows
  rows = read_rows(args.corpus)
  vectors = Embedder(args.device)([f"# File: {r['path']}\n{r['text']}" for r in rows])
  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  torch.save({'vectors': vectors, 'corpus_sha256': corpus_sha(rows), 'model': EMBEDDER, 'task': TASK}, args.output)
  print(json.dumps({'chunks': len(rows), 'dimensions': vectors.shape[1], 'output': str(args.output)}))


def evaluate(args):
  from qos_ai_data import read_rows
  from qos_ai_retrieval import load
  docs, qa = load(args)
  index = Hybrid(docs, args.embeddings, corpus_rows=read_rows(args.corpus), device=args.device)
  ks = (1, 3, 5); hits = {m: {k: 0 for k in ks} for m in ('bm25', 'dense', 'hybrid')}
  for r in qa:
    for mode in hits:
      found = [d['path'] for d in index.search(r['question'], max(ks), mode)]
      for k in ks: hits[mode][k] += r['path'] in found[:k]
  report = {'questions': len(qa), 'chunks': len(docs), 'embedder': EMBEDDER,
            **{f'{m}_top{k}_source_file': round(v[k] / len(qa), 4) for m, v in hits.items() for k in ks}}
  print(json.dumps(report, indent=2))
  if args.output: Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  for name in ('embed', 'evaluate'):
    c = s.add_parser(name); c.add_argument('--corpus', default='qos_ai/corpus.jsonl'); c.add_argument('--device', default='auto')
  s.choices['embed'].add_argument('--output', required=True)
  e = s.choices['evaluate']; e.add_argument('--embeddings', required=True); e.add_argument('--output')
  e.add_argument('--qa', default='qos_ai/pod_pt/qa.jsonl'); e.add_argument('--n', type=int, default=300)
  e.add_argument('--splits', default='qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/splits.json')
  a = p.parse_args()
  {'embed': embed, 'evaluate': evaluate}[a.command](a)
