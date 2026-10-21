"""Use the fine-tuned Qwen3.5-0.8B Sol model locally (M1/M4 MPS or CPU): answer, execute, check, retrieve.

  ask        answer questions, then run the answer's Sol code (or the question's program) with `qos_ai_execute`
  mps-check  re-answer a deterministic sample of the frozen prompts locally and compare with the GPU-pod answers
  rag-loss   answer-only perplexity on held-out dev questions with 0, 1 or 3 BM25-retrieved chunks in the user turn,
             using Qwen's longer context (references trimmed to --reference-tokens)

  python qos_ai_qwen_chat.py ask --model DIR --fpr FPR --retrieve 3 --question "..."
  python qos_ai_qwen_chat.py mps-check --model DIR --prompts prompts.jsonl --reference tuned_answers.jsonl --n 120
  python qos_ai_qwen_chat.py rag-loss --model DIR --n 300
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

BASE = 'Qwen/Qwen3.5-0.8B'
CODE_REQUEST = re.compile(r'\b(write|program|expression|implement|fix)\b', re.I)


def chat_text(question, answer=None):
  text = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  return text if answer is None else text + f'{answer}<|im_end|>'


def load(model, device_name, dtype_name='float32'):
  import torch
  from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
  device = torch.device(device_name if device_name != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
  dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_name]
  tok = AutoTokenizer.from_pretrained(BASE)
  m = Qwen3_5ForConditionalGeneration.from_pretrained(model, dtype=dtype).to(device).eval()
  return m, tok, device


def generate(m, tok, device, questions, budget, batch_size=8):
  import torch
  stop = [tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')]
  out = []
  for s in range(0, len(questions), batch_size):
    part = questions[s:s + batch_size]
    enc = [tok.encode(chat_text(q), add_special_tokens=False) for q in part]; w = max(map(len, enc))
    ids = torch.full((len(part), w), stop[1], dtype=torch.long); att = torch.zeros_like(ids)
    for j, e in enumerate(enc): ids[j, w - len(e):] = torch.tensor(e); att[j, w - len(e):] = 1
    with torch.inference_mode():
      g = m.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=budget,
                     do_sample=False, eos_token_id=stop, pad_token_id=stop[1])
    for row in g[:, w:].tolist():
      end = next((i for i, t in enumerate(row) if t in stop), None)
      out.append(tok.decode(row if end is None else row[:end], skip_special_tokens=True).strip())
  return out


def ask(args):
  """Answer, then execute. With --retrieve K, questions without a code block get the top-K BM25 chunks of the
  documentation corpus as a reference (the retrieval-training format). Questions that contain code or ask for a
  program or expression are asked as written: on the 2026-10-22 demo, references turned a correct program into a
  string-interpolation one."""
  from qos_ai_execute import BLOCK, augment
  m, tok, device = load(args.model, args.device, args.dtype)
  questions = [args.question] if args.question else [q.strip() for q in Path(args.questions).read_text().split('\n---\n') if q.strip()]
  prompts, sources = list(questions), [[] for _ in questions]
  if args.retrieve:
    from qos_ai_data import read_rows
    from qos_ai_retrieval import BM25, with_reference
    index = BM25([{'path': r['path'], 'text': f"# File: {r['path']}\n{r['text']}"} for r in read_rows(args.corpus)])
    for i, q in enumerate(questions):
      if BLOCK.search(q) or CODE_REQUEST.search(q): continue  # references distract program writing
      hits = index.search(q, args.retrieve)
      if not hits: continue
      ref = tok.decode(tok.encode('\n\n'.join(h['text'] for h in hits), add_special_tokens=False)[:args.reference_tokens])
      prompts[i], sources[i] = with_reference(q, ref), [h['path'] for h in hits]
  for q, src, a in zip(questions, sources, generate(m, tok, device, prompts, args.max_new_tokens)):
    r = augment(q, a, args.fpr, drop_wrong_steps=True) if args.fpr else {'answer': a, 'note': 'not executed'}
    ref = f'--- references: {", ".join(src)} ---\n' if src else ''
    print(f'Q: {q}\n{ref}--- model ---\n{a}\n--- executed ({r["note"]}) ---\n{r["answer"]}\n', flush=True)


def mps_check(args):
  rows = [json.loads(l) for l in open(args.prompts)]
  reference = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(args.reference)}
  random.Random(20261022).shuffle(rows); rows = rows[:args.n]
  m, tok, device = load(args.model, args.device, args.dtype)
  started = time.perf_counter(); results = []
  for budget in sorted({r['max_new_tokens'] for r in rows}):
    part = [r for r in rows if r['max_new_tokens'] == budget]
    for r, a in zip(part, generate(m, tok, device, [r['question'] for r in part], budget)):
      ref = reference[r['id']]
      results.append({'id': r['id'], 'local': a, 'pod': ref, 'identical': a == ref,
                      'same_first_line': a.split('\n')[0] == ref.split('\n')[0]})
  seconds = time.perf_counter() - started
  report = {'device': str(device), 'dtype': args.dtype, 'n': len(results), 'identical': sum(r['identical'] for r in results),
            'same_first_line': sum(r['same_first_line'] for r in results), 'seconds': round(seconds, 1),
            'seconds_per_answer': round(seconds / len(results), 2)}
  Path(args.output).write_text(json.dumps({'report': report, 'answers': results}, indent=1) + '\n')
  print(json.dumps(report, indent=2))


def rag_loss(args):
  """Same 300 dev questions, index and user-turn format as the 557M Stage 5 measurement (`qos retrieval loss
  --format user`). Conditions: no reference; top-1 chunk cut to fit 512 tokens in all (what the 557M can see); top-1
  chunk and top-3 chunks up to --reference-tokens; a random chunk of the question's own source file."""
  import torch
  from torch.nn import functional as F
  from qos_ai_retrieval import BM25, load as load_questions, with_reference
  docs, qa = load_questions(args)
  index = BM25(docs)
  m, tok, device = load(args.model, args.device, args.dtype)
  lm, head = m.model.language_model, m.lm_head
  enc = lambda t: tok.encode(t, add_special_tokens=False)
  names = ('plain', 'top1_fit512', 'top1', 'top3', 'oracle_file')
  totals = {c: [0., 0] for c in names}; hit = used = 0; started = time.perf_counter()
  for r in qa:
    answer_ids = enc(r['answer'] + '<|im_end|>')
    if len(enc(chat_text(r['question'], r['answer']))) > 400: continue
    used += 1
    hits = index.search(r['question'], 3); hit += bool(hits) and hits[0]['path'] == r['path']
    same = [d for d in docs if d['path'] == r['path']]
    bare = len(enc(chat_text(with_reference(r['question'], '')))) + len(answer_ids)
    refs = {'plain': None,
            'top1_fit512': enc(hits[0]['text'])[:max(0, 512 - bare)] if hits else [],
            'top1': enc(hits[0]['text'])[:args.reference_tokens] if hits else [],
            'top3': enc('\n\n'.join(h['text'] for h in hits))[:args.reference_tokens],
            'oracle_file': enc(random.Random(len(r['question'])).choice(same)['text'])[:args.reference_tokens] if same else []}
    for name, ref in refs.items():
      q = r['question'] if ref is None else with_reference(r['question'], tok.decode(ref))
      prompt = enc(chat_text(q))
      ids = torch.tensor([prompt + answer_ids], device=device)
      with torch.inference_mode():
        h = lm(input_ids=ids).last_hidden_state[0, len(prompt) - 1:-1]
        loss = F.cross_entropy(head(h).float(), ids[0, len(prompt):], reduction='sum')
      totals[name][0] += loss.item(); totals[name][1] += len(answer_ids)
  report = {'model': str(args.model), 'device': str(device), 'questions': used, 'top1_source_file': hit / max(1, used),
            'reference_tokens': args.reference_tokens, 'seconds': round(time.perf_counter() - started),
            **{f'{c}_answer_perplexity': math.exp(v[0] / v[1]) for c, v in totals.items()}}
  Path(args.output).write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  for name in ('ask', 'mps-check', 'rag-loss'):
    c = s.add_parser(name); c.add_argument('--model', required=True)
    c.add_argument('--device', default='auto'); c.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'])
  a = s.choices['ask']; a.add_argument('--fpr'); a.add_argument('--question'); a.add_argument('--questions')
  a.add_argument('--max-new-tokens', type=int, default=200)
  a.add_argument('--retrieve', type=int, default=0, help='top-K documentation chunks for questions without code')
  a.add_argument('--reference-tokens', type=int, default=1500); a.add_argument('--corpus', default='qos_ai/corpus.jsonl')
  c = s.choices['mps-check']; c.add_argument('--prompts', required=True); c.add_argument('--reference', required=True)
  c.add_argument('--n', type=int, default=120); c.add_argument('--output', required=True)
  r = s.choices['rag-loss']; r.add_argument('--n', type=int, default=300); r.add_argument('--reference-tokens', type=int, default=1500)
  r.add_argument('--corpus', default='qos_ai/corpus.jsonl'); r.add_argument('--qa', default='qos_ai/pod_pt/qa.jsonl')
  r.add_argument('--splits', default='qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/splits.json')
  r.add_argument('--output', required=True)
  args = p.parse_args()
  {'ask': ask, 'mps-check': mps_check, 'rag-loss': rag_loss}[args.command](args)
