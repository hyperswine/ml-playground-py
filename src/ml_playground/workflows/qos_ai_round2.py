"""Round 2 data for the fine-tuned Qwen: find what it gets wrong, keep what the compiler confirms, add long references.

Program tasks (predict the output, write, repair) have one right answer, which the generator already knows with an
exact trace; sampling there only finds the tasks the model fails, and those failures (plus a share of successes)
keep the generator's reference answer. Expression tasks have many right answers; there the model's own distinct
answers that pass all six compiler-checked cases are kept, falling back to the reference. Conditional expressions
are generated on purpose and must take both branches with different results on the cases.

  tasks   fresh tasks (new seeds; excludes the frozen tests, all training programs and earlier self-sampling tasks)
  sample  answer each task with the fine-tuned Qwen: greedy for programs, K samples for expressions      [Mac, MPS]
  select  judge every answer with fpr and write the round-2 rows
  rag     retrieval examples with up to --reference-tokens of top-1..3 references (training partition only)

  uv run --locked qos round2 tasks --fpr FPR --output DIR
  python qos_ai_round2.py sample --model MODEL --tasks DIR/tasks.jsonl --output DIR/samples.0.jsonl --shard 0/2
  uv run --locked qos round2 select --fpr FPR --tasks DIR/tasks.jsonl --samples DIR/samples.*.jsonl --output DIR/rows.jsonl
  uv run --locked qos round2 rag --output DIR/rag.jsonl [--embeddings EMB]
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATED = re.compile(r'It prints (.*?)\.(?:\s+Steps:|\s*$)', re.S)
BLOCK = re.compile(r'```(?:sol)?\s*\n(.*?)```', re.S)
BIG = re.compile(r'\b(6[1-9]|[7-9]\d|1[0-4]\d|150)\b')  # the frozen large-number stratum


def param_answer(expr, case, expected):
  c = case
  return (f'```sol\n{expr}\n```\nFor xs = [{", ".join(map(str, c["xs"]))}], k = {c["k"]} and limit = {c["limit"]} '
          f'it evaluates to {expected}.')


def conditional(gen, rng, cmp_ops):
  """An if-expression whose branches are full integer expressions (sum, abs, max, min, clamp)."""
  cond = (rng.choice(cmp_ops), rng.choice([('var', 'k'), ('var', 'limit'), ('sum', ('xs',))]), rng.randint(-3, 10))
  return ('if', cond, gen.int_(rng.choice([1, 2, 2])), gen.int_(rng.choice([1, 2, 2])))


def build_tasks(args):
  import qos_ai_stage4_eval as ev
  import qos_ai_stage4_param as s4
  import qos_ai_stage5_param as s5
  import qos_ai_solgen as sg
  if args.output.exists(): raise ValueError('choose a new output directory')
  fpr = str(Path(args.fpr).expanduser())
  suite = json.loads(s4.SUITE.read_text()); refs = {r['reference'] for r in suite['rows']}
  seen_exprs, frozen = set(), set()
  for path in args.exclude_param:
    seen_exprs |= {json.loads(l)['expression'] for l in open(path)}
  for t in map(json.loads, open(args.exclude_tasks)):
    (seen_exprs if t['kind'] == 'param' else frozen).add(t['reference'])
  for name in ('train', 'test_fresh', 'test_held_pairs', 'test_big_numbers'):
    frozen |= {json.loads(l)['code'] for l in open(Path(args.solgen) / f'{name}.jsonl')}
  banned = {s4.skeleton(t) for t in [
    ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('filter', '>', 'limit', ('xs',))),
    ('sum', ('map', ('clamp', 0, 10), ('xs',))),
    ('if', ('>', ('var', 'k'), 0), ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('xs',))),
    ('sum', ('map', ('op', '*', 'k'), ('filter', '>', 'limit', ('xs',)))),
    ('if', ('>', ('var', 'k'), 0), ('max', ('var', 'k'), ('sum', ('xs',))), ('abs', ('sum', ('xs',))))]}
  grams = s5.suite_ngrams()
  rng = random.Random(args.seed); gen = s4.Gen(rng); phr = s5.Phraser(random.Random(args.seed + 1))
  tasks, counts = [], {'param_conditional': 0, 'param': 0}
  while counts['param_conditional'] < args.conditional or counts['param'] < args.param:
    want_cond = counts['param_conditional'] < args.conditional
    t = conditional(gen, rng, list(s4.CMP)) if want_cond else gen.int_(rng.choice([1, 2, 2, 3]))
    if not want_cond and t[0] == 'if': continue
    expr = s4.render(t)
    if expr in seen_exprs or expr in refs or s4.skeleton(t) in banned: continue
    cs = s4.cases(rng)
    try:
      expected = [str(s4.evaluate(t, c)) for c in cs]
      if want_cond:  # both branches must matter on these cases
        then_v = [str(s4.evaluate(t[2], c)) for c in cs]; else_v = [str(s4.evaluate(t[3], c)) for c in cs]
        if expected == then_v or expected == else_v: continue
    except Exception:
      continue
    if any(abs(int(v)) > 999 for v in expected) or len(set(expected)) < 2: continue
    try: d, mood = phr.describe(t)
    except Exception: continue
    words = re.findall(r'\w+', d.lower())
    if any(tuple(words[i:i + 6]) in grams for i in range(len(words) - 5)): continue
    seen_exprs.add(expr)
    key = 'param_conditional' if want_cond else 'param'; counts[key] += 1
    q = rng.choice(s5.STEP_PROMPTS if mood == 'steps' else s5.PROMPTS).format(d=d) + ' Return the expression in a sol code block.'
    tasks.append({'id': f'r2-{key}-{len(tasks):05d}', 'kind': 'param', 'conditional': want_cond, 'question': q,
                  'reference': expr, 'answer': param_answer(expr, cs[1], expected[1]),
                  'cases': [{'inputs': c, 'expected': e} for c, e in zip(cs, expected)]})
  g = sg.Gen(random.Random(args.seed + 2), banned=frozenset(sg.HELD_PAIRS)); trng = random.Random(args.seed + 3)
  kinds = {'output': .5, 'generate': .3, 'repair': .2}
  while sum(t['kind'] != 'param' for t in tasks) < args.programs:
    try: node = g.program()
    except sg.Bad: continue
    kind = trng.choices(list(kinds), list(kinds.values()))[0]
    item = sg.task(node, kind, trng)
    if not item or item['code'] in frozen or BIG.search(item['code']): continue
    frozen.add(item['code'])
    tasks.append({'id': f'r2-{kind}-{len(tasks):05d}', 'kind': kind, 'question': item['question'],
                  'expected': item['expected'], 'reference': item['code'], 'answer': item['answer']})
  def ok(t):
    if t['kind'] == 'param':
      try: node = ev.parse_expression(f"```sol\n{t['reference']}\n```", ('xs', 'k', 'limit'))
      except Exception: return False
      return all(ev.run_expression(fpr, node, c['inputs']) == c['expected'] for c in t['cases'])
    return sg.run_fpr(fpr, t['reference']) == t['expected']
  with ThreadPoolExecutor(8) as pool:
    flags = list(pool.map(ok, tasks))
  tasks = [t for t, f in zip(tasks, flags) if f]
  args.output.mkdir(parents=True)
  with open(args.output / 'tasks.jsonl', 'w') as f:
    for t in tasks: f.write(json.dumps(t) + '\n')
  kinds_out = {}
  for t in tasks:
    k = 'param_conditional' if t.get('conditional') else t['kind']; kinds_out[k] = kinds_out.get(k, 0) + 1
  report = {'tasks': len(tasks), 'kinds': kinds_out, 'reference_failures': flags.count(False),
            'seed': args.seed, 'compiler_sha256': ev.sha(fpr), 'tasks_sha256': ev.sha(args.output / 'tasks.jsonl')}
  (args.output / 'tasks.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


def sample(args):
  import time
  import torch
  from qos_ai_qwen_chat import generate, load
  tasks = [json.loads(l) for l in open(args.tasks)]
  i, n = map(int, args.shard.split('/')); tasks = tasks[i::n]
  done = {json.loads(l)['id'] for l in open(args.output)} if args.output.exists() else set()
  tasks = [t for t in tasks if t['id'] not in done]
  m, tok, device = load(args.model, args.device, 'float32')
  torch.manual_seed(args.seed + i)
  started, total = time.perf_counter(), 0
  with open(args.output, 'a') as out:
    for greedy in (True, False):
      part = [t for t in tasks if (t['kind'] != 'param') == greedy]
      step = args.batch_size if greedy else max(1, args.batch_size // args.k)
      for s in range(0, len(part), step):
        chunk = part[s:s + step]; k = 1 if greedy else args.k
        answers = generate(m, tok, device, [t['question'] for t in chunk], args.max_new_tokens, batch_size=len(chunk),
                           k=k, temperature=None if greedy else args.temperature, stop_loops=True)
        for j, t in enumerate(chunk):
          out.write(json.dumps({'id': t['id'], 'samples': answers[j * k:(j + 1) * k]}) + '\n')
        out.flush(); total += len(chunk)
        print(f'{total}/{len(tasks)} tasks, {(time.perf_counter() - started) / total:.2f} s/task', flush=True)


def select(args):
  import qos_ai_stage4_eval as ev
  import qos_ai_solgen as sg
  fpr = str(Path(args.fpr).expanduser())
  tasks = {t['id']: t for t in map(json.loads, open(args.tasks))}
  samples = {}
  for path in args.samples:
    for r in map(json.loads, open(path)): samples[r['id']] = r['samples']
  missing = set(tasks) - set(samples)
  if missing and not args.allow_missing: raise ValueError(f'{len(missing)} tasks have no samples')
  def judge(item):
    t, a = item
    if t['kind'] == 'param':
      try: node = ev.parse_expression(a, ('xs', 'k', 'limit'))
      except Exception: return False
      return all(ev.run_expression(fpr, node, c['inputs']) == c['expected'] for c in t['cases'])
    m = STATED.search(a)
    if m is None or m.group(1).strip() != t['expected']: return False
    if t['kind'] == 'output': return True
    blocks = BLOCK.findall(a)
    return len(blocks) == 1 and sg.run_fpr(fpr, blocks[0].strip()) == t['expected']
  pairs = [(tasks[i], a) for i in tasks if i in samples for a in samples[i]]
  with ThreadPoolExecutor(8) as pool:
    verdicts = list(pool.map(judge, pairs))
  right = {}
  for (t, a), v in zip(pairs, verdicts):
    right.setdefault(t['id'], []).append((a, v))
  rng = random.Random(args.seed); rows, stats = [], {}
  for tid, t in tasks.items():
    if tid not in right: continue
    results = right[tid]; correct = [a for a, v in results if v]
    key = ('param_conditional' if t.get('conditional') else 'param') if t['kind'] == 'param' else t['kind']
    s = stats.setdefault(key, {'tasks': 0, 'solved': 0, 'samples': 0, 'correct_samples': 0, 'rows': 0})
    s['tasks'] += 1; s['solved'] += bool(correct); s['samples'] += len(results); s['correct_samples'] += len(correct)
    if t['kind'] == 'param':
      keep = list(dict.fromkeys(correct))[:args.per_task] or [t['answer']]
      for j, a in enumerate(keep):
        rows.append({'id': f'{tid}-{j}', 'question': t['question'], 'answer': a,
                     'source': 'self_sample' if correct else 'reference'})
      s['rows'] += len(keep)
    elif not correct or rng.random() < args.keep_solved:
      rows.append({'id': tid, 'question': t['question'], 'answer': t['answer'],
                   'source': 'reference_failed' if not correct else 'reference_solved'})
      s['rows'] += 1
  with open(args.output, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
  report = {'rows': len(rows), 'by_kind': stats, 'sources': {k: sum(r['source'] == k for r in rows) for k in
            ('self_sample', 'reference', 'reference_failed', 'reference_solved')},
            'keep_solved': args.keep_solved, 'per_task': args.per_task, 'compiler_sha256': ev.sha(fpr),
            'rows_sha256': ev.sha(args.output)}
  Path(str(args.output) + '.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


def rag(args):
  from tokenizers import Tokenizer
  from qos_ai_data import read_rows, row_split
  from qos_ai_retrieval import BM25, with_reference
  manifest = json.loads(Path(args.splits).read_text())
  corpus = read_rows(args.corpus)
  docs = [{'path': r['path'], 'text': f"# File: {r['path']}\n{r['text']}", 'index': i} for i, r in enumerate(corpus)
          if manifest['files'].get(r['path']) == 'train']
  if args.embeddings:
    from qos_ai_hybrid import Hybrid
    index = Hybrid(docs, args.embeddings, corpus_rows=corpus)
  else:
    index = BM25(docs)
  used = {r['question'].split('\n\nQuestion: ', 1)[-1] for r in read_rows(args.exclude)} if args.exclude else set()
  qa = [r for r in read_rows(args.qa) if row_split(r, manifest) == 'train' and r['question'] not in used]
  rng = random.Random(args.seed); rng.shuffle(qa)
  tok = Tokenizer.from_file(str(Path(args.tokenizer) / 'tokenizer.json'))
  out, same = [], 0
  for r in qa:
    k = rng.choice([1, 2, 3, 3])
    hits = index.search(r['question'], k)
    if not hits: continue
    ids = tok.encode('\n\n'.join(h['text'] for h in hits), add_special_tokens=False).ids[:args.reference_tokens]
    if len(ids) < 64: continue
    same += hits[0]['path'] == r['path']
    out.append({'id': f'r2-rag-{len(out):05d}', 'question': with_reference(r['question'], tok.decode(ids)),
                'answer': r['answer'], 'source': 'rag', 'references': len(hits), 'retrieved_path': hits[0]['path'],
                'source_path': r['path']})
    if len(out) == args.count: break
  with open(args.output, 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
  report = {'examples': len(out), 'retriever': 'hybrid' if args.embeddings else 'bm25', 'chunks': len(docs),
            'top1_from_source_file': same / max(1, len(out)), 'reference_tokens': args.reference_tokens,
            'excluded_stage5_questions': len(used), 'index': 'training-partition chunks and questions only'}
  Path(str(args.output) + '.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  t = s.add_parser('tasks'); t.add_argument('--fpr', required=True); t.add_argument('--output', type=Path, required=True)
  t.add_argument('--programs', type=int, default=3000); t.add_argument('--param', type=int, default=600)
  t.add_argument('--conditional', type=int, default=900); t.add_argument('--seed', type=int, default=20261101)
  t.add_argument('--solgen', default='qos_ai/solgen_20261020'); t.add_argument('--exclude-tasks', default='qos_ai/rft_20261021/tasks.jsonl')
  t.add_argument('--exclude-param', nargs='*', default=['qos_ai_benchmark/stage4_20261021/param/train.jsonl',
                                                         'qos_ai_benchmark/stage5_20261021/param/train.jsonl'])
  sm = s.add_parser('sample'); sm.add_argument('--model', required=True); sm.add_argument('--tasks', type=Path, required=True)
  sm.add_argument('--output', type=Path, required=True); sm.add_argument('--shard', default='0/1')
  sm.add_argument('--k', type=int, default=4); sm.add_argument('--temperature', type=float, default=0.8)
  sm.add_argument('--max-new-tokens', type=int, default=200); sm.add_argument('--batch-size', type=int, default=16)
  sm.add_argument('--seed', type=int, default=7); sm.add_argument('--device', default='auto')
  v = s.add_parser('select'); v.add_argument('--fpr', required=True); v.add_argument('--tasks', type=Path, required=True)
  v.add_argument('--samples', type=Path, nargs='+', required=True); v.add_argument('--output', type=Path, required=True)
  v.add_argument('--per-task', type=int, default=2); v.add_argument('--keep-solved', type=float, default=0.25)
  v.add_argument('--seed', type=int, default=20261102); v.add_argument('--allow-missing', action='store_true')
  r = s.add_parser('rag'); r.add_argument('--output', type=Path, required=True); r.add_argument('--count', type=int, default=3000)
  r.add_argument('--reference-tokens', type=int, default=1500); r.add_argument('--embeddings')
  r.add_argument('--corpus', default='qos_ai/corpus.jsonl'); r.add_argument('--qa', default='qos_ai/pod_pt/qa.jsonl')
  r.add_argument('--splits', default='qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/splits.json')
  r.add_argument('--exclude', default='qos_ai/stage5_20261021/rag_train.jsonl'); r.add_argument('--seed', type=int, default=20261103)
  r.add_argument('--tokenizer', default='runpod/557m/bundle/tokenizer')
  a = p.parse_args()
  {'tasks': build_tasks, 'sample': sample, 'select': select, 'rag': rag}[a.command](a)
