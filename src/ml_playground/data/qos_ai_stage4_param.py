"""Parameterized-expression curriculum for Stage 4: one reusable Sol expression over xs, k and limit.

The multi-input suite (qos_ai_benchmark/stage4_20261021/multi_input.json) asks for this answer shape, which no
earlier training data teaches. Each generated task is checked with the suite's own restricted parser and runner
(`qos_ai_stage4_eval.parse_expression` / `run_expression`) on six input sets against a Python oracle; any
disagreement is dropped. The suite stays a held-out test:
  * the six suite reference expressions and anything with the same skeleton (same functions, nesting, variables,
    with arithmetic operators, comparisons and constants abstracted) are excluded;
  * none of the suite's three prompt wordings are used.

  build   python3 qos_ai_stage4_param.py build --fpr ~/Documents/GitHub/fprisc/fpr --output DIR
  extend  python3 qos_ai_stage4_param.py extend --pools .../pools.pt --param DIR/train.jsonl --output NEWDIR
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import qos_ai_stage4_eval as ev

SUITE = Path('qos_ai_benchmark/stage4_20261021/multi_input.json')
PROMPTS = [
  'Give a single Sol expression in terms of xs, k and limit that computes {d}.',
  'Using the variables xs, k and limit, write one Sol expression for {d}.',
  'Write a Sol expression over xs, k and limit, with no bindings and no print, for {d}.',
  'Write one Sol expression that works for any xs, k and limit and computes {d}.',
  'Turn this into one Sol expression over xs, k and limit: {d}.',
]
CMP = {'>': 'greater than', '<': 'less than', '>=': 'at least', '<=': 'at most', '!=': 'not equal to'}
PYCMP = {'>': lambda a, b: a > b, '<': lambda a, b: a < b, '>=': lambda a, b: a >= b,
         '<=': lambda a, b: a <= b, '!=': lambda a, b: a != b}


class Bad(Exception):
  pass


def lit(v):
  return f'({v})' if v < 0 else str(v)


def atom(name, env):
  return env[name] if name in ('k', 'limit') else name


# Expression trees are tuples. Rendering, evaluation and description are kept together per node kind.
def render(t):
  kind = t[0]
  if kind == 'xs': return 'xs'
  if kind == 'map': return f'List.map (fn x -> {fn_text(t[1])}) {wrap(render(t[2]))}'
  if kind == 'filter': return f'List.filter (fn x -> x {t[1]} {operand(t[2])}) {wrap(render(t[3]))}'
  if kind in ('take', 'drop'): return f'List.{kind} {t[1]} {wrap(render(t[2]))}'
  if kind == 'rev': return f'List.rev {wrap(render(t[1]))}'
  if kind == 'sum': return f'List.sum {wrap(render(t[1]))}'
  if kind == 'abs': return f'Numeric.abs {wrap(render(t[1]))}'
  if kind in ('max', 'min'): return f'Numeric.{kind} {wrap(render_int(t[1]))} {wrap(render_int(t[2]))}'
  if kind == 'clamp': return f'Numeric.clamp {lit(t[1])} {lit(t[2])} {wrap(render(t[3]))}'
  if kind == 'if': return f'if {render_cond(t[1])} then {render(t[2])} else {render(t[3])}'
  raise ValueError(kind)


def render_int(t):
  return t[1] if t[0] == 'var' else render(t)


def render_cond(c):
  return f'{render_int(c[1])} {c[0]} {operand(c[2])}'


def operand(o):
  return o if isinstance(o, str) else lit(o)


def fn_text(f):
  kind = f[0]
  if kind == 'op': return f'x {f[1]} {operand(f[2])}'
  if kind == 'abs': return 'Numeric.abs x'
  if kind == 'clamp': return f'Numeric.clamp {lit(f[1])} {lit(f[2])} x'
  if kind in ('max', 'min'): return f'Numeric.{kind} x {operand(f[1])}'
  raise ValueError(kind)


def wrap(s):
  return s if re.fullmatch(r'\w+|\(-?\d+\)', s) else f'({s})'


def value(o, env):
  return env[o] if isinstance(o, str) else o


def apply_fn(f, x, env):
  kind = f[0]
  if kind == 'op':
    b = value(f[2], env); return {'+': x + b, '-': x - b, '*': x * b}[f[1]]
  if kind == 'abs': return abs(x)
  if kind == 'clamp': return min(max(x, f[1]), f[2])
  if kind == 'max': return max(x, value(f[1], env))
  if kind == 'min': return min(x, value(f[1], env))


def evaluate(t, env):
  kind = t[0]
  if kind == 'xs': return list(env['xs'])
  if kind == 'var': return env[t[1]]
  if kind == 'map': return [apply_fn(t[1], x, env) for x in evaluate(t[2], env)]
  if kind == 'filter': return [x for x in evaluate(t[3], env) if PYCMP[t[1]](x, value(t[2], env))]
  if kind == 'take': return evaluate(t[2], env)[:t[1]]
  if kind == 'drop': return evaluate(t[2], env)[t[1]:]
  if kind == 'rev': return evaluate(t[1], env)[::-1]
  if kind == 'sum': return sum(evaluate(t[1], env))
  if kind == 'abs': return abs(evaluate(t[1], env))
  if kind == 'max': return max(evaluate(t[1], env), evaluate(t[2], env))
  if kind == 'min': return min(evaluate(t[1], env), evaluate(t[2], env))
  if kind == 'clamp': return min(max(evaluate(t[3], env), t[1]), t[2])
  if kind == 'if':
    c = t[1]; return evaluate(t[2], env) if PYCMP[c[0]](evaluate(c[1], env), value(c[2], env)) else evaluate(t[3], env)
  raise ValueError(kind)


def describe(t):
  kind = t[0]
  if kind == 'xs': return 'xs'
  if kind == 'var': return t[1]
  if kind == 'map': return f'{group(t[2])} with each element {describe_fn(t[1])}'
  if kind == 'filter': return f'the elements of {group(t[3])} that are {CMP[t[1]]} {t[2]}'
  if kind == 'take': return f'the first {t[1]} elements of {group(t[2])}'
  if kind == 'drop': return f'{group(t[2])} without its first {t[1]} elements'
  if kind == 'rev': return f'{group(t[1])} reversed'
  if kind == 'sum': return f'the sum of {group(t[1])}'
  if kind == 'abs': return f'the absolute value of {group(t[1])}'
  if kind == 'max': return f'the larger of {group(t[1])} and {group(t[2])}'
  if kind == 'min': return f'the smaller of {group(t[1])} and {group(t[2])}'
  if kind == 'clamp': return f'{group(t[3])} clamped to the range {t[1]} through {t[2]}'
  if kind == 'if':
    c = t[1]
    return f'{group(t[2])} when {describe(c[1])} is {CMP[c[0]]} {c[2]}, otherwise {group(t[3])}'


def describe_fn(f):
  kind = f[0]
  if kind == 'op': return {'+': 'increased by', '-': 'decreased by', '*': 'multiplied by'}[f[1]] + f' {f[2]}'
  if kind == 'abs': return 'replaced by its absolute value'
  if kind == 'clamp': return f'clamped to {f[1]} through {f[2]}'
  return f'replaced by the {"larger" if kind == "max" else "smaller"} of itself and {f[1]}'


def group(t):
  return describe(t) if t[0] in ('xs', 'var') else f'({describe(t)})'


def skeleton(t):
  """Structure with operators, comparisons and constants abstracted; variable names kept."""
  if isinstance(t, tuple):
    kind = t[0]
    if kind == 'op': return ('op', 'OP', skeleton(t[2]))
    if kind in ('filter',): return ('filter', 'CMP', skeleton(t[2]), skeleton(t[3]))
    if kind in ('take', 'drop'): return (kind, 'N', skeleton(t[2]))
    if kind == 'clamp' and len(t) == 4: return ('clamp', 'N', 'N', skeleton(t[3]))
    if kind == 'clamp': return ('fclamp', 'N', 'N')
    if kind == 'if': c = t[1]; return ('if', ('CMP', skeleton(c[1]), skeleton(c[2])), skeleton(t[2]), skeleton(t[3]))
    return tuple(skeleton(x) for x in t)
  if isinstance(t, int): return 'N'
  return t


def from_suite(node):
  """Translate a parsed suite reference into this module's tree form, for skeleton exclusion."""
  text = ev.render(node)
  return text


class Gen:
  def __init__(self, rng):
    self.r = rng

  def operand(self, allow_vars=True):
    r = self.r
    return r.choice(['k', 'limit']) if allow_vars and r.random() < .6 else r.randint(-5, 12)

  def fn(self):
    r = self.r
    kind = r.choices(['op', 'abs', 'clamp', 'max', 'min'], [.6, .1, .12, .09, .09])[0]
    if kind == 'op': return ('op', r.choice(['+', '-', '*']), self.operand())
    if kind == 'clamp': a = r.randint(-5, 5); return ('clamp', a, a + r.randint(1, 15))
    if kind in ('max', 'min'): return (kind, self.operand())
    return ('abs',)

  def list_(self, depth):
    r = self.r
    if depth <= 0 or r.random() < .3: return ('xs',)
    kind = r.choices(['map', 'filter', 'take', 'drop', 'rev'], [.35, .35, .1, .1, .1])[0]
    inner = self.list_(depth - 1)
    if kind == 'map': return ('map', self.fn(), inner)
    if kind == 'filter': return ('filter', r.choice(list(CMP)), self.operand(), inner)
    if kind in ('take', 'drop'): return (kind, r.randint(1, 3), inner)
    return ('rev', inner)

  def int_(self, depth):
    r = self.r
    kind = r.choices(['sum', 'abs', 'max', 'min', 'clamp', 'if'], [.5, .1, .12, .08, .08, .12])[0]
    if kind == 'sum' or depth <= 1: return ('sum', self.list_(depth))
    if kind == 'abs': return ('abs', ('sum', self.list_(depth - 1)))
    if kind in ('max', 'min'): return (kind, ('var', r.choice(['k', 'limit'])), ('sum', self.list_(depth - 1)))
    if kind == 'clamp': a = r.randint(-20, 10); return ('clamp', a, a + r.randint(5, 60), ('sum', self.list_(depth - 1)))
    cond = (r.choice(list(CMP)), r.choice([('var', 'k'), ('var', 'limit'), ('sum', ('xs',))]), r.randint(-3, 10))
    return ('if', cond, ('sum', self.list_(depth - 1)), ('sum', self.list_(depth - 1)))


def cases(rng):
  out = [{'xs': [], 'k': rng.randint(-3, 5), 'limit': rng.randint(-3, 10)}]
  for _ in range(5):
    xs = [rng.randint(-20, 60) for _ in range(rng.randint(1, 5))]
    limit = rng.choice(xs) if rng.random() < .4 else rng.randint(-10, 40)
    out.append({'xs': xs, 'k': rng.choice([0, rng.randint(-5, -1), rng.randint(1, 6)]), 'limit': limit})
  return out


def build(args):
  if args.output.exists(): raise ValueError('choose a new output directory')
  fpr = str(Path(args.fpr).expanduser())
  suite = json.loads(SUITE.read_text())
  suite_refs = {r['reference'] for r in suite['rows']}
  suite_prompts = {r['question'].split(':')[0][:40] for r in suite['rows']}
  rng = random.Random(20261021); gen = Gen(rng)
  # Suite skeletons, generated from the same tree vocabulary where possible.
  suite_trees = [
    ('sum', ('map', ('op', '*', 'k'), ('xs',))),
    ('sum', ('filter', '>', 'limit', ('xs',))),
    ('sum', ('map', ('clamp', 0, 10), ('xs',))),
    ('if', ('>', ('var', 'k'), 0), ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('xs',))),
    ('sum', ('map', ('op', '*', 'k'), ('filter', '>', 'limit', ('xs',)))),
    ('if', ('>', ('var', 'k'), 0), ('max', ('var', 'k'), ('sum', ('xs',))), ('abs', ('sum', ('xs',)))),
  ]
  for t in suite_trees:
    if render(t).replace(' ', '') not in {s.replace(' ', '') for s in suite_refs}:
      raise ValueError('suite skeleton translation drifted: ' + render(t))
  banned = {skeleton(t) for t in suite_trees}
  candidates, seen, excluded = [], set(), 0
  while len(candidates) < args.count * 2:
    t = gen.int_(rng.choice([1, 2, 2, 3]))
    expr = render(t)
    if expr in seen: continue
    seen.add(expr)
    if expr in suite_refs or skeleton(t) in banned:
      excluded += 1; continue
    cs = cases(rng)
    try:
      expected = [str(evaluate(t, c)) for c in cs]
    except Exception:
      continue
    if any(abs(int(v)) > 999 for v in expected) or len(set(expected)) < 2:
      continue  # avoid huge values and trivially constant expressions
    candidates.append((t, expr, cs, expected))

  def check(item):
    t, expr, cs, expected = item
    try:
      node = ev.parse_expression(f'```sol\n{expr}\n```', ('xs', 'k', 'limit'))
    except Exception as e:
      return None, f'parse: {e}'
    for c, want in zip(cs, expected):
      if ev.run_expression(fpr, node, c) != want:
        return None, 'compiler disagreement'
    return item, None

  with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(check, candidates))
  kept = [r for r, _ in results if r][:args.count]
  reasons = {}
  for _, why in results:
    if why: reasons[why.split(':')[0]] = reasons.get(why.split(':')[0], 0) + 1
  args.output.mkdir(parents=True)
  with open(args.output / 'train.jsonl', 'w') as f:
    for i, (t, expr, cs, expected) in enumerate(kept):
      c = cs[1]
      q = rng.choice(PROMPTS).format(d=describe(t)) + ' Return the expression in a sol code block.'
      a = (f'```sol\n{expr}\n```\nFor xs = [{", ".join(map(str, c["xs"]))}], k = {c["k"]} and limit = {c["limit"]} '
           f'it evaluates to {expected[1]}.')
      f.write(json.dumps({'id': f'param-{i:05d}', 'question': q, 'answer': a, 'expression': expr,
                          'cases': [{'inputs': c, 'expected': e} for c, e in zip(cs, expected)]}) + '\n')
  report = {'kept': len(kept), 'candidates_checked': len(candidates), 'rejected': reasons,
            'excluded_suite_skeletons': excluded, 'suite_hash': suite['hash'],
            'compiler_sha256': ev.sha(fpr), 'cases_per_task': 6,
            'train_sha256': ev.sha(args.output / 'train.jsonl'),
            'prompt_overlap_with_suite': any(p.split('{')[0][:40] in suite_prompts for p in PROMPTS)}
  (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


def extend(args):
  """Append the encoded parameterized tasks to the prepared pools' verified set (replay pools unchanged)."""
  import numpy as np
  import torch
  from tokenizers import Tokenizer
  import qos_ai as ai
  from qos_ai_data import digest
  from qos_ai_stage4_data import targets
  if args.output.exists(): raise ValueError('choose a new output directory')
  pools = torch.load(args.pools, weights_only=True)
  if pools['hash'] != digest({n: pools[n] for n in ('verified', 'qa_pool', 'docs')}):
    raise ValueError('stale pool contents')
  v = torch.load(args.vocabulary, weights_only=True)
  if ev.sha(args.vocabulary) != pools['vocabulary_sha256']: raise ValueError('vocabulary mismatch')
  qwen = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
  lookup = np.zeros(max(qwen.get_vocab_size(), int(v['qwen_ids'].max()) + 1), dtype=np.int64)
  lookup[v['qwen_ids'][1:].numpy()] = np.arange(1, len(v['qwen_ids']))
  enc = lambda s: lookup[np.asarray(qwen.encode(s, add_special_tokens=False).ids, dtype=np.int64)].tolist()
  added, dropped = [], 0
  for line in open(args.param):
    r = json.loads(line)
    prompt = enc(ai.chat_text(r['question'])); ids = enc(ai.chat_text(r['question'], r['answer']))
    if 0 in ids or len(ids) > 512: dropped += 1; continue
    example = (ids, [i >= len(prompt) for i in range(len(ids))]); targets(example); added.append(example)
  pools['verified'] = list(pools['verified']) + added
  pools['hash'] = digest({n: pools[n] for n in ('verified', 'qa_pool', 'docs')})
  args.output.mkdir(parents=True)
  torch.save(pools, args.output / 'pools.pt')
  report = {'base_pools_sha256': ev.sha(args.pools), 'param_train_sha256': ev.sha(args.param),
            'added': len(added), 'dropped_unknown_or_long': dropped,
            'added_target_tokens': sum(targets(e) for e in added),
            'verified_examples': len(pools['verified']), 'verified_target_tokens': sum(targets(e) for e in pools['verified']),
            'pools_hash': pools['hash'], 'pools_file_sha256': ev.sha(args.output / 'pools.pt')}
  (args.output / 'extension.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  b = s.add_parser('build'); b.add_argument('--fpr', required=True); b.add_argument('--output', type=Path, required=True)
  b.add_argument('--count', type=int, default=2000)
  e = s.add_parser('extend'); e.add_argument('--pools', type=Path, required=True); e.add_argument('--param', type=Path, required=True)
  e.add_argument('--output', type=Path, required=True)
  e.add_argument('--vocabulary', type=Path, default=Path('runpod/557m/bundle/vocabulary.pt'))
  e.add_argument('--tokenizer', type=Path, default=Path('runpod/557m/bundle/tokenizer'))
  a = p.parse_args()
  build(a) if a.command == 'build' else extend(a)
