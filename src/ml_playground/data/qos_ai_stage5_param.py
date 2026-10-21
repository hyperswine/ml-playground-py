"""Stage 5 parameterized curriculum: Stage 4's verified xs/k/limit expressions with varied natural wording.

Stage 4 described every task in one formulaic style ("the sum of (xs with each element multiplied by k)"). On the
multi-input suite the models then misread natural requests such as "multiply each element of xs by k and add the
products" (dropping the sum) or "if k is positive ..." (copying one branch into both). This generator keeps the
Stage 4 tree generator, oracle, suite-skeleton exclusion and compiler/runner verification unchanged and only varies
the English: several phrasings per operation, imperative and descriptive sentence shapes, and composed
"first ..., then ..." pipelines. Any description containing six consecutive words of a suite description is
rejected, and the suite's prompt templates remain unused.

  uv run --locked qos stage5_param build --fpr FPR --output DIR [--count 4000]
  uv run --locked qos stage4_param extend --pools POOLS --param DIR/train.jsonl --output NEWDIR
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import qos_ai_stage4_eval as ev
import qos_ai_stage4_param as s4

PROMPTS = s4.PROMPTS + [
  'Express the following as a single Sol expression over xs, k and limit: {d}.',
  'xs, k and limit are already bound. Give one Sol expression for {d}.',
]
STEP_PROMPTS = [
  'Write one Sol expression over xs, k and limit that does the following: {d}.',
  'xs, k and limit are already bound. In one Sol expression, {d}.',
  'Using xs, k and limit in a single Sol expression: {d}.',
  'Steps to encode as one Sol expression over xs, k and limit: {d}.',
]
CMP_WORDS = {'>': ['greater than', 'strictly above', 'larger than', 'over'],
             '<': ['less than', 'strictly below', 'smaller than', 'under'],
             '>=': ['at least', 'greater than or equal to', 'no smaller than'],
             '<=': ['at most', 'less than or equal to', 'no larger than'],
             '!=': ['not equal to', 'different from']}
OP_WORDS = {'*': ['multiplied by', 'times', 'scaled by'], '+': ['increased by', 'plus', 'shifted up by'],
            '-': ['decreased by', 'minus', 'reduced by']}


class Phraser:
  """Random but meaning-preserving English for a Stage 4 expression tree."""
  def __init__(self, rng):
    self.r = rng

  def pick(self, options):
    return self.r.choice(options)

  def part(self, t):
    """A child phrase, parenthesized when it is itself compound so nesting stays unambiguous."""
    simple = t[0] in ('xs', 'var') or (len(t) > 1 and isinstance(t[-1], tuple) and t[-1][0] == 'xs' and t[0] != 'if')
    return self.noun(t) if simple else f'({self.noun(t)})'

  def noun(self, t):
    """A noun phrase naming the value of t."""
    k, r = t[0], self.r
    if k == 'xs': return self.pick(['xs', 'the list xs', 'the values in xs'])
    if k == 'var': return t[1]
    if k == 'map': return self.pick([f'{self.part(t[2])} with every element {self.fn(t[1])}',
                                     f'each element of {self.part(t[2])} {self.fn(t[1])}'])
    if k == 'filter':
      w = self.pick(CMP_WORDS[t[1]])
      return self.pick([f'the elements of {self.part(t[3])} that are {w} {t[2]}',
                        f'only those values of {self.part(t[3])} {w} {t[2]}',
                        f'{self.part(t[3])} keeping just the elements {w} {t[2]}'])
    if k == 'take': return self.pick([f'the first {t[1]} elements of {self.part(t[2])}', f'the leading {t[1]} values of {self.part(t[2])}'])
    if k == 'drop': return self.pick([f'{self.part(t[2])} without its first {t[1]} elements', f'{self.part(t[2])} after skipping {t[1]} elements'])
    if k == 'rev': return self.pick([f'{self.part(t[1])} in reverse order', f'{self.part(t[1])} reversed'])
    if k == 'sum': return self.pick([f'the sum of {self.part(t[1])}', f'the total of {self.part(t[1])}',
                                     f'{self.part(t[1])} added together'])
    if k == 'abs': return self.pick([f'the absolute value of {self.part(t[1])}', f'the magnitude of {self.part(t[1])}'])
    if k in ('max', 'min'):
      word = 'larger' if k == 'max' else 'smaller'
      return self.pick([f'the {word} of {self.part(t[1])} and {self.part(t[2])}',
                        f'whichever is {word}: {self.part(t[1])} or {self.part(t[2])}'])
    if k == 'clamp': return self.pick([f'{self.part(t[3])} clamped to {t[1]} through {t[2]}',
                                       f'{self.part(t[3])} limited to the range {t[1]} to {t[2]}'])
    if k == 'if':
      c = t[1]; w = self.pick(CMP_WORDS[c[0]])
      return self.pick([f'{self.part(t[2])} when {self.part(c[1])} is {w} {c[2]}, otherwise {self.part(t[3])}',
                        f'{self.part(t[2])} if {self.part(c[1])} is {w} {c[2]}, else {self.part(t[3])}'])
    raise ValueError(k)

  def fn(self, f):
    k = f[0]
    if k == 'op': return f'{self.pick(OP_WORDS[f[1]])} {f[2]}'
    if k == 'abs': return self.pick(['replaced by its absolute value', 'made non-negative'])
    if k == 'clamp': return self.pick([f'clamped to {f[1]} through {f[2]}', f'kept within {f[1]} to {f[2]}'])
    word = 'larger' if k == 'max' else 'smaller'
    return self.pick([f'replaced by the {word} of itself and {f[1]}', f'raised or lowered to the {word} of itself and {f[1]}'])

  def imperative(self, t):
    """Step-by-step instructions for a reduction over a list pipeline, when the shape allows it."""
    if t[0] != 'sum':
      return None
    steps, node = [], t[1]
    while node[0] != 'xs':
      k = node[0]
      if k == 'map': steps.append(self.pick([f'{self.verb_fn(node[1])} each element', f'{self.verb_fn(node[1])} every value']))
      elif k == 'filter': steps.append(f'keep only the elements {self.pick(CMP_WORDS[node[1]])} {node[2]}')
      elif k == 'take': steps.append(f'take the first {node[1]} elements')
      elif k == 'drop': steps.append(f'drop the first {node[1]} elements')
      elif k == 'rev': steps.append('reverse the list')
      node = node[-1]
    steps.reverse()
    finish = self.pick(['add them up', 'sum the results', 'return the total', 'add the remaining values'])
    if not steps:
      return self.pick(['add up every element of xs', 'sum the elements of xs'])
    return 'starting from xs, ' + ', then '.join(steps) + ', and finally ' + finish

  def verb_fn(self, f):
    k = f[0]
    if k == 'op':
      return {'*': f'multiply by {f[2]}', '+': f'add {f[2]} to', '-': f'subtract {f[2]} from'}[f[1]]
    if k == 'abs': return 'take the absolute value of'
    if k == 'clamp': return f'clamp to {f[1]} through {f[2]}'
    return f'take the {"larger" if k == "max" else "smaller"} of {f[1]} and'

  def describe(self, t):
    """(text, mood): step-by-step instructions where the shape allows (preferred), otherwise a noun phrase."""
    imp = self.imperative(t)
    if imp and self.r.random() < .7:
      return imp, 'steps'
    return self.noun(t), 'noun'


def suite_ngrams(n=6):
  rows = json.loads(s4.SUITE.read_text())['rows']
  grams = set()
  for r in rows:
    words = re.findall(r'\w+', r['question'].split(': ')[-1].lower())
    grams |= {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
  return grams


def build(args):
  if args.output.exists(): raise ValueError('choose a new output directory')
  fpr = str(Path(args.fpr).expanduser())
  suite = json.loads(s4.SUITE.read_text())
  refs = {r['reference'] for r in suite['rows']}
  banned_grams = suite_ngrams()
  rng = random.Random(20261022); gen = s4.Gen(rng); phr = Phraser(random.Random(20261023))
  suite_trees = [  # identical to Stage 4's exclusion list
    ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('filter', '>', 'limit', ('xs',))),
    ('sum', ('map', ('clamp', 0, 10), ('xs',))),
    ('if', ('>', ('var', 'k'), 0), ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('xs',))),
    ('sum', ('map', ('op', '*', 'k'), ('filter', '>', 'limit', ('xs',)))),
    ('if', ('>', ('var', 'k'), 0), ('max', ('var', 'k'), ('sum', ('xs',))), ('abs', ('sum', ('xs',))))]
  banned = {s4.skeleton(t) for t in suite_trees}
  candidates, seen, excluded, wording_rejects = [], set(), 0, 0
  while len(candidates) < args.count * 2:
    t = gen.int_(rng.choice([1, 2, 2, 3]))
    expr = s4.render(t)
    if expr in seen: continue
    seen.add(expr)
    if expr in refs or s4.skeleton(t) in banned:
      excluded += 1; continue
    cs = s4.cases(rng)
    try:
      expected = [str(s4.evaluate(t, c)) for c in cs]
    except Exception:
      continue
    if any(abs(int(v)) > 999 for v in expected) or len(set(expected)) < 2:
      continue
    d, mood = phr.describe(t)
    words = re.findall(r'\w+', d.lower())
    if any(tuple(words[i:i + 6]) in banned_grams for i in range(len(words) - 5)):
      wording_rejects += 1; continue
    candidates.append((t, expr, cs, expected, d, mood))

  def check(item):
    t, expr, cs, expected, d, mood = item
    try:
      node = ev.parse_expression(f'```sol\n{expr}\n```', ('xs', 'k', 'limit'))
    except Exception:
      return None
    return item if all(ev.run_expression(fpr, node, c) == w for c, w in zip(cs, expected)) else None

  with ThreadPoolExecutor(8) as pool:
    kept = [r for r in pool.map(check, candidates) if r][:args.count]
  args.output.mkdir(parents=True)
  with open(args.output / 'train.jsonl', 'w') as f:
    for i, (t, expr, cs, expected, d, mood) in enumerate(kept):
      c = cs[1]
      q = rng.choice(STEP_PROMPTS if mood == 'steps' else PROMPTS).format(d=d) + ' Return the expression in a sol code block.'
      a = (f'```sol\n{expr}\n```\nFor xs = [{", ".join(map(str, c["xs"]))}], k = {c["k"]} and limit = {c["limit"]} '
           f'it evaluates to {expected[1]}.')
      f.write(json.dumps({'id': f'param5-{i:05d}', 'question': q, 'answer': a, 'expression': expr, 'description': d,
                          'cases': [{'inputs': c, 'expected': e} for c, e in zip(cs, expected)]}) + '\n')
  report = {'kept': len(kept), 'candidates_checked': len(candidates), 'excluded_suite_skeletons': excluded,
            'wording_rejected_suite_ngrams': wording_rejects, 'suite_hash': suite['hash'],
            'compiler_sha256': ev.sha(fpr), 'cases_per_task': 6, 'train_sha256': ev.sha(args.output / 'train.jsonl'),
            'step_share': sum(1 for k in kept if k[5] == 'steps') / max(1, len(kept))}
  (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  b = s.add_parser('build'); b.add_argument('--fpr', required=True); b.add_argument('--output', type=Path, required=True)
  b.add_argument('--count', type=int, default=4000)
  a = p.parse_args(); build(a)
