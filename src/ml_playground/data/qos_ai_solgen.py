"""Varied, executed Sol programs for teaching and for a frozen generalization test.

Random well-typed expression trees over the Sol builtins that exist today (taught and untaught), rendered as
programs, evaluated by a Python reference and confirmed by running the real `fpr` compiler; disagreements are
dropped. Each program becomes one of three tasks:

  output    "What does this print?"           -> "It prints V." plus name=value steps for nested programs
  generate  an English description            -> a program plus "It prints V."
  repair    a description and a buggy program -> the fixed program plus "It prints V."

The test set is built first and frozen. It has three strata, none of whose exact programs appear in training:
  fresh        same distribution as training
  held_pairs   contains a parent/child operation pair that never occurs in training
  big_numbers  integer literals from 61-150 (training uses -40..60)

  python3 qos_ai_solgen.py build --fpr ~/Documents/GitHub/fprisc/fpr --output qos_ai/solgen_20261020
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INT, LIST, BOOL, STR = 'Int', 'List', 'Bool', 'Str'
WORDS = ['sol', 'qos', 'river', 'stone', 'lamp', 'cedar', 'orbit', 'pixel', 'maple', 'harbor', 'violet', 'copper',
         'meadow', 'signal', 'falcon', 'garden', 'kernel', 'window', 'rocket', 'silver', 'thunder', 'basket']
SEPS = [', ', '-', ' ', '/', '']
CMPS = ['>', '<', '>=', '<=', '==', '!=']
MAPOPS = ['+', '-', '*']
HELD_PAIRS = {('abs', 'sum'), ('drop', 'rev'), ('any', 'map'), ('upper', 'slice'), ('max', 'last'), ('filter', 'range')}
LIMIT = 999


class Bad(Exception):
  pass


def lit(v):
  if isinstance(v, bool):
    return 'True' if v else 'False'
  if isinstance(v, int):
    return f'({v})' if v < 0 else str(v)
  if isinstance(v, str):
    return '"' + v + '"'
  return '[' + ', '.join(lit(x) for x in v) + ']'


def show(v):
  """How `print (str v)` renders a value."""
  if isinstance(v, bool):
    return 'True' if v else 'False'
  if isinstance(v, int):
    return str(v)
  if isinstance(v, str):
    return v
  if v and isinstance(v[0], tuple):
    return '[' + ', '.join(f'({a}, {b})' for a, b in v) + ']'
  return '[' + ', '.join(show(x) for x in v) + ']'


def tdiv(a, b):
  q = abs(a) // abs(b)
  return q if (a >= 0) == (b > 0) else -q


def tmod(a, b):
  return a - b * tdiv(a, b)


def cmp(op, a, b):
  return {'>': a > b, '<': a < b, '>=': a >= b, '<=': a <= b, '==': a == b, '!=': a != b}[op]


def mapop(op, a, k):
  return {'+': a + k, '-': a - k, '*': a * k}[op]


class Node:
  """op, type, children (Nodes), params (literals), value, sol text, English description."""
  def __init__(self, op, typ, kids=(), **params):
    self.op, self.type, self.kids, self.p = op, typ, list(kids), params
    self.value = self.evaluate()
    self.sol = self.render()

  def evaluate(self):
    o, p = self.op, self.p
    v = [k.value for k in self.kids]
    if o == 'lit': r = p['v']
    elif o == 'sum': r = sum(v[0])
    elif o == 'last':
      if not v[0]: raise Bad('last of empty')
      r = v[0][-1]
    elif o == 'index':
      if not 0 <= p['i'] < len(v[0]): raise Bad('index')
      r = v[0][p['i']]
    elif o == 'clamp': r = min(max(v[0], p['lo']), p['hi'])
    elif o == 'max': r = max(v[0], v[1])
    elif o == 'min': r = min(v[0], v[1])
    elif o == 'abs': r = abs(v[0])
    elif o == 'add': r = v[0] + v[1]
    elif o == 'sub': r = v[0] - v[1]
    elif o == 'mul': r = v[0] * v[1]
    elif o == 'mod': r = tmod(v[0], p['k'])
    elif o == 'div': r = tdiv(v[0], p['k'])
    elif o == 'if': r = v[1] if v[0] else v[2]
    elif o == 'find': r = v[0].find(chr(p['c']))
    elif o == 'at':
      if not 0 <= p['i'] < len(v[0]): raise Bad('at')
      r = ord(v[0][p['i']])
    elif o == 'map': r = [mapop(p['f'], x, p['k']) for x in v[0]]
    elif o == 'filter': r = [x for x in v[0] if cmp(p['c'], x, p['k'])]
    elif o == 'take': r = v[0][:p['n']]
    elif o == 'drop': r = v[0][p['n']:]
    elif o == 'rev': r = v[0][::-1]
    elif o == 'range': r = list(range(p['a'], p['b'] + 1))
    elif o == 'any': r = any(cmp(p['c'], x, p['k']) for x in v[0])
    elif o == 'all': r = all(cmp(p['c'], x, p['k']) for x in v[0])
    elif o == 'cmp': r = cmp(p['c'], v[0], v[1])
    elif o == 'upper': r = v[0].upper()
    elif o == 'slice': r = v[0][max(0, p['a']):max(p['a'], p['b'])]
    elif o == 'trim': r = v[0].strip()
    elif o == 'join': r = p['sep'].join(p['items'])
    elif o == 'zip': r = list(zip(v[0], v[1]))
    else: raise ValueError(o)
    for x in (r if isinstance(r, list) else [r]):
      if isinstance(x, int) and not isinstance(x, bool) and abs(x) > LIMIT: raise Bad('too large')
    if isinstance(r, list) and len(r) > 10: raise Bad('too long')
    return r

  def render(self):
    o, p = self.op, self.p
    s = [k.sol for k in self.kids]
    fn = lambda body: f'(fn x -> {body})'
    return {
      'lit': lambda: lit(p['v']),
      'sum': lambda: f'List.sum {wrap(s[0])}',
      'last': lambda: f'List.last {wrap(s[0])}',
      'index': lambda: f'{wrap(s[0])} ! {p["i"]}',
      'clamp': lambda: f'Numeric.clamp {lit(p["lo"])} {lit(p["hi"])} {wrap(s[0])}',
      'max': lambda: f'Numeric.max {wrap(s[0])} {wrap(s[1])}',
      'min': lambda: f'Numeric.min {wrap(s[0])} {wrap(s[1])}',
      'abs': lambda: f'Numeric.abs {wrap(s[0])}',
      'add': lambda: f'{wrap(s[0])} + {wrap(s[1])}',
      'sub': lambda: f'{wrap(s[0])} - {wrap(s[1])}',
      'mul': lambda: f'{wrap(s[0])} * {wrap(s[1])}',
      'mod': lambda: f'{wrap(s[0])} % {p["k"]}',
      'div': lambda: f'{wrap(s[0])} / {p["k"]}',
      'if': lambda: f'if {s[0]} then {s[1]} else {s[2]}',
      'find': lambda: f'Str.find {p["c"]} {wrap(s[0])}',
      'at': lambda: f'Str.at {wrap(s[0])} {p["i"]}',
      'map': lambda: f'List.map {fn("x " + p["f"] + " " + lit(p["k"]))} {wrap(s[0])}',
      'filter': lambda: f'List.filter {fn("x " + p["c"] + " " + lit(p["k"]))} {wrap(s[0])}',
      'take': lambda: f'List.take {p["n"]} {wrap(s[0])}',
      'drop': lambda: f'List.drop {p["n"]} {wrap(s[0])}',
      'rev': lambda: f'List.rev {wrap(s[0])}',
      'range': lambda: f'List.range {lit(p["a"])} {lit(p["b"])}',
      'any': lambda: f'List.any {fn("x " + p["c"] + " " + lit(p["k"]))} {wrap(s[0])}',
      'all': lambda: f'List.all {fn("x " + p["c"] + " " + lit(p["k"]))} {wrap(s[0])}',
      'cmp': lambda: f'{wrap(s[0])} {p["c"]} {wrap(s[1])}',
      'upper': lambda: f'Str.upper {wrap(s[0])}',
      'slice': lambda: f'Str.slice {wrap(s[0])} {p["a"]} {p["b"]}',
      'trim': lambda: f'Str.trim {wrap(s[0])}',
      'join': lambda: f'Str.join {lit(p["sep"])} {lit(p["items"])}',
      'zip': lambda: f'List.zip {wrap(s[0])} {wrap(s[1])}',
    }[o]()

  def describe(self):
    o, p = self.op, self.p
    d = [k.describe() if k.op == 'lit' else f'({k.describe()})' for k in self.kids]
    words = {'>': 'greater than', '<': 'less than', '>=': 'at least', '<=': 'at most', '==': 'equal to', '!=': 'not equal to'}
    verbs = {'+': 'increased by', '-': 'decreased by', '*': 'multiplied by'}
    return {
      'lit': lambda: lit(p['v']) if not isinstance(p['v'], str) else f'the string "{p["v"]}"',
      'sum': lambda: f'the sum of {d[0]}',
      'last': lambda: f'the last element of {d[0]}',
      'index': lambda: f'the element at index {p["i"]} of {d[0]}',
      'clamp': lambda: f'{d[0]} clamped into [{p["lo"]}, {p["hi"]}]',
      'max': lambda: f'the larger of {d[0]} and {d[1]}',
      'min': lambda: f'the smaller of {d[0]} and {d[1]}',
      'abs': lambda: f'the absolute value of {d[0]}',
      'add': lambda: f'{d[0]} plus {d[1]}',
      'sub': lambda: f'{d[0]} minus {d[1]}',
      'mul': lambda: f'{d[0]} times {d[1]}',
      'mod': lambda: f'the remainder of {d[0]} divided by {p["k"]}',
      'div': lambda: f'{d[0]} divided by {p["k"]} (integer division)',
      'if': lambda: f'{d[1]} if {d[0]}, otherwise {d[2]}',
      'find': lambda: f'the position of byte {p["c"]} in {d[0]}',
      'at': lambda: f'the byte value at index {p["i"]} of {d[0]}',
      'map': lambda: f'each value of {d[0]} {verbs[p["f"]]} {p["k"]}',
      'filter': lambda: f'the values of {d[0]} that are {words[p["c"]]} {p["k"]}',
      'take': lambda: f'the first {p["n"]} values of {d[0]}',
      'drop': lambda: f'{d[0]} without its first {p["n"]} values',
      'rev': lambda: f'{d[0]} reversed',
      'range': lambda: f'the numbers from {p["a"]} to {p["b"]}',
      'any': lambda: f'whether any value of {d[0]} is {words[p["c"]]} {p["k"]}',
      'all': lambda: f'whether every value of {d[0]} is {words[p["c"]]} {p["k"]}',
      'cmp': lambda: f'whether {d[0]} is {words[p["c"]]} {d[1]}',
      'upper': lambda: f'{d[0]} in upper case',
      'slice': lambda: f'the characters {p["a"]} to {p["b"]} (end exclusive) of {d[0]}',
      'trim': lambda: f'{d[0]} with surrounding spaces removed',
      'join': lambda: f'the strings {lit(p["items"])} joined with "{p["sep"]}"',
      'zip': lambda: f'{d[0]} paired element by element with {d[1]}',
    }[o]()

  def walk(self):
    yield self
    for k in self.kids:
      yield from k.walk()

  def pairs(self):
    return {(n.op, k.op) for n in self.walk() for k in n.kids}

  def steps(self):
    """Post-order values of non-literal nodes, as name=value."""
    out, seen = [], {}
    def visit(n):
      for k in n.kids:
        visit(k)
      if n.op != 'lit':
        seen[n.op] = seen.get(n.op, 0) + 1
        name = n.op if seen[n.op] == 1 else f'{n.op}{seen[n.op]}'
        out.append(f'{name}={show(n.value)}')
    visit(self)
    return out


def wrap(s):
  return s if re.fullmatch(r'\w+|\(-?\d+\)|\[[^\[\]]*\]|"[^"]*"', s) else f'({s})'


class Gen:
  def __init__(self, rng, lo=-40, hi=60, banned=frozenset(), big=False):
    self.r, self.lo, self.hi, self.banned, self.big = rng, lo, hi, banned, big

  def num(self, a=None, b=None):
    if self.big and self.r.random() < .5:
      return self.r.randint(61, 150)
    return self.r.randint(self.lo if a is None else a, self.hi if b is None else b)

  def small(self, a=-9, b=12):
    return self.r.randint(a, b)

  def make(self, typ, depth, parent=None):
    for _ in range(50):
      try:
        n = self._make(typ, depth)
        if parent and (parent, n.op) in self.banned:
          continue
        return n
      except Bad:
        continue
    raise Bad('give up')

  def _make(self, typ, depth):
    r = self.r
    leaf = depth <= 0 or r.random() < .25
    if typ == LIST:
      if leaf:
        if r.random() < .15:
          a = self.small(-5, 10); return Node('range', LIST, a=a, b=a + r.randint(0, 6))
        return Node('lit', LIST, v=[self.num() for _ in range(r.randint(2, 6))])
      op = r.choice(['map', 'filter', 'take', 'drop', 'rev', 'map', 'filter'])
      k = self.make(LIST, depth - 1, op)
      if op == 'map': return Node(op, LIST, [k], f=r.choice(MAPOPS), k=self.small(-4, 6) if True else 0)
      if op == 'filter': return Node(op, LIST, [k], c=r.choice(CMPS), k=self.num())
      if op in ('take', 'drop'): return Node(op, LIST, [k], n=r.randint(0, 4))
      return Node(op, LIST, [k])
    if typ == INT:
      if leaf:
        return Node('lit', INT, v=self.num())
      op = r.choice(['sum', 'sum', 'last', 'index', 'clamp', 'max', 'min', 'abs', 'add', 'sub', 'mul', 'mod', 'div',
                     'if', 'find', 'at'])
      if op in ('sum', 'last'): return Node(op, INT, [self.make(LIST, depth - 1, op)])
      if op == 'index':
        k = self.make(LIST, depth - 1, op)
        if not k.value: raise Bad('empty')
        return Node(op, INT, [k], i=r.randrange(len(k.value)))
      if op == 'clamp':
        a, b = sorted([self.num(), self.num()]); return Node(op, INT, [self.make(INT, depth - 1, op)], lo=a, hi=b)
      if op in ('max', 'min', 'add', 'sub'): return Node(op, INT, [self.make(INT, depth - 1, op), self.make(INT, depth - 1, op)])
      if op == 'mul': return Node(op, INT, [self.make(INT, depth - 1, op), Node('lit', INT, v=self.small())])
      if op == 'abs': return Node(op, INT, [self.make(INT, depth - 1, op)])
      if op in ('mod', 'div'): return Node(op, INT, [self.make(INT, depth - 1, op)], k=r.randint(2, 9))
      if op == 'if': return Node(op, INT, [self.make(BOOL, depth - 1, op), self.make(INT, depth - 1, op), self.make(INT, depth - 1, op)])
      s = Node('lit', STR, v=r.choice(WORDS))
      if op == 'find': return Node(op, INT, [s], c=ord(r.choice(s.value + 'xyz')))
      return Node(op, INT, [s], i=r.randrange(len(s.value)))
    if typ == BOOL:
      op = r.choice(['any', 'all', 'cmp'])
      if op == 'cmp': return Node(op, BOOL, [self.make(INT, depth - 1, op), self.make(INT, depth - 1, op)], c=r.choice(CMPS))
      return Node(op, BOOL, [self.make(LIST, depth - 1, op)], c=r.choice(CMPS), k=self.num())
    if typ == STR:
      op = r.choice(['upper', 'slice', 'trim', 'join', 'upper', 'slice'])
      if op == 'join': return Node(op, STR, sep=r.choice(SEPS), items=r.sample(WORDS, r.randint(2, 4)))
      if op == 'trim': return Node(op, STR, [Node('lit', STR, v=' ' * r.randint(1, 3) + r.choice(WORDS) + ' ' * r.randint(0, 3))])
      inner = self.make(STR, depth - 1, op) if r.random() < .3 else Node('lit', STR, v=r.choice(WORDS))
      if op == 'upper': return Node(op, STR, [inner])
      a = r.randint(0, 4); return Node(op, STR, [inner], a=a, b=a + r.randint(1, 5))
    raise ValueError(typ)

  def program(self):
    r = self.r
    typ = r.choices([INT, LIST, BOOL, STR, 'zip'], [.5, .25, .12, .1, .03])[0]
    depth = r.choices([1, 2, 3], [.35, .4, .25])[0]
    if typ == 'zip':
      return Node('zip', LIST, [self.make(LIST, depth - 1, 'zip'), self.make(LIST, depth - 1, 'zip')])
    n = self.make(typ, depth)
    if n.op == 'lit':
      raise Bad('literal program')
    return n


def program_text(node, rng):
  """One-line form, or a binding form that names the innermost list literal."""
  printer = (lambda e: f'> print ({e}).') if node.type == STR else (lambda e: f'> print (str {wrap(e)}).')
  lits = [n for n in node.walk() if n.op == 'lit' and n.type == LIST]
  if lits and rng.random() < .4:
    name = rng.choice(['xs', 'nums', 'values', 'data', 'items'])
    res = rng.choice(['result', 'total', 'answer', 'out'])
    target = lits[0]
    body = node.sol.replace(target.sol, name, 1)
    return f'{name} = {target.sol}.\n{res} = {body}.\n' + printer(res)
  return printer(node.sol)


OUTPUT_Q = ['What does this Sol program print?', 'What is printed by the following Sol script?',
            'Give the output of this Sol code.', 'Work out what this Sol program prints.',
            'What will this Sol program output?', 'Predict the stdout of this Sol file.']
GEN_Q = ['Write a Sol program that prints {d}.', 'In Sol, compute and print {d}.',
         'Write Sol code that outputs {d}. Then state what it prints.', 'Implement in Sol: print {d}.']
FIX_Q = ['This Sol program should print {d}, but it prints {bad}. Fix it and state what the fixed program prints.',
         'A Sol script is meant to print {d}. It currently prints {bad}. Correct it.']


def mutate(node, rng):
  """A copy of the tree with one parameter changed (a plausible bug)."""
  import copy
  for _ in range(20):
    t = copy.deepcopy(node)
    cands = [n for n in t.walk() if n.op in ('map', 'filter', 'take', 'drop', 'index', 'clamp', 'mod', 'div', 'any', 'all', 'cmp', 'lit')
             and not (n.op == 'lit' and n.type == STR)]
    if not cands:
      return None
    n = rng.choice(cands)
    if n.op == 'map': n.p['f'] = rng.choice([f for f in MAPOPS if f != n.p['f']])
    elif n.op in ('filter', 'any', 'all', 'cmp'): n.p['c'] = rng.choice([c for c in CMPS if c != n.p['c']])
    elif n.op in ('take', 'drop'): n.p['n'] = n.p['n'] + rng.choice([-1, 1]) if n.p['n'] > 0 else 1
    elif n.op == 'index': n.p['i'] = n.p['i'] + rng.choice([-1, 1])
    elif n.op in ('mod', 'div'): n.p['k'] = n.p['k'] + 1
    elif n.op == 'clamp': n.p['hi'] = n.p['hi'] + rng.randint(1, 5)
    elif n.op == 'lit' and n.type == INT: n.p['v'] = n.p['v'] + rng.choice([-2, -1, 1, 2])
    elif n.op == 'lit' and n.type == LIST and n.p['v']:
      v = list(n.p['v']); i = rng.randrange(len(v)); v[i] += rng.choice([-3, 3]); n.p['v'] = v
    else:
      continue
    try:
      rebuild(t)
    except Bad:
      continue
    if t.value != node.value:
      return t
  return None


def rebuild(n):
  for k in n.kids:
    rebuild(k)
  n.value = n.evaluate(); n.sol = n.render()


def task(node, kind, rng):
  code = program_text(node, rng)
  v = show(node.value)
  steps = node.steps()
  trace = f' Steps: {"; ".join(steps)}.' if len(steps) > 1 else ''
  if kind == 'output':
    q = f'{rng.choice(OUTPUT_Q)}\n```sol\n{code}\n```'
    a = f'It prints {v}.{trace}'
  elif kind == 'generate':
    q = rng.choice(GEN_Q).format(d=node.describe())
    a = f'```sol\n{code}\n```\nIt prints {v}.{trace}'
  else:
    bad = mutate(node, rng)
    if bad is None:
      return None
    q = rng.choice(FIX_Q).format(d=node.describe(), bad=show(bad.value)) + f'\n```sol\n{program_text(bad, rng)}\n```'
    a = f'```sol\n{code}\n```\nIt prints {v}.{trace}'
  return {'question': q, 'answer': a, 'code': code, 'expected': v, 'kind': kind, 'ops': sorted({n.op for n in node.walk()} - {'lit'}),
          'pairs': sorted(f'{a}>{b}' for a, b in node.pairs() if b != 'lit'), 'depth': max(len(s) for s in [node.steps()])}


def run_fpr(fpr, code):
  with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 't.sol'); open(p, 'w').write(code + '\n')
    try:
      r = subprocess.run([fpr, 'sol', p], capture_output=True, text=True, timeout=20, cwd=d)
    except subprocess.TimeoutExpired:
      return None
    return r.stdout.rstrip('\n') if r.returncode == 0 else None


def verify(rows, fpr, workers=8):
  with ThreadPoolExecutor(workers) as ex:
    outs = list(ex.map(lambda r: run_fpr(fpr, r['code']), rows))
  keep = [r for r, o in zip(rows, outs) if o is not None and o == r['expected']]
  return keep, len(rows) - len(keep)


def build(args):
  out = args.output
  if out.exists():
    raise ValueError('Choose a new output directory')
  out.mkdir(parents=True)
  fpr = os.path.expanduser(str(args.fpr))
  # Programs that must never be trained on: existing benchmark/probe programs.
  protected = set()
  for path in args.protect:
    for line in open(path):
      row = json.loads(line) if path.endswith('.jsonl') else None
      text = row['question'] if row else ''
      protected.update(c.strip() for c in re.findall(r'```sol\n(.*?)```', text, re.S))
  if args.probe:
    protected.update((q.get('reference_code') or '').strip() for q in json.load(open(args.probe)))
  kinds = ['output'] * 70 + ['generate'] * 20 + ['repair'] * 10

  def collect(gen, n, rng, accept):
    rows, seen = [], set()
    while len(rows) < n:
      try:
        node = gen.program()
      except Bad:
        continue
      t = task(node, rng.choice(kinds), rng)
      if t and t['code'] not in seen and t['code'] not in protected and accept(t):
        seen.add(t['code']); rows.append(t)
    return rows

  rng = random.Random(20261020)
  test = {
    'fresh': collect(Gen(random.Random(1), banned=frozenset(HELD_PAIRS)), args.test_each, random.Random(11), lambda t: True),
    'held_pairs': collect(Gen(random.Random(2)), args.test_each, random.Random(12),
                          lambda t: any(tuple(p.split('>')) in HELD_PAIRS for p in t['pairs'])),
    'big_numbers': collect(Gen(random.Random(3), banned=frozenset(HELD_PAIRS), big=True), args.test_each, random.Random(13),
                           lambda t: bool(re.search(r'\b(6[1-9]|[7-9]\d|1[0-4]\d|150)\b', t['code']))),
  }
  test_codes = {r['code'] for rows in test.values() for r in rows}
  stats = {}
  for name, rows in test.items():
    rows, dropped = verify(rows, fpr)
    test[name] = rows; stats[f'test_{name}'] = {'kept': len(rows), 'fpr_disagreed': dropped}
    with open(out / f'test_{name}.jsonl', 'w') as f:
      for i, r in enumerate(rows):
        f.write(json.dumps({'id': f'{name}-{i:03d}', 'stratum': name, **r}) + '\n')
  big = re.compile(r'\b(6[1-9]|[7-9]\d|1[0-4]\d|150)\b')
  train = collect(Gen(rng, banned=frozenset(HELD_PAIRS)), args.train, random.Random(14),
                  lambda t: t['code'] not in test_codes and not big.search(t['code']))
  train, dropped = verify(train, fpr)
  stats['train'] = {'kept': len(train), 'fpr_disagreed': dropped}
  with open(out / 'train.jsonl', 'w') as f:
    for i, r in enumerate(train):
      f.write(json.dumps({'id': f'train-{i:05d}', **r}) + '\n')
  ops = {}
  for r in train:
    for o in r['ops']:
      ops[o] = ops.get(o, 0) + 1
  stats.update(op_counts=dict(sorted(ops.items())), held_pairs=sorted(f'{a}>{b}' for a, b in HELD_PAIRS),
               fpr_sha256=hashlib.sha256(open(fpr, 'rb').read()).hexdigest(), protected_programs=len(protected),
               kinds={k: sum(r['kind'] == k for r in train) for k in ('output', 'generate', 'repair')})
  (out / 'build.json').write_text(json.dumps(stats, indent=2) + '\n')
  print(json.dumps(stats, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  b = s.add_parser('build')
  b.add_argument('--fpr', type=Path, required=True)
  b.add_argument('--output', type=Path, required=True)
  b.add_argument('--train', type=int, default=60000)
  b.add_argument('--test-each', type=int, default=120)
  b.add_argument('--protect', nargs='*', default=['runpod/557m/bundle/benchmarks/benchmark.jsonl',
                                                  'runpod/557m/bundle/benchmarks/coding_diagnostic.jsonl',
                                                  'runpod/557m/bundle/benchmarks/generalization_diagnostic.jsonl'])
  b.add_argument('--probe', default='qos_ai/probe_sol_20261020/questions.json')
  a = p.parse_args()
  build(a)
