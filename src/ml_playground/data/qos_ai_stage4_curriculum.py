"""Build a bounded arithmetic supplement with compiler-checked intermediate values.

Training, validation and paraphrases of an input case share one group assignment.
Existing benchmark/reference programs are excluded. This supplements Stage 3;
it does not replace its broader operation coverage.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qos_ai_stage4_eval import fingerprint, literal, parse_expression, run_expression, sha


TRAIN_WORDING = [
  'Calculate the output of this Sol program. Show the intermediate values.',
  'Evaluate this script and explain the arithmetic using exact intermediate lists.',
  'What does the code print? Include the calculation steps.',
  'Trace the calculation below and give its printed result.',
]
VALID_WORDING = [
  'Check this calculation by hand: give the resulting lists and final stdout.',
  'Determine the value printed here, accounting for every element in the calculation.',
]


def record(family, xs, k, limit):
  x, scalar, threshold = literal(xs), literal(k), literal(limit)
  if family == 'map_sum':
    values = [v * k for v in xs]
    stages = [('products', f'List.map (fn x -> x * {scalar}) {x}', values)]
  elif family == 'filter_sum':
    values = [v for v in xs if v > limit]
    stages = [('selected', f'List.filter (fn x -> x > {threshold}) {x}', values)]
  elif family == 'abs_sum':
    values = [abs(v) for v in xs]
    stages = [('magnitudes', f'List.map (fn x -> Numeric.abs x) {x}', values)]
  elif family == 'clamp_sum':
    values = [min(10, max(0, v)) for v in xs]
    stages = [('clamped', f'List.map (fn x -> Numeric.clamp 0 10 x) {x}', values)]
  else:
    raise ValueError('unknown family')
  intermediate = stages[0][1]
  partials = [sum(values[:i]) for i in range(1, len(values) + 1)]
  partial_expr = '[' + ', '.join(f'List.sum (List.take {i} ({intermediate}))' for i in range(1, len(values) + 1)) + ']'
  stages += [('partial_sums', partial_expr, partials), ('sum', f'List.sum ({intermediate})', sum(values))]
  # Trace every individual multiplication, including those whose products sum to zero.
  if family == 'map_sum':
    stages += [(f'product{i}', f'{literal(v)} * {scalar}', v * k) for i, v in enumerate(xs)]
  expression = f'List.sum ({intermediate})'
  program = f'> print (str ({expression})).'
  return {'family': family, 'inputs': {'xs': xs, 'k': k, 'limit': limit},
          'program': program, 'expression': expression, 'expected': str(sum(values)),
          'stages': stages}


def normalized(code):
  return ' '.join(code.split())


def protected_programs(paths):
  import re
  programs, questions = set(), set()
  for path in paths:
    text = path.read_text()
    rows = json.loads(text) if path.suffix == '.json' else [json.loads(line) for line in text.splitlines()]
    if not isinstance(rows, list): raise ValueError('protected input must contain rows')
    for r in rows:
      questions.add(' '.join(r.get('question', '').split()).casefold())
      for key in ('code', 'program', 'reference_code'):
        if isinstance(r.get(key), str): programs.add(normalized(r[key]))
        elif isinstance(r.get(key), dict):
          for value in r[key].values():
            if isinstance(value, str): programs.add(normalized(value))
      programs.update(normalized(c) for c in re.findall(r'```sol\n(.*?)```', r.get('question', ''), re.S))
  return programs, questions


def build(fpr, output, protect, count=1024, seed=20261021):
  if output.exists(): raise ValueError('choose a new output directory')
  if count < 40 or count % 4: raise ValueError('count must be at least 40 and divisible by four')
  protected, questions = protected_programs(protect)
  rng = random.Random(seed); records = []; seen = set()
  families = ('map_sum', 'filter_sum', 'abs_sum', 'clamp_sum')
  attempts = 0
  while len(records) < count:
    attempts += 1
    if attempts > count * 100: raise ValueError('insufficient unique cases')
    family = families[len(records) % len(families)]
    xs = [rng.randint(-40, 60) for _ in range(rng.randint(1, 5))]
    row = record(family, xs, rng.randint(-8, 8), rng.randint(-10, 10))
    key = normalized(row['program'])
    if key in seen or key in protected: continue
    seen.add(key)
    group = fingerprint([family, key])
    split = 'validation' if int(group[:8], 16) % 10 == 0 else 'train'
    wording = rng.choice(VALID_WORDING if split == 'validation' else TRAIN_WORDING)
    row.update(id=group[:20], group=group, split=split,
               question=wording + '\n```sol\n' + row['program'] + '\n```')
    if ' '.join(row['question'].split()).casefold() in questions: continue
    row['answer'] = f'It prints {row["expected"]}. Steps: ' + '; '.join(
      f'{name}={value}' for name, _, value in row['stages']) + '.'
    records.append(row)

  def verify(row):
    for name, expression, expected in row['stages']:
      node = parse_expression(expression, ())
      if run_expression(fpr, node, {}) != str(expected):
        raise ValueError(f'compiler disagrees with {row["id"]}:{name}')
    return row

  with ThreadPoolExecutor(8) as pool:
    verified = list(pool.map(verify, records))
  output.mkdir(parents=True)
  for split in ('train', 'validation'):
    (output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in verified if r['split'] == split))
  report = {'schema': 1, 'seed': seed, 'compiler_sha256': sha(fpr), 'groups': len(records),
            'train': sum(r['split'] == 'train' for r in records),
            'validation': sum(r['split'] == 'validation' for r in records),
            'verified_intermediate_executions': sum(len(r['stages']) for r in records),
            'protected_inputs': {str(p): sha(p) for p in protect},
            'files': {s: sha(output / f'{s}.jsonl') for s in ('train', 'validation')},
            'scope': 'arithmetic supplement; validation holds out input groups and wording, not all concepts'}
  report['hash'] = fingerprint(report)
  (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
  return report


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--fpr', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--count', type=int, default=1024)
  p.add_argument('--protect', type=Path, nargs='+', required=True)
  a = p.parse_args()
  print(json.dumps(build(a.fpr, a.output, a.protect, a.count), indent=2))


if __name__ == '__main__': main()
