"""Score answers to the frozen Sol generalization tests (qos_ai/solgen_20261020/test_*.jsonl).

output    correct when the stated value ("It prints V.") equals the compiler's stdout exactly
generate  / repair: `program` = the answer's code block runs under fpr and prints the expected stdout;
          `stated` = the stated value matches; `correct` = program and stated both right
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATED = re.compile(r'It prints (.*?)\.(?:\s+Steps:|\s*$)', re.S)


def stated(answer):
  m = STATED.search(answer)
  if m:
    return m.group(1).strip()
  bare = answer.strip()
  # Lenient for models that answer with the bare value: a single short line with no code block.
  if bare and '\n' not in bare and '```' not in bare and len(bare) < 80:
    return bare.rstrip('.').strip().strip('`"')
  return None


def run(fpr, code):
  with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 't.sol'); open(p, 'w').write(code.strip() + '\n')
    try:
      r = subprocess.run([fpr, 'sol', p], capture_output=True, text=True, timeout=15, cwd=d)
    except subprocess.TimeoutExpired:
      return None
    return r.stdout.rstrip('\n') if r.returncode == 0 else None


def score(tests, answers, fpr):
  def one(t):
    a = answers.get(t['id'], '')
    s = stated(a)
    row = {'id': t['id'], 'stratum': t['stratum'], 'kind': t['kind'], 'stated_ok': s == t['expected']}
    if t['kind'] == 'output':
      row['correct'] = row['stated_ok']
    else:
      blocks = re.findall(r'```(?:sol)?\n(.*?)```', a, re.S)
      out = run(fpr, blocks[0]) if blocks else None
      row['program_ok'] = out == t['expected']
      row['correct'] = row['program_ok'] and row['stated_ok']
    return row
  with ThreadPoolExecutor(8) as ex:
    return list(ex.map(one, tests))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--tests', type=Path, default=Path('qos_ai/solgen_20261020'))
  p.add_argument('--fpr', default=os.path.expanduser('~/Documents/GitHub/fprisc/fpr'))
  p.add_argument('answers', type=Path, nargs='+', help='run_probe.py output files (name.json)')
  a = p.parse_args()
  tests = [json.loads(l) for s in ('fresh', 'held_pairs', 'big_numbers') for l in open(a.tests / f'test_{s}.jsonl')]
  summary = {}
  for path in a.answers:
    answers = {r['id']: r['answer'] for r in json.load(open(path))}
    rows = score(tests, answers, a.fpr)
    res = {}
    for s in ('fresh', 'held_pairs', 'big_numbers'):
      sel = [r for r in rows if r['stratum'] == s]
      res[s] = {'correct': sum(r['correct'] for r in sel), 'n': len(sel)}
      for k in ('output', 'generate', 'repair'):
        sk = [r for r in sel if r['kind'] == k]
        res[s][k] = f"{sum(r['correct'] for r in sk)}/{len(sk)}"
    gen = [r for r in rows if r['kind'] != 'output']
    res['programs_run_correctly'] = f"{sum(r.get('program_ok', False) for r in gen)}/{len(gen)}"
    summary[path.stem] = res
    path.with_suffix('.scored.json').write_text(json.dumps(rows, indent=1))
  print(json.dumps(summary, indent=1))


if __name__ == '__main__':
  main()
