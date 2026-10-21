import json, re, subprocess, tempfile, os
Q = {q['id']: q for q in json.load(open('questions.json'))}
FPR = os.path.expanduser('~/Documents/GitHub/fprisc/fpr')
def run(code):
  with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 't.sol'); open(p, 'w').write(code + '\n')
    try:
      r = subprocess.run([FPR, 'sol', p], capture_output=True, text=True, timeout=15, cwd=d)
      return r.stdout.strip() if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
      return None
def states(ans, exp):
  # expected value stated as its own token (not inside a longer number/word)
  return re.search(r'(?<![\w.-])' + re.escape(exp) + r'(?!\w|\.\w)', ans) is not None
models = ['m557_1b_pilot', 'm557_1b_mix', 'm168_coding_parent', 'm168_seed7', 'm557_sol']
table = {}
for m in models:
  for a in json.load(open(f'answers/{m}.json')):
    q = Q[a['id']]; ans = a['answer']
    if isinstance(q['expected'], list):
      ok = all(re.search(g, ans, re.I) for g in q['expected']); kind = 'keywords'
    elif q['id'] in ('t1-gen-double-sum', 't2-gen-filter', 't1-fix-index'):
      blocks = re.findall(r'```(?:sol)?\n(.*?)```', ans, re.S) or ([ans] if ans.strip().startswith(('>', 'x', 'r', 't')) else [])
      outs = [run(b.strip()) for b in blocks]
      ok = q['expected'] in outs; kind = 'runs+output'
      a['executed'] = outs
    else:
      ok = states(ans, q['expected']); kind = 'stated output'
    table.setdefault(a['id'], {})[m] = ok
    a['ok'] = ok; a['check'] = kind
  json.dump(json.load(open(f'answers/{m}.json')), open(f'answers/{m}.json', 'w'))
tiers = {}
for qid, r in table.items():
  t = Q[qid]['tier']
  for m, ok in r.items():
    tiers.setdefault(t, {}).setdefault(m, 0); tiers[t][m] += ok
print('tier          n   ' + '  '.join(f'{m:>18s}' for m in models))
for t in ('familiar', 'new-op', 'composition', 'concept'):
  n = sum(1 for q in Q.values() if q['tier'] == t)
  print(f'{t:12s} {n:3d}   ' + '  '.join(f'{tiers[t][m]:>18d}' for m in models))
print(f'{"total":12s} {len(Q):3d}   ' + '  '.join(f'{sum(tiers[t][m] for t in tiers):>18d}' for m in models))
print()
for qid, r in table.items():
  print(f'{qid:24s} expect={str(Q[qid]["expected"])[:28]:28s} ' + ' '.join('Y' if r[m] else '.' for m in models))
