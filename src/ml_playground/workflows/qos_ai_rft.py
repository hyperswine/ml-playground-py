"""Rejection-sampling fine-tuning data: sample the model, keep only answers the real Sol compiler confirms.

Earlier self-play failed because nothing judged the student's own answers. Here every kept answer is executed:
parameterized expressions must match all six oracle cases through the multi-input suite's runner, and program
generation/repair answers must run and print the expected value *and* state it.

  tasks   build fresh tasks (new seeds; excludes the frozen tests and all training programs)     [local]
  sample  draw K answers per task from a checkpoint at temperature T                           [M1, MPS]
  verify  check every sample with fpr and keep up to N distinct correct answers per task        [local]

  uv run --locked qos rft tasks --fpr FPR --output DIR
  python qos_ai_rft.py sample --load CKPT --tasks DIR/tasks.jsonl --output DIR/samples.jsonl    (on the M1)
  uv run --locked qos rft verify --fpr FPR --tasks DIR/tasks.jsonl --samples DIR/samples.jsonl --output DIR/accepted.jsonl
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATED = re.compile(r'It prints (.*?)\.(?:\s+Steps:|\s*$)', re.S)
BLOCK = re.compile(r'```(?:sol)?\s*\n(.*?)```', re.S)


def build_tasks(args):
  import qos_ai_stage4_eval as ev
  import qos_ai_stage4_param as s4
  import qos_ai_stage5_param as s5
  import qos_ai_solgen as sg
  if args.output.exists(): raise ValueError('choose a new output directory')
  fpr = str(Path(args.fpr).expanduser())
  suite = json.loads(s4.SUITE.read_text()); refs = {r['reference'] for r in suite['rows']}
  seen_exprs = set()
  for path in args.exclude_param:
    seen_exprs |= {json.loads(l)['expression'] for l in open(path)}
  banned = {s4.skeleton(t) for t in [
    ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('filter', '>', 'limit', ('xs',))),
    ('sum', ('map', ('clamp', 0, 10), ('xs',))),
    ('if', ('>', ('var', 'k'), 0), ('sum', ('map', ('op', '*', 'k'), ('xs',))), ('sum', ('xs',))),
    ('sum', ('map', ('op', '*', 'k'), ('filter', '>', 'limit', ('xs',)))),
    ('if', ('>', ('var', 'k'), 0), ('max', ('var', 'k'), ('sum', ('xs',))), ('abs', ('sum', ('xs',))))]}
  grams = s5.suite_ngrams()
  rng = random.Random(20261031); gen = s4.Gen(rng); phr = s5.Phraser(random.Random(20261032))
  tasks = []
  while sum(t['kind'] == 'param' for t in tasks) < args.param:
    t = gen.int_(rng.choice([1, 2, 2, 3])); expr = s4.render(t)
    if expr in seen_exprs or expr in refs or s4.skeleton(t) in banned: continue
    seen_exprs.add(expr)
    cs = s4.cases(rng)
    try: expected = [str(s4.evaluate(t, c)) for c in cs]
    except Exception: continue
    if any(abs(int(v)) > 999 for v in expected) or len(set(expected)) < 2: continue
    d, mood = phr.describe(t); words = re.findall(r'\w+', d.lower())
    if any(tuple(words[i:i + 6]) in grams for i in range(len(words) - 5)): continue
    q = rng.choice(s5.STEP_PROMPTS if mood == 'steps' else s5.PROMPTS).format(d=d) + ' Return the expression in a sol code block.'
    tasks.append({'id': f'rft-param-{len(tasks):05d}', 'kind': 'param', 'question': q, 'reference': expr,
                  'cases': [{'inputs': c, 'expected': e} for c, e in zip(cs, expected)]})
  # Program generation / repair tasks from the Stage 3 generator with a new seed.
  frozen = set()
  for name in ('train', 'test_fresh', 'test_held_pairs', 'test_big_numbers'):
    frozen |= {json.loads(l)['code'] for l in open(Path(args.solgen) / f'{name}.jsonl')}
  g = sg.Gen(random.Random(20261033), banned=frozenset(sg.HELD_PAIRS)); trng = random.Random(20261034)
  while sum(t['kind'] != 'param' for t in tasks) < args.programs:
    try: node = g.program()
    except sg.Bad: continue
    kind = trng.choice(['generate', 'generate', 'repair'])
    item = sg.task(node, kind, trng)
    if not item or item['code'] in frozen or re.search(r'\b(6[1-9]|[7-9]\d|1[0-4]\d|150)\b', item['code']): continue
    frozen.add(item['code'])
    tasks.append({'id': f'rft-{kind}-{len(tasks):05d}', 'kind': kind, 'question': item['question'],
                  'expected': item['expected'], 'reference': item['code']})
  # Confirm every reference with the compiler before sampling.
  def ok(t):
    if t['kind'] == 'param':
      try:
        node = ev.parse_expression(f"```sol\n{t['reference']}\n```", ('xs', 'k', 'limit'))
      except Exception:
        return False  # outside the suite's admitted subset, so it could never be scored
      return all(ev.run_expression(fpr, node, c['inputs']) == c['expected'] for c in t['cases'])
    return sg.run_fpr(fpr, t['reference']) == t['expected']
  with ThreadPoolExecutor(8) as pool:
    flags = list(pool.map(ok, tasks))
  tasks = [t for t, f in zip(tasks, flags) if f]
  args.output.mkdir(parents=True)
  with open(args.output / 'tasks.jsonl', 'w') as f:
    for t in tasks: f.write(json.dumps(t) + '\n')
  report = {'tasks': len(tasks), 'kinds': {k: sum(t['kind'] == k for t in tasks) for k in ('param', 'generate', 'repair')},
            'reference_failures': flags.count(False), 'compiler_sha256': ev.sha(fpr)}
  (args.output / 'tasks.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


def sample(args):
  import torch
  import qos_ai as ai
  device = ai.choose_device(args.device)
  model, ck = ai.load_student(args.load, device); tok = ai.StudentTokenizer(ck['qwen_ids'])
  tasks = [json.loads(l) for l in open(args.tasks)]
  done = set()
  if args.output.exists():
    done = {json.loads(l)['id'] for l in open(args.output)}
  torch.manual_seed(args.seed)
  with open(args.output, 'a') as out:
    for i, t in enumerate(tasks):
      if t['id'] in done: continue
      prompt = tok.encode(ai.chat_text(t['question']))
      with torch.inference_mode():
        results = model.generate_batch([prompt] * args.k, args.max_new_tokens, ck['im_end_id'], args.temperature, 40, 0)
      out.write(json.dumps({'id': t['id'], 'samples': [tok.decode(ids).strip() for ids, _ in results]}) + '\n'); out.flush()
      if i % 50 == 0: print(f'{i}/{len(tasks)}', flush=True)


def verify(args):
  import qos_ai_stage4_eval as ev
  import qos_ai_solgen as sg
  fpr = str(Path(args.fpr).expanduser())
  tasks = {json.loads(l)['id']: json.loads(l) for l in open(args.tasks)}
  rows = [json.loads(l) for l in open(args.samples)]
  def judge(item):
    t, answer = item
    if t['kind'] == 'param':
      try:
        node = ev.parse_expression(answer, ('xs', 'k', 'limit'))
      except Exception:
        return False
      return all(ev.run_expression(fpr, node, c['inputs']) == c['expected'] for c in t['cases'])
    blocks = BLOCK.findall(answer); m = STATED.search(answer)
    return len(blocks) == 1 and m is not None and m.group(1).strip() == t['expected'] and sg.run_fpr(fpr, blocks[0].strip()) == t['expected']
  pairs = [(tasks[r['id']], a) for r in rows for a in r['samples']]
  with ThreadPoolExecutor(8) as pool:
    verdicts = list(pool.map(judge, pairs))
  accepted, per_task, solved = [], {}, set()
  for (t, a), ok in zip(pairs, verdicts):
    if not ok: continue
    solved.add(t['id'])
    seen = per_task.setdefault(t['id'], set())
    if a in seen or len(seen) >= args.per_task: continue
    seen.add(a); accepted.append({'id': f"{t['id']}-a{len(seen)}", 'question': t['question'], 'answer': a, 'kind': t['kind']})
  with open(args.output, 'w') as f:
    for r in accepted: f.write(json.dumps(r) + '\n')
  by_kind = {}
  for t in tasks.values():
    k = by_kind.setdefault(t['kind'], [0, 0]); k[1] += 1; k[0] += t['id'] in solved
  report = {'samples': len(pairs), 'correct_samples': sum(verdicts), 'tasks_solved': {k: f'{v[0]}/{v[1]}' for k, v in by_kind.items()},
            'accepted': len(accepted), 'per_task_limit': args.per_task}
  Path(str(args.output) + '.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  t = s.add_parser('tasks'); t.add_argument('--fpr', required=True); t.add_argument('--output', type=Path, required=True)
  t.add_argument('--param', type=int, default=1500); t.add_argument('--programs', type=int, default=1500)
  t.add_argument('--solgen', default='qos_ai/solgen_20261020')
  t.add_argument('--exclude-param', nargs='*', default=['qos_ai_benchmark/stage4_20261021/param/train.jsonl',
                                                         'qos_ai_benchmark/stage5_20261021/param/train.jsonl'])
  sm = s.add_parser('sample'); sm.add_argument('--load', type=Path, required=True); sm.add_argument('--tasks', type=Path, required=True)
  sm.add_argument('--output', type=Path, required=True); sm.add_argument('--k', type=int, default=8)
  sm.add_argument('--temperature', type=float, default=0.8); sm.add_argument('--max-new-tokens', type=int, default=200)
  sm.add_argument('--seed', type=int, default=7); sm.add_argument('--device', default='mps')
  v = s.add_parser('verify'); v.add_argument('--fpr', required=True); v.add_argument('--tasks', type=Path, required=True)
  v.add_argument('--samples', type=Path, required=True); v.add_argument('--output', type=Path, required=True)
  v.add_argument('--per-task', type=int, default=2)
  a = p.parse_args()
  {'tasks': build_tasks, 'sample': sample, 'verify': verify}[a.command](a)
