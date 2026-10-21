"""Execution-augmented answers: run the model's Sol code (or the question's program) and report real output.

Only side-effect-free code passes `admit`: pure List/Str/Numeric calls, local names and `> print`; no `use`,
process, file or network calls, and no string interpolation. The compiler runs in a temporary directory with JIT, GPU and cache disabled and an
8-second limit. Everything else is left as the model wrote it.

  ask      generate an answer with a checkpoint, then execute it
  rescore  apply execution to answers already saved for the frozen Sol tests (no generation) and compare

  uv run --locked qos execute ask --load CKPT --fpr FPR --question "What does this print? ..."
  uv run --locked qos execute rescore --fpr FPR --tests qos_ai/solgen_20261020 ANSWERS.json ...
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


STATED = re.compile(r'It prints (.*?)\.(?:\s+Steps:|\s*$)', re.S)
BLOCK = re.compile(r'```(?:sol)?\s*\n(.*?)```', re.S)


PURE = {'List': {'sum', 'map', 'filter', 'range', 'take', 'drop', 'rev', 'any', 'all', 'zip', 'last'},
        'Str': {'upper', 'slice', 'find', 'at', 'join', 'trim'},
        'Numeric': {'abs', 'max', 'min', 'clamp'}}
KEYWORDS = {'fn', 'if', 'then', 'else', 'True', 'False', 'str', 'print', 'and', 'or', 'not'}


def admit(code):
  """Allow-list check for side-effect-free programs: pure List/Str/Numeric calls, local names, `> print` only."""
  if '{' in code or '`' in code:
    return 'interpolation or quoting not admitted'
  body = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', code)       # string literals carry no names
  if '"' in body.replace('""', ''):
    return 'unterminated string'
  bound = set(re.findall(r'^\s*([a-z_][a-z0-9_]*)\s*=(?!=)', body, re.M)) | set(re.findall(r'\bfn\s+([a-z_]\w*)', body))
  for line in body.splitlines():
    if line.lstrip().startswith('>') and not re.match(r'\s*>\s*print\s*\(', line):
      return 'only print effects are admitted'
  for module, name in re.findall(r'\b([A-Z]\w*)\.([a-z]\w*)', body):
    if name not in PURE.get(module, ()):
      return f'{module}.{name} not admitted'
  for word in re.findall(r'(?<![\w.])([A-Za-z_]\w*)(?![\w.])', body):
    if word not in KEYWORDS and word not in bound and word not in PURE:
      return f'name {word} not admitted'
  return None


def run_sol(code, fpr, timeout=8):
  """Execute an admitted pure-Sol program. Returns (stdout, None) or (None, reason)."""
  reason = admit(code)
  if reason:
    return None, f'not executed: {reason}'
  with tempfile.TemporaryDirectory(prefix='qos-exec-') as d:
    path = Path(d) / 'answer.sol'; path.write_text(code.strip() + '\n')
    env = {**os.environ, 'SOL_JIT': '0', 'SOL_GPU': '0', 'SOL_CACHE': '0', 'SOL_CACHE_DIR': d + '/cache'}
    try:
      r = subprocess.run([str(fpr), 'sol', str(path)], cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
      return None, 'timed out'
  if r.returncode != 0:
    return None, 'compiler error'
  return r.stdout.rstrip('\n'), None


def stated(answer):
  m = STATED.search(answer)
  return m.group(1).strip() if m else None


def augment(question, answer, fpr):
  """Return the answer with its stated output replaced by real execution where possible."""
  blocks = BLOCK.findall(answer)
  source, code = None, None
  if len(blocks) == 1:
    source, code = 'answer', blocks[0]
  elif not blocks:
    q = BLOCK.findall(question)
    if len(q) == 1:
      source, code = 'question', q[0]
  claim = stated(answer)
  if code is None:
    return {'answer': answer, 'executed': None, 'source': None, 'claimed': claim, 'note': 'no single code block'}
  out, why = run_sol(code, fpr)
  if out is None:
    return {'answer': answer, 'executed': None, 'source': source, 'claimed': claim, 'note': why}
  line = f'It prints {out}.'
  if claim is not None:
    m = STATED.search(answer)
    final = answer[:m.start(1)] + out + answer[m.end(1):] if claim != out else answer
  else:
    final = (answer.rstrip() + '\n' + line) if source == 'answer' else line
  return {'answer': final, 'executed': out, 'source': source, 'claimed': claim,
          'note': 'claim corrected' if claim not in (None, out) else 'claim confirmed' if claim == out else 'output added'}


def ask(args):
  import torch
  import qos_ai as ai
  device = ai.choose_device(args.device)
  model, ck = ai.load_student(args.load, device)
  tok = ai.StudentTokenizer(ck['qwen_ids'])
  questions = [args.question] if args.question else [q for q in open(args.questions).read().split('\n---\n') if q.strip()]
  for q in questions:
    torch.manual_seed(7)
    with torch.inference_mode():
      ids, _ = model.generate_batch([tok.encode(ai.chat_text(q))], args.max_new_tokens, ck['im_end_id'], 0, 20, 0)[0]
    raw = tok.decode(ids).strip()
    r = augment(q, raw, args.fpr)
    print(f'Q: {q}\n--- model ---\n{raw}\n--- executed ({r["note"]}) ---\n{r["answer"]}\n')


def rescore(args):
  """Write execution-augmented copies of saved answers, then score both with the unchanged Stage 3 scorer."""
  from qos_ai_solgen_score import score
  tests = []
  for s in ('fresh', 'held_pairs', 'big_numbers'):
    tests += [json.loads(line) for line in open(Path(args.tests) / f'test_{s}.jsonl')]
  by_id = {t['id']: t for t in tests}
  summary = {}
  for path in args.answers:
    rows = [r for r in json.load(open(path)) if r['id'] in by_id]
    plain = {r['id']: r['answer'] for r in rows}
    aug = {r['id']: augment(by_id[r['id']]['question'], r['answer'], args.fpr) for r in rows}
    tool = {i: a['answer'] for i, a in aug.items()}
    result = {}
    for label, answers in (('without_execution', plain), ('with_execution', tool)):
      scored = score(tests, answers, str(args.fpr))
      result[label] = {'correct': sum(r['correct'] for r in scored),
                       'by_kind': {k: sum(r['correct'] for r in scored if r['kind'] == k) for k in ('output', 'generate', 'repair')}}
    notes = {}
    for a in aug.values():
      key = a['note'].split(':')[0] + ('' if a['source'] is None else f' ({a["source"]} code)')
      notes[key] = notes.get(key, 0) + 1
    result.update(n=len(rows), kinds={k: sum(t['kind'] == k for t in tests) for k in ('output', 'generate', 'repair')}, notes=notes)
    summary[Path(path).stem] = result
  print(json.dumps(summary, indent=1))
  if args.output:
    Path(args.output).write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  a = s.add_parser('ask')
  a.add_argument('--load', type=Path, required=True)
  a.add_argument('--fpr', type=Path, required=True)
  a.add_argument('--question'); a.add_argument('--questions', type=Path, help='file of questions separated by lines of ---')
  a.add_argument('--max-new-tokens', type=int, default=200)
  a.add_argument('--device', default='mps')
  r = s.add_parser('rescore')
  r.add_argument('--fpr', type=Path, required=True)
  r.add_argument('--tests', default='qos_ai/solgen_20261020')
  r.add_argument('--output')
  r.add_argument('answers', nargs='+')
  args = p.parse_args()
  ask(args) if args.command == 'ask' else rescore(args)
