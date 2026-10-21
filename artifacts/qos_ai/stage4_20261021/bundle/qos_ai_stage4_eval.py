"""Prospective multi-input Sol diagnostic. Historical scores are never rewritten.

Answers must be one pure expression over the supplied variables. Parse the
restricted subset, reconstruct it, and run the same expression on every case.
This is a conservative admission policy, not a general Sol sandbox.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from qos_ai_pure_sol import Parser, Unsupported, validate_tree, children


def sha(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ExpressionParser(Parser):
  def starts_atom(self, token):
    return token not in ('then', 'else') and super().starts_atom(token)

  def atom(self):
    if self.peek() != 'if':
      return super().atom()
    self.pop()
    condition = self.expr(); self.expect('then')
    yes = self.expr(); self.expect('else')
    no = self.expr()
    return ('if', condition, yes, no)


def validate(node, in_lambda=False):
  if node[0] == 'if':
    for branch in node[1:]:
      validate(branch, in_lambda)
  else:
    validate_tree(node, in_lambda)
    # The historical validator predates conditionals. Check nested branches too.
    for child in children(node):
      validate(child, in_lambda or node[0] == 'lambda')


def parse_expression(answer, variables):
  blocks = re.findall(r'```(?:sol)?\s*\n(.*?)```', answer, re.S)
  if len(blocks) > 1:
    raise Unsupported('multiple code blocks')
  code = blocks[0].strip() if blocks else answer.strip()
  p = ExpressionParser(code)
  p.locals = set(variables)
  node = p.expr()
  if p.peek() is not None:
    raise Unsupported('expected only an expression, without declarations or effects')
  validate(node)
  return node


def render(node):
  kind = node[0]
  if kind == 'int': return str(node[1]) if node[1] >= 0 else f'({node[1]})'
  if kind == 'string': return json.dumps(node[1])
  if kind in ('var', 'name'): return node[1]
  if kind == 'neg': return f'(-{render(node[1])})'
  if kind == 'list': return '[' + ', '.join(render(x) for x in node[1]) + ']'
  if kind == 'app': return f'({render(node[1])} {render(node[2])})'
  if kind == 'bin': return f'({render(node[2])} {node[1]} {render(node[3])})'
  if kind == 'lambda': return f'(fn {" ".join(node[1])} -> {render(node[2])})'
  if kind == 'if': return f'(if {render(node[1])} then {render(node[2])} else {render(node[3])})'
  raise Unsupported('unsupported AST')


def literal(value):
  if isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 1000:
    return str(value) if value >= 0 else f'({value})'
  if isinstance(value, list) and len(value) <= 64:
    return '[' + ', '.join(literal(x) for x in value) + ']'
  raise ValueError('diagnostic inputs must be bounded integers or lists')


def run_expression(fpr, node, inputs):
  if any(not re.fullmatch(r'[a-z][a-z0-9_]*', name) for name in inputs):
    raise ValueError('invalid input name')
  code = '\n'.join(f'{name} = {literal(value)}.' for name, value in inputs.items())
  code += f'\n> print (str ({render(node)})).\n'
  with tempfile.TemporaryDirectory(prefix='qos-eval-') as directory:
    script = Path(directory) / 'case.sol'; script.write_text(code)
    env = {**os.environ, 'SOL_CACHE_DIR': directory + '/cache', 'SOL_JIT': '0', 'SOL_GPU': '0'}
    try:
      result = subprocess.run([str(Path(fpr).resolve()), 'sol', str(script)], cwd=directory,
                              env=env, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
      return None
    return result.stdout.rstrip('\n') if result.returncode == 0 else None


# Cases include zero/negative parameters, equality boundaries and both if arms.
INPUTS = [dict(xs=xs, k=k, limit=lim) for xs, k, lim in [
  ([2, 7, 4], 5, 4), ([-3, 0, 8], -2, 0), ([], 0, 0),
  ([4, 4, 5], 0, 4), ([61, 73, -5], 2, 61), ([-8, -2, -1], -1, -2)]]
FAMILIES = [
  ('map_sum', 'List.sum (List.map (fn x -> x * k) xs)',
   'multiply each element of xs by k and add the products',
   lambda v: sum(x * v['k'] for x in v['xs'])),
  ('filter_sum', 'List.sum (List.filter (fn x -> x > limit) xs)',
   'add only the elements of xs strictly greater than limit',
   lambda v: sum(x for x in v['xs'] if x > v['limit'])),
  ('clamp_sum', 'List.sum (List.map (fn x -> Numeric.clamp 0 10 x) xs)',
   'clamp each element of xs to the inclusive interval 0 through 10, then add them',
   lambda v: sum(min(10, max(0, x)) for x in v['xs'])),
  ('signed_branch', 'if k > 0 then List.sum (List.map (fn x -> x * k) xs) else List.sum xs',
   'when k is positive, sum the elements of xs multiplied by k; otherwise sum xs unchanged',
   lambda v: sum(x * v['k'] for x in v['xs']) if v['k'] > 0 else sum(v['xs'])),
  ('filter_map_sum', 'List.sum (List.map (fn x -> x * k) (List.filter (fn x -> x > limit) xs))',
   'keep elements strictly above limit, multiply the retained elements by k, and sum them',
   lambda v: sum(x * v['k'] for x in v['xs'] if x > v['limit'])),
  ('branch_max', 'if k > 0 then Numeric.max k (List.sum xs) else Numeric.abs (List.sum xs)',
   'if k is positive, choose the larger of k and the sum of xs; otherwise take the absolute sum of xs',
   lambda v: max(v['k'], sum(v['xs'])) if v['k'] > 0 else abs(sum(v['xs']))),
]
PROMPTS = [
  'Return a reusable Sol expression over xs, k and limit to {description}.',
  'The bindings xs, k and limit will vary between calls. Express this rule in Sol: {description}.',
  'Write this computation as a Sol expression using the supplied variable names: {description}.',
]


def build_suite(fpr, output):
  if output.exists(): raise ValueError('choose a new suite path')
  rows = []
  for family, reference, description, oracle in FAMILIES:
    node = parse_expression(reference, ('xs', 'k', 'limit'))
    cases = []
    for values in INPUTS:
      expected = str(oracle(values))
      actual = run_expression(fpr, node, values)
      if actual != expected:
        raise ValueError(f'compiler/reference disagreement: {family} {values}: {actual} != {expected}')
      cases.append({'inputs': values, 'expected': expected})
    for wording, template in enumerate(PROMPTS):
      rows.append({'id': f'{family}-p{wording}', 'family': family, 'wording': wording,
                   'question': template.format(description=description) +
                   ' Return one expression in a sol code block, without bindings or print statements.',
                   'reference': reference, 'cases': cases})
  suite = {'schema': 1, 'compiler_sha256': sha(fpr), 'rows': rows,
           'scope': 'prospective development diagnostic; shared concepts, new variable-input contract; not a final test'}
  suite['hash'] = fingerprint(suite)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(suite, indent=2) + '\n')
  return suite


def validate_suite(suite, fpr):
  if suite['hash'] != fingerprint({k: v for k, v in suite.items() if k != 'hash'}):
    raise ValueError('suite hash mismatch')
  if sha(fpr) != suite['compiler_sha256']:
    raise ValueError('compiler identity differs from frozen suite')


def score(suite, answers, fpr):
  validate_suite(suite, fpr)
  ids = [r['id'] for r in answers]
  expected_ids = {r['id'] for r in suite['rows']}
  if len(ids) != len(set(ids)) or set(ids) != expected_ids:
    raise ValueError('answers must cover every suite ID exactly once')
  lookup = {r['id']: r['answer'] for r in answers}
  rows = []
  for task in suite['rows']:
    try:
      node = parse_expression(lookup[task['id']], task['cases'][0]['inputs'])
      passed = [run_expression(fpr, node, case['inputs']) == case['expected'] for case in task['cases']]
      rows.append({'id': task['id'], 'family': task['family'], 'passed_cases': sum(passed),
                   'total_cases': len(passed), 'correct': all(passed), 'admitted': True})
    except (Unsupported, RecursionError) as error:
      rows.append({'id': task['id'], 'family': task['family'], 'correct': False,
                   'admitted': False, 'reason': str(error)})
  return {'suite_hash': suite['hash'], 'answers_hash': fingerprint(answers),
          'compiler_sha256': sha(fpr), 'correct': sum(r['correct'] for r in rows),
          'total': len(rows), 'rows': rows,
          'scope': 'finite-case behavioral checks; unsupported answers count as unmatched, not proof of incorrect Sol'}


def generate(suite_path, checkpoint, output, device_name, tokenizer_path):
  import platform
  import torch
  import qos_ai as ai
  from qos_ai_runpod_data import FrozenTokenizer
  suite = json.loads(suite_path.read_text())
  if suite['hash'] != fingerprint({k: v for k, v in suite.items() if k != 'hash'}):
    raise ValueError('suite hash mismatch')
  summary_path = output.with_name(output.stem + '_generation.json')
  if output.exists() or summary_path.exists(): raise ValueError('choose new answer and summary paths')
  tokenizer_hash = sha(tokenizer_path / 'tokenizer.json')
  if tokenizer_hash != '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523':
    raise ValueError('tokenizer differs from the frozen preparation tokenizer')
  device = ai.choose_device(device_name)
  model, parent = ai.load_student(checkpoint, device)
  tokenizer = ai.StudentTokenizer(parent['qwen_ids'], FrozenTokenizer(tokenizer_path))
  answers = []; torch.manual_seed(7)
  for row in suite['rows']:
    prompt = tokenizer.encode(ai.chat_text(row['question']))
    if 0 in prompt or len(prompt) + 200 > model.config.context_length:
      raise ValueError('unknown or overlong prompt: ' + row['id'])
    with torch.inference_mode():
      ids, finished = model.generate_batch([prompt], 200, parent['im_end_id'], 0, 20, 0)[0]
    answers.append({'id': row['id'], 'answer': tokenizer.decode(ids).strip(),
                    'finished': finished, 'tokens': len(ids), 'severe_loop': ai.loops(ids, phrase=4, times=4)})
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(answers, indent=2) + '\n')
  summary = {'checkpoint_sha256': sha(checkpoint), 'suite_hash': suite['hash'], 'answers_sha256': sha(output),
             'tokenizer_sha256': tokenizer_hash,
             'seed': 7, 'temperature': 0, 'top_k': 20, 'no_repeat': 0, 'max_new_tokens': 200,
             'device': str(device), 'host': platform.node(), 'torch': str(torch.__version__),
             'count': len(answers), 'finished': sum(a['finished'] for a in answers),
             'severe_loops': sum(a['severe_loop'] for a in answers)}
  summary_path.write_text(json.dumps(summary, indent=2) + '\n')
  return summary


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('command', choices=['build', 'generate', 'score'])
  p.add_argument('--fpr', type=Path)
  p.add_argument('--suite', type=Path, required=True)
  p.add_argument('--answers', type=Path)
  p.add_argument('--output', type=Path)
  p.add_argument('--checkpoint', type=Path)
  p.add_argument('--tokenizer', type=Path, default=Path('runpod/557m/bundle/tokenizer'))
  p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
  a = p.parse_args()
  if a.command in ('build', 'score') and not a.fpr: p.error('build and score require --fpr')
  if a.command == 'build':
    suite = build_suite(a.fpr, a.suite)
    print(json.dumps({'suite_hash': suite['hash'], 'tasks': len(suite['rows']), 'reference_cases': 36}))
  elif a.command == 'generate':
    if not a.checkpoint or not a.output: p.error('generate requires --checkpoint and --output')
    print(json.dumps(generate(a.suite, a.checkpoint, a.output, a.device, a.tokenizer), indent=2))
  else:
    if not a.answers or not a.output or a.output.exists():
      p.error('score requires answers and a new output path')
    result = score(json.loads(a.suite.read_text()), json.loads(a.answers.read_text()), a.fpr)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__': main()
