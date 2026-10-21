"""Paired comparison of fixed development benchmark runs, with explicit proxy labels."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from qos_ai_data import file_hash, read_rows


def pair_runs(parent, tuned, benchmark, manifest_extension=None):
  left, right = read_rows(parent / 'answers.jsonl'), read_rows(tuned / 'answers.jsonl')
  before = json.loads((parent / 'summary.json').read_text())
  after = json.loads((tuned / 'summary.json').read_text())
  settings = ('benchmark_sha256', 'split_manifest_hash', 'split', 'seed', 'temperature', 'top_k',
              'no_repeat', 'requested_max_new_tokens', 'device', 'code_sha256', 'evaluator_sha256', 'split_policy_code_sha256')
  if manifest_extension is not None:
    from qos_ai_curriculum import validate_extension
    chain=list(manifest_extension)
    if len(chain)<2:raise ValueError('A manifest extension chain needs at least two manifests')
    for base,expanded in zip(chain,chain[1:]):validate_extension(base,expanded)
    base,expanded=chain[0],chain[-1]
    if before['split_manifest_hash'] != base['hash'] or after['split_manifest_hash'] != expanded['hash']:
      raise ValueError('Run split hashes do not match the verified extension')
    settings = tuple(k for k in settings if k != 'split_manifest_hash')
  if any(before.get(k) != after.get(k) for k in settings):
    raise ValueError('Comparison settings, data or evaluator differ')
  expected = {r['id'] for r in benchmark if r['split'] == 'dev'}
  if before['split'] != 'dev' or {r['id'] for r in left} != expected or {r['id'] for r in right} != expected:
    raise ValueError('Comparison must cover the complete development partition only')
  if len(left) != len(expected) or len(right) != len(expected):
    raise ValueError('Duplicate/missing benchmark rows')
  questions = {r['id']: r['question'] for r in benchmark if r['split'] == 'dev'}
  pairs = []
  for a, b in zip(left, right):
    if (a['id'], a['question'], a['effective_token_budget']) != (b['id'], b['question'], b['effective_token_budget']):
      raise ValueError('Prompt order/content/token budget differ')
    if a['question'] != questions[a['id']]:
      raise ValueError('Saved prompt differs from the benchmark')
    pairs.append({'id': a['id'], 'group': a['group'], 'category': a['category'], 'track': a.get('track', 'unknown'), 'question': a['question'],
                  'reference_answer': a['reference_answer'], 'parent_answer': a['answer'], 'tuned_answer': b['answer'],
                  'parent_proxy': a['rubric_proxy_pass'], 'tuned_proxy': b['rubric_proxy_pass'],
                  'parent_finished': a['finished'], 'tuned_finished': b['finished'],
                  'parent_severe_loop': a['severe_loop'], 'tuned_severe_loop': b['severe_loop']})
  return before, after, pairs


def stated_output(answer):
  """Extract a stated printed literal, excluding code echoes; null if ambiguous.

  This is a narrower automatic diagnostic, not a full factual/code judgment.
  """
  prose = re.sub(r'```.*?```', '', answer, flags=re.S).replace('**','').replace('`','')
  bare = re.fullmatch(r'\s*(?:"([^"\n]*)"|\x27([^\x27\n]*)\x27|([-+]?\d+(?:\.\d+)?|[A-Za-z]+))\s*', prose)
  if bare:
    return next(v for v in bare.groups() if v is not None)
  captures = re.findall(r'\b(?:it|this sol program|the program|this program)\s+prints?\s+([^\n]+)', prose, re.I)
  captures += re.findall(r'\boutput\b[^\n]{0,100}?\bis\s+([^\n]+)', prose, re.I)
  values = set()
  for value in captures:
    value = re.sub(r'^the\s+(?:string|value|number)\s+', '', value.strip(), flags=re.I)
    match = re.match(r'"([^"\n]*)"|\'([^\'\n]*)\'|([-+]?\d+(?:\.\d+)?|[A-Za-z]+)', value)
    if match:
      literal = next(v for v in match.groups() if v is not None)
      if literal.casefold() not in ('not','the','a','an'):
        values.add(literal)
  return next(iter(values)) if len(values) == 1 else None


def output_diagnostic(pairs):
  selected = [p for p in pairs if p['category'] == 'output_prediction']
  details=[]
  for pair in selected:
    expected=stated_output(pair['reference_answer'])
    if expected is None:
      raise ValueError('Output reference lacks an unambiguous literal')
    before,after=stated_output(pair['parent_answer']),stated_output(pair['tuned_answer'])
    details.append({'id':pair['id'],'expected':expected,'parent_stated':before,'tuned_stated':after,
                    'parent_match':before==expected,'tuned_match':after==expected})
  return {'count':len(details),'parent_matches':sum(r['parent_match'] for r in details),
          'tuned_matches':sum(r['tuned_match'] for r in details),
          'parent_parseable':sum(r['parent_stated'] is not None for r in details),
          'tuned_parseable':sum(r['tuned_stated'] is not None for r in details),
          'note':'Post-hoc diagnostic of explicitly stated output literals; code echoes excluded. Ambiguous answers are unmatched. Not full factual accuracy.',
          'details':details}


def metrics(summary):
  return {k: summary[k] for k in ('checkpoint_sha256','ancestry','count','concept_groups','rubric_proxy_pass_rate',
                                 'finished_rate','severe_loop_rate','mean_tokens','context_limited_count')}


def compare(args):
  benchmark = read_rows(args.benchmark)
  training = json.loads(args.training.read_text())
  result = {'training': training, 'benchmark_sha256': file_hash(args.benchmark),
            'scope': 'complete development partition; no model evaluation of test partition',
            'comparison_code_sha256': file_hash(Path(__file__)),
            'accuracy_note': 'Rubric matching is a lexical proxy. Factual judgments are separate, with explicit reviewer coverage.',
            'runs': {}}
  for condition in ('raw','guard'):
    before, after, pairs = pair_runs(args.reports / f'sft5_parent_{condition}', args.reports / f'sft5_tuned_{condition}', benchmark)
    if before['benchmark_sha256'] != result['benchmark_sha256']:
      raise ValueError('Benchmark changed after evaluation')
    if before['checkpoint_sha256'] != training['parent_sha256'] or after['checkpoint_sha256'] != training['checkpoint_sha256']:
      raise ValueError('Reports do not match fine-tune lineage')
    categories = defaultdict(list)
    groups = defaultdict(list)
    tracks = defaultdict(list)
    for pair in pairs:
      categories[pair['category']].append(pair)
      groups[pair['group']].append(pair)
      tracks[pair['track']].append(pair)
    summarize = lambda rows: {'count':len(rows),'parent_proxy_passes':sum(r['parent_proxy'] for r in rows),
                              'tuned_proxy_passes':sum(r['tuned_proxy'] for r in rows)}
    details = {'no_repeat': before['no_repeat'], 'max_new_tokens': before['requested_max_new_tokens'],
               'parent': metrics(before), 'tuned': metrics(after),
               'explicit_output_diagnostic': output_diagnostic(pairs),
               'proxy_improved': sum(not p['parent_proxy'] and p['tuned_proxy'] for p in pairs),
               'proxy_regressed': sum(p['parent_proxy'] and not p['tuned_proxy'] for p in pairs),
               'categories': {k:summarize(v) for k,v in categories.items()},
               'tracks': {k:summarize(v) for k,v in tracks.items()},
               'concept_groups': {k:summarize(v) for k,v in groups.items()}}
    result['runs'][condition] = details
    (args.reports / f'sft5_paired_{condition}.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in pairs))
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({'training': training, 'runs': {k:{f:v for f,v in r.items() if f not in ('categories','concept_groups')} for k,r in result['runs'].items()}},indent=2))


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--reports',type=Path,required=True)
  parser.add_argument('--training',type=Path,required=True)
  parser.add_argument('--benchmark',type=Path,default=Path(__file__).parent/'qos_ai_benchmark/benchmark.jsonl')
  parser.add_argument('--output',type=Path,required=True)
  compare(parser.parse_args())


if __name__=='__main__':main()
