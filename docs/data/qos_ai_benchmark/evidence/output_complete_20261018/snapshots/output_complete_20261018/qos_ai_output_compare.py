"""Compare the fixed output/complete-repair follow-up with its saved coding parent."""
import json
from pathlib import Path
from qos_ai_data import digest,file_hash,read_rows,rows_fingerprint
from qos_ai_compare import pair_runs,metrics,output_diagnostic
from qos_ai_coding_diag import grade_answer


def load(path):return json.loads(path.read_text())


def compare():
  reports=Path('reports/output_complete_20261018');parent_reports=Path('reports/balanced_repair_20261018')
  data=Path('curricula/output_complete_20261018');previous=Path('curricula/balanced_repair_20261018')
  experiment=Path('experiments/output_complete_20261018');training=load(experiment/'result.json')
  benchmark=read_rows(Path('qos_ai_benchmark/benchmark.jsonl'));diag=read_rows(previous/'coding_diagnostic.jsonl')
  if file_hash(experiment/'student.pt')!=training['checkpoint_sha256']:raise ValueError('Evaluated checkpoint identity differs')
  result={'training':training,'post_training_loss':load(experiment/'post_training_loss.json'),
    'comparison_code_sha256':file_hash(Path(__file__)),'benchmark_sha256':file_hash(Path('qos_ai_benchmark/benchmark.jsonl')),
    'diagnostic_hash':rows_fingerprint(diag),'scope':'90 development prompts and fixed 96-prompt coding diagnostic; no test model inference',
    'runs':{},'diagnostics':{}}
  cache={}
  for condition in ('raw','guard'):
    left=parent_reports/f'codegen_after_{condition}';right=reports/f'tuned_{condition}'
    before,after,pairs=pair_runs(left,right,benchmark,[load(previous/'splits.json'),load(data/'splits.json')])
    if before['checkpoint_sha256']!=training['parent_sha256'] or after['checkpoint_sha256']!=training['checkpoint_sha256']:
      raise ValueError('Official benchmark lineage differs')
    if any(before[k]!=after[k] for k in ('host','machine','torch')) or before['benchmark_sha256']!=result['benchmark_sha256']:
      raise ValueError('Official benchmark/platform differs')
    coding={}
    for label,path in (('before',left),('after',right)):
      answers={r['id']:r['answer'] for r in read_rows(path/'answers.jsonl')};grades=[]
      for row in benchmark:
        if row['split']=='dev' and row['category'] in ('debugging','code_generation'):
          grade=grade_answer(row,answers[row['id']],Path('source_snapshot/fprisc/fpr'),Path('source_snapshot'),cache)
          grade['answer_hash']=digest(answers[row['id']]);grade['task_pass']=grade['behavior_pass'] and (row['id']!='debug-at' or grade['stated_output_match']) and (grade['stated_output'] is None or grade['stated_output_match'])
          grades.append(grade)
      coding[label]={'answers_sha256':file_hash(path/'answers.jsonl'),'details':grades,
        **{c:sum(g['task_pass'] for g in grades if g['category']==c) for c in ('code_generation','debugging')}}
    result['runs'][condition]={'before':metrics(before),'after':metrics(after),'output_diagnostic':output_diagnostic(pairs),
      'lexical_improvements':sum(not p['parent_proxy'] and p['tuned_proxy'] for p in pairs),
      'lexical_regressions':sum(p['parent_proxy'] and not p['tuned_proxy'] for p in pairs),'official_coding':coding,
      'tracks':{track:{'count':sum(p['track']==track for p in pairs),'before':sum(p['parent_proxy'] for p in pairs if p['track']==track),
        'after':sum(p['tuned_proxy'] for p in pairs if p['track']==track)} for track in ('familiar','generalization')}}
    (reports/f'paired_{condition}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
  reference=None
  for label,path in (('before',parent_reports/'codegen_after_diag'),('after',reports/'tuned_diag')):
    summary=load(path/'summary.json');grading=load(path/'grading.json');answers=read_rows(path/'answers.jsonl')
    keys=('diagnostic_hash','diagnostic_sha256','seed','temperature','top_k','no_repeat','max_new_tokens','device',
      'host','machine','torch','model_code_sha256','generation_code_sha256')
    grading_keys=('diagnostic_hash','grader_sha256','parser_sha256','compiler_sha256','output_parser_sha256')
    identity=({k:summary[k] for k in keys},{k:grading[k] for k in grading_keys},[(r['id'],r['question'],r['effective_token_budget']) for r in answers])
    if reference is not None and reference!=identity:raise ValueError('Coding diagnostic conditions differ')
    reference=identity
    if [(r['id'],r['question']) for r in answers]!=[(r['id'],r['question']) for r in diag] or summary['count']!=96:
      raise ValueError('Coding diagnostic coverage differs')
    expected=training['parent_sha256' if label=='before' else 'checkpoint_sha256']
    if grading['checkpoint_sha256']!=expected or summary['checkpoint_sha256']!=expected or grading['answers_sha256']!=file_hash(path/'answers.jsonl'):
      raise ValueError('Diagnostic identity or answer hashes differ')
    result['diagnostics'][label]={'summary':summary,'grading':{k:v for k,v in grading.items() if k!='results'}}
  (reports/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({c:{'before_proxy':round(r['before']['rubric_proxy_pass_rate']*90),'after_proxy':round(r['after']['rubric_proxy_pass_rate']*90),
    'before_output':r['output_diagnostic']['parent_matches'],'after_output':r['output_diagnostic']['tuned_matches'],
    'coding':r['official_coding']} for c,r in result['runs'].items()},indent=2))
  print(json.dumps({label:d['grading']['categories'] for label,d in result['diagnostics'].items()},indent=2))


if __name__=='__main__':compare()
