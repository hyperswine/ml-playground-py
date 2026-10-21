"""Validate and compare both matched balanced trials, saved controls and coding diagnostics."""
import argparse
import json
from pathlib import Path
from qos_ai_data import digest,file_hash,read_rows,rows_fingerprint
from qos_ai_compare import pair_runs,metrics,output_diagnostic
from qos_ai_coding_diag import grade_answer


def load(path):return json.loads(path.read_text())


def compare(root):
  reports=root/'reports/balanced_repair_20261018';data=root/'curricula/balanced_repair_20261018'
  benchmark=read_rows(root/'qos_ai_benchmark/benchmark.jsonl');diagnostic=read_rows(data/'coding_diagnostic.jsonl')
  broad=load(root/'curricula/verified_expanded/splits.json');codegen=load(root/'curricula/input_fidelity_v2/splits.json');new=load(data/'splits.json')
  result={'scope':'unchanged complete 90-prompt development benchmark and separate 96-prompt familiar-source coding diagnostic',
    'benchmark_sha256':file_hash(root/'qos_ai_benchmark/benchmark.jsonl'),'diagnostic_hash':rows_fingerprint(diagnostic),
    'comparison_code_sha256':file_hash(Path(__file__)),'trials':{},'diagnostics':{}}
  matched=('data_sha256','split_hash','epochs_requested','batch_size','grad_accumulation','learning_rate','weight_decay',
    'warmup_steps','gradient_clip','max_minutes','seed','device','host','torch','examples','training_code_sha256',
    'answer_tokens','unknown_answer_tokens_skipped','longest_example','pad_multiple','mps_cache_interval','segment_steps',
    'optimizer_plan_hash','resume_code_sha256','model_code_sha256','fixed_padding','padding_width','selection_sha256',
    'selection_hash','optimizer_steps','epochs_completed','examples_seen','supervised_tokens','budget_stop')
  previous=None;cache={}
  for name,stem,chain in (('broad','expanded_tuned',[broad,codegen,new]),('codegen','inputs_tuned',[codegen,new])):
    experiment=root/f'experiments/balanced_from_{name}_20261018';training=load(experiment/'result.json')
    if file_hash(experiment/'student.pt')!=training['checkpoint_sha256']:raise ValueError('Candidate checkpoint hash differs')
    if previous and any(previous[k]!=training[k] for k in matched):raise ValueError('Training conditions differ between branches')
    if training['optimizer_steps']!=168 or training['epochs_completed']!=1 or training['examples_seen']!=672 or training['budget_stop']:
      raise ValueError('Matched epoch incomplete')
    previous=training;trial={'training':training,'post_training_loss':load(experiment/'post_training_loss.json'),'runs':{}}
    for condition in ('raw','guard'):
      parent=root/f'reports/{stem}_{condition}';tuned=reports/f'{name}_after_{condition}'
      before,after,pairs=pair_runs(parent,tuned,benchmark,chain)
      if before['benchmark_sha256']!=result['benchmark_sha256'] or any(before[k]!=after[k] for k in ('host','machine','torch')):
        raise ValueError('Benchmark/platform identity differs')
      if before['checkpoint_sha256']!=training['parent_sha256'] or after['checkpoint_sha256']!=training['checkpoint_sha256']:
        raise ValueError('Official benchmark lineage differs')
      official={}
      for label,path in (('before',parent),('after',tuned)):
        answers={r['id']:r['answer'] for r in read_rows(path/'answers.jsonl')};grades=[]
        for row in benchmark:
          if row['split']=='dev' and row['category'] in ('debugging','code_generation'):
            grade=grade_answer(row,answers[row['id']],root/'source_snapshot/fprisc/fpr',root/'source_snapshot',cache)
            grade['answer_hash']=digest(answers[row['id']])
            grade['task_pass']=grade['behavior_pass'] and (row['id']!='debug-at' or grade['stated_output_match']) and (grade['stated_output'] is None or grade['stated_output_match'])
            grades.append(grade)
        official[label]={'answers_sha256':file_hash(path/'answers.jsonl'),'details':grades,
          **{category:sum(g['task_pass'] for g in grades if g['category']==category) for category in ('debugging','code_generation')}}
      trial['runs'][condition]={'before':metrics(before),'after':metrics(after),'output_diagnostic':output_diagnostic(pairs),
        'lexical_improvements':sum(not p['parent_proxy'] and p['tuned_proxy'] for p in pairs),
        'lexical_regressions':sum(p['parent_proxy'] and not p['tuned_proxy'] for p in pairs),
        'tracks':{track:{'count':sum(p['track']==track for p in pairs),
          'before':sum(p['parent_proxy'] for p in pairs if p['track']==track),'after':sum(p['tuned_proxy'] for p in pairs if p['track']==track)} for track in ('familiar','generalization')},
        'official_coding':official}
      (reports/f'{name}_paired_{condition}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    result['trials'][name]=trial
  reference_settings=None;reference_prompts=None;reference_grading=None
  keys=('diagnostic_hash','diagnostic_sha256','seed','temperature','top_k','no_repeat','max_new_tokens','device',
    'host','machine','torch','model_code_sha256','generation_code_sha256')
  grading_keys=('diagnostic_hash','grader_sha256','parser_sha256','compiler_sha256','output_parser_sha256')
  for name in ('broad','codegen'):
    for phase in ('before','after'):
      path=reports/f'{name}_{phase}_diag';summary=load(path/'summary.json');grading=load(path/'grading.json');answers=read_rows(path/'answers.jsonl')
      settings={k:summary[k] for k in keys};grader={k:grading[k] for k in grading_keys}
      prompts=[(r['id'],r['question'],r['effective_token_budget']) for r in answers]
      if reference_settings and (settings!=reference_settings or prompts!=reference_prompts or grader!=reference_grading):
        raise ValueError('Coding diagnostic conditions differ')
      reference_settings=settings;reference_prompts=prompts;reference_grading=grader
      if [(r['id'],r['question']) for r in answers]!=[(r['id'],r['question']) for r in diagnostic] or summary['count']!=96:
        raise ValueError('Diagnostic coverage differs')
      if grading['answers_sha256']!=file_hash(path/'answers.jsonl'):raise ValueError('Grading is stale')
      expected=result['trials'][name]['training']['parent_sha256' if phase=='before' else 'checkpoint_sha256']
      if summary['checkpoint_sha256']!=expected or grading['checkpoint_sha256']!=expected:raise ValueError('Diagnostic lineage differs')
      result['diagnostics'][f'{name}_{phase}']={'summary':summary,'grading':{k:v for k,v in grading.items() if k!='results'},
        'families':{family:{'count':sum(r['family']==family for r in grading['results']),
          'complete_pass':sum(r['complete_pass'] for r in grading['results'] if r['family']==family)} for family in sorted({r['family'] for r in diagnostic})}}
  (reports/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({name:{'initial_loss':trial['training']['initial_training_loss'],
    'final_loss':trial['post_training_loss']['final_training_loss'],
    'official':{condition:{'before_proxy':round(run['before']['rubric_proxy_pass_rate']*90),
      'after_proxy':round(run['after']['rubric_proxy_pass_rate']*90),'before_output':run['output_diagnostic']['parent_matches'],
      'after_output':run['output_diagnostic']['tuned_matches'],'coding':run['official_coding']} for condition,run in trial['runs'].items()}}
    for name,trial in result['trials'].items()},indent=2))
  print(json.dumps({name:r['grading']['categories'] for name,r in result['diagnostics'].items()},indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args();compare(a.root)
