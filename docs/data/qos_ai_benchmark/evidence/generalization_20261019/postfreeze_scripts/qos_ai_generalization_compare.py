"""Identity-checked aggregation of the predeclared broader two-seed experiment."""
import argparse
import json
import math
from pathlib import Path
from qos_ai_data import file_hash,read_rows
from qos_ai_compare import pair_runs,output_diagnostic
from qos_ai_coding_diag import grade_answer
from qos_ai_output_claim_audit import claim_status
from qos_ai_generalization_trace_audit import strict_trace
from qos_ai_resume import learning_rate
from qos_ai_generalization_update_audit import audit as accepted_log_audit

LABELS=('parent','seed7_step0336','seed7_final','seed17_step0336','seed17_final')


def load(path):return json.loads(path.read_text())


def compare(root):
  reports=root/'reports/generalization_20261019';data=root/'curricula/generalization_20261019';snapshot=root/'source_snapshot'
  if load(reports/'status.json')['phase']!='complete':raise ValueError('Fixed workflow is incomplete')
  frozen=load(reports/'frozen_inputs.json')
  for path,sha in frozen.items():
    if file_hash(root/path)!=sha:raise ValueError('Frozen input changed '+path)
  benchmark=read_rows(data/'benchmark.jsonl');curves=read_rows(reports/'learning_curve.jsonl')
  if [r['label'] for r in curves]!=list(LABELS):raise ValueError('Incomplete curve')
  result={'comparison_code_sha256':file_hash(Path(__file__)),'protocol':load(reports/'protocol.json'),
    'benchmark_sha256':file_hash(data/'benchmark.jsonl'),'generalization_hash':load(data/'build.json')['generalization_hash'],
    'separation_audit':load(data/'separation_audit.json'),'runs':{},'scope':'Known-data-controlled development diagnostics; unknown legacy exposure remains. Trace fields are strict canonical matches, not a complete prose judge.'}
  identities={};cache={}
  for label in LABELS:
    path=reports/label;dev={};official={}
    for condition in ('guard','raw'):
      if not (path/f'dev_{condition}').exists():continue
      before,after,pairs=pair_runs(reports/'parent'/f'dev_{condition}',path/f'dev_{condition}',benchmark)
      if any(before[k]!=after[k] for k in ('host','machine','torch')):raise ValueError('Development platform differs')
      outputs=output_diagnostic(pairs)
      dev[condition]={'lexical_matches':sum(p['tuned_proxy'] for p in pairs),'count':len(pairs),'output_matches':outputs['tuned_matches'],'output_count':outputs['count'],
        'finished':sum(p['tuned_finished'] for p in pairs),'severe_loops':sum(p['tuned_severe_loop'] for p in pairs),
        'lexical_regressions':sum(p['parent_proxy'] and not p['tuned_proxy'] for p in pairs),
        'tracks':{track:{'count':sum(p['track']==track for p in pairs),'matches':sum(p['tuned_proxy'] for p in pairs if p['track']==track)} for track in ('familiar','generalization')}}
      ans={r['id']:r['answer'] for r in read_rows(path/f'dev_{condition}'/'answers.jsonl')};grades=[]
      for row in benchmark:
        if row['split']=='dev' and row['category'] in ('debugging','code_generation'):
          g=grade_answer(row,ans[row['id']],snapshot/'fprisc/fpr',snapshot,cache)
          status,literal=claim_status(ans[row['id']],g['expected_stdout'])
          g.update(claim_status=status,task_pass=g['behavior_pass'] and (row['id']!='debug-at' or g['stated_output_match']) and status in ('absent','correct'))
          grades.append(g)
      official[condition]=grades
    coding=load(path/'coding/grading.json');coding_settings=load(path/'coding/summary.json');gen=load(path/'generalization/grading.json');gen_settings=load(path/'generalization/summary.json')
    for name,settings,g,p in [('coding',coding_settings,coding,path/'coding'),('generalization',gen_settings,gen,path/'generalization')]:
      identity={k:v for k,v in settings.items() if k not in ('checkpoint_sha256','checkpoint','checkpoint_label','count','finished','severe_loops','mean_tokens','ancestry')}
      if name in identities and identities[name]!=identity:raise ValueError('Evaluation identity differs '+name)
      identities[name]=identity
      if g['checkpoint_sha256']!=after['checkpoint_sha256'] or g['answers_sha256']!=file_hash(p/'answers.jsonl'):raise ValueError('Stale/mismatched grading '+label)
    code_answers={r['id']:r['answer'] for r in read_rows(path/'coding/answers.jsonl')}
    claim_details=[]
    for g in coding['results']:
      if g['category']!='code_generation':continue
      claim,literal=claim_status(code_answers[g['id']],g['expected_stdout'])
      claim_details.append({'id':g['id'],'working_program':g['complete_pass'],'claim_status':claim,'stated_output':literal,'expected_stdout':g['expected_stdout']})
    generation_claims={'count':len(claim_details),'working_programs':sum(r['working_program'] for r in claim_details),
      **{'working_with_'+status+'_claim':sum(r['working_program'] and r['claim_status']==status for r in claim_details) for status in ('correct','incorrect','ambiguous','absent')},'details':claim_details}
    result['runs'][label]={'checkpoint_sha256':after['checkpoint_sha256'],'development':dev,'official_coding':official,
      'coding':{k:v for k,v in coding.items() if k!='results'},'generalization':{k:v for k,v in gen.items() if k!='results'},'loss':load(path/'loss.json'),'generation_claim_audit':generation_claims}
  for seed in (7,17):
    training=load(root/f'experiments/generalization_20261019_seed{seed}/result.json')
    if training['checkpoint_sha256']!=result['runs'][f'seed{seed}_final']['checkpoint_sha256'] or training['optimizer_steps']!=696 or training['examples_seen']!=2784 or training['budget_stop']:raise ValueError('Incomplete training treatment')
    result.setdefault('training',{})[str(seed)]=training
  recovery={}
  for seed in (7,17):
    experiment=root/f'experiments/generalization_20261019_seed{seed}';records,log_audit=accepted_log_audit(experiment)
    if [r['step'] for r in records]!=list(range(1,697)):raise ValueError('Accepted optimizer records are not canonical')
    if any(abs(r['learning_rate']-learning_rate(1e-5,r['step'],696))>1e-15 for r in records):raise ValueError('Accepted LR schedule differs')
    if records[-1]['supervised_tokens']!=97693 or any(a['supervised_tokens']>=b['supervised_tokens'] for a,b in zip(records,records[1:])):raise ValueError('Accepted token counts differ')
    entry={'accepted_updates':696,'accepted_log_sha256':file_hash(experiment/'accepted_training.jsonl'),'accepted_learning_rate_schedule_verified':True,'log_reconstruction':log_audit}
    if (experiment/'operational_recovery.json').exists():
      entry['operational_recovery']=load(experiment/'operational_recovery.json')
      entry['failed_attempt_logs']=[{'path':str(p),'sha256':file_hash(p)} for p in sorted(experiment.glob('failed_training_*.jsonl'))]
    recovery[str(seed)]=entry
  result['accepted_update_audit']=recovery
  if (reports/'raw_protocol.json').exists():
    if load(reports/'raw_status.json')['phase']!='complete':raise ValueError('Supplemental decoding control is incomplete')
    protocol=load(reports/'raw_protocol.json');raw_runs={};identity=None
    for path,sha in protocol['frozen_inputs'].items():
      if file_hash(root/path)!=sha:raise ValueError('Supplemental diagnostic input changed '+path)
    for label in LABELS:
      path=reports/label/'generalization_raw';summary=load(path/'summary.json');g=load(path/'grading.json')
      same={k:v for k,v in summary.items() if k not in ('checkpoint_sha256','count','finished','severe_loops','ancestry')}
      if identity is not None and same!=identity:raise ValueError('Supplemental settings differ')
      identity=same
      if summary['no_repeat']!=0 or summary['checkpoint_sha256']!=result['runs'][label]['checkpoint_sha256'] or g['answers_sha256']!=file_hash(path/'answers.jsonl'):raise ValueError('Supplemental identity differs')
      if g['diagnostic_hash']!=result['generalization_hash']:raise ValueError('Supplemental prompts differ')
      raw_runs[label]={'summary':summary,**{k:v for k,v in g.items() if k!='results'}}
    result['supplemental_decoding']={'protocol':protocol,'feasibility':load(reports/'repetition_feasibility.json'),'runs':raw_runs}
  diag_rows={r['id']:r for r in read_rows(data/'generalization_diagnostic.jsonl')}
  trace_audits={}
  for condition,folder in [('guard','generalization'),('raw','generalization_raw')]:
    trace_audits[condition]={}
    for label in LABELS:
      path=reports/label/folder
      if not path.exists():continue
      g=load(path/'grading.json');answers={r['id']:r['answer'] for r in read_rows(path/'answers.jsonl')}
      details=[]
      for r in g['results']:
        row=diag_rows[r['id']]
        if row['mode']=='output_only':continue
        audit=strict_trace(row,answers[r['id']])
        details.append({'id':r['id'],'stratum':r['stratum'],'primary_trace_prefix_match':r['canonical_trace_match'],
          'answer_pass':r['complete_pass'],'answer_and_strict_trace_pass':r['complete_pass'] and audit['strict_canonical_trace_match'],**audit})
      trace_audits[condition][label]={'requested_trace_tasks':len(details),'strict_trace_passes':sum(d['strict_canonical_trace_match'] for d in details),
        'answer_and_strict_trace_passes':sum(d['answer_and_strict_trace_pass'] for d in details),
        'prefix_false_positives':sum(d['primary_trace_prefix_match'] and not d['strict_canonical_trace_match'] for d in details),'details':details}
  result['supplemental_strict_trace_audit']={'code_sha256':file_hash(Path('qos_ai_generalization_trace_audit.py')),
    'note':'Post-freeze inspection found malformed numeric prefixes accepted by the primary trace proxy. This audit parses each whole canonical field as JSON. Different valid prose or field names remain unmatched; primary scores are preserved.','runs':trace_audits}
  inference_logs=[]
  for path in sorted(reports.glob('*.log')):
    if not any(marker in path.name for marker in ('_generalization','_dev_','_coding','_loss')):continue
    text=path.read_text(errors='replace').casefold()
    if any(marker in text for marker in ('command buffer exited with error','kiogpucommandbuffer','device target count mismatch')):
      raise ValueError('GPU failure in an inference/loss run: '+str(path))
    inference_logs.append({'path':str(path),'sha256':file_hash(path)})
  for r in result['runs'].values():
    for key in ('train_probe','wording','inputs','combinations'):
      value=r['loss'][key]['answer_cross_entropy']
      if not math.isfinite(value) or value<0:raise ValueError('Invalid loss probe')
  result['inference_gpu_error_log_audit']={'checked_logs':len(inference_logs),'gpu_errors_detected':0,'logs':inference_logs}
  (reports/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
  compact={label:{'dev':r['development']['guard'],'coding':r['coding']['categories'],'generalization':r['generalization']['strata'],'loss':r['loss']} for label,r in result['runs'].items()}
  print(json.dumps(compact,indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args();compare(a.root)
