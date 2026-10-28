"""Confirm the development-selected curriculum with a new seed and frozen regression gates."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
root=Path('/Users/jq/qos_ai_eval');work=root/'experiments/qwen2b_controlled_20261028';old=root/'experiments/qwen2b_repair_20261027'
os.chdir(work);sys.path[:0]=[str(work),str(old),str(root)]
from qos_ai_qwen_repair import read,write,score,digest
from qos_ai_qwen_controlled import ordered_schedule
from qos_ai_generalization_trace_audit import strict_trace
from transformers import AutoTokenizer
mlx=root/'mlx_venv/bin/python';py=root/'.venv/bin/python';model=root/'experiments/mlx_20261027/mlx_2b_round4_q8'
fpr=root/'experiments/stage4_20261021/preparation/toolchain/fpr'
env={**os.environ,'PYTHONPATH':':'.join([str(work),str(old),str(root)])}

def status(phase,state='running',**kw):
 (work/'confirmation_status.json').write_text(json.dumps({'phase':phase,'state':state,'time':time.time(),**kw},indent=2)+'\n')

def run(label,argv):
 status(label)
 with open(label+'.log','w') as log:subprocess.run([str(x) for x in argv],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

try:
 winner=json.loads((work/'selection.json').read_text())['winner'];assert winner
 seed=20261029;label=winner['tag'].split('_')[0];budget=int(winner['tag'].split('_')[1])
 data=read(work/'prepared/replay_pool.jsonl')+(read(work/'prepared/new_pool.jsonl') if label=='mixed' else [])
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 schedule,points=ordered_schedule(data,tok,[b for b in (2048,4096,8192) if b<=budget],seed,label=='mixed')
 write(work/'repeat_schedule.jsonl',schedule)
 floors={'solgen_correct':339,'program_correct':111,'multi_correct':18,
  'bench':{'dev_outputs_33':29,'working_generation_24':24,'applied_repairs_48':45,'repair_plus_output_48':34,
   'no_change_plus_output_24':18,'gen_wording_48':42,'gen_inputs_48':41,'gen_combinations_48':40,'strict_trace_120':106}}
 plan={'winner':winner,'repeat_seed':seed,'milestones':points,'repeat_schedule_sha256':digest(work/'repeat_schedule.jsonl'),
  'same_development_screen':True,'full_regression_floors':floors,
  'final_policy':'only generate final holdout if primary and repeat pass full regression and repeat passes development; no automatic default change',
  'created':time.time()}
 (work/'confirmation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 count=len(schedule);adapter=work/'adapter_repeat'
 run('repeat_train',[mlx,work/'mlx_repair.py','train','--model',model,'--train',work/'repeat_schedule.jsonl','--valid',work/'prepared/valid.jsonl',
  '--output',adapter,'--iters',count,'--learning-rate','1e-5','--seed',seed,'--ordered','--checkpoints',count])
 receipt=json.loads((adapter/f'step_{count:04d}/receipt.json').read_text());assert receipt['trained_tokens']==points[-1]['actual_tokens']
 for suite in ('dev','guard'):
  run('repeat_'+suite,[mlx,work/'mlx_repair.py','answer','--model',model,'--adapter',adapter,'--prompts',work/f'{suite}_prompts.jsonl','--output',work/f'repeat_{suite}.jsonl'])
 score(SimpleNamespace(rows=work/'dev_rows.jsonl',answers=work/'repeat_dev.jsonl',fpr=str(fpr),output=work/'repeat_scores.json'))
 s=json.loads((work/'repeat_scores.json').read_text());base=json.loads((work/'parent_scores.json').read_text());guard=read(work/'guard_rows.jsonl')
 answers={r['id']:r['answer'] for r in read(work/'repeat_guard.jsonl')}
 trace=sum(strict_trace(r,answers['gen/'+r['id']])['strict_canonical_trace_match'] for r in guard)
 baseguard=json.loads((work/'plan.json').read_text())['parent_guard_strict_trace']
 passed=(s['code_complete']>=base['code_complete']+2 and s['code_programs_correct']>=base['code_programs_correct']
   and all(s['by_kind'][k]['correct']>=v['correct']-1 for k,v in base['by_kind'].items()) and trace>=baseguard-1)
 repeat={'complete':s['code_complete'],'programs':s['code_programs_correct'],'outputs':s['code_claims_correct'],'strict_trace_guard':trace,'passes_screen':passed,'tokens':points[-1]['actual_tokens']}
 (work/'repeat_comparison.json').write_text(json.dumps(repeat,indent=2)+'\n')
 full={}
 for label,ad in [('selected',Path(winner['adapter'])),('repeat',adapter)]:
  if label=='repeat' and (not passed or not full['selected']['passes']):break
  run(label+'_frozen',[mlx,work/'mlx_repair.py','answer','--model',model,'--adapter',ad,'--prompts',old/'frozen_prompts.jsonl','--output',work/f'{label}_frozen.jsonl'])
  run(label+'_frozen_score',[py,old/'score_qwen_frozen.py','--root',root,'--answers',work/f'{label}_frozen.jsonl','--output',work/f'{label}_frozen_scored'])
  summary=json.loads((work/f'{label}_frozen_scored/summary.json').read_text())
  failures={k:{'minimum':v,'actual':summary[k]} for k,v in floors.items() if k!='bench' and summary[k]<v}
  failures.update({k:{'minimum':v,'actual':summary['bench'][k]} for k,v in floors['bench'].items() if summary['bench'][k]<v})
  if summary['pure_admission_declines']:failures['execution_admission']='some programs declined'
  full[label]={'passes':not failures,'failures':failures}
  (work/'full_regression_decision.json').write_text(json.dumps(full,indent=2)+'\n')
 final_run=False
 if passed and full.get('selected',{}).get('passes') and full.get('repeat',{}).get('passes'):
  # The original selected checkpoint remains the candidate; no seed cherry-picking.
  for label,ad in [('parent',None),('selected',Path(winner['adapter']))]:
   argv=[mlx,work/'mlx_repair.py','answer','--model',model,'--prompts',old/'prepared_v3/final_source_prompts.jsonl','--output',work/f'{label}_final.jsonl']
   if ad:argv+=['--adapter',ad]
   run(label+'_final',argv)
   score(SimpleNamespace(rows=old/'prepared_v3/final_rows.jsonl',answers=work/f'{label}_final.jsonl',fpr=str(fpr),output=work/f'{label}_final_scores.json'))
  final_run=True
 decision={'repeat':repeat,'full':full,'final_holdout_run':final_run,'promoted':False,
  'next':'manual final documentation review and checkpoint decision' if final_run else 'retain round 4; confirmation gates not all met'}
 (work/'confirmation_decision.json').write_text(json.dumps(decision,indent=2)+'\n');status('complete','complete')
except Exception as e:
 status('failed','failed',error=repr(e));raise
