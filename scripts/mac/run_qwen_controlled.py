"""Run the two predeclared M1 pilots sequentially; select on development only."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/Users/jq/qos_ai_eval');work=root/'experiments/qwen2b_controlled_20261028'
old=root/'experiments/qwen2b_repair_20261027';os.chdir(work)
sys.path[:0]=[str(work),str(old),str(root)]
from qos_ai_qwen_repair import read,write,score,digest
from qos_ai_generalization_trace_audit import strict_trace
from types import SimpleNamespace
mlx=root/'mlx_venv/bin/python';py=root/'.venv/bin/python'
model=root/'experiments/mlx_20261027/mlx_2b_round4_q8';fpr=root/'experiments/stage4_20261021/preparation/toolchain/fpr'
env={**os.environ,'PYTHONPATH':':'.join([str(work),str(old),str(root)])}

def status(phase,state='running',**kw):
 (work/'status.json').write_text(json.dumps({'phase':phase,'state':state,'time':time.time(),**kw},indent=2)+'\n')

def run(label,argv):
 status(label)
 with open(label+'.log','w') as log:
  subprocess.run([str(x) for x in argv],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

try:
 status('waiting_for_verified_build')
 deadline=time.monotonic()+1200
 while not (work/'prepared/manifest.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('data verification did not finish')
  time.sleep(5)
 manifest=json.loads((work/'prepared/manifest.json').read_text())
 dev=[r for r in read(old/'prepared_v3/dev_rows.jsonl') if r['kind']!='docs']
 prompts=[r for r in read(old/'prepared_v3/dev_source_prompts.jsonl') if r['id'] in {x['id'] for x in dev}]
 write(work/'dev_rows.jsonl',dev);write(work/'dev_prompts.jsonl',prompts)
 general=read(root/'curricula/generalization_20261019/generalization_diagnostic.jsonl')
 guard=[]
 for stratum in ('inputs','combinations','wording'):
  # Fixed first eight trace-requesting items; no candidate outputs used for selection.
  guard.extend([r for r in general if r['stratum']==stratum and r['mode']!='output_only'][:8])
 write(work/'guard_rows.jsonl',guard)
 frozen=read(old/'frozen_prompts.jsonl')
 guardids={'gen/'+r['id'] for r in guard}
 write(work/'guard_prompts.jsonl',[r for r in frozen if r['id'] in guardids])
 assert len(read(work/'guard_prompts.jsonl'))==24
 parent={r['id']:r['answer'] for r in read(root/'experiments/mlx_20261027/answers_2b_round4_q8.jsonl')}
 baseguard=sum(strict_trace(r,parent['gen/'+r['id']])['strict_canonical_trace_match'] for r in guard)
 parentfresh=read(old/'parent_dev_v2.jsonl');write(work/'parent_dev.jsonl',[r for r in parentfresh if r['id'] in {x['id'] for x in dev}])
 score(SimpleNamespace(rows=work/'dev_rows.jsonl',answers=work/'parent_dev.jsonl',fpr=str(fpr),output=work/'parent_scores.json'))
 base=json.loads((work/'parent_scores.json').read_text())
 plan={'learning_rate':1e-5,'seed':20261028,'budget_targets':[2048,4096,8192],'conditions':['replay','mixed'],
  'selection':'complete coding >= parent+2; programs >= parent; no category loss >1; guard strict trace >= parent-1. Rank by complete coding, programs, then lower token exposure.',
  'parent_guard_strict_trace':baseguard,'guard_n':24,'parent_complete':base['code_complete'],
  'final_holdout_policy':'never use unless a repeated-seed winner passes full original regression checks',
  'prepared_manifest_sha256':digest(work/'prepared/manifest.json'),'dev_prompts_sha256':digest(work/'dev_prompts.jsonl'),
  'guard_prompts_sha256':digest(work/'guard_prompts.jsonl'),'created':time.time()}
 (work/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 results=[]
 for label in ('replay','mixed'):
  points=manifest['schedules'][label];last=points[-1]['micro_batches']
  run(label+'_train',[mlx,work/'mlx_repair.py','train','--model',model,'--train',work/f'prepared/{label}_schedule.jsonl',
   '--valid',work/'prepared/valid.jsonl','--output',work/f'adapter_{label}','--iters',last,
   '--learning-rate','1e-5','--seed','20261028','--ordered','--checkpoints',','.join(str(p['micro_batches']) for p in points)])
  for point in points:
   tag=f'{label}_{point["budget"]}';adapter=work/f'adapter_{label}/step_{point["micro_batches"]:04d}'
   receipt=json.loads((adapter/'receipt.json').read_text())
   assert receipt['trained_tokens']==point['actual_tokens'],(receipt,point)
   for suite in ('dev','guard'):
    run(tag+'_'+suite,[mlx,work/'mlx_repair.py','answer','--model',model,'--adapter',adapter,'--prompts',work/f'{suite}_prompts.jsonl','--output',work/f'{tag}_{suite}.jsonl'])
   score(SimpleNamespace(rows=work/'dev_rows.jsonl',answers=work/f'{tag}_dev.jsonl',fpr=str(fpr),output=work/f'{tag}_scores.json'))
   s=json.loads((work/f'{tag}_scores.json').read_text())
   answers={r['id']:r['answer'] for r in read(work/f'{tag}_guard.jsonl')}
   trace=sum(strict_trace(r,answers['gen/'+r['id']])['strict_canonical_trace_match'] for r in guard)
   passed=(s['code_complete']>=base['code_complete']+2 and s['code_programs_correct']>=base['code_programs_correct']
     and all(s['by_kind'][k]['correct']>=v['correct']-1 for k,v in base['by_kind'].items()) and trace>=baseguard-1)
   results.append({'tag':tag,'adapter':str(adapter),'tokens':point['actual_tokens'],'complete':s['code_complete'],
     'programs':s['code_programs_correct'],'outputs':s['code_claims_correct'],'by_kind':s['by_kind'],'strict_trace_guard':trace,'passes_screen':passed})
   (work/'comparison.json').write_text(json.dumps({'parent':{'complete':base['code_complete'],'programs':base['code_programs_correct'],'outputs':base['code_claims_correct'],'strict_trace_guard':baseguard},'runs':results},indent=2)+'\n')
 eligible=[r for r in results if r['passes_screen']]
 winner=max(eligible,key=lambda r:(r['complete'],r['programs'],-r['tokens'])) if eligible else None
 (work/'selection.json').write_text(json.dumps({'winner':winner,'promoted':False,'final_holdout_run':False,
  'next':'repeat winning recipe and run full regression before holdout' if winner else 'retain parent; no candidate passed the development screen'},indent=2)+'\n')
 status('complete','complete')
except Exception as e:
 status('failed','failed',error=repr(e));raise
