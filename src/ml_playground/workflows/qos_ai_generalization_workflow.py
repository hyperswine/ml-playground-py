"""Two-seed, fixed-treatment M1 experiment; never replaces a parent/default model."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from qos_ai_data import file_hash,read_rows

DATA=Path('curricula/generalization_20261019')
PREVIOUS=Path('curricula/output_complete_20261018')
PARENT=Path('experiments/balanced_from_codegen_20261018/student.pt')
PARENT_SHA='5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767'
REPORTS=Path('reports/generalization_20261019')


def status(phase):
  r={'phase':phase,'unix_time':time.time()};(REPORTS/'status.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)


def launch(args,log,retries=0):
  for attempt in range(retries+1):
    with (REPORTS/log).open('a') as out:
      out.write('\nCOMMAND '+json.dumps(list(map(str,args)))+'\n');out.flush()
      r=subprocess.run([sys.executable,*map(str,args)],stdout=out,stderr=subprocess.STDOUT)
    if not r.returncode:return
    if attempt<retries:status(f'retrying fresh process after failure: {log}, attempt {attempt+2}')
  raise RuntimeError('Command failed; see '+str(REPORTS/log))


def export_progress(experiment):
  import torch
  progress=torch.load(experiment/'progress_latest.pt',map_location='cpu',weights_only=True,mmap=True)
  if progress['settings']['parent_sha256']!=PARENT_SHA:raise ValueError('Wrong intermediate parent')
  step=progress['progress']['step'];path=experiment/f'step_{step:04d}.pt'
  if path.exists():raise ValueError('Intermediate already exists')
  parent=torch.load(PARENT,map_location='cpu',weights_only=True,mmap=True)
  payload={**parent,'model_state':progress['model_state'],'step':parent['step']+step,
    'split_hash':progress['settings']['split_hash'],'parent_sha256':PARENT_SHA,
    'verified_sft':{**progress['settings'],**progress['progress'],'intermediate':True}}
  tmp=path.with_suffix('.partial.pt');torch.save(payload,tmp)
  torch.load(tmp,map_location='cpu',weights_only=True,mmap=True);tmp.replace(path)
  return path


def evaluate(label,checkpoint,final=False):
  status('evaluate '+label)
  output=REPORTS/label;output.mkdir()
  launch(['qos_ai_generalization_eval.py','loss','--data',DATA,'--load',checkpoint,'--output',output/'loss.json'],label+'_loss.log')
  launch(['qos_ai_generalization_eval.py','run','--data',DATA,'--load',checkpoint,'--output',output/'generalization'],label+'_generalization.log')
  launch(['qos_ai_generalization_eval.py','grade','--data',DATA,'--output',output/'generalization'],label+'_generalization.log')
  for condition,guard in [('guard',3)]+([('raw',0)] if final else []):
    launch(['qos_ai_bench.py','run','--data','qos_ai_benchmark','--load',checkpoint,'--splits',DATA/'splits.json',
      '--split','dev','--seed','7','--temperature','0','--top-k','20','--no-repeat',guard,'--max-new-tokens','200','--device','mps','--output',output/f'dev_{condition}'],label+'_dev_'+condition+'.log')
  launch(['qos_ai_coding_diag.py','run','--data',DATA,'--load',checkpoint,'--output',output/'coding'],label+'_coding.log')
  launch(['qos_ai_coding_diag.py','grade','--data',DATA,'--reports',output/'coding'],label+'_coding.log')
  item={'label':label,'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint),
        'generalization':json.loads((output/'generalization/grading.json').read_text()),
        'loss':json.loads((output/'loss.json').read_text()),'coding':json.loads((output/'coding/grading.json').read_text()),
        'development':json.loads((output/'dev_guard/summary.json').read_text())}
  # Keep all raw generations/details in their own files; this summary is compact.
  for key in ('generalization','coding'):item[key].pop('results',None)
  with (REPORTS/'learning_curve.jsonl').open('a') as out:out.write(json.dumps(item)+'\n')


def main():
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
  REPORTS.mkdir(parents=True,exist_ok=False)
  if file_hash(PARENT)!=PARENT_SHA:raise ValueError('Wrong parent')
  if json.loads((DATA/'build.json').read_text())['new_cases']!=2400:raise ValueError('Unexpected data size')
  for name in ('benchmark.jsonl','coding_diagnostic.jsonl','coding_verification.json'):
    if (DATA/name).read_bytes()!=(PREVIOUS/name).read_bytes():raise ValueError('Old benchmark changed')
  status('tests and authored grader controls')
  launch(['-m','unittest','test_qos_ai_data','test_qos_ai_curriculum','test_qos_ai_compare','test_qos_ai_finetune',
    'test_qos_ai_inputs','test_qos_ai_balanced','test_qos_ai_resume','test_qos_ai_output','test_qos_ai_output_claim_audit','test_qos_ai_generalization'], 'tests.log')
  launch(['qos_ai_generalization_controls.py','--data',DATA], 'controls.log')
  frozen={str(p):file_hash(p) for p in list(DATA.glob('*'))+list(Path('.').glob('qos_ai*.py')) if p.is_file()}
  (REPORTS/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
  treatment={'seeds':[7,17],'epochs':1,'learning_rate':1e-5,'batch_size':2,'gradient_accumulation':2,'segment_steps':48,
    'intermediate_step':336,'final_step':696,'examples':2784,'max_training_minutes_per_seed':60,
    'checkpoint_selection':'report both seeds and fixed intermediate/final checkpoints; no adaptive choice from new diagnostic',
    'scope':'coverage and trace curriculum changed together; this is not a trace-only causal ablation; all checkpoints retain legacy_or_unknown ancestry'}
  (REPORTS/'protocol.json').write_text(json.dumps(treatment,indent=2)+'\n')
  evaluate('parent',PARENT,True)
  for seed in (7,17):
    experiment=Path(f'experiments/generalization_20261019_seed{seed}')
    train=['qos_ai_finetune.py','--load',PARENT,'--curriculum',DATA,'--data',DATA/'balanced_train_ready.jsonl',
      '--selection',DATA/'training_selection.json','--splits',DATA/'splits.json','--root','fprisc=source_snapshot/fprisc',
      '--root','qos=source_snapshot/qos','--output',experiment,'--epochs','1','--batch-size','2','--grad-accumulation','2',
      '--learning-rate','0.00001','--max-minutes','60','--seed',seed,'--pad-multiple','32','--fixed-padding',
      '--mps-cache-interval','12','--defer-final-loss','--segment-steps','48','--resume','--device','mps']
    for segment in range(1,16):
      status(f'train seed {seed}, segment {segment}/15');launch(train,f'seed{seed}_training.log',retries=2)
      if (experiment/'student.pt').exists():break
      import torch
      progress=torch.load(experiment/'progress_latest.pt',map_location='cpu',weights_only=True,mmap=True)
      step=progress['progress']['step'];del progress
      if step==336:
        checkpoint=export_progress(experiment);evaluate(f'seed{seed}_step0336',checkpoint)
    else:raise ValueError('Fixed training failed to finish')
    result=json.loads((experiment/'result.json').read_text())
    if result['optimizer_steps']!=696 or result['examples_seen']!=2784 or result['epochs_completed']!=1 or result['budget_stop']:raise ValueError('Incomplete fixed treatment')
    evaluate(f'seed{seed}_final',experiment/'student.pt',True)
  if file_hash(PARENT)!=PARENT_SHA:raise ValueError('Parent changed')
  for path,sha in frozen.items():
    if file_hash(Path(path))!=sha:raise ValueError('Frozen input changed '+path)
  status('complete')


if __name__=='__main__':
  try:main()
  except Exception as e:
    if REPORTS.exists():status('failed: '+str(e))
    raise
