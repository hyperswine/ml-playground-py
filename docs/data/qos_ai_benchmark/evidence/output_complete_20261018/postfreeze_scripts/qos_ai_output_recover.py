"""Resume a failed GPU segment without changing the frozen experiment treatment."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
import torch
import qos_ai_output_workflow as job
from qos_ai_data import file_hash,read_rows
from qos_ai_curriculum import write_rows


def main(attempt):
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
  if (job.EXPERIMENT/'student.pt').exists():raise ValueError('Checkpoint already published')
  progress=torch.load(job.EXPERIMENT/'progress_latest.pt',map_location='cpu',weights_only=True)['progress']
  step=progress['step']
  records=read_rows(job.EXPERIMENT/'training.jsonl');prefix=[r for r in records if r['step']<=step]
  if [r['step'] for r in prefix]!=list(range(1,step+1)):raise ValueError('Saved-step log prefix differs')
  failed=job.EXPERIMENT/f'failed_training_attempt{attempt}.jsonl'
  if failed.exists():raise ValueError('Recovery attempt already recorded')
  shutil.copy2(job.EXPERIMENT/'training.jsonl',failed)
  shutil.copy2(job.REPORTS/'status.json',job.EXPERIMENT/f'failed_status_attempt{attempt}.json')
  write_rows(job.EXPERIMENT/'training.jsonl',prefix)
  metadata={'attempt':attempt,'saved_step':step,'discarded_unsaved_steps':[r['step'] for r in records if r['step']>step],
    'reason':'Metal command-buffer failure and device target-count guard; no checkpoint published',
    'treatment_change':False,'code_sha256':file_hash(Path(__file__))}
  (job.EXPERIMENT/f'recovery{attempt}.json').write_text(json.dumps(metadata,indent=2)+'\n')
  train=['qos_ai_finetune.py','--load',job.PARENT,'--curriculum',job.DATA,'--data',job.DATA/'balanced_train_ready.jsonl',
    '--selection',job.DATA/'training_selection.json','--splits',job.DATA/'splits.json',
    '--root','fprisc=source_snapshot/fprisc','--root','qos=source_snapshot/qos','--output',job.EXPERIMENT,
    '--epochs','1','--batch-size','2','--grad-accumulation','2','--learning-rate','0.00002','--max-minutes','10',
    '--seed','7','--pad-multiple','32','--fixed-padding','--mps-cache-interval','20','--defer-final-loss',
    '--segment-steps','64','--resume','--device','mps']
  job.status(f'GPU recovery {attempt} from optimizer step {step}')
  job.launch(train,f'training_retry{attempt}.log')
  result=json.loads((job.EXPERIMENT/'result.json').read_text())
  if result['optimizer_steps']!=192 or result['examples_seen']!=768 or result['epochs_completed']!=1 or result['budget_stop']:
    raise ValueError('Fixed epoch incomplete')
  starts=[]
  for line in Path('output_complete_workflow_20261018.log').read_text().splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('phase')=='training segment 1':starts.append(r['unix_time'])
  timing={'training_wall_seconds_including_segments_failure_and_recovery':time.time()-starts[0],
    'successful_training_cumulative_seconds':result['elapsed_seconds'],'gpu_recovery_attempts':attempt}
  (job.EXPERIMENT/'workflow_timing.json').write_text(json.dumps(timing,indent=2)+'\n')
  job.status('fresh-process final training loss')
  job.launch(['qos_ai_post_loss.py','--load',job.EXPERIMENT/'student.pt','--data',job.DATA,'--output',job.EXPERIMENT/'post_training_loss.json'],'post_loss.log')
  for condition,guard in (('raw',0),('guard',3)):
    job.status('original development benchmark '+condition)
    job.launch(['qos_ai_bench.py','run','--data','qos_ai_benchmark','--load',job.EXPERIMENT/'student.pt','--splits',job.DATA/'splits.json',
      '--split','dev','--seed','7','--temperature','0','--top-k','20','--no-repeat',guard,'--max-new-tokens','200',
      '--device','mps','--output',job.REPORTS/f'tuned_{condition}'],f'tuned_{condition}.log')
  job.status('unchanged coding diagnostic')
  job.launch(['qos_ai_coding_diag.py','run','--data',job.PREVIOUS,'--load',job.EXPERIMENT/'student.pt','--output',job.REPORTS/'tuned_diag'],'tuned_diag.log')
  job.launch(['qos_ai_coding_diag.py','grade','--data',job.PREVIOUS,'--reports',job.REPORTS/'tuned_diag'],'tuned_diag.log')
  if file_hash(job.PARENT)!=job.PARENT_SHA:raise ValueError('Parent changed')
  frozen=json.loads((job.REPORTS/'frozen_inputs.json').read_text())
  for name,sha in frozen.items():
    if file_hash(Path(name))!=sha:raise ValueError('Frozen input changed: '+name)
  job.launch(['qos_ai_output_compare.py'],'comparison.log');job.status('complete')


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
  try:main(a.attempt)
  except Exception as error:
    job.status('failed recovery: '+str(error));raise
