"""Complete the fixed seed-17 treatment after failed 48-update MPS processes."""
import json
import os
import shutil
from pathlib import Path
import torch
import qos_ai_generalization_workflow as job
from qos_ai_data import file_hash,read_rows
from qos_ai_curriculum import write_rows

EXPERIMENT=Path('experiments/generalization_20261019_seed17')


def canonicalize_log(tag):
  saved=torch.load(EXPERIMENT/'progress_latest.pt',map_location='cpu',weights_only=True,mmap=True)['progress']['step']
  records=read_rows(EXPERIMENT/'training.jsonl');prefix=[r for r in records if r['step']<=saved]
  if [r['step'] for r in prefix]!=list(range(1,saved+1)):raise ValueError('Saved update log prefix is not canonical')
  target=EXPERIMENT/f'failed_training_{tag}.jsonl'
  if target.exists():raise ValueError('Recovery audit already exists '+tag)
  shutil.copy2(EXPERIMENT/'training.jsonl',target);write_rows(EXPERIMENT/'training.jsonl',prefix)
  return {'saved_step':saved,'discarded_unsaved_records':len(records)-len(prefix)}


def main():
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
  if (EXPERIMENT/'student.pt').exists():raise ValueError('Seed-17 checkpoint already published')
  metadata=canonicalize_log('before_short_process_recovery')
  if metadata['saved_step']!=384:raise ValueError('Unexpected initial recovery state')
  metadata.update(reason='Three 48-update MPS attempts failed near 35-40 updates; target-count safety checks rejected them.',
    original_process_chunk=48,recovery_process_chunk=16,data_order_lr_optimizer_loss_and_total_updates_unchanged=True,
    code_sha256=file_hash(Path(__file__)),step_code_sha256=file_hash(Path('qos_ai_generalization_recovery_step.py')))
  (EXPERIMENT/'operational_recovery.json').write_text(json.dumps(metadata,indent=2)+'\n')
  job.status('seed 17 operational recovery: 16-update processes from saved step 384')
  job.launch(['-m','unittest','test_qos_ai_generalization_recovery'],'recovery_tests.log')
  for chunk in range(1,21):
    job.status(f'seed 17 recovery process {chunk}/20')
    for attempt in range(1,4):
      try:
        job.launch(['qos_ai_generalization_recovery_step.py','--data',job.DATA,'--experiment',EXPERIMENT,'--chunk-steps','16'],'seed17_recovery.log');break
      except RuntimeError:
        canonicalize_log(f'chunk{chunk}_attempt{attempt}')
        if attempt==3:raise
    if (EXPERIMENT/'student.pt').exists():break
  else:raise ValueError('Recovery did not complete the fixed epoch')
  result=json.loads((EXPERIMENT/'result.json').read_text())
  if result['optimizer_steps']!=696 or result['examples_seen']!=2784 or result['supervised_tokens']!=97693 or result['epochs_completed']!=1 or result['budget_stop']:raise ValueError('Canonical treatment incomplete')
  if [r['step'] for r in read_rows(EXPERIMENT/'training.jsonl')]!=list(range(1,697)):raise ValueError('Final accepted update log differs')
  job.evaluate('seed17_final',EXPERIMENT/'student.pt',True)
  if file_hash(job.PARENT)!=job.PARENT_SHA:raise ValueError('Parent changed')
  for path,sha in json.loads((job.REPORTS/'frozen_inputs.json').read_text()).items():
    if file_hash(Path(path))!=sha:raise ValueError('Frozen input changed '+path)
  job.status('complete')


if __name__=='__main__':
  try:main()
  except Exception as e:job.status('failed recovery: '+str(e));raise
