"""One fixed calculation/complete-repair follow-up from the stronger coding checkpoint."""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from qos_ai_data import file_hash

DATA=Path('curricula/output_complete_20261018')
PREVIOUS=Path('curricula/balanced_repair_20261018')
PARENT=Path('experiments/balanced_from_codegen_20261018/student.pt')
PARENT_SHA='5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767'
REPORTS=Path('reports/output_complete_20261018')
EXPERIMENT=Path('experiments/output_complete_20261018')


def status(phase):
  r={'phase':phase,'unix_time':time.time()};(REPORTS/'status.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)


def launch(args,log):
  with (REPORTS/log).open('a') as output:r=subprocess.run([sys.executable,*map(str,args)],stdout=output,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('Command failed; see '+str(REPORTS/log))


def main():
  import torch
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
  REPORTS.mkdir(parents=True,exist_ok=False)
  if file_hash(PARENT)!=PARENT_SHA:raise ValueError('Wrong starting checkpoint')
  for name in ('benchmark.jsonl','coding_diagnostic.jsonl','coding_verification.json'):
    if (DATA/name).read_bytes()!=(PREVIOUS/name).read_bytes():raise ValueError('Frozen benchmark/reference changed')
  parent_control=json.loads(Path('reports/balanced_repair_20261018/codegen_after_raw/summary.json').read_text())
  current={'code_sha256':file_hash(Path('qos_ai.py')),'evaluator_sha256':file_hash(Path('qos_ai_bench.py')),
    'split_policy_code_sha256':file_hash(Path('qos_ai_data.py')),'host':platform.node(),'machine':platform.machine(),
    'torch':str(torch.__version__),'checkpoint_sha256':PARENT_SHA,'benchmark_sha256':file_hash(Path('qos_ai_benchmark/benchmark.jsonl'))}
  if any(parent_control[k]!=v for k,v in current.items()):raise ValueError('Saved parent benchmark identity/environment differs')
  diag_control=json.loads(Path('reports/balanced_repair_20261018/codegen_after_diag/summary.json').read_text())
  if diag_control['generation_code_sha256']!=file_hash(Path('qos_ai_coding_diag.py')):raise ValueError('Diagnostic generator changed')
  proof=json.loads((DATA/'verification.json').read_text())
  if proof['passed']!=proof['total']:raise ValueError('Reference verification incomplete')
  frozen={str(p):file_hash(p) for p in list(DATA.glob('*'))+list(Path('.').glob('qos_ai*.py')) if p.is_file()}
  (REPORTS/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
  status('tests')
  launch(['-m','unittest','test_qos_ai_data','test_qos_ai_curriculum','test_qos_ai_compare','test_qos_ai_finetune',
    'test_qos_ai_inputs','test_qos_ai_balanced','test_qos_ai_resume','test_qos_ai_output'],'tests.log')
  train=['qos_ai_finetune.py','--load',PARENT,'--curriculum',DATA,'--data',DATA/'balanced_train_ready.jsonl',
    '--selection',DATA/'training_selection.json','--splits',DATA/'splits.json',
    '--root','fprisc=source_snapshot/fprisc','--root','qos=source_snapshot/qos','--output',EXPERIMENT,
    '--epochs','1','--batch-size','2','--grad-accumulation','2','--learning-rate','0.00002','--max-minutes','10',
    '--seed','7','--pad-multiple','32','--fixed-padding','--mps-cache-interval','20','--defer-final-loss',
    '--segment-steps','64','--resume','--device','mps']
  started=time.time()
  for segment in range(1,5):
    status(f'training segment {segment}');launch(train,'training.log')
    if (EXPERIMENT/'student.pt').exists():break
  else:raise ValueError('Unexpected number of training segments')
  result=json.loads((EXPERIMENT/'result.json').read_text())
  if result['optimizer_steps']!=192 or result['examples_seen']!=768 or result['epochs_completed']!=1 or result['budget_stop']:
    raise ValueError('Fixed training treatment incomplete')
  (EXPERIMENT/'workflow_timing.json').write_text(json.dumps({'training_wall_seconds_including_segments':time.time()-started})+'\n')
  status('fresh-process final training loss')
  launch(['qos_ai_post_loss.py','--load',EXPERIMENT/'student.pt','--data',DATA,'--output',EXPERIMENT/'post_training_loss.json'],'post_loss.log')
  for condition,guard in (('raw',0),('guard',3)):
    status('original development benchmark '+condition)
    launch(['qos_ai_bench.py','run','--data','qos_ai_benchmark','--load',EXPERIMENT/'student.pt','--splits',DATA/'splits.json',
      '--split','dev','--seed','7','--temperature','0','--top-k','20','--no-repeat',guard,'--max-new-tokens','200',
      '--device','mps','--output',REPORTS/f'tuned_{condition}'],f'tuned_{condition}.log')
  status('unchanged coding diagnostic')
  launch(['qos_ai_coding_diag.py','run','--data',PREVIOUS,'--load',EXPERIMENT/'student.pt','--output',REPORTS/'tuned_diag'],'tuned_diag.log')
  launch(['qos_ai_coding_diag.py','grade','--data',PREVIOUS,'--reports',REPORTS/'tuned_diag'],'tuned_diag.log')
  if file_hash(PARENT)!=PARENT_SHA:raise ValueError('Parent checkpoint changed')
  for name,sha in frozen.items():
    if file_hash(Path(name))!=sha:raise ValueError('Frozen input changed: '+name)
  launch(['qos_ai_output_compare.py'],'comparison.log')
  status('complete')


if __name__=='__main__':
  try:main()
  except Exception as error:
    if REPORTS.exists():status('failed: '+str(error))
    raise
