"""Run the preregistered two-parent balanced experiment on the M1, sequentially."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from qos_ai_data import file_hash

DATA=Path('curricula/balanced_repair_20261018')
EVIDENCE=Path('reports/balanced_repair_20261018')
PARENTS={'broad':Path('experiments/verified_expanded_sft2_retry/student.pt'),
         'codegen':Path('experiments/input_fidelity_sft1/student.pt')}
EXPECTED={'broad':'0707da88d15ea674ebbf1395ad948e9dad5438a9f308e7611b07fde53e80d83d',
          'codegen':'70e1fa8b26391b16156170ce121706c2abd4e187f2f12b7037630c10d43f42db'}


def launch(argv,log):
  with log.open('a') as output:
    completed=subprocess.run([sys.executable,*map(str,argv)],stdout=output,stderr=subprocess.STDOUT)
  if completed.returncode:raise RuntimeError(f'Command failed ({completed.returncode}); see {log}')


def status(phase):
  record={'phase':phase,'unix_time':time.time()}
  (EVIDENCE/'status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


def main():
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
  EVIDENCE.mkdir(parents=True,exist_ok=False)
  for name,path in PARENTS.items():
    if file_hash(path)!=EXPECTED[name]:raise ValueError('Starting checkpoint identity changed')
  if file_hash(Path('qos_ai_benchmark/benchmark.jsonl'))!='1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7':
    raise ValueError('Official benchmark changed')
  frozen={str(path):file_hash(path) for path in list(DATA.glob('*')) if path.is_file()}
  frozen.update({str(path):file_hash(path) for path in Path('.').glob('qos_ai*.py')})
  (EVIDENCE/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
  status('tests and reference grader')
  launch(['-m','unittest','test_qos_ai_data','test_qos_ai_curriculum','test_qos_ai_compare','test_qos_ai_finetune',
    'test_qos_ai_inputs','test_qos_ai_balanced','test_qos_ai_resume'],EVIDENCE/'tests.log')
  # Every authored response must satisfy the actual frozen grader before model answers exist.
  from qos_ai_coding_diag import diagnostic,grade_answer
  rows=diagnostic(DATA,Path('source_snapshot'),Path('source_snapshot/fprisc/fpr'));cache={}
  reference=[grade_answer(r,r['answer'],Path('source_snapshot/fprisc/fpr'),Path('source_snapshot'),cache) for r in rows]
  (EVIDENCE/'reference_grader.json').write_text(json.dumps(reference,indent=2)+'\n')
  if not all(r['complete_pass'] for r in reference):raise ValueError('Authored reference fails diagnostic grader')
  for name,parent in PARENTS.items():
    status(f'{name} baseline coding diagnostic')
    launch(['qos_ai_coding_diag.py','run','--data',DATA,'--load',parent,'--output',EVIDENCE/f'{name}_before_diag'],EVIDENCE/f'{name}_before_diag.log')
    launch(['qos_ai_coding_diag.py','grade','--data',DATA,'--reports',EVIDENCE/f'{name}_before_diag'],EVIDENCE/f'{name}_before_diag.log')
  for name,parent in PARENTS.items():
    experiment=Path(f'experiments/balanced_from_{name}_20261018');started=time.time()
    train=['qos_ai_finetune.py','--load',parent,'--curriculum',DATA,'--data',DATA/'balanced_train_ready.jsonl',
      '--selection',DATA/'training_selection.json','--splits',DATA/'splits.json',
      '--root','fprisc=source_snapshot/fprisc','--root','qos=source_snapshot/qos','--output',experiment,
      '--epochs','1','--batch-size','2','--grad-accumulation','2','--learning-rate','0.00002','--max-minutes','10',
      '--seed','7','--pad-multiple','32','--fixed-padding','--mps-cache-interval','20','--defer-final-loss',
      '--segment-steps','64','--resume','--device','mps']
    for segment in range(1,5):
      status(f'{name} training segment {segment}')
      launch(train,EVIDENCE/f'{name}_training.log')
      if (experiment/'student.pt').exists():break
    else:raise ValueError('Too many optimizer segments')
    result=json.loads((experiment/'result.json').read_text())
    if result['optimizer_steps']!=168 or result['examples_seen']!=672 or result['budget_stop']:raise ValueError('Incomplete matched training')
    (experiment/'workflow_timing.json').write_text(json.dumps({'training_wall_seconds_including_segments':time.time()-started})+'\n')
    status(f'{name} fresh-process final training loss')
    launch(['qos_ai_post_loss.py','--load',experiment/'student.pt','--data',DATA,'--output',experiment/'post_training_loss.json'],EVIDENCE/f'{name}_post_loss.log')
    for condition,guard in (('raw',0),('guard',3)):
      status(f'{name} official development benchmark {condition}')
      launch(['qos_ai_bench.py','run','--data','qos_ai_benchmark','--load',experiment/'student.pt','--splits',DATA/'splits.json',
        '--split','dev','--seed','7','--temperature','0','--top-k','20','--no-repeat',guard,'--max-new-tokens','200',
        '--device','mps','--output',EVIDENCE/f'{name}_after_{condition}'],EVIDENCE/f'{name}_after_{condition}.log')
    status(f'{name} tuned coding diagnostic')
    launch(['qos_ai_coding_diag.py','run','--data',DATA,'--load',experiment/'student.pt','--output',EVIDENCE/f'{name}_after_diag'],EVIDENCE/f'{name}_after_diag.log')
    launch(['qos_ai_coding_diag.py','grade','--data',DATA,'--reports',EVIDENCE/f'{name}_after_diag'],EVIDENCE/f'{name}_after_diag.log')
  for name,path in PARENTS.items():
    if file_hash(path)!=EXPECTED[name]:raise ValueError('Parent changed during experiment')
  for path,expected in frozen.items():
    if file_hash(Path(path))!=expected:raise ValueError('Frozen experiment input changed: '+path)
  status('complete')


if __name__=='__main__':
  try:main()
  except Exception as error:
    if EVIDENCE.exists():status('failed: '+str(error))
    raise
