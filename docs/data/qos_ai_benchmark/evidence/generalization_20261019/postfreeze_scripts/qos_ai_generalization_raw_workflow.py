"""Wait for the fixed primary experiment, then add the decoding-feasibility control."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from qos_ai_data import file_hash

ROOT=Path('reports/generalization_20261019')
DATA=Path('curricula/generalization_20261019')


def status(phase):
  record={'phase':phase,'unix_time':time.time()};(ROOT/'raw_status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


def main():
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
  protocol={'reason':'94 authored diagnostic answers contain token trigrams prohibited by no-repeat three; this is a decoding feasibility control, not an adaptive data change.',
    'primary_metrics_unchanged':True,'no_repeat':0,'temperature':0,'top_k':20,'max_new_tokens':128,'seed':7,
    'labels':['parent','seed7_step0336','seed7_final','seed17_step0336','seed17_final'],
    'frozen_inputs':{str(p):file_hash(p) for p in DATA.glob('*') if p.is_file()},
    'code_sha256':file_hash(Path('qos_ai_generalization_raw.py')),'workflow_sha256':file_hash(Path(__file__))}
  (ROOT/'raw_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
  status('waiting for primary experiment; no concurrent model execution')
  while True:
    phase=json.loads((ROOT/'status.json').read_text())['phase']
    if phase=='complete':break
    if phase.startswith('failed:'):raise ValueError('Primary experiment failed: '+phase)
    time.sleep(10)
  curves={r['label']:r for r in map(json.loads,(ROOT/'learning_curve.jsonl').read_text().splitlines())}
  for label in protocol['labels']:
    status('unguarded diagnostic '+label);checkpoint=Path(curves[label]['checkpoint'])
    if file_hash(checkpoint)!=curves[label]['checkpoint_sha256']:raise ValueError('Checkpoint changed '+label)
    for command in ('run','grade'):
      args=[sys.executable,'qos_ai_generalization_raw.py',command,'--data',str(DATA),'--output',str(ROOT/label/'generalization_raw')]
      if command=='run':args+=['--load',str(checkpoint)]
      with (ROOT/(label+'_generalization_raw.log')).open('a') as log:proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
      if proc.returncode:raise ValueError('Supplementary diagnostic failed '+label+' '+command)
  for path,sha in protocol['frozen_inputs'].items():
    if file_hash(Path(path))!=sha:raise ValueError('Diagnostic data changed '+path)
  status('complete')


if __name__=='__main__':
  try:main()
  except Exception as e:status('failed: '+str(e));raise
