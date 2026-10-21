"""Run one bounded data/profile/pretrain/SFT/evaluation job on the mounted volume."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from qos_ai_data import file_hash
from qos_ai_scale import write_json


def main():
  p = argparse.ArgumentParser(); p.add_argument('--created-unix', type=float, required=True)
  args = p.parse_args()
  root = Path('experiment'); root.mkdir(exist_ok=True)
  budget = {'created_unix': args.created_unix, 'hard_gpu_deadline_unix': args.created_unix + 8 * 3600,
            'usd_cap': 10.45, 'gpu_usd_hour': .89, 'network_volume_gb': 40,
            'policy': 'External MCP controller stops compute at 8h; pretraining ends by 6.5h; no credentials injected into pod'}
  write_json(Path('budget.json'), budget)
  os.environ.update(PYTHONUNBUFFERED='1', HF_HOME='/workspace/hf', TOKENIZERS_PARALLELISM='true')
  def phase(name, **extra):
    write_json(root / 'status.json', {'phase': name, 'unix_time': time.time(), **extra})
  def run(command, log, timeout):
    with (root / log).open('a') as out:
      out.write('COMMAND ' + json.dumps(command) + '\n'); out.flush()
      r = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
      raise RuntimeError(log + ' exited ' + str(r.returncode))
  try:
    phase('verify_bundle')
    manifest = json.loads(Path('bundle_manifest.json').read_text())
    for path, sha in manifest.items():
      if file_hash(Path(path)) != sha:
        raise ValueError('Bundle file changed: ' + path)
    phase('prepare_data')
    if not Path('pretraining_data/dataset.json').exists():
      run([sys.executable, '-u', 'qos_ai_runpod_data.py', '--vocabulary', 'vocabulary.pt', '--tokenizer', 'tokenizer',
           '--benchmark', 'benchmarks/benchmark.jsonl', '--benchmark', 'benchmarks/coding_diagnostic.jsonl',
           '--benchmark', 'benchmarks/generalization_diagnostic.jsonl', '--output', 'pretraining_data'], 'prepare.log', 1800)
    phase('train')
    left = int(budget['hard_gpu_deadline_unix'] - time.time() - 1200)
    if left <= 0:
      raise TimeoutError('No remaining run budget')
    run([sys.executable, '-u', 'qos_ai_runpod_train.py', '--config', 'qos_ai_557m.json', '--data', 'pretraining_data',
         '--sft', 'verified_sft.pt', '--tokenizer', 'tokenizer', '--benchmarks', 'benchmarks',
         '--output', 'experiment', '--budget', 'budget.json'], 'training.log', left)
  except Exception as e:
    phase('failed', error=type(e).__name__ + ': ' + str(e))
    raise


if __name__ == '__main__':
  main()
