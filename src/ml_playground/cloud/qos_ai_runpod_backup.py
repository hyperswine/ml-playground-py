"""Foreground recovery of the approved 557M artifacts from an owned CPU pod.

No infrastructure or authentication settings are changed. Keep the calling
turn active; stop the CPU through managed MCP immediately when this returns.
"""
import argparse
import ipaddress
import json
import subprocess
import time
from pathlib import Path

from qos_ai_data import file_hash

ROOT = Path(__file__).absolute().parent
EVIDENCE = ROOT / 'qos_ai_benchmark/evidence/runpod_557m_20261019'
MODELS = ROOT / 'qos_ai/runpod_557m_20261019'
REMOTE = '/workspace/qos_ai_557m'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--host', required=True); p.add_argument('--port', type=int, required=True)
  args = p.parse_args(); ipaddress.IPv4Address(args.host)
  if not 1 <= args.port <= 65535:
    raise ValueError('Invalid SSH port')
  deadline = time.monotonic() + 7200
  local = EVIDENCE / 'remote'; local.mkdir(parents=True, exist_ok=True)
  MODELS.mkdir(parents=True, exist_ok=True)
  options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
             '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
             '-i', str(Path.home() / '.ssh/id_ed25519'), '-i', str(Path.home() / '.ssh/id_ed25519_hyperswine')]
  scp = ['scp', *options, '-P', str(args.port)]
  remote = f'root@{args.host}:' + REMOTE

  def phase(name, **fields):
    record = {'phase': name, 'unix_time': time.time(), **fields}
    (EVIDENCE / 'backup_status.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)

  def run(command):
    subprocess.run(command, cwd=ROOT, check=True, timeout=max(1, deadline - time.monotonic()))

  def copy(paths, destination, recursive=False):
    run([*scp, *(['-r'] if recursive else []), *[remote + '/' + s for s in paths], str(destination)])

  try:
    phase('metadata')
    copy(['experiment/*.json', 'experiment/*.jsonl'], local)
    terminal = json.loads((local / 'status.json').read_text())
    pre = json.loads((local / 'pretraining_result.json').read_text())
    sft = json.loads((local / 'sft_result.json').read_text())
    if terminal['phase'] != 'complete' or pre['processed_tokens'] != 459243520 or pre['budget_stop']:
      raise ValueError('The expected complete training run is required')
    if terminal['pretrained_sha256'] != pre['checkpoint_sha256'] or terminal['student_sha256'] != sft['checkpoint_sha256']:
      raise ValueError('Terminal checkpoint identities differ')
    copy(['experiment/evaluation'], local, recursive=True)
    for folder in ['corpus', 'setup']:
      (local / folder).mkdir(exist_ok=True)
    copy(['pretraining_data/dataset.json', 'pretraining_data/documents.jsonl.gz'], local / 'corpus')
    copy(['budget.json', 'bundle_manifest.json', 'io_fix.json', 'environment.txt', 'tests.log', 'setup.log', 'job.log'], local / 'setup')
    copy(['experiment/training.log', 'experiment/prepare.log'], local / 'setup')
    expected = {'student.pt': sft['checkpoint_sha256'], 'pretrained.pt': pre['checkpoint_sha256'], 'progress_pretrain.pt': pre['progress_sha256']}
    for name, sha in expected.items():
      path = MODELS / name
      if path.exists() and file_hash(path) == sha:
        phase('checkpoint_already_verified', file=name); continue
      phase('download', file=name)
      temporary = path.with_name(path.name + '.partial')
      copy(['experiment/' + name], temporary)
      phase('verify', file=name, bytes=temporary.stat().st_size)
      if file_hash(temporary) != sha:
        raise ValueError('Copied checkpoint checksum differs: ' + name)
      temporary.replace(path)
      phase('checkpoint_verified', file=name, sha256=sha, bytes=path.stat().st_size)
    (MODELS / 'checksums.json').write_text(json.dumps(expected, indent=2) + '\n')
    phase('complete', backup_verified=True, checksums=expected)
  except Exception as error:
    phase('failed', error=type(error).__name__ + ': ' + str(error))
    raise


if __name__ == '__main__':
  main()
