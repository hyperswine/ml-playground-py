"""Read-only backup checks and conservative disk headroom for atomic checkpoint saves."""
import argparse
import json
import shutil
from pathlib import Path

from qos_ai_stage4_eval import sha

GIB = 1024 ** 3
REQUIRED_ROLES = {'domain_parent', 'stage3_specialist', 'pretrained', 'optimizer', 'document_ledger'}


def require_space(directory, new_bytes, reserve_bytes=2 * GIB):
  if new_bytes < 0 or reserve_bytes < 0: raise ValueError('negative storage budget')
  free = shutil.disk_usage(directory).free
  # Free bytes already exclude the old checkpoint. The entire replacement must fit alongside it.
  needed = new_bytes + reserve_bytes
  if free < needed:
    raise OSError(f'insufficient checkpoint headroom: {free} free bytes, {needed} required')
  return {'free_bytes': free, 'new_write_bytes': new_bytes, 'reserve_bytes': reserve_bytes}


def verify_backups(manifest):
  artifacts = manifest['artifacts']; roles = [a['role'] for a in artifacts]
  if len(roles) != len(set(roles)) or set(roles) != REQUIRED_ROLES:
    raise ValueError('backup inventory must contain each required role exactly once')
  results = []
  for a in artifacts:
    path = Path(a['backup_path']) if a.get('backup_path') else None
    expected = a.get('sha256')
    exists = path is not None and path.is_file()
    actual = sha(path) if exists else None
    ok = bool(expected and actual == expected)
    results.append({'role': a['role'], 'verified': ok, 'actual_sha256': actual,
                    'reason': 'verified' if ok else 'missing backup, expected digest, or matching bytes'})
  return {'complete': all(r['verified'] for r in results), 'artifacts': results,
          'scope': 'byte identity only; load/checkpoint-structure and off-volume independence require separate checks'}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--inventory', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--checkpoint-directory', type=Path, required=True)
  p.add_argument('--new-bytes', type=int, default=9 * GIB)
  args = p.parse_args()
  if args.output.exists(): p.error('choose a new output path')
  result = verify_backups(json.loads(args.inventory.read_text()))
  result['storage'] = require_space(args.checkpoint_directory, args.new_bytes)
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))
  if not result['complete']: raise SystemExit(2)


if __name__ == '__main__': main()
