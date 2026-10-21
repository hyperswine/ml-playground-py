"""Build a local compatibility workspace without copying or rewriting experiment data."""
import json
import os
from pathlib import Path


def project_root():
  root = Path(__file__).resolve().parents[2]
  if not (root / 'configs/layout.json').is_file():
    raise RuntimeError('Use this package from the ml-playground-py checkout.')
  return root


def catalog(root=None):
  root = Path(root) if root is not None else project_root()
  return json.loads((root / 'configs/layout.json').read_text())


def link_file(link, target):
  if link.is_symlink():
    if link.resolve() != target.resolve():
      raise RuntimeError(f'Existing workspace link differs: {link}')
    return
  if link.exists():
    raise RuntimeError(f'Refusing to replace an existing workspace file: {link}')
  link.parent.mkdir(parents=True, exist_ok=True)
  link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())


def prepare_workspace(root=None):
  root = Path(root) if root is not None else project_root()
  spec = catalog(root)
  workspace = root / '.runtime/workspace'
  workspace.mkdir(parents=True, exist_ok=True)
  for alias, relative in spec['directories'].items():
    target = root / relative
    target.mkdir(parents=True, exist_ok=True)
    link_file(workspace / alias, target)
  for section in ('sources', 'tests', 'scripts', 'files', 'documents'):
    for alias, relative in spec[section].items():
      target = root / relative
      if not target.is_file(): raise FileNotFoundError(f'Missing {section} input: {target}')
      link_file(workspace / alias, target)
  if (root / '.venv').is_dir(): link_file(workspace / '.venv', root / '.venv')
  return workspace


def environment(root, workspace):
  env = dict(os.environ)
  paths = [str(workspace), str(root / 'src')]
  if env.get('PYTHONPATH'): paths.append(env['PYTHONPATH'])
  env['PYTHONPATH'] = os.pathsep.join(paths)
  env['ML_PLAYGROUND_PROJECT_ROOT'] = str(root)
  env['ML_PLAYGROUND_WORKSPACE'] = str(workspace)
  return env
