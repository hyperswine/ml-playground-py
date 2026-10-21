"""Run categorized ML scripts through their preserved experiment interfaces."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

from .layout import catalog, environment, prepare_workspace, project_root


HELP = '''Usage: qos SCRIPT [arguments...]
       qos list
       qos paths
       qos test [--pattern 'test_*.py']
       qos shell SCRIPT [arguments...]

Examples:
  qos model --help
  qos stage4_eval --help
  qos stage4_train --help
  qos qwen08 --help
  qos neural-lite-transformer --help

Run from this checkout with `uv run --locked qos ...`.
Scripts retain their original argument interfaces. No training or cloud job is
started by `list`, `paths`, `test`, or this help message.
'''


def script_names(spec):
  names = {}
  for filename in spec['sources']:
    stem = Path(filename).stem
    names[stem] = filename
    if stem.startswith('qos_ai_'): names[stem.removeprefix('qos_ai_')] = filename
    if stem == 'qos_ai': names['model'] = filename
  return names


def main(argv=None):
  args = list(sys.argv[1:] if argv is None else argv)
  if not args or args[0] in ('-h', '--help', 'help'):
    print(HELP); return 0
  root = project_root(); spec = catalog(root)
  if args[0] == 'list':
    for filename, relative in sorted(spec['sources'].items(), key=lambda item: (Path(item[1]).parent.name, item[0])):
      print(f'{Path(relative).parent.name:12} {Path(filename).stem.removeprefix("qos_ai_"):32} {relative}')
    return 0
  if args[0] == 'paths':
    print(f'Project: {root}\nSource: {root / "src/ml_playground"}\nDocumentation: {root / "docs"}\nArtifacts: {root / "artifacts"}\nCompatibility workspace: {root / ".runtime/workspace"}')
    return 0
  names = script_names(spec)
  if args[0] not in ('test', 'shell') and args[0] not in names:
    print(f'Unknown script: {args[0]}. Run `qos list`.', file=sys.stderr); return 2
  workspace = prepare_workspace(root)
  os.environ.update(environment(root, workspace)); sys.path.insert(0, str(workspace))
  os.chdir(workspace)
  if args[0] == 'test':
    parser = argparse.ArgumentParser(prog='qos test')
    parser.add_argument('--pattern', default='test_*.py')
    settings = parser.parse_args(args[1:])
    suite = unittest.defaultTestLoader.discover(str(workspace), pattern=settings.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1
  if args[0] == 'shell':
    if len(args) < 2:
      print('Choose a script, for example: qos shell runpod/pull.sh ...', file=sys.stderr); return 2
    name = args[1]
    if name not in spec['scripts']:
      print('Unknown shell script. Use its runpod/... legacy path.', file=sys.stderr); return 2
    return subprocess.run(['bash', str(workspace / name), *args[2:]], cwd=workspace).returncode
  filename = names[args[0]]
  sys.argv = [str(workspace / filename), *args[1:]]
  runpy.run_path(str(workspace / filename), run_name='__main__')
  return 0
