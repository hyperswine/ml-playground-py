import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ml_playground.layout import catalog, prepare_workspace, project_root
from ml_playground.cli import script_names


class LayoutTests(unittest.TestCase):
  def test_model_defaults_resolve_to_moved_artifacts_and_benchmarks(self):
    import qos_ai as ai
    root = project_root()
    self.assertEqual(ai.WORK_DIR.resolve(), root / 'artifacts/qos_ai')
    self.assertEqual(ai.BENCHMARK_FILE.resolve(), root / 'docs/data/qos_ai_benchmark/benchmark.jsonl')

  def test_source_catalog_has_unique_targets_and_useful_aliases(self):
    spec = catalog(); names = script_names(spec)
    self.assertEqual(len(set(spec['sources'].values())), len(spec['sources']))
    self.assertEqual(names['model'], 'qos_ai.py')
    self.assertEqual(names['stage4_eval'], 'qos_ai_stage4_eval.py')
    self.assertEqual(names['qwen08'], 'qwen08.py')
    self.assertTrue(all((project_root() / p).is_file() for p in spec['sources'].values()))

  def test_workspace_does_not_expose_credential_files(self):
    workspace = prepare_workspace()
    for name in ('.env', '.secrets', 'runpod/API_KEY'):
      self.assertFalse((workspace / name).exists(), name)

  def test_workspace_never_overwrites_existing_files(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp); (root / 'configs').mkdir(); (root / 'src').mkdir()
      (root / 'src/example.py').write_text('original source')
      spec = {key: {} for key in ('directories', 'sources', 'tests', 'scripts', 'files', 'documents')}
      spec['sources']['example.py'] = 'src/example.py'
      (root / 'configs/layout.json').write_text(json.dumps(spec))
      workspace = prepare_workspace(root)
      self.assertTrue((workspace / 'example.py').is_symlink())
      (workspace / 'example.py').unlink(); (workspace / 'example.py').write_text('user work')
      with self.assertRaises(RuntimeError): prepare_workspace(root)
      self.assertEqual((workspace / 'example.py').read_text(), 'user work')

  def test_frozen_development_benchmark_identity_is_unchanged(self):
    path = project_root() / 'artifacts/runpod/557m/bundle/benchmarks/benchmark.jsonl'
    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), '1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7')
