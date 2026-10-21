"""Check the source catalog and local narrative links; optionally hash migrated artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote


def check(root, full=False):
  catalog = json.loads((root / 'configs/layout.json').read_text())
  missing = []
  for group in ('sources', 'tests', 'scripts', 'files', 'documents'):
    for path in catalog[group].values():
      if not (root / path).is_file(): missing.append(path)
  pages = [root / 'README.md', root / 'docs/README.md', root / 'docs/data/README.md', *root.glob('docs/20*.md')]
  broken, checked = [], 0
  for page in pages:
    for link in re.findall(r'\]\(([^\n]+?)\)', page.read_text()):
      value = link.strip('<>').split('#')[0]
      if not value or re.match(r'\w+:', value): continue
      checked += 1
      if not (page.parent / unquote(value)).exists():
        broken.append({'page': str(page.relative_to(root)), 'target': value})
  artifacts, mismatches = 0, []
  if full:
    manifest = json.loads((root / 'docs/data/migration/2026-10-21/files.json').read_text())
    for record in manifest['records']:
      if not record['new'].startswith(('artifacts/', 'docs/data/')): continue
      path = root / record['new']
      if not path.is_file(): mismatches.append(record['new']); continue
      with path.open('rb') as f: value = hashlib.file_digest(f, 'sha256').hexdigest()
      artifacts += 1
      if value != record['sha256']: mismatches.append(record['new'])
  return {'catalog_missing': missing, 'local_links_checked': checked, 'broken_links': broken,
          'artifact_hashes_checked': artifacts, 'artifact_mismatches': mismatches,
          'ok': not missing and not broken and not mismatches}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--full', action='store_true', help='Read every original model/evidence file; about 21 GB.')
  p.add_argument('--output', type=Path)
  args = p.parse_args(); result = check(Path(__file__).resolve().parents[1], args.full)
  if args.output: args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))
  raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__': main()
