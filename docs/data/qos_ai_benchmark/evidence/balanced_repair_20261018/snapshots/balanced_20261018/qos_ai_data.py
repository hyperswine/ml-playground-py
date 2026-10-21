"""Frozen, file-grouped splits and benchmark exclusions. No model dependencies."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 2
SPLITS = ('train', 'dev', 'test')
PRIORITY = {'train': 0, 'dev': 1, 'test': 2}


def read_rows(path):
  path = Path(path)
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
      h.update(block)
  return h.hexdigest()


def normalize(question):
  return ' '.join(re.findall(r'\w+|[^\w\s]', question.casefold()))


def question_id(question):
  return digest(normalize(question))


def shingles(question):
  words = re.findall(r'\w+', question.casefold())
  return {tuple(words[i:i + 3]) for i in range(len(words) - 2)} or {tuple(words)}


def similar_pairs(questions, threshold=0.75):
  """Exact Jaccard matches using a rarity-ordered prefix index (no quadratic scan).

  This detects lexical near duplicates; it does not claim semantic equivalence.
  Whole-file grouping also keeps all generated question styles/rounds together.
  """
  sets = [shingles(q) for q in questions]
  freq = Counter(s for ss in sets for s in ss)
  index = defaultdict(list)
  for i in sorted(range(len(sets)), key=lambda i: (len(sets[i]), questions[i])):
    ss = sets[i]
    ordered = sorted(ss, key=lambda s: (freq[s], s))
    prefix = ordered[:len(ss) - math.ceil(threshold * len(ss)) + 1]
    candidates = set()
    for s in prefix:
      candidates.update(j for j in index[s] if len(sets[j]) >= threshold * len(ss))
    for j in sorted(candidates):
      other = sets[j]
      if len(ss & other) / len(ss | other) >= threshold:
        yield j, i
    for s in prefix:
      index[s].append(i)


def rows_fingerprint(rows):
  return digest(sorted(digest(row) for row in rows))


def make_manifest(corpus, qa, verified, benchmark):
  """Keep whole files together; exclude cross-partition duplicate Q&A components.

  Files are not merged through generic repeated questions: that can collapse a
  whole repository into one split. Instead those questions are removed from all
  partitions. Generalization reserves named source files, not every mention of
  their concepts elsewhere in the repository.
  """
  rows = qa + verified
  paths = sorted({r['path'] for r in corpus + rows} | {s['path'] for b in benchmark for s in b['sources']})
  forced = defaultdict(set)
  for b in benchmark:
    if b['track'] == 'generalization':
      for source in b['sources']:
        forced[source['path']].add(b['split'])
  if any(len(v) > 1 for v in forced.values()):
    raise ValueError('Generalization sources overlap dev/test; resolve benchmark groups')
  assignments = {}
  for path in paths:
    value = int(digest(path)[:8], 16) % 100
    assignments[path] = next(iter(forced[path])) if forced[path] else 'test' if value < 5 else 'dev' if value < 10 else 'train'

  text, memberships = {}, defaultdict(set)
  for row in rows:
    qid = question_id(row['question'])
    text[qid] = row['question']
    memberships[qid].add(assignments[row['path']])
  reserved = {question_id(b['question']) for b in benchmark}
  for b in benchmark:
    text[question_id(b['question'])] = b['question']
  keys = sorted(text)
  parent = {k: k for k in keys}
  def find(k):
    while parent[k] != k:
      parent[k] = parent[parent[k]]
      k = parent[k]
    return k
  near_count = 0
  for a, b in similar_pairs([text[k] for k in keys]):
    ra, rb = find(keys[a]), find(keys[b])
    parent[max(ra, rb)] = min(ra, rb)
    near_count += 1
  group_splits, benchmark_groups = defaultdict(set), set()
  for k in keys:
    group_splits[find(k)].update(memberships[k])
    if k in reserved:
      benchmark_groups.add(find(k))
  questions = {}
  excluded_reasons = Counter()
  for k in keys:
    root = find(k)
    if root in benchmark_groups:
      questions[k] = 'exclude'
      excluded_reasons['benchmark_or_near_duplicate'] += 1
    elif len(group_splits[root]) > 1:
      questions[k] = 'exclude'
      excluded_reasons['cross_partition_duplicate'] += 1
    else:
      questions[k] = next(iter(group_splits[root]))
  excluded_chunks = set()
  for a, b in similar_pairs([r['text'] for r in corpus]):
    left, right = corpus[a], corpus[b]
    if assignments[left['path']] != assignments[right['path']]:
      excluded_chunks.update((digest([left['path'], left['chunk']]), digest([right['path'], right['chunk']])))
  # Exact/near copied source chunks must not re-enter through generated Q&A.
  for row in qa:
    if digest([row['path'], row['chunk']]) in excluded_chunks:
      questions[question_id(row['question'])] = 'exclude'

  result = {
    'schema': SCHEMA, 'policy': 'whole-file+exclude-cross-split-exact-and-trigram-jaccard-0.75',
    'inputs': {n: rows_fingerprint(r) for n, r in [('corpus', corpus), ('qa', qa), ('verified', verified), ('benchmark', benchmark)]},
    'files': assignments, 'questions': questions, 'excluded_chunks': sorted(excluded_chunks),
    'benchmark_questions': [b['question'] for b in benchmark],
    'counts': {'files': dict(Counter(assignments.values())), 'qa': dict(Counter(questions[question_id(r['question'])] for r in qa)),
               'verified': dict(Counter(questions[question_id(r['question'])] for r in verified)),
               'near_duplicate_links': near_count, 'excluded_copied_source_chunks': len(excluded_chunks), 'excluded_unique_question_reasons': dict(excluded_reasons)},
  }
  result['hash'] = digest(result)
  return result


def validate_manifest(manifest, corpus=None, qa=None, verified=None, benchmark=None):
  if 'excluded_chunks' not in manifest or manifest.get('schema') != SCHEMA or manifest.get('hash') != digest({k: v for k, v in manifest.items() if k != 'hash'}):
    raise ValueError('Missing, corrupt or obsolete split manifest; run splits first')
  for name, rows in [('corpus', corpus), ('qa', qa), ('verified', verified), ('benchmark', benchmark)]:
    if rows is not None and manifest['inputs'][name] != rows_fingerprint(rows):
      raise ValueError(f'{name} changed since split preparation; run splits into a new experiment directory')


def row_split(row, manifest):
  """Unknown questions/files fail closed; self/repair inherit their question split."""
  qid = question_id(row['question'])
  if qid not in manifest['questions']:
    raise ValueError('Question absent from split manifest; rebuild splits before teaching')
  split = manifest['questions'][qid]
  if split == 'exclude':
    return split
  if row.get('path') and manifest['files'].get(row['path']) != split:
    raise ValueError('Question/source split mismatch')
  return split


def pack_examples(examples, length):
  """Pack complete examples; reject oversize items rather than orphaning answers."""
  output, stream = [], []
  for example in examples:
    if len(example) > length:
      raise ValueError(f'Example has {len(example)} tokens, exceeds context {length}; shorten it or increase --context-length')
    if stream and len(stream) + len(example) > length:
      output.append(window(stream))
      stream = []
    stream.extend(example)
  if stream:
    output.append(window(stream))
  return output


def window(stream):
  return ([t for t, _ in stream], [i + 1 < len(stream) and stream[i + 1][1] for i in range(len(stream))])


def source_check(row, roots):
  """Check exact source bytes and recorded snippets; commits alone miss dirty edits."""
  for s in row['sources']:
    repo, rel = s['path'].split('/', 1)
    root = Path(roots[repo]).resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
      raise ValueError('Source path escapes repository')
    if file_hash(path) != s['sha256']:
      raise ValueError(f'Stale source for {row["id"]}: {s["path"]}')
    text = path.read_text()
    if not s.get('evidence') or s['evidence'] not in text:
      raise ValueError(f'Missing source evidence for {row["id"]}: {s["path"]}')


def lexical_score(row, answer):
  """Transparent rubric proxy; never label substring matching as factual accuracy."""
  checks = row.get('rubric', {})
  required = checks.get('required', [])  # AND of OR groups, case-insensitive regexes
  forbidden = checks.get('forbidden', [])
  passed = [any(re.search(p, answer, re.I | re.S) for p in group) for group in required]
  rejected = [p for p in forbidden if re.search(p, answer, re.I | re.S)]
  return {'rubric_proxy_pass': bool(answer.strip()) and bool(required) and all(passed) and not rejected,
          'required_groups_passed': passed, 'forbidden_matches': rejected,
          'factual_accuracy': None, 'human_review_required': True}
