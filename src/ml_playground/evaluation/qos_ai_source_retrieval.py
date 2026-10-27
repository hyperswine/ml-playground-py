"""Opt-in retrieval for current implementation questions; legacy BM25 stays frozen."""
import re
from qos_ai_retrieval import BM25, tokens

PATH = re.compile(r'(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+')
IDENT = re.compile(r'\b[A-Za-z_]\w*\.\w+\b')
HISTORICAL = re.compile(r'(?:prelim|proposal|(?:^|[/_-])plan(?:[._/-]|$))', re.I)
GROUNDING = ('Answer briefly using only the supplied reference. Use implementation excerpts for current behavior. '
             'Do not invent APIs, proposals, comparisons or examples. If the reference does not establish a behavior, '
             'say that it is not established. End with "Source: <file path>" citing the relevant reference file.')


class SourceBM25(BM25):
  """Lexical ranking with exact path/identifier boosts and explicit source authority.

  Authority is supplied by a curated source manifest, never inferred from date alone.
  Explicit requests for a historical path are still honored. No benchmark answers enter the index.
  """
  def __init__(self, docs):
    if not docs:
      raise ValueError('source index must not be empty')
    super().__init__(docs)

  def search(self, query, k=3):
    q = tokens(query)
    paths = [p.rstrip('.') for p in PATH.findall(query)]
    identifiers = IDENT.findall(query)
    hosted = bool(re.search(r'\bSol\b', query, re.I))
    native = bool(re.search(r'\b(native|FP-RISC)\b', query, re.I))
    scores = []
    for i, (d, c) in enumerate(zip(self.docs, self.tf)):
      score = 0.
      for w in q:
        if w in c:
          f = c[w]
          score += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
      exact_paths = sum(d['path'] == p or d['path'].endswith('/' + p) for p in paths)
      if not score and not exact_paths:
        continue
      authority = d.get('authority')
      historical = authority == 'historical' if authority else bool(HISTORICAL.search(d['path']))
      score *= .25 if historical else (1.3 if d.get('authority') == 'implementation' else 1.)
      # For otherwise ambiguous JSON/library queries, retain the requested runtime profile.
      if not paths and not historical:
        if hosted and not native and '/sol/' in d['path']: score += 6
        if native and not hosted and '/std/' in d['path']: score += 6
      score += 40 * exact_paths
      score += 4 * sum(bool(re.search(r'(?<![\w.])' + re.escape(ident) + r'(?![\w.])', d['text'])) for ident in identifiers)
      scores.append((score, -i))
    return [self.docs[-j] for _, j in sorted(scores, reverse=True)[:k]]


def is_documentation_question(question):
  # An explicit module path is sufficient even when the user says "write" or includes code.
  return bool(PATH.search(question) or re.search(r'\b(API|documentation|module|library|reference)\b', question, re.I))


def grounded_question(question):
  return f'{question}\n\n{GROUNDING}'
