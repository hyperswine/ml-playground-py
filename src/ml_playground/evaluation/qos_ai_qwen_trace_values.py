"""Supplementary ordered-value audit; never replaces frozen canonical trace scores.

Equal ordered values with renamed fields can explain a strict-field miss. This is
still a proxy: it does not establish semantic equivalence of arbitrary explanations,
program correctness, or that an answer contains no unsupported extra prose.
"""
import json
import re


def ordered_trace(answer):
  prose=re.sub(r'```.*?```','',answer,flags=re.S)
  m=re.search(r'\bSteps:\s*(.*?)(?:\nIt prints|$)',prose,re.S)
  if not m:return None
  rest=m[1].strip();fields=[];decoder=json.JSONDecoder()
  while rest:
    field=re.match(r'([A-Za-z_]\w*)\s*=\s*',rest)
    if not field:return None
    rest=rest[field.end():]
    try:value,end=decoder.raw_decode(rest)
    except ValueError:return None
    fields.append((field[1],value));rest=rest[end:].strip()
    if rest=='.':return fields
    if not rest.startswith(';'):return None
    rest=rest[1:].strip()
  return None


def audit(row,answer):
  parsed=ordered_trace(answer);expected=[(t['name'],t['value']) for t in row['traces']]
  match=parsed is not None and len(parsed)==len(expected) and all(type(a[1]) is type(b[1]) and a[1]==b[1] for a,b in zip(parsed,expected))
  return {'ordered_values_match':bool(match),'renamed_fields_only':bool(match and [a[0] for a in parsed]!=[b[0] for b in expected]),
          'parsed':parsed,'expected':expected}
