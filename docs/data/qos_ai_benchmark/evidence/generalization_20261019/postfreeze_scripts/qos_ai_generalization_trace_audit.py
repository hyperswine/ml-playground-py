"""Supplementary strict audit of whole canonical trace fields in saved answers.

The frozen primary trace parser recognizes value prefixes. This audit refuses
extra tokens such as input=4 4, decimals parsed as integer prefixes, duplicate
fields and malformed JSON. It does not replace the frozen primary scores.
"""
import json
import re


def same_value(actual,expected):
  if type(actual) is not type(expected):return False
  if isinstance(expected,list):return len(actual)==len(expected) and all(same_value(a,b) for a,b in zip(actual,expected))
  return actual==expected


def strict_trace(row,answer):
  prose=re.sub(r'```.*?```','',answer,flags=re.S)
  fields={}
  for t in row['traces']:
    # Each authored assignment ends at a semicolon, newline or end of prose.
    # Remove a single sentence-final period, then require the entire remaining
    # value to decode as JSON, rather than accepting a numeric prefix.
    found=re.findall(r'\b'+re.escape(t['name'])+r'\s*=\s*([^;\n]*)',prose)
    values=[]
    for value in found:
      value=value.strip()
      if value.endswith('.'):value=value[:-1].rstrip()
      try:values.append(json.loads(value))
      except ValueError:values.append(None)
    fields[t['name']]=len(values)==1 and same_value(values[0],t['value'])
  return {'strict_canonical_trace_match':all(fields.values()),'fields':fields}
