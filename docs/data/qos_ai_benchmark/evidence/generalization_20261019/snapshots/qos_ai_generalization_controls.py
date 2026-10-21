"""Prove authored diagnostic answers pass the same restricted grader before inference."""
import argparse
from pathlib import Path
from qos_ai_generalization_eval import checked_diagnostic
from qos_ai_generalization import trace_grade
from qos_ai_coding_diag import grade_answer
from qos_ai_compare import stated_output
from qos_ai_data import read_rows,rows_fingerprint
from qos_ai_pure_sol import Parser,Unsupported
import json
from collections import Counter


def audit_data(data):
  canonical=read_rows(data/'verified_train.jsonl');train=read_rows(data/'balanced_train_ready.jsonl');diag=read_rows(data/'generalization_diagnostic.jsonl')
  def shape(node):
    if isinstance(node,tuple):
      if node[0] in ('int','string','list'):return (node[0],'literal')
      return tuple(shape(v) for v in node)
    return node
  asts=[]
  for r in canonical:
    for field in ('program','debug_program'):
      if field not in r:continue
      try:asts.append(Parser(r[field]['code']).program())
      except Unsupported:pass
  known={repr(a) for a in asts};shapes={repr(shape(a)) for a in asts}
  lookup={r['id']:r for r in train};report={}
  for stratum in ('wording','inputs','combinations'):
    rows=[r for r in diag if r['stratum']==stratum]
    modes=dict(Counter(r['mode'] for r in rows));overlap=sum(repr(Parser(r['program']['code']).program()) in known for r in rows)
    shape_overlap=sum(repr(shape(Parser(r['program']['code']).program())) in shapes for r in rows)
    if len(rows)!=48 or set(modes.values())!={8}:raise ValueError('Diagnostic mode imbalance')
    if stratum=='wording':
      for r in rows:
        pair=lookup[r['training_pair_id']]
        if any(r[k]!=pair[k] for k in ('answer','program','trace_program','mode','traces')):raise ValueError('Wording changes more than instructions')
      if overlap!=48:raise ValueError('Missing wording pairs')
    elif overlap:raise ValueError('Held-out input program leaked into known curriculum')
    if stratum=='combinations' and shape_overlap:raise ValueError('Held-out operation shape leaked into known curriculum')
    report[stratum]={'count':len(rows),'modes':modes,'known_program_overlap':overlap,'known_operation_shape_overlap':shape_overlap}
  report['diagnostic_hash']=rows_fingerprint(diag)
  report['scope']='Known canonical programs only; wording overlap is intentional; unknown historical teacher/pretraining exposure is not excluded.'
  return report


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
  snapshot=Path('source_snapshot');rows=checked_diagnostic(a.data,snapshot);cache={}
  audit=audit_data(a.data);(a.data/'separation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
  for r in rows:
    if r['category']=='output_prediction':passed=stated_output(r['answer'])==r['program']['stdout'].strip()
    else:
      g=grade_answer(r,r['answer'],snapshot/'fprisc/fpr',snapshot,cache)
      passed=g['complete_pass'] and g['stated_output_match']
    if not passed or (r['mode']!='output_only' and not trace_grade(r,r['answer'])['canonical_trace_match']):raise ValueError('Authored grader control failed '+r['id'])
  print('All',len(rows),'authored diagnostic answers passed, including required canonical traces.')
