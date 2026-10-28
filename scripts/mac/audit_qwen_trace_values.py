import json,hashlib,sys
from pathlib import Path
r=Path('/Users/jq/qos_ai_eval/experiments/qwen2b_controlled_20261028');sys.path.insert(0,str(r))
from qos_ai_qwen_trace_values import audit
full='--full' in sys.argv
row_path=r.parent.parent/'curricula/generalization_20261019/generalization_diagnostic.jsonl' if full else r/'guard_rows.jsonl'
rows=[json.loads(l) for l in row_path.read_text().splitlines()]
if full:rows=[row for row in rows if row['mode']!='output_only']
suffix='_frozen' if full else '_guard'
paths={'parent':r.parent/'mlx_20261027/answers_2b_round4_q8.jsonl',**{p.stem.removesuffix(suffix):p for p in sorted(r.glob('*'+suffix+'.jsonl'))}}
out={}
for label,path in paths.items():
 answers={a['id']:a['answer'] for a in map(json.loads,path.read_text().splitlines())}
 if not all('gen/'+row['id'] in answers for row in rows):continue
 details=[{'id':row['id'],'answer':answers['gen/'+row['id']],**audit(row,answers['gen/'+row['id']])} for row in rows]
 out[label]={'n':len(rows),'ordered_values_match':sum(d['ordered_values_match'] for d in details),
  'renamed_fields_only':sum(d['renamed_fields_only'] for d in details),'answers_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'details':details}
report={'policy':'Post-hoc supplementary proxy; promotion criteria and frozen canonical scores unchanged. Exact ordered intermediate values permit renamed labels but do not prove arbitrary prose semantics or program correctness.',
 'scorer_sha256':hashlib.sha256((r/'qos_ai_qwen_trace_values.py').read_bytes()).hexdigest(),'runs':out}
(r/('full_trace_value_audit.json' if full else 'trace_value_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{x:y for x,y in v.items() if x!='details'} for k,v in out.items()},indent=2))
