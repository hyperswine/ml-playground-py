import json,sys
from pathlib import Path
r=Path('/Users/jq/qos_ai_eval/experiments/qwen_internals_20261028');old=r.parent/'qwen2b_repair_20261027';root=r.parent.parent
sys.path[:0]=[str(old),str(root)]
from qos_ai_qwen_repair import read,score_row,digest
rows={x['id']:x for x in read(r.parent/'qwen2b_controlled_20261028/dev_rows.jsonl')}
fpr=str(root/'experiments/stage4_20261021/preparation/toolchain/fpr')
report={'policy':'Exploratory four-question development comparison; not statistical or promotion evidence. Strict final-output score treats an unclosed thinking section as no final answer. Raw-output score is a separate diagnostic, not protocol-compliant success.','models':{}}
for model in ('base','parent'):
 data=json.loads((r/f'{model}_probe.json').read_text());result={}
 for condition in ('historical_budget','longer_no_thinking','thinking'):
  answers=[x for x in data['answers'] if x['condition']==condition];details=[]
  for ans in answers:
   row=rows[ans['id']];grade=score_row(row,ans['answer'],fpr);raw=score_row(row,ans['raw'],fpr)
   details.append({**ans,'expected':row['expected'],'strict_final_grade':grade,'raw_grade':raw})
  result[condition]={'n':len(details),'complete_final':sum(x['strict_final_grade']['correct'] for x in details),
    'complete_raw':sum(x['raw_grade']['correct'] for x in details),'correct_final_outputs':sum(x['strict_final_grade']['claim_correct'] for x in details),
    'working_final_programs':sum(x['strict_final_grade'].get('code_correct',False) for x in details),'finished':sum(x['finished'] for x in details),
    'thinking_closed':sum(x['thinking_closed'] for x in details),'generated_tokens':sum(x['generated_tokens'] for x in details),'details':details}
 report['models'][model]={'label':'untuned Qwen3.5-2B instruction checkpoint (not Qwen3.5-2B-Base)' if model=='base' else 'round-4 Qwen3.5-2B q8','answers_sha256':digest(r/f'{model}_probe.json'),'conditions':result}
(r/'probe_scores.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({m:{c:{k:v for k,v in d.items() if k!='details'} for c,d in info['conditions'].items()} for m,info in report['models'].items()},indent=2))
