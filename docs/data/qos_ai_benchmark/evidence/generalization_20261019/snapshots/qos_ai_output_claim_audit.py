"""Supplement the unchanged coding score with explicit output-claim consistency."""
import json
import re
from pathlib import Path
from qos_ai_data import file_hash,read_rows
from qos_ai_compare import stated_output


def claim_status(answer,expected):
  prose=re.sub(r'```.*?```','',answer,flags=re.S)
  # Preserve the primary scorer, but flag conflicting same-line claims here.
  claims={stated_output(part) for part in re.split(r'\n|(?<=[.!?])\s+(?=[A-Z])',prose)}-{None}
  if len(claims)>1:return 'ambiguous',None
  literal=stated_output(answer)
  if literal is not None:return ('correct' if literal==expected else 'incorrect'),literal
  if re.search(r'\b(?:prints?|stdout|output|result)\b',prose,re.I):return 'ambiguous',None
  return 'absent',None


def audit(root=Path('.')):
  reports=root/'reports/output_complete_20261018';result={'primary_benchmark_unchanged':True,
    'code_sha256':file_hash(Path(__file__)),'note':'Supplemental audit of stated literals in existing generation answers; ambiguous claims remain unassessed. This does not assess all explanation prose.', 'runs':{}}
  for label,path in (('before',root/'reports/balanced_repair_20261018/codegen_after_diag'),('after',reports/'tuned_diag')):
    grading=json.loads((path/'grading.json').read_text());answers={r['id']:r['answer'] for r in read_rows(path/'answers.jsonl')}
    if grading['answers_sha256']!=file_hash(path/'answers.jsonl'):raise ValueError('Stale code grading')
    details=[]
    for row in grading['results']:
      if row['category']!='code_generation':continue
      status,literal=claim_status(answers[row['id']],row['expected_stdout'])
      details.append({'id':row['id'],'working_program':row['complete_pass'],'expected_stdout':row['expected_stdout'],'stated_output':literal,'claim_status':status})
    result['runs'][label]={'checkpoint_sha256':grading['checkpoint_sha256'],'answers_sha256':grading['answers_sha256'],'count':len(details),
      'working_programs':sum(d['working_program'] for d in details),
      'working_with_consistent_or_absent_claim':sum(d['working_program'] and d['claim_status'] in ('correct','absent') for d in details),
      **{'working_with_'+status+'_claim':sum(d['working_program'] and d['claim_status']==status for d in details) for status in ('correct','incorrect','ambiguous','absent')},'details':details}
  (reports/'output_claim_audit.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({label:{k:v for k,v in r.items() if k!='details'} for label,r in result['runs'].items()},indent=2))


if __name__=='__main__':audit()
