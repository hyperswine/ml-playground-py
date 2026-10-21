import json,os,platform,re,subprocess,tempfile
from pathlib import Path
from qos_ai_data import digest,file_hash,read_rows,source_check
root=Path('/Users/jq/qos_ai_eval')
bench={r['id']:r for r in read_rows(root/'qos_ai_benchmark/benchmark.jsonl')}
# These observed snippets were manually inspected: pure List/Str/Numeric operations and print.
approved={
 'parent':{
  'debug-at':'result = Str.at "ABC" 1.\n> print (str result).\n',
  'debug-list':'> print (str (List.rev [4, 5, 6] ! 1)).\n',
  'write-map':'result = List.sum (List.map (fn v -> v * 3) [2, 3, 2]).\n> print (str result).\n',
  'write-clamp':'result = Numeric.clamp 20 8.\n> print (str result).\n'},
 'tuned':{
  'debug-at':'result = Str.at "ABC" 1.\n> print (str result).\n',
  'debug-list':'result = (List.rev [4, 5, 6]) ! 1.\n> print (str result).\n',
  'write-map':'result = List.sum (List.map (fn x -> x * 2) [2, 3]).\n> print (str result).\n',
  'write-clamp':'result = Numeric.clamp 1 8 20.\n> print (str result).\n'}}
records=[];cache={}
for model,stem in (('parent','expanded_tuned'),('tuned','inputs_tuned')):
 for condition in ('raw','guard'):
  answers={r['id']:r for r in read_rows(root/f'reports/{stem}_{condition}/answers.jsonl')}
  for ident,code in approved[model].items():
   row=answers[ident];answer=row['answer'];b=bench[ident]
   source_check(b,{'fprisc':str(root/'source_snapshot/fprisc'),'qos':str(root/'source_snapshot/qos')})
   blocks=re.findall(r'```sol\n(.*?)```',answer,re.S)
   extraction='entire generated program, unchanged'
   if model=='parent' and ident=='debug-list':
    if answer!='List.rev [4, 5, 6] ! 1 is 6.':raise ValueError('Unreviewed parent expression')
    extraction='print wrapper around the exact stated expression'
   elif model=='tuned' and condition=='guard' and ident=='debug-at':
    if blocks or answer!='The Sol string uses a zero-based byte offset. The Sol string has a byte length of 1.':raise ValueError('Unreviewed no-code answer')
    code=None;extraction='no corrected expression or program was supplied'
   elif blocks!=[code]:raise ValueError('Unreviewed generated code')
   contract=model=='tuned' and ident in ('write-map','write-clamp')
   if code is not None and code not in cache:
    with tempfile.TemporaryDirectory(prefix='qos-ai-inputs-generated-') as tmp:
     path=Path(tmp)/'generated.sol';path.write_text(code)
     env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=str(root/'source_snapshot/fprisc'))
     run=subprocess.run([str(root/'source_snapshot/fprisc/fpr'),'sol',str(path)],cwd=tmp,env=env,capture_output=True,text=True,timeout=30)
     cache[code]={'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
   outcome=cache[code] if code is not None else {'exit_code':None,'stdout':None,'stderr':None}
   expected=b['program']['stdout'].strip()
   matches=outcome['exit_code']==0 and outcome['stdout'].strip()==expected if code is not None else False
   records.append({'model':model,'condition':condition,'id':ident,'category':b['category'],'answer_hash':digest(answer),
    'code':code,'extraction':extraction,'expected_stdout':expected,'requested_input_and_operation_contract':contract,
    'behavior_matches':matches,'task_behavior_correct':matches and contract,**outcome})
summary={}
for model in ('parent','tuned'):
 for condition in ('raw','guard'):
  selected=[r for r in records if r['model']==model and r['condition']==condition]
  summary[model+'_'+condition]={category:{'count':sum(r['category']==category for r in selected),
    'correct':sum(r['task_behavior_correct'] for r in selected if r['category']==category)} for category in ('debugging','code_generation')}
result={'host':platform.node(),'machine':platform.machine(),'fpr_sha256':file_hash(root/'source_snapshot/fprisc/fpr'),
        'distinct_executions':len(cache),'reviewer':'Codex source/input-contract review with actual M1 execution; not independent human review',
        'summary':summary,'records':records}
(root/'reports/inputs_generated_execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
