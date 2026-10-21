import json,os,subprocess,tempfile,platform
from pathlib import Path
from qos_ai_data import digest,read_rows,file_hash
root=Path('/Users/jq/qos_ai_eval')
# These four observed snippets were inspected: pure numeric/list/string work and print only.
approved={
 'debug-at':'result = Str.at "ABC" 1.\n> print (str result).\n',
 'debug-list':'> print (str (List.rev [4, 5, 6] ! 1)).\n',
 'write-map':'result = List.sum (List.map (fn v -> v * 3) [2, 3, 2]).\n> print (str result).\n',
 'write-clamp':'result = Numeric.clamp 20 8.\n> print (str result).\n'}
expected={'debug-at':'65','debug-list':'6','write-map':'10','write-clamp':'8'}
cache={};records=[]
for condition in ('raw','guard'):
 rows={r['id']:r for r in read_rows(root/f'reports/expanded_tuned_{condition}/answers.jsonl')}
 for ident,code in approved.items():
  answer=rows[ident]['answer']
  if ident=='debug-list':
   if answer!='List.rev [4, 5, 6] ! 1 is 6.':raise ValueError('Unreviewed expression')
  elif answer != '```sol\n'+code+'```':raise ValueError('Unreviewed program')
  if code not in cache:
   with tempfile.TemporaryDirectory(prefix='qos-ai-reviewed-generated-') as tmp:
    script=Path(tmp)/'generated.sol';script.write_text(code)
    env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=str(root/'source_snapshot/fprisc'))
    run=subprocess.run([str(root/'source_snapshot/fprisc/fpr'),'sol',str(script)],cwd=tmp,env=env,capture_output=True,text=True,timeout=30)
    cache[code]={'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
  outcome=cache[code]
  records.append({'condition':condition,'id':ident,'answer_hash':digest(answer),'code':code,
    'extraction':'printed wrapper around the exact stated expression' if ident=='debug-list' else 'entire generated program, unchanged',
    'expected_stdout':expected[ident],'behavior_correct':outcome['exit_code']==0 and outcome['stdout'].strip()==expected[ident],**outcome})
result={'host':platform.node(),'machine':platform.machine(),'fpr_sha256':file_hash(root/'source_snapshot/fprisc/fpr'),
        'distinct_executions':len(cache),'records':records}
(root/'reports/expanded_generated_execution.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
