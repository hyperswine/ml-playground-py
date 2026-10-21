"""Frozen coding diagnostic generation and conservative, separately repeatable grading."""
import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from qos_ai_data import file_hash, read_rows, rows_fingerprint, source_check
from qos_ai_compare import stated_output
from qos_ai_pure_sol import Parser, Unsupported, extract


def diagnostic(data, snapshot, fpr=None):
  rows=read_rows(data/'coding_diagnostic.jsonl')
  build=json.loads((data/'build.json').read_text())
  proof=json.loads((data/'coding_verification.json').read_text())
  if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate diagnostic')
  if rows_fingerprint(rows)!=build['diagnostic_hash'] or proof['diagnostic_hash']!=build['diagnostic_hash']:
    raise ValueError('Diagnostic fingerprint differs from frozen reference proof')
  expected={(r['id'],field) for r in rows for field in ('program','debug_program') if field in r}
  passed={(r['id'],r['fixture']) for r in proof['results'] if r['passed']}
  if passed!=expected or proof['passed']!=proof['total']:raise ValueError('Missing passing reference execution')
  if fpr and file_hash(fpr)!=proof['compiler_sha256']:raise ValueError('Compiler differs from reference proof')
  roots={n:str(snapshot/n) for n in ('fprisc','qos')}
  for row in rows:source_check(row,roots)
  return rows


def run(args):
  import torch
  import qos_ai as ai
  rows=diagnostic(args.data,args.snapshot)
  if args.output.exists():raise ValueError('Choose a new diagnostic output')
  torch.manual_seed(7);device=ai.choose_device('mps')
  model,parent=ai.load_student(args.load,device);tokenizer=ai.StudentTokenizer(parent['qwen_ids'])
  settings={'checkpoint_sha256':file_hash(args.load),'diagnostic_hash':rows_fingerprint(rows),
    'diagnostic_sha256':file_hash(args.data/'coding_diagnostic.jsonl'),'seed':7,'temperature':0,'top_k':20,
    'no_repeat':3,'max_new_tokens':128,'device':str(device),'host':platform.node(),'machine':platform.machine(),
    'torch':str(torch.__version__),'model_code_sha256':file_hash(Path(ai.__file__)),
    'generation_code_sha256':file_hash(Path(__file__)),'ancestry':parent.get('benchmark_ancestry','legacy_or_unknown'),
    'scope':'auxiliary familiar-source novel-input diagnostic; no clean legacy pretraining exposure claim'}
  args.output.mkdir(parents=True);results=[]
  (args.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
  for row in rows:
    prompt=tokenizer.encode(ai.chat_text(row['question']));budget=min(128,model.config.context_length-len(prompt))
    if budget<=0:raise ValueError('Diagnostic prompt exceeds context')
    started=time.perf_counter()
    with torch.inference_mode():
      ids,finished=model.generate_batch([prompt],budget,parent['im_end_id'],0,20,3)[0]
    record={'id':row['id'],'question':row['question'],'category':row['category'],'family':row['family'],
      'answer':tokenizer.decode(ids).strip(),'finished':finished,'tokens':len(ids),'effective_token_budget':budget,
      'severe_loop':ai.loops(ids,phrase=4,times=4),'seconds':time.perf_counter()-started}
    results.append(record)
    with (args.output/'answers.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(f'{len(results)}/{len(rows)} {row["id"]}',flush=True)
  summary={**settings,'count':len(results),'finished':sum(r['finished'] for r in results),
    'severe_loops':sum(r['severe_loop'] for r in results),'mean_tokens':sum(r['tokens'] for r in results)/len(results)}
  (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2))


def grade_answer(row,answer,fpr,snapshot,cache):
  expected=row['program']['stdout'].strip();category=row['category']
  result={'id':row['id'],'category':category,'family':row.get('family','official'),
    'expected_stdout':expected,'stated_output':stated_output(answer),'contract_match':False,
    'executed':False,'behavior_pass':False,'complete_pass':False}
  result['stated_output_match']=result['stated_output']==expected
  no_change=bool(re.search(r'\bno change (?:is )?(?:needed|required)\b',answer,re.I))
  has_code='```' in answer or '`' in answer or answer.strip().startswith('>')
  if category=='no_change' and not has_code:
    result.update(no_change_claim=no_change,complete_pass=no_change and result['stated_output_match'])
    return result
  try:
    code,ast,kind=extract(answer)
    reference=Parser(row['program']['code']).program()
    result.update(extraction=kind,code=code,contract_match=ast==reference)
    # Only reviewed input/operation structures reach the compiler. Incorrect or
    # alternative structures remain unmatched, rather than executing arbitrary code.
    if not result['contract_match']:
      result['reason']='supported syntax but reference operation/input structure differs';return result
    if category=='code_generation' and kind!='program':
      result['reason']='complete runnable program requested';return result
    if code not in cache:
      with tempfile.TemporaryDirectory(prefix='qos-ai-coding-') as tmp:
        path=Path(tmp)/'candidate.sol';path.write_text(code)
        env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=str((snapshot/'fprisc').resolve()))
        try:
          proc=subprocess.run([str(fpr.resolve()),'sol',str(path)],cwd=tmp,env=env,text=True,capture_output=True,timeout=8)
          cache[code]={'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr[:2000]}
        except subprocess.TimeoutExpired:cache[code]={'timeout':True}
    execution=cache[code];result.update(executed=True,execution=execution)
    result['behavior_pass']=execution.get('returncode')==0 and execution.get('stdout','').strip()==expected
    result['complete_pass']=result['behavior_pass'] and (category=='code_generation' or result['stated_output_match'])
    if category=='no_change':result.update(no_change_claim=no_change,complete_pass=result['complete_pass'] and no_change)
  except (Unsupported,RecursionError) as error:result['reason']='restricted harness: '+str(error)
  return result


def grade(args):
  rows=diagnostic(args.data,args.snapshot,args.fpr);answers=read_rows(args.reports/'answers.jsonl')
  summary=json.loads((args.reports/'summary.json').read_text())
  if summary['diagnostic_hash']!=rows_fingerprint(rows):raise ValueError('Generation diagnostic differs')
  if [(r['id'],r['question']) for r in rows]!=[(r['id'],r['question']) for r in answers]:
    raise ValueError('Generation order/questions/coverage differ')
  cache={};results=[grade_answer(r,a['answer'],args.fpr,args.snapshot,cache) for r,a in zip(rows,answers)]
  categories={}
  for category in sorted({r['category'] for r in results}):
    selected=[r for r in results if r['category']==category]
    categories[category]={'count':len(selected),**{key:sum(r[key] for r in selected) for key in
      ('contract_match','executed','behavior_pass','stated_output_match','complete_pass')}}
  output={'checkpoint_sha256':summary['checkpoint_sha256'],'diagnostic_hash':rows_fingerprint(rows),
    'answers_sha256':file_hash(args.reports/'answers.jsonl'),'grader_sha256':file_hash(Path(__file__)),
    'parser_sha256':file_hash(Path(__file__).with_name('qos_ai_pure_sol.py')),'compiler_sha256':file_hash(args.fpr),
    'output_parser_sha256':file_hash(Path(__file__).with_name('qos_ai_compare.py')),
    'coverage_note':'Restricted structural contract plus executed stdout; unsupported or different algorithms are unmatched. No-change prose uses verified reference proof, not generated execution.',
    'categories':categories,'count':len(results),'complete_pass':sum(r['complete_pass'] for r in results),
    'distinct_executions':len(cache),'results':results}
  (args.reports/'grading.json').write_text(json.dumps(output,indent=2)+'\n')
  print(json.dumps({k:v for k,v in output.items() if k!='results'},indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
  for command,fn in (('run',run),('grade',grade)):
    sub=s.add_parser(command);sub.set_defaults(fn=fn)
    sub.add_argument('--data',type=Path,required=True);sub.add_argument('--snapshot',type=Path,default=Path('source_snapshot'))
    if command=='run':sub.add_argument('--load',type=Path,required=True);sub.add_argument('--output',type=Path,required=True)
    else:sub.add_argument('--reports',type=Path,required=True);sub.add_argument('--fpr',type=Path,default=Path('source_snapshot/fprisc/fpr'))
  a=p.parse_args();a.fn(a)
