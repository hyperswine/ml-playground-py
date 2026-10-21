"""Balanced repair/retention teaching and a separately frozen coding diagnostic."""
import argparse
import json
import random
import re
from collections import Counter,defaultdict
from pathlib import Path
from qos_ai_curriculum import screen_extension,source_for_builtin,write_rows,verify as verify_programs
from qos_ai_inputs_curriculum import render as input_render,goal as input_goal,mistakes,program,atom,quoted
from qos_ai_data import digest,file_hash,question_id,read_rows,rows_fingerprint,similar_pairs,source_check

FAMILIES=('map','clamp','byte','reverse','slice','filter','sum','take','length','min','range','missing')
MODES=('repair_inline','repair_short','repair_program','correct_check','output_only','output_stated','generate','generate_alt')
TRAIN_WORDS=('Amber','Birch','Cedar','Delta','Fable','Orbit','planet','garden','silver','window','yellow','market','flower','little','monkey','rabbit','castle','pencil','orange','basket','stream','winter','summer','spring')
DIAG_WORDS=('Grape','Maple','Honey','Raven','Tulip','Lemon','forest','bridge','clouds','button','circle','school','travel','island','bright','camera','record','candle','pepper','ribbon','pocket','singer','sailor','cotton')
NEEDLES={'map':['map = fn f xs','sum = fn xs'],'clamp':['clamp = fn lo hi a'],'byte':['at = fn s i'],'reverse':['rev = fn xs'],
 'slice':['slice = fn s i j'],'filter':['filter = fn p xs','sum = fn xs'],'sum':['sum = fn xs'],'take':['take = fn n xs','sum = fn xs'],
 'length':['len = fn s'],'min':['min = fn a b'],'range':['range = fn lo hi','sum = fn xs'],'missing':['find = fn c s']}


def render(family,p):
  if family=='length':return f'Str.len {quoted(p["text"])}',str(len(p['text']))
  if family=='min':return f'Numeric.min {atom(p["a"])} {atom(p["b"])}',str(min(p['a'],p['b']))
  if family=='range':return f'List.sum (List.range {p["lower"]} {p["upper"]})',str(sum(range(p['lower'],p['upper']+1)))
  if family=='missing':return f'Str.find 126 {quoted(p["text"])}','-1'
  return input_render(family,p)


def task(family,p):
  if family=='length':return f'print the byte length of the ASCII string {quoted(p["text"])} using Str.len'
  if family=='min':return f'use Numeric.min to choose the smaller of {p["a"]} and {p["b"]}, then print it'
  if family=='range':return f'sum the inclusive List.range from {p["lower"]} through {p["upper"]} and print the total'
  if family=='missing':return f'find byte code 126 in {quoted(p["text"])} with Str.find and print the returned offset'
  return input_goal(family,p,0)


def wrong(family,p,index):
  expression,value=render(family,p)
  if family=='length':return expression+' + 1',str(int(value)+1)
  if family=='min':return f'Numeric.max {atom(p["a"])} {atom(p["b"])}',str(max(p['a'],p['b']))
  if family=='range':
    q={**p,'upper':p['upper']+1};return render(family,q)
  if family=='missing':return expression+' + 1','0'
  m=mistakes(family,p)[index%2];return m[1],m[2]


def case_pool(seed,words):
  rows=[]
  for f in FAMILIES:
    rng=random.Random(f'{seed}:{f}');seen=set()
    for i in range(60):
      xs=rng.sample(range(2,30),2+i%3);word=words[i%len(words)]
      if f=='map':p={'values':xs,'factor':2+i%4}
      elif f=='clamp':
        lo=rng.randrange(-5,8);hi=lo+rng.randrange(3,11);value=(lo-4,hi+5,lo+1)[i%3]
        p={'lower':lo,'upper':hi,'value':value}
      elif f in ('byte','length','missing'):p={'text':word}
      elif f in ('reverse','sum'):p={'values':xs}
      elif f=='slice':p={'text':word,'start':i%2,'end':i%2+2+i%2}
      elif f=='filter':
        t=rng.randrange(3,12);p={'values':[t-2,t,t+1,t+rng.randrange(4,12)],'threshold':t}
      elif f=='take':p={'values':xs+[31],'count':1+i%2}
      elif f=='min':
        a,b=rng.sample(range(2,30),2);p={'a':a,'b':b}
      elif f=='range':
        lo=rng.randrange(1,16);p={'lower':lo,'upper':lo+rng.randrange(1,9)}
      key=digest(p)
      if key in seen:continue
      seen.add(key);rows.append({'family':f,'inputs':p,'serial':i,'case_id':f'{seed}-{f}-{i}'})
  return rows


def make_row(case,mode,sources,diagnostic=False):
  f=case['family'];p=case['inputs'];expr,value=render(f,p);bad,bad_value=wrong(f,p,case['serial']);goal=task(f,p)
  good=program(expr);broken=program(bad)
  if mode=='repair_inline':
    q=(f'Repair the faulty part of `{bad}`. Intended task: {goal}. Show the replacement expression and its printed result.' if diagnostic else
       f'A Sol script uses `{bad}` to {goal}. Fix the mistake and give the resulting printed value.')
    answer=f'Use `{expr}`. It prints {value}.';category='debugging'
  elif mode=='repair_short':
    q=f'Correct only the wrong part of this Sol expression: `{bad}`. It should {goal}. Give the corrected expression and stdout.'
    answer=f'`{expr}`\nIt prints {value}.';category='debugging'
  elif mode=='repair_program':
    q=(f'Please correct the logic below, keeping the task data unchanged. Specification: {goal}. Return the fixed file and say what it prints.' if diagnostic else
       f'Find and fix the error in this Sol program. Required behavior: {goal}. Return the corrected program and its output.')+'\n```sol\n'+broken+'```'
    answer='```sol\n'+good+f'```\nIt prints {value}.';category='debugging'
  elif mode=='correct_check':
    q=(f'An engineer proposes the following implementation to {goal}. Is a repair necessary? If it is correct, reply No change needed and state stdout.' if diagnostic else
       f'Check whether this Sol program correctly meets the task: {goal}. If correct, say No change needed and state its printed value.')+'\n```sol\n'+good+'```'
    answer=f'No change needed. It prints {value}.';category='no_change'
  elif mode=='output_only':
    q='What exactly does this Sol program print? Reply with only the stdout line.\n```sol\n'+good+'```'
    answer=value;category='output_prediction'
  elif mode=='output_stated':
    q='State the output produced by this program. Do not edit or rewrite it.\n```sol\n'+good+'```'
    answer=f'It prints {value}.';category='output_prediction'
  elif mode in ('generate','generate_alt'):
    q=(f'Produce a runnable Sol file satisfying this specification: {goal}. Use the stated data and print the result.' if diagnostic else
       f'Write a complete Sol program to {goal}.' if mode=='generate' else
       f'Implement this hosted-script task using precisely the supplied values: {goal}. Provide runnable code.')
    answer='```sol\n'+good+'```';category='code_generation'
  else:raise ValueError('Unknown mode')
  row={'id':('codingdiag' if diagnostic else 'balanced')+'-'+case['case_id']+'-'+mode,
       'path':'fprisc/compiler/Sol/Preamble.hs','chunk':0,'style':'verified','round':0,'category':category,
       'family':f,'case_id':case['case_id'],'mode':mode,'question':q,'answer':answer,'expression':expr,
       'sources':sources[f],'program':{'code':good,'stdout':value+'\n','exit_code':0},
       'balanced_contract':{'family':f,'inputs':p,'serial':case['serial'],'diagnostic':diagnostic},
       'verification':'independent Python result oracle, exact response format/input contract and M1 reference execution'}
  if category=='debugging':row['debug_program']={'code':broken,'stdout':bad_value+'\n','exit_code':0}
  return row


def validate_contract(row):
  c=row['balanced_contract'];case={'family':c['family'],'inputs':c['inputs'],'serial':c['serial'],'case_id':row['case_id']}
  rebuilt=make_row(case,row['mode'],{c['family']:row['sources']},c['diagnostic'])
  for key in ('question','answer','expression','program','debug_program'):
    if row.get(key)!=rebuilt.get(key):raise ValueError(f'Balanced {key} does not match its supplied input/response contract')
  if row.get('debug_program') and row['debug_program']['stdout']==row['program']['stdout']:
    raise ValueError('Broken program has no behavioral difference')


def compact(s):return re.sub(r'\s+','',s)


def lexical_audit(left,right):
  texts=[r['question'] for r in left+right];n=len(left)
  return [(a,b-n) for a,b in similar_pairs(texts) if a<n<=b] + [(b,a-n) for a,b in similar_pairs(texts) if b<n<=a]


def build(args):
  if args.output.exists():raise ValueError('Choose a fresh balanced curriculum directory')
  import torch
  import qos_ai as ai
  old=read_rows(args.previous/'verified_train.jsonl');base=json.loads((args.previous/'splits.json').read_text())
  official=read_rows(args.benchmark);qa=read_rows(args.qa)
  parents=[torch.load(p,map_location='cpu',weights_only=True) for p in args.parent]
  if len(parents)!=2 or not torch.equal(parents[0]['qwen_ids'],parents[1]['qwen_ids']):raise ValueError('Parents need an identical student vocabulary')
  tokenizer=ai.StudentTokenizer(parents[0]['qwen_ids'])
  sources={f:[source_for_builtin(args.snapshot,old,n) for n in ns] for f,ns in NEEDLES.items()}
  def fits(rows):
    return all((lambda ids:0 not in ids and len(ids)<=192)(tokenizer.encode(ai.chat_text(r['question'],r['answer']))) for r in rows)
  old_code=[compact(r[field]['code']) for r in old+official for field in ('program','debug_program') if field in r]
  pool=[]
  for case in case_pool(args.seed,TRAIN_WORDS):
    rows=[make_row(case,m,sources) for m in MODES]
    if fits(rows):pool.extend(rows)
  # Diagnostic inputs are chosen without observing any checkpoint generations.
  all_code=old_code+[compact(r[field]['code']) for r in pool for field in ('program','debug_program') if field in r]
  all_qids={question_id(r['question']) for r in qa+old+official+pool}
  diagnostic=[];chosen=Counter();diagnostic_modes=('repair_inline','repair_program','generate','correct_check')
  for case in case_pool(args.diagnostic_seed,DIAG_WORDS):
    f=case['family']
    if chosen[f]>=2:continue
    rows=[make_row(case,m,sources,True) for m in diagnostic_modes]
    expr,_=render(f,case['inputs']);bad,_=wrong(f,case['inputs'],case['serial'])
    if any(compact(e) in code for e in (expr,bad) for code in all_code):continue
    if not fits(rows) or any(question_id(r['question']) in all_qids for r in rows):continue
    if lexical_audit(old+pool,rows):continue
    diagnostic.extend(rows);chosen[f]+=1
  if any(chosen[f]!=2 for f in FAMILIES):raise ValueError(f'Could not fill independent diagnostic: {chosen}')
  kept,rejected,_=screen_extension(base,old,pool,qa,official,[r['question'] for r in diagnostic])
  accepted={r['id'] for r in kept};groups=defaultdict(list)
  for r in pool:groups[r['case_id']].append(r)
  selected_new=[];per_family=Counter()
  for group in groups.values():
    f=group[0]['family']
    if per_family[f]<6 and all(r['id'] in accepted for r in group):selected_new.extend(group);per_family[f]+=1
  if any(per_family[f]!=6 for f in FAMILIES):raise ValueError(f'Insufficient balanced groups after filtering: {per_family}')
  kept,_,manifest=screen_extension(base,old,selected_new,qa,official,[r['question'] for r in diagnostic])
  replay=[r for r in old if not r.get('program')];random.Random(args.seed).shuffle(replay);replay=replay[:96]
  selected=selected_new+replay
  if lexical_audit(selected,diagnostic):raise ValueError('Diagnostic has lexical overlap with selected training')
  manifest.pop('hash');manifest['auxiliary_diagnostics']={'coding':rows_fingerprint(diagnostic),'coding_questions':[question_id(r['question']) for r in diagnostic]};manifest['hash']=digest(manifest)
  selection={'schema':1,'canonical_hash':rows_fingerprint(kept),'split_hash':manifest['hash'],
    'policy':'balanced minimal repair, correct-program controls, explicit output answers, generation and factual replay; identical for both parents',
    'ids':[r['id'] for r in selected]};selection['hash']=digest(selection)
  args.output.mkdir(parents=True)
  for name,rows in (('verified_train.jsonl',kept),('balanced_train_ready.jsonl',selected),('coding_diagnostic.jsonl',diagnostic),('rejected.jsonl',rejected)):
    write_rows(args.output/name,rows)
  (args.output/'benchmark.jsonl').write_bytes(args.benchmark.read_bytes())
  (args.output/'splits.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
  (args.output/'training_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
  summary={'training_examples':len(selected),'new_examples':len(selected_new),'factual_replay':len(replay),'canonical_examples':len(kept),
    'training_categories':dict(Counter(r.get('category','facts') for r in selected)),
    'training_modes':dict(Counter(r.get('mode','fact_replay') for r in selected)),'new_cases':dict(per_family),
    'diagnostic_examples':len(diagnostic),'diagnostic_categories':dict(Counter(r['category'] for r in diagnostic)),
    'diagnostic_cases':dict(chosen),'diagnostic_hash':rows_fingerprint(diagnostic),'split_hash':manifest['hash'],'base_split_hash':base['hash'],
    'benchmark_sha256':file_hash(args.benchmark),'parent_sha256':[file_hash(p) for p in args.parent],
    'generator_sha256':file_hash(Path(__file__)),'seed':args.seed,'diagnostic_seed':args.diagnostic_seed,
    'context_cap':192,'new_unknown_tokens':0,'overlap_policy':'exact historical-QA questions, lexical verified-training questions and complete correct/broken expressions excluded; legacy checkpoint exposure remains unknown'}
  (args.output/'build.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


def verify(args):
  import os,subprocess,tempfile,platform
  rows=read_rows(args.data/'verified_train.jsonl');diagnostic=read_rows(args.data/'coding_diagnostic.jsonl')
  for r in rows+diagnostic:
    if 'balanced_contract' in r:validate_contract(r)
  verify_programs(args)
  results=[];cache={};roots={'fprisc':str(args.snapshot/'fprisc'),'qos':str(args.snapshot/'qos')}
  for r in diagnostic:
    source_check(r,roots)
    for field in ('program','debug_program'):
      if field not in r:continue
      p=r[field];key=digest(p)
      if key not in cache:
        with tempfile.TemporaryDirectory(prefix='qos-ai-diag-reference-') as tmp:
          path=Path(tmp)/'reference.sol';path.write_text(p['code'])
          run=subprocess.run([str(args.fpr.resolve()),'sol',str(path)],cwd=tmp,env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=roots['fprisc']),text=True,capture_output=True,timeout=30)
          cache[key]={'passed':run.returncode==p['exit_code'] and run.stdout.strip()==p['stdout'].strip(),'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
      results.append({'id':r['id'],'fixture':field,**cache[key]})
  proof={'diagnostic_hash':rows_fingerprint(diagnostic),'compiler_sha256':file_hash(args.fpr),'host':platform.node(),
    'passed':sum(r['passed'] for r in results),'total':len(results),'distinct_executions':len(cache),'results':results}
  (args.data/'coding_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
  if proof['passed']!=proof['total']:raise ValueError('Diagnostic reference behavior failed')
  from qos_ai_finetune import selected_verified_rows
  selection=json.loads((args.data/'training_selection.json').read_text());manifest=json.loads((args.data/'splits.json').read_text())
  selected=selected_verified_rows(rows,manifest,selection);write_rows(args.data/'balanced_train_ready.jsonl',selected)
  print(json.dumps({k:v for k,v in proof.items() if k!='results'},indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
  b=s.add_parser('build');b.set_defaults(fn=build)
  for name in ('previous','qa','snapshot','output'):b.add_argument('--'+name,required=True,type=Path)
  b.add_argument('--benchmark',type=Path,default=Path('qos_ai_benchmark/benchmark.jsonl'));b.add_argument('--parent',type=Path,action='append',required=True)
  b.add_argument('--seed',type=int,default=137);b.add_argument('--diagnostic-seed',type=int,default=991)
  v=s.add_parser('verify');v.set_defaults(fn=verify)
  for name in ('data','snapshot','fpr'):v.add_argument('--'+name,required=True,type=Path)
  a=p.parse_args();a.fn(a)
