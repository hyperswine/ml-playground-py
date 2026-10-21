"""Broader verified Sol teaching, scalar-checked traces, and three frozen diagnostic strata."""
import argparse
import copy
import json
import os
import random
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from qos_ai_balanced import FAMILIES, NEEDLES
from qos_ai_curriculum import screen_extension, source_for_builtin, write_rows, verify as verify_programs, validate_extension
from qos_ai_data import digest, file_hash, read_rows, rows_fingerprint, question_id, source_check
from qos_ai_inputs_curriculum import atom, quoted, list_text
from qos_ai_pure_sol import Parser

TRAIN_FAMILIES=FAMILIES+('map_take','filter_reverse','clamp_sum')
HELD_COMBINATIONS=('map_filter','map_reverse','take_filter')
MODES=('output_only','output_trace','repair_expr','repair_program','no_change','generate')
PARENT_SHA='5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767'


def spec(f,p):
  """Python oracles and explicit intermediate expressions, independent of Sol execution."""
  xs=p['values'];lit=list_text(xs);n=p['factor'];k=p['count'];t=p['threshold'];text=p['text'];word=quoted(text)
  traces=[]
  def stage(name,value,expression):traces.append({'name':name,'value':value,'expression':expression});return value
  if f in ('map','map_take','map_filter','map_reverse'):
    mapped=stage('products',[x*n for x in xs],f'List.map (fn x -> x * {atom(n)}) {lit}')
    if f=='map':vals=mapped;expr=f'List.sum ({traces[-1]["expression"]})';goal=f'multiply every value in {lit} by {n} and sum the products'
    elif f=='map_take':
      vals=stage('prefix',mapped[:k],f'List.take {k} ({traces[-1]["expression"]})');expr=f'List.sum ({traces[-1]["expression"]})';goal=f'multiply {lit} by {n}, take the first {k} products, and sum them'
    elif f=='map_filter':
      vals=stage('kept',[x for x in mapped if x>=t],f'List.filter (fn x -> x >= {atom(t)}) ({traces[-1]["expression"]})');expr=f'List.sum ({traces[-1]["expression"]})';goal=f'multiply {lit} by {n}, keep products at least {t}, and sum them'
    else:
      vals=stage('reversed',mapped[::-1],f'List.rev ({traces[-1]["expression"]})');expr=f'({traces[-1]["expression"]}) ! 0';goal=f'multiply {lit} by {n}, reverse the products, and select index zero'
    value=vals[0] if f=='map_reverse' else sum(vals)
  elif f in ('filter','filter_reverse','take_filter'):
    source=lit;values=xs
    if f=='take_filter':values=stage('prefix',xs[:k],f'List.take {k} {lit}');source='('+traces[-1]['expression']+')'
    kept=stage('kept',[x for x in values if x>=t],f'List.filter (fn x -> x >= {atom(t)}) {source}')
    if f=='filter_reverse':
      rev=stage('reversed',kept[::-1],f'List.rev ({traces[-1]["expression"]})');expr=f'List.sum ({traces[-1]["expression"]})';value=sum(rev);goal=f'keep values at least {t} in {lit}, reverse them, and sum the result'
    else:expr=f'List.sum ({traces[-1]["expression"]})';value=sum(kept);goal=(f'take the first {k} values of {lit}, keep values at least {t}, and sum them' if f=='take_filter' else f'sum values at least {t} in {lit}')
  elif f in ('clamp','clamp_sum'):
    lo=p['lower'];hi=p['upper'];v=p['value'];clamp=lambda x:min(hi,max(lo,x))
    if f=='clamp':
      expr=f'Numeric.clamp {atom(lo)} {atom(hi)} {atom(v)}';value=clamp(v);goal=f'clamp {v} into the inclusive bounds [{lo},{hi}] using Numeric.clamp'
      stage('lower',lo,atom(lo));stage('upper',hi,atom(hi));stage('input',v,atom(v))
    else:
      values=stage('bounded',[clamp(x) for x in xs],f'List.map (fn x -> Numeric.clamp {atom(lo)} {atom(hi)} x) {lit}')
      expr=f'List.sum ({traces[-1]["expression"]})';value=sum(values);goal=f'clamp each value of {lit} into [{lo},{hi}] and sum the bounded values'
  elif f=='byte':
    i=p['index'];expr=f'Str.at {word} {i}';value=ord(text[i]);goal=f'read the numeric ASCII byte at zero-based offset {i} of {word} with Str.at';stage('offset',i,str(i))
  elif f=='reverse':
    vals=stage('reversed',xs[::-1],f'List.rev {lit}');expr=f'(List.rev {lit}) ! 0';value=vals[0];goal=f'reverse {lit} and select index zero'
  elif f=='slice':
    i=p['index'];end=p['end'];expr=f'Str.slice {word} {i} {end}';value=text[i:end];goal=f'take Str.slice of {word} from byte offset {i} to exclusive end offset {end}'
    stage('start',i,str(i));stage('end',end,str(end))
  elif f=='sum':expr=f'List.sum {lit}';value=sum(xs);goal=f'add every value in {lit} using List.sum';stage('values',xs,lit)
  elif f=='take':
    vals=stage('prefix',xs[:k],f'List.take {k} {lit}');expr=f'List.sum (List.take {k} {lit})';value=sum(vals);goal=f'sum the first {k} values of {lit} with List.take'
  elif f=='length':expr=f'Str.len {word}';value=len(text);goal=f'count the ASCII bytes of {word} using Str.len';stage('text',text,word)
  elif f=='min':
    a=p['a'];b=p['b'];expr=f'Numeric.min {atom(a)} {atom(b)}';value=min(a,b);goal=f'select the smaller of {a} and {b} using Numeric.min';stage('left',a,atom(a));stage('right',b,atom(b))
  elif f=='range':
    lo=p['lower'];hi=p['upper'];vals=stage('values',list(range(lo,hi+1)),f'List.range {atom(lo)} {atom(hi)}');expr=f'List.sum (List.range {atom(lo)} {atom(hi)})';value=sum(vals);goal=f'sum the inclusive List.range from {lo} through {hi}'
  elif f=='missing':
    needle=p['needle'];expr=f'Str.find {needle} {word}';value=text.find(chr(needle));goal=f'find the zero-based offset of byte {needle} in {word} with Str.find';stage('needle',needle,str(needle))
  else:raise ValueError(f)
  # A wrong-result fixture with an explicit, repairable off-by-one error. The
  # operation/inputs stay visible, rather than fabricating a compiler failure.
  bad='('+expr+') + 1' if isinstance(value,int) else expr+' + "x"'
  badvalue=value+1 if isinstance(value,int) else value+'x'
  stage('result',value,expr)
  return {'expression':expr,'value':value,'goal':goal,'traces':traces,'bad':bad,'bad_value':badvalue}


def trace_text(traces):
  return 'Steps: '+'; '.join(t['name']+'='+json.dumps(t['value'],ensure_ascii=False) for t in traces)+'.'


def make_row(case,mode,sources,wording=0,stratum='train'):
  f=case['family'];s=spec(f,case['inputs']);expr=s['expression'];value=str(s['value']);var=case['variable']
  code=f'{var} = {expr}.\n> print (str {var}).\n'
  good='```sol\n'+code+'```';broken='> print (str ('+s['bad']+')).\n';trace=trace_text(s['traces'])
  calc=' State stdout and short name=value calculation steps.'
  g=s['goal']
  if mode=='output_only':
    q=('Find the single stdout line from the supplied inputs. Reply with that line only.' if not wording else 'Without modifying the file, give exactly the text that its print writes.')+'\n'+good
    answer=value;cat='output_prediction'
  elif mode=='output_trace':
    q=('Work out the stdout of this Sol file.' if not wording else 'Trace this hosted script using its actual data and determine what reaches stdout.')+calc+'\n'+good
    answer=f'It prints {value}. '+trace;cat='output_prediction'
  elif mode in ('repair_expr','repair_program'):
    q=(f'Fix the extra +1 or appended x. Intended task: {g}.' if not wording else
       f'Undo the unnecessary result adjustment. Preserve the inputs and meet this specification: {g}.')
    q+=(' Return the fixed Sol file.' if mode=='repair_program' else ' Give the fixed expression in backticks.')+calc+'\n```sol\n'+broken+'```'
    answer=(good if mode=='repair_program' else '`'+expr+'`')+f'\nIt prints {value}. '+trace;cat='debugging'
  elif mode=='no_change':
    q=(f'Test this implementation against the requirement: {g}. If it works, say No change needed.' if not wording else
       f'Decide whether the file below already satisfies this specification: {g}. Confirm correct code with No change needed.')+calc+'\n'+good
    answer=f'No change needed. It prints {value}. '+trace;cat='no_change'
  elif mode=='generate':
    q=(f'Build a runnable Sol file for this computation: {g}. Bind the answer to {var} and print it.' if not wording else
       f'Implement this requirement in Sol, preserving every input: {g}. Use a binding named {var} and explicit printing.')+calc
    answer=good+f'\nIt prints {value}. '+trace;cat='code_generation'
  else:raise ValueError(mode)
  p={'code':code,'stdout':value+'\n','exit_code':0}
  # Batch all scalar intermediate assertions into one authored fixture. Lists
  # are checked element by element, avoiding reliance on list pretty-printing.
  checks=[];outputs=[]
  for t in s['traces']:
    vals=t['value'] if isinstance(t['value'],list) else [t['value']]
    for i,v in enumerate(vals):
      e=f'({t["expression"]}) ! {i}' if isinstance(t['value'],list) else t['expression']
      checks.append(f'> print (str ({e})).');outputs.append(str(v))
    if isinstance(t['value'],list):checks.append(f'> print (str (List.len ({t["expression"]}))).');outputs.append(str(len(vals)))
  row={'id':f'general-{stratum}-{case["id"]}-{mode}-w{wording}','path':'fprisc/compiler/Sol/Preamble.hs',
    'chunk':0,'style':'verified','round':0,'category':cat,'family':f,'mode':mode,'stratum':stratum,
    'case_id':case['id'],'question':q,'answer':answer,'expression':expr,'sources':sources[f],
    'program':p,'trace_program':{'code':'\n'.join(checks)+'\n','stdout':'\n'.join(outputs)+'\n','exit_code':0},
    'traces':s['traces'],'generalization_contract':{'case':copy.deepcopy(case),'mode':mode,'wording':wording,'stratum':stratum},
    'verification':'reconstructed Python oracle and scalar-by-scalar M1 Sol intermediate/final executions'}
  if cat=='debugging':row['debug_program']={'code':broken,'stdout':str(s['bad_value'])+'\n','exit_code':0}
  return row


def validate_contract(row):
  c=row['generalization_contract'];rebuilt=make_row(c['case'],c['mode'],{row['family']:row['sources']},c['wording'],c['stratum'])
  if rebuilt!=row:raise ValueError('Generalization input/answer/intermediate contract differs')


def case_pool(f,seed):
  rng=random.Random(f'{f}:{seed}')
  for serial in range(20000):
    length=2+serial%4;xs=[rng.randrange(-24,41) for _ in range(length)]
    text=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3+serial%6))
    lo=rng.randrange(-15,5);hi=lo+rng.randrange(1,9);i=rng.randrange(len(text)-1)
    p={'values':xs,'factor':rng.choice([-3,-2,2,3,4]),'count':serial%(length+1),
       'threshold':rng.randrange(-10,21),'text':text,'index':i,'end':rng.randrange(i+1,len(text)+1),
       'lower':lo,'upper':hi,'value':rng.choice([lo-2,lo,hi,hi+3]),'a':rng.randrange(-30,40),'b':rng.randrange(-30,40),
       'needle':ord(text[serial%len(text)]) if serial%2 else 126}
    if f=='range':
      p['lower']=rng.randrange(-45,46);p['upper']=p['lower']+serial%6
    yield {'id':f'{seed}-{f}-{serial}','family':f,'inputs':p,'variable':('result','total','value','answer')[serial%4]}


def code_key(row,field='program'):return repr(Parser(row[field]['code']).program())


def source_table(snapshot,old):
  needles=copy.deepcopy(NEEDLES)
  for f in TRAIN_FAMILIES+HELD_COMBINATIONS:
    if f not in needles:
      names={'map_take':('map','take'),'filter_reverse':('filter','reverse'),'clamp_sum':('clamp','map'),
             'map_filter':('map','filter'),'map_reverse':('map','reverse'),'take_filter':('take','filter')}[f]
      needles[f]=list(dict.fromkeys(n for name in names for n in NEEDLES[name]))
  return {f:[source_for_builtin(snapshot,old,n) for n in ns] for f,ns in needles.items()}


def trace_grade(row,answer):
  # Strict canonical field accuracy is a diagnostic, not a semantic prose judge.
  matches={};ok=True
  prose=re.sub(r'```.*?```','',answer,flags=re.S)
  for t in row['traces']:
    found=re.findall(r'\b'+re.escape(t['name'])+r'\s*=\s*(\[[^\]\n]*\]|"[^"\n]*"|-?\d+)',' '+prose)
    try:values=[json.loads(v) for v in found]
    except ValueError:values=[]
    match=len(values)==1 and type(values[0]) is type(t['value']) and values[0]==t['value']
    matches[t['name']]=match;ok &= match
  return {'canonical_trace_match':bool(ok),'trace_fields':matches}


def build(args):
  import torch
  import qos_ai as ai
  if args.output.exists():raise ValueError('Choose a fresh output directory')
  old=read_rows(args.previous/'verified_train.jsonl');base=json.loads((args.previous/'splits.json').read_text())
  bench=read_rows(args.previous/'benchmark.jsonl');old_diag=read_rows(args.previous/'coding_diagnostic.jsonl');qa=read_rows(args.qa)
  if file_hash(args.parent)!=PARENT_SHA:raise ValueError('Wrong balanced starting checkpoint')
  parent=torch.load(args.parent,weights_only=True,map_location='cpu',mmap=True);tok=ai.StudentTokenizer(parent['qwen_ids'])
  sources=source_table(args.snapshot,old)
  def fits(row):
    ids=tok.encode(ai.chat_text(row['question'],row['answer']));return 0 not in ids and len(ids)<=192
  # Treat all historical canonical/benchmark programs and gallery prompts as
  # protected. The new wording stratum intentionally reuses new-training cases.
  forbidden={code_key(r,field) for r in old+bench+old_diag for field in ('program','debug_program') if field in r and _supported(r,field)}
  candidates=[];seen=set(forbidden);counts=Counter()
  for f in TRAIN_FAMILIES:
    for case in case_pool(f,419):
      mode=MODES[counts[f]%len(MODES)];row=make_row(case,mode,sources)
      if code_key(row) in seen or not fits(row):continue
      candidates.append(row);seen.add(code_key(row));counts[f]+=1
      if counts[f]>=args.per_family+(160 if f=='length' else 32):break
  # Freeze genuinely novel input and composition cases before screening training.
  diagnostic=[];diag_seen=set(seen);diag_seen.update(forbidden)
  for stratum,families,n in [('inputs',FAMILIES,4),('combinations',HELD_COMBINATIONS,16)]:
    serial=0
    for f in families:
      added=0
      for case in case_pool(f,941):
        mode=MODES[serial%6];row=make_row(case,mode,sources,stratum=stratum)
        if code_key(row) in diag_seen or not fits(row):continue
        diagnostic.append(row);diag_seen.add(code_key(row));added+=1;serial+=1
        if added==n:break
      if added!=n:raise ValueError('Incomplete diagnostic '+f)
  # Novel-input diagnostics deliberately reuse training wording. Protect their
  # complete programs above, rather than rejecting shared task templates as
  # lexical leakage. Historical benchmark/diagnostic screens remain unchanged.
  protected=[r['question'] for r in old_diag]
  gallery=Path('qos_ai_samples/2026-10-18/prompts.jsonl')
  if gallery.exists():protected += [r['question'] for r in read_rows(gallery)]
  canonical,rejected,_=screen_extension(base,old,candidates,qa,bench,protected)
  accepted={r['id'] for r in canonical};selected=[];counts=Counter()
  for r in candidates:
    if r['id'] in accepted and counts[r['family']]<args.per_family:selected.append(r);counts[r['family']]+=1
  if any(counts[f]!=args.per_family for f in TRAIN_FAMILIES):raise ValueError('Insufficient accepted cases '+str(counts))
  # Wording-only items share the exact inputs, mode, program and trace with a
  # selected training case. This deliberate overlap is labelled explicitly.
  paired=set()
  for fi,f in enumerate(FAMILIES):
    for j in range(4):
      wanted=MODES[(fi*4+j)%6]
      # Historical protection can remove a whole template in one small family.
      # Fill that response-mode slot from another family without weakening it.
      options=[r for family in (f,)+tuple(x for x in FAMILIES if x!=f) for r in selected if r['family']==family and r['mode']==wanted and r['id'] not in paired]
      for source in options:
        c=source['generalization_contract'];row=make_row(c['case'],wanted,sources,1,'wording')
        if fits(row):break
      else:raise ValueError('No vocabulary/context-safe wording pair '+wanted)
      row['training_pair_id']=source['id'];diagnostic.append(row);paired.add(source['id'])
  # Contract reconstruction permits only the explicit pairing metadata.
  replay_facts=[r for r in old if not r.get('program') and fits(r)]
  replay_code=[r for r in old if r.get('program') and fits(r)]
  random.Random(419).shuffle(replay_facts);random.Random(420).shuffle(replay_code)
  replay=replay_facts[:96]+replay_code[:288]
  if len(replay)!=384:raise ValueError('Replay unavailable')
  # Only append the selected new cases. Intermediate candidates are not taught.
  canonical,_,manifest=screen_extension(base,old,selected,qa,bench,protected)
  if len(canonical)!=len(old)+len(selected):raise ValueError('Selection changed on final screening')
  manifest.pop('hash');manifest['auxiliary_diagnostics']=copy.deepcopy(base.get('auxiliary_diagnostics',{}))
  manifest['auxiliary_diagnostics']['generalization']={'hash':rows_fingerprint(diagnostic),'questions':[question_id(r['question']) for r in diagnostic],
    'scope':'wording intentionally shares new training cases; inputs/combinations are protected novel programs; legacy ancestry unknown'}
  manifest['hash']=digest(manifest);validate_extension(base,manifest)
  selection={'schema':1,'canonical_hash':rows_fingerprint(canonical),'split_hash':manifest['hash'],'policy':'2400 distinct broader cases with scalar-verified traces plus 384 replay; one epoch, identical data for two seeds','ids':[r['id'] for r in selected+replay]};selection['hash']=digest(selection)
  args.output.mkdir(parents=True)
  for name,rows in [('verified_train.jsonl',canonical),('balanced_train_ready.jsonl',selected+replay),('generalization_diagnostic.jsonl',diagnostic),('rejected.jsonl',rejected)]:write_rows(args.output/name,rows)
  for name in ('benchmark.jsonl','coding_diagnostic.jsonl','coding_verification.json'):(args.output/name).write_bytes((args.previous/name).read_bytes())
  for name,data in [('splits.json',manifest),('training_selection.json',selection)]:
    (args.output/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
  summary={'new_cases':len(selected),'training_examples':len(selected)+len(replay),'families':dict(counts),'modes':dict(Counter(r['mode'] for r in selected)),
    'strata':dict(Counter(r['stratum'] for r in diagnostic)),'generalization_hash':rows_fingerprint(diagnostic),'diagnostic_hash':rows_fingerprint(old_diag),'parent_sha256':PARENT_SHA,
    'base_split_hash':base['hash'],'split_hash':manifest['hash'],'benchmark_sha256':file_hash(args.output/'benchmark.jsonl'),
    'context_cap':192,'unknown_tokens':0,'generator_sha256':file_hash(Path(__file__)),'old_diagnostic_hash':rows_fingerprint(old_diag)}
  (args.output/'build.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


def _supported(row,field):
  try:Parser(row[field]['code']).program();return True
  except ValueError:return False


def check_row(row):
  plain={k:v for k,v in row.items() if k!='training_pair_id'};validate_contract(plain)


def verify(args):
  from qos_ai_finetune import selected_verified_rows
  canonical=read_rows(args.data/'verified_train.jsonl');diag=read_rows(args.data/'generalization_diagnostic.jsonl')
  selected=selected_verified_rows(canonical,json.loads((args.data/'splits.json').read_text()),json.loads((args.data/'training_selection.json').read_text()))
  for row in canonical+diag:
    if 'generalization_contract' in row:check_row(row)
  roots={n:str((args.snapshot/n).resolve()) for n in ('fprisc','qos')}
  # All authored fixtures are pure, bounded source generated by this module or
  # inherited from the verified curriculum. Parallelize their independent runs;
  # never feed model-generated programs into this reference verifier.
  from concurrent.futures import ThreadPoolExecutor,as_completed
  from qos_ai_data import validate_manifest,row_split
  rows=canonical+diag;fixtures={}
  for row in rows+read_rows(args.data/'benchmark.jsonl'):source_check(row,roots)
  for row in rows:
    for field in ('program','debug_program','trace_program'):
      if field in row:fixtures.setdefault(digest(row[field]),row[field])
  def execute(item):
    key,p=item
    with tempfile.TemporaryDirectory(prefix='qos-ai-general-ref-') as tmp:
      path=Path(tmp)/'fixture.sol';path.write_text(p['code'])
      run=subprocess.run([str(args.fpr.resolve()),'sol',str(path)],cwd=tmp,env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=roots['fprisc']),capture_output=True,text=True,timeout=20)
      result={'passed':run.returncode==p['exit_code'] and run.stdout.strip()==p['stdout'].strip(),'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr[:2000]}
      return key,result
  cache={}
  with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(execute,item) for item in fixtures.items()]
    for future in as_completed(futures):
      key,result=future.result();cache[key]=result
      if len(cache)%200==0:print('Distinct reference executions',len(cache),'/',len(fixtures),flush=True)
  results=[];canonical_results=[]
  canonical_ids={r['id'] for r in canonical}
  for row in rows:
    for field in ('program','debug_program','trace_program'):
      if field not in row:continue
      result={'id':row['id'],'fixture':field,**cache[digest(row[field])]};results.append(result)
      if not result['passed']:raise ValueError('Reference mismatch: '+json.dumps(result))
      if row['id'] in canonical_ids and field!='trace_program':
        canonical_results.append({**result,'id':row['id']+(':broken' if field=='debug_program' else '')})
  proof={'diagnostic_hash':rows_fingerprint(diag),'new_cases':sum('generalization_contract' in r for r in canonical),'compiler_sha256':file_hash(args.fpr),
    'passed':sum(r['passed'] for r in results),'total':len(results),'distinct_executions':len(cache),'results':results}
  (args.data/'generalization_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
  canonical_proof={'verified_data_hash':rows_fingerprint(canonical),'benchmark_data_hash':rows_fingerprint(read_rows(args.data/'benchmark.jsonl')),
    'fpr_sha256':file_hash(args.fpr),'passed':sum(r['passed'] for r in canonical_results),'total':len(canonical_results),
    'distinct_executions':len({digest(r[field]) for r in canonical for field in ('program','debug_program') if field in r}),'results':canonical_results}
  (args.data/'verification.json').write_text(json.dumps(canonical_proof,indent=2)+'\n')
  manifest=json.loads((args.data/'splits.json').read_text());validate_manifest(manifest,verified=canonical,benchmark=read_rows(args.data/'benchmark.jsonl'))
  if any(row_split(r,manifest)!='train' for r in canonical):raise ValueError('Non-training canonical row')
  write_rows(args.data/'balanced_train_ready.jsonl',selected)
  for row in diag:
    if row['mode']!='output_only' and not trace_grade(row,row['answer'])['canonical_trace_match']:raise ValueError('Authored trace grading failed')
  print(json.dumps({k:v for k,v in proof.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
  b=s.add_parser('build');b.set_defaults(fn=build)
  for name in ('previous','parent','snapshot','qa','output'):b.add_argument('--'+name,type=Path,required=True)
  b.add_argument('--per-family',type=int,default=160)
  v=s.add_parser('verify');v.set_defaults(fn=verify)
  for name in ('data','snapshot','fpr'):v.add_argument('--'+name,type=Path,required=True)
  a=p.parse_args();a.fn(a)
