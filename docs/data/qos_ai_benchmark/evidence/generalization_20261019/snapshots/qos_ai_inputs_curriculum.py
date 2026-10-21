"""Verified contrast pairs for using supplied inputs in Sol generation and repairs."""
import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from qos_ai_curriculum import screen_extension, source_for_builtin, write_rows, verify as verify_programs
from qos_ai_data import digest, file_hash, read_rows, rows_fingerprint, row_split, validate_manifest

FAMILIES=('map','clamp','byte','reverse','slice','filter','sum','take')


def atom(n):return f'({n})' if n<0 else str(n)
def list_text(xs):return '['+', '.join(atom(x) for x in xs)+']'
def quoted(s):return json.dumps(s)


def render(family, inputs):
  """Render the requested expression and independently calculate its result."""
  p=inputs
  if family=='map':return f'List.sum (List.map (fn x -> x * {p["factor"]}) {list_text(p["values"])})',str(sum(v*p['factor'] for v in p['values']))
  if family=='clamp':return f'Numeric.clamp {atom(p["lower"])} {atom(p["upper"])} {atom(p["value"])}',str(min(p['upper'],max(p['lower'],p['value'])))
  if family=='byte':return f'Str.at {quoted(p["text"])} 0',str(ord(p['text'][0]))
  if family=='reverse':return f'(List.rev {list_text(p["values"])}) ! 0',str(p['values'][-1])
  if family=='slice':return f'Str.slice {quoted(p["text"])} {p["start"]} {p["end"]}',p['text'][p['start']:p['end']]
  if family=='filter':return f'List.sum (List.filter (fn x -> x >= {p["threshold"]}) {list_text(p["values"])})',str(sum(v for v in p['values'] if v>=p['threshold']))
  if family=='sum':return f'List.sum {list_text(p["values"])}',str(sum(p['values']))
  if family=='take':return f'List.sum (List.take {p["count"]} {list_text(p["values"])})',str(sum(p['values'][:p['count']]))
  raise ValueError('Unknown input family')


def goal(family,p,wording):
  xs=list_text(p.get('values',[]))
  if family=='map':
    factor=p['factor'];verb={2:'double',3:'triple',4:'quadruple'}.get(factor,f'multiply by {factor}')
    return (f'multiply each entry of {xs} by {factor}, sum those products, and print the total' if wording%2 or factor>4 else f'{verb} every value in {xs}, add the transformed values, and print the sum')
  if family=='clamp':return f'clamp the value {p["value"]} into the inclusive range {p["lower"]} through {p["upper"]} and print the bounded value'
  if family=='byte':return f'read the first byte of {quoted(p["text"])} with Str.at and print its numeric code'
  if family=='reverse':return f'reverse the list {xs}, select its first element, and print that element'
  if family=='slice':return f'print the slice of {quoted(p["text"])} starting at byte offset {p["start"]} and ending before offset {p["end"]}'
  if family=='filter':return f'keep entries of {xs} that are at least {p["threshold"]}, sum the kept entries, and print the sum'
  if family=='sum':return f'add every entry of {xs} exactly once and print their sum'
  if family=='take':return f'take the first {p["count"]} entries of {xs}, sum that prefix, and print the sum'
  raise ValueError('Unknown input family')


def contrast_cases(seed=83):
  rng=random.Random(seed);cases=[]
  for family in FAMILIES:
    for group in range(8):
      xs=rng.sample(range(2,20),3 if family=='reverse' or group%2 else 2)
      if family=='map':
        a={'values':xs,'factor':2+group%4};b=copy.deepcopy(a)
        if group%2:b['factor']+=1
        else:b['values'][-1]+=1
      elif family=='clamp':
        lo=rng.randrange(-7,5);hi=lo+rng.randrange(4,10)
        a={'lower':lo,'upper':hi,'value':lo-3 if group%3==0 else hi+4 if group%3==1 else lo+1};b=dict(a)
        field=('lower','upper','value')[group%3];b[field]+=1
      elif family=='byte':
        text=chr(67+group*2)+str(group)+'rQ'
        a={'text':text};b={'text':chr(ord(text[0])+1)+text[1:]}
      elif family=='reverse':
        a={'values':xs};b=copy.deepcopy(a);b['values'][-1]+=1
      elif family=='slice':
        text=('lantern','orchard','gateway','library','crimson','puzzles','station','scarlet')[group];start=group%2;end=start+2
        a={'text':text,'start':start,'end':end};b=dict(a)
        if group%2:b['end']+=1
        else:b['text']=text[:start]+chr(ord(text[start])+1)+text[start+1:]
      elif family=='filter':
        t=3+group;a={'values':[t-2,t,t+1,t+3],'threshold':t};b=copy.deepcopy(a);b['threshold']+=1
      elif family=='sum':
        a={'values':xs};b=copy.deepcopy(a);b['values'][-1]+=2
      elif family=='take':
        a={'values':xs+[23],'count':1};b=copy.deepcopy(a);b['count']=2
      assert sum(a[k]!=b[k] for k in a)==1
      assert render(family,a)[1]!=render(family,b)[1]
      for variant,p in enumerate((a,b)):
        cases.append({'family':family,'group':f'{family}-{group}','variant':variant,'inputs':p,'wording':group})
  return cases


def mistakes(family,p):
  good,value=render(family,p)
  other=copy.deepcopy(p)
  if family=='map':
    other['factor']+=1;wrong=render(family,other)
    other=copy.deepcopy(p);other['values'].append(p['values'][-1]);copied=render(family,other)
    reason='Use the multiplier in the request, not a multiplier from a different example.'
  elif family=='clamp':
    wrong=(f'Numeric.clamp {atom(p["lower"])} {atom(p["upper"])}','<fn Numeric.clamp/1>')
    other['value']=p['upper']+3 if value==str(p['lower']) else p['lower']-3;copied=render(family,other)
    reason='Numeric.clamp needs all three arguments: lower bound, upper bound, then the supplied value.'
  elif family=='byte':
    wrong=(f'Str.at {quoted(p["text"])} 1',str(ord(p['text'][1])))
    other['text']='Z'+p['text'][1:];copied=render(family,other)
    reason='The first byte has offset zero. Keep the requested string.'
  elif family=='reverse':
    wrong=(f'(List.rev {list_text(p["values"])}) ! 1',str(p['values'][-2]))
    other['values']=p['values'][:-1];copied=render(family,other)
    reason='Select index zero after reversing the complete supplied list.'
  elif family=='slice':
    other['end']+=1;wrong=render(family,other)
    other=copy.deepcopy(p);s=p['text'];i=p['start'];other['text']=s[:i]+'0'+s[i+1:];copied=render(family,other)
    reason='The final slice argument is the exclusive end offset, and the string must match the request.'
  elif family=='filter':
    wrong=(f'List.sum (List.filter (fn x -> x > {p["threshold"]}) {list_text(p["values"])})',str(sum(v for v in p['values'] if v>p['threshold'])))
    other['values'][-1]+=7;copied=render(family,other)
    reason='At least includes equality. Preserve all list entries and the supplied threshold.'
  elif family=='sum':
    other['values']=p['values'][1:];wrong=render(family,other)
    other=copy.deepcopy(p);other['values'].append(p['values'][0]);copied=render(family,other)
    reason='Use every supplied element once, without dropping or adding entries.'
  elif family=='take':
    other['count']+=1;wrong=render(family,other)
    other=copy.deepcopy(p);other['values'][0]+=1;copied=render(family,other)
    reason='Keep the requested prefix count and the original list values.'
  assert wrong[1]!=value and copied[1]!=value
  return [('operation',*wrong,reason),('substituted-input',*copied,'The program uses values that do not match the task. Copy the supplied inputs exactly.')]


def program(expr):return f'result = {expr}.\n> print (str result).\n'


def validate_input_contract(row):
  c=row['input_contract'];expr,value=render(c['family'],c['inputs'])
  required=goal(c['family'],c['inputs'],c['wording'])
  if required not in row['question']:raise ValueError('Question no longer contains the supplied input contract')
  if row['expression']!=expr or row['program']!={'code':program(expr),'stdout':value+'\n','exit_code':0}:
    raise ValueError('Reference program does not use the supplied inputs')
  blocks=re.findall(r'```sol\n(.*?)```',row['answer'],re.S)
  if blocks!=[program(expr)]:raise ValueError('Answer program differs from the verified input-preserving reference')
  if row.get('debug_program') and row['debug_program']['stdout'].strip()==value:
    raise ValueError('Broken example does not demonstrate a behavioral error')
  if row.get('debug_program'):
    kind,expr,observed,_=next(m for m in mistakes(c['family'],c['inputs']) if m[0]==row['teaching_mode'])
    if row['debug_program']!={'code':program(expr),'stdout':observed+'\n','exit_code':0} or re.findall(r'```sol\n(.*?)```',row['question'],re.S)!=[program(expr)]:
      raise ValueError('Broken program differs from the authored input mistake')


def candidates(snapshot,old,seed):
  needles={'map':['map = fn f xs','sum = fn xs'],'clamp':['clamp = fn lo hi a'],'byte':['at = fn s i'],
           'reverse':['rev = fn xs'],'slice':['slice = fn s i j'],'filter':['filter = fn p xs','sum = fn xs'],
           'sum':['sum = fn xs'],'take':['take = fn n xs','sum = fn xs']}
  sources={family:[source_for_builtin(snapshot,old,n) for n in ns] for family,ns in needles.items()}
  rows=[]
  for case in contrast_cases(seed):
    family=case['family'];p=case['inputs'];expr,value=render(family,p);task=goal(family,p,case['wording'])
    base={'path':'fprisc/compiler/Sol/Preamble.hs','chunk':0,'style':'verified','round':0,'sources':sources[family],
          'expression':expr,'family':family,'contrast_group':case['group'],'contrast_variant':case['variant'],
          'input_contract':{'family':family,'inputs':p,'wording':case['wording']},
          'program':{'code':program(expr),'stdout':value+'\n','exit_code':0},
          'verification':'Python input oracle, exact input-preserving answer, and M1 Sol execution required'}
    name=f'inputs-{case["group"]}-{case["variant"]}'
    for mode in ('direct','plan'):
      q=(f'Write a complete Sol script to {task}.' if mode=='direct' else f'Use the exact values in this task: {task}. First restate the input values briefly, then provide runnable Sol code.')
      answer=('Inputs: '+', '.join(k+'='+json.dumps(v) for k,v in p.items())+'.\n' if mode=='plan' else '')+'```sol\n'+program(expr)+'```'
      rows.append({**base,'id':name+'-'+mode,'teaching_mode':mode,'category':'code_generation','question':q,'answer':answer})
    for kind,bad,bad_value,reason in mistakes(family,p):
      code=program(bad)
      q=f'The task is to {task}. The following attempt prints {bad_value}, which violates that task. Repair it while preserving the requested values and explain the change.\n```sol\n'+code+'```'
      answer=reason+'\n```sol\n'+program(expr)+'```'
      rows.append({**base,'id':name+'-'+kind,'teaching_mode':kind,'category':'debugging','question':q,'answer':answer,
                   'debug_program':{'code':code,'stdout':bad_value+'\n','exit_code':0}})
  for row in rows:validate_input_contract(row)
  return rows


def build(args):
  if args.output.exists():raise ValueError('Use a fresh curriculum directory')
  old=read_rows(args.previous/'verified_train.jsonl');bench=read_rows(args.benchmark)
  base=json.loads((args.previous/'splits.json').read_text())
  proposed=candidates(args.snapshot,old,args.seed)
  import torch
  import qos_ai as ai
  checkpoint=torch.load(args.tokenizer_checkpoint,map_location='cpu',weights_only=True)
  tokenizer=ai.StudentTokenizer(checkpoint['qwen_ids'])
  vocabulary_blocked={r['contrast_group'] for r in proposed if 0 in tokenizer.encode(ai.chat_text(r['question'],r['answer']))}
  vocabulary_rejected=[{'id':r['id'],'question':r['question'],'reason':'contrast group contains text outside the existing student vocabulary'} for r in proposed if r['contrast_group'] in vocabulary_blocked]
  original_candidate_count=len(proposed)
  proposed=[r for r in proposed if r['contrast_group'] not in vocabulary_blocked]
  kept,rejected,manifest=screen_extension(base,old,proposed,read_rows(args.qa),bench)
  new=[r for r in kept if 'input_contract' in r]
  # Drop a whole contrast group if any variant/teaching mode was screened out.
  group_ids={}
  for r in new:group_ids.setdefault(r['contrast_group'],set()).add((r['contrast_variant'],r['teaching_mode']))
  expected={(v,m) for v in (0,1) for m in ('direct','plan','operation','substituted-input')}
  incomplete={g for g,variants in group_ids.items() if variants!=expected}
  total_candidates=original_candidate_count
  if incomplete:
    removed=[{'id':r['id'],'question':r['question'],'reason':'incomplete contrast group after screening'} for r in new if r['contrast_group'] in incomplete]
    original_rejections=rejected
    proposed=[r for r in proposed if r['contrast_group'] not in incomplete]
    kept,rejected,manifest=screen_extension(base,old,proposed,read_rows(args.qa),bench)
    rejected=original_rejections+removed
  rejected=vocabulary_rejected+rejected
  args.output.mkdir(parents=True)
  write_rows(args.output/'verified_train.jsonl',kept);write_rows(args.output/'rejected.jsonl',rejected)
  (args.output/'benchmark.jsonl').write_bytes(args.benchmark.read_bytes())
  (args.output/'splits.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
  new=[r for r in kept if 'input_contract' in r]
  replay=[r for r in old if not r.get('program')]
  selected=new+replay
  selection={'schema':1,'canonical_hash':rows_fingerprint(kept),'split_hash':manifest['hash'],
             'policy':'all new input-contract examples plus all previous source-only facts; omit older program templates from this focused pass',
             'ids':[r['id'] for r in selected]}
  selection['hash']=digest(selection)
  (args.output/'training_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
  summary={'old_examples_retained':len(old),'candidates':total_candidates,'new_examples':len(new),'rejected':len(rejected),
           'canonical_examples':len(kept),'training_examples':len(selected),'factual_replay':len(replay),
           'training_categories':dict(Counter(r.get('category','original_fact') for r in selected)),
           'contrast_groups':len({r['contrast_group'] for r in new}),'families':dict(Counter(r['family'] for r in new)),
           'base_split_hash':base['hash'],'split_hash':manifest['hash'],'benchmark_sha256':file_hash(args.benchmark),
           'generator_sha256':file_hash(Path(__file__)),'seed':args.seed}
  summary.update(tokenizer_checkpoint_sha256=file_hash(args.tokenizer_checkpoint),vocabulary_hash=digest(checkpoint['qwen_ids'].tolist()),
                 vocabulary_rejected_groups=len(vocabulary_blocked),new_task_unknown_tokens=0)
  (args.output/'build.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


def verify(args):
  rows=read_rows(args.data/'verified_train.jsonl')
  for row in rows:
    if 'input_contract' in row:validate_input_contract(row)
  verify_programs(args)
  manifest=json.loads((args.data/'splits.json').read_text());selection=json.loads((args.data/'training_selection.json').read_text())
  from qos_ai_finetune import selected_verified_rows
  selected=selected_verified_rows(rows,manifest,selection)
  write_rows(args.data/'focused_train_ready.jsonl',selected)
  print(json.dumps({'focused_prepared_examples':len(selected),'input_contracts_verified':sum('input_contract' in r for r in rows)}))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
  b=s.add_parser('build');b.set_defaults(fn=build)
  b.add_argument('--previous',required=True,type=Path);b.add_argument('--benchmark',type=Path,default=Path('qos_ai_benchmark/benchmark.jsonl'))
  b.add_argument('--qa',required=True,type=Path);b.add_argument('--snapshot',required=True,type=Path);b.add_argument('--output',required=True,type=Path);b.add_argument('--seed',type=int,default=83)
  b.add_argument('--tokenizer-checkpoint',required=True,type=Path)
  v=s.add_parser('verify');v.set_defaults(fn=verify)
  for name in ('data','snapshot','fpr'):v.add_argument('--'+name,required=True,type=Path)
  a=p.parse_args();a.fn(a)
