"""Verified calculation steps, complete repairs and input contrasts, preserving both benchmarks."""
import argparse
import copy
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from qos_ai_balanced import FAMILIES,NEEDLES,render as basic_render,task as basic_task,wrong as basic_wrong,compact,lexical_audit
from qos_ai_inputs_curriculum import atom,quoted,list_text,program
from qos_ai_curriculum import screen_extension,source_for_builtin,write_rows,verify as verify_programs,validate_extension
from qos_ai_data import digest,file_hash,read_rows,rows_fingerprint

WORDS=('Acorn','Bison','Coral','Daisy','Eagle','Flint','Globe','Haven','Ivory','Jewel','Koala','Lilac',
 'Mango','Noble','Ocean','Pearl','Quartz','Robin','Satin','Tiger','Urban','Venus','Whale','Xenon','Yacht','Zebra',
 'acorn','bison','coral','daisy','eagle','flint','Book','Lion','Desk','Anchor','Cobalt','Copper','Lantern','Dolphin','Compass','Elephant','Sapphire')
MODES=('output_only','output_calculate','repair_inline','repair_program','correct_check','generate_complete')


def arithmetic(values):return ' + '.join(map(str,values))+' = '+str(sum(values))+'.'


def specification(family,p):
  if family=='byte':
    index=p['index'];value=str(ord(p['text'][index]));expr=f'Str.at {quoted(p["text"])} {index}'
    goal=(f'read the first byte of {quoted(p["text"])} with Str.at and print its numeric code' if index==0 else
      f'print the numeric byte code at zero-based offset {index} of {quoted(p["text"])} using Str.at')
    reason=f'At offset {index}, {quoted(p["text"][index])} has ASCII code {value}.'
    bad=f'Str.at {quoted(p["text"])} {index+1}';bad_value=str(ord(p['text'][index+1]))
  elif family=='missing':
    value=str(p['text'].find(chr(p['needle'])));expr=f'Str.find {p["needle"]} {quoted(p["text"])}'
    goal=f'print the offset returned by Str.find for byte {p["needle"]} in {quoted(p["text"])}'
    reason=(f'Byte {p["needle"]} is absent, so the returned offset is -1.' if value=='-1' else
      f'Byte {p["needle"]} first occurs at zero-based offset {value}.')
    bad=expr+' + 1';bad_value=str(int(value)+1)
  else:
    expr,value=basic_render(family,p);goal=basic_task(family,p);bad,bad_value=basic_wrong(family,p,0)
    if family=='map':reason='The products are '+list_text([v*p['factor'] for v in p['values']])+'. '+arithmetic([v*p['factor'] for v in p['values']])
    elif family=='clamp':
      reason=(f'{p["value"]} < {p["lower"]}, so choose the lower bound {p["lower"]}.' if p['value']<p['lower'] else
        f'{p["value"]} > {p["upper"]}, so choose the upper bound {p["upper"]}.' if p['value']>p['upper'] else
        f'{p["value"]} is inside the bounds, so it stays unchanged.')
    elif family=='reverse':reason='Reversing gives '+list_text(list(reversed(p['values'])))+f'; index zero is {value}.'
    elif family=='slice':reason=f'Offsets {p["start"]} through {p["end"]-1} give {quoted(value)}; the end offset is exclusive.'
    elif family=='filter':
      kept=[v for v in p['values'] if v>=p['threshold']];reason='The retained values are '+list_text(kept)+'. '+arithmetic(kept)
    elif family=='sum':reason=arithmetic(p['values'])
    elif family=='take':
      kept=p['values'][:p['count']];reason='The prefix is '+list_text(kept)+'. '+arithmetic(kept)
    elif family=='length':reason='Count the ASCII bytes: '+', '.join(p['text'])+f'; total {value}.'
    elif family=='min':reason=f'{min(p["a"],p["b"])} < {max(p["a"],p["b"])}, so the smaller value is {value}.'
    elif family=='range':reason='The inclusive values are '+list_text(list(range(p['lower'],p['upper']+1)))+'. '+arithmetic(list(range(p['lower'],p['upper']+1)))
  if value==bad_value:raise ValueError('Broken fixture must change behavior')
  return expr,value,goal,reason,bad,bad_value


def contrast_groups(seed):
  for family in FAMILIES:
    rng=random.Random(f'{seed}:{family}')
    for group in range(24):
      xs=rng.sample(range(2,45),2+group%2);text=rng.choice(WORDS)
      if family=='map':a={'values':xs,'factor':2+group%3};b={**a,'factor':a['factor']+1}
      elif family=='clamp':
        lower=rng.randrange(-6,9);upper=lower+rng.randrange(4,12)
        a={'lower':lower,'upper':upper,'value':lower-3};b={**a,'value':upper+4}
      elif family=='byte':
        index=group%3;other=rng.choice([w for w in WORDS if w[index]!=text[index]])
        a={'text':text,'index':index};b={**a,'text':other}
        if a['text'][index]==a['text'][index+1] or b['text'][index]==b['text'][index+1]:continue
      elif family in ('reverse','sum'):
        a={'values':xs};b=copy.deepcopy(a);b['values'][-1]+=3
        if family=='reverse' and b['values'][-1]==b['values'][-2]:b['values'][-1]+=1
      elif family=='slice':
        start=group%2 if len(text)>4 else 0
        a={'text':text,'start':start,'end':start+2};b={**a,'end':a['end']+1}
      elif family=='filter':
        threshold=rng.randrange(4,15);a={'values':[threshold-2,threshold,threshold+3],'threshold':threshold}
        b=copy.deepcopy(a);b['values'][-1]+=4
      elif family=='take':a={'values':xs+[47],'count':1};b={**a,'count':2}
      elif family=='length':a={'text':text};b={'text':rng.choice([w for w in WORDS if len(w)!=len(text)])}
      elif family=='min':
        low=rng.randrange(2,30);a={'a':low,'b':low+rng.randrange(2,9)};b={**a,'a':a['b']+5}
      elif family=='range':
        lower=rng.randrange(20,45);a={'lower':lower,'upper':lower+2};b={**a,'upper':lower+3}
      elif family=='missing':a={'text':text,'needle':126};b={**a,'needle':ord(text[group%len(text)])}
      if sum(a[k]!=b[k] for k in a)!=1:raise ValueError('Contrast must change one input field')
      if specification(family,a)[1]==specification(family,b)[1]:raise ValueError('Contrast must change output')
      yield [{'family':family,'inputs':p,'group':f'{seed}-{family}-{group}','variant':variant} for variant,p in enumerate((a,b))]


def make_row(case,mode,sources):
  family=case['family'];expr,value,goal,reason,bad,bad_value=specification(family,case['inputs']);good=program(expr)
  if mode=='output_only':
    question='Calculate the stdout line for this file. Answer only with that line.\n```sol\n'+good+'```';answer=value;category='output_prediction'
  elif mode=='output_calculate':
    question='Calculate this file\'s output from its actual inputs. State the output and show a short calculation.\n```sol\n'+good+'```'
    answer=f'It prints {value}. {reason}';category='output_prediction'
  elif mode=='repair_inline':
    question=f'The expression `{bad}` should {goal}. Give the corrected expression, its output and a short calculation.'
    answer=f'Use `{expr}`. It prints {value}. {reason}';category='debugging'
  elif mode=='repair_program':
    question=f'This file should {goal}. Return the repaired file, the resulting stdout and a short calculation.\n```sol\n'+program(bad)+'```'
    answer='```sol\n'+good+f'```\nIt prints {value}. {reason}';category='debugging'
  elif mode=='correct_check':
    question=f'Review this file against the goal: {goal}. If it already works, say No change needed. State its output and the calculation.\n```sol\n'+good+'```'
    answer=f'No change needed. It prints {value}. {reason}';category='no_change'
  elif mode=='generate_complete':
    question=f'Create a complete Sol file to {goal}. Also state the printed result and show a short calculation.'
    answer='```sol\n'+good+f'```\nIt prints {value}. {reason}';category='code_generation'
  else:raise ValueError('Unknown output curriculum mode')
  row={'id':f'calculation-{case["group"]}-{case["variant"]}-{mode}','path':'fprisc/compiler/Sol/Preamble.hs',
    'chunk':0,'style':'verified','round':0,'category':category,'family':family,'mode':mode,
    'contrast_group':case['group'],'contrast_variant':case['variant'],'question':question,'answer':answer,
    'expression':expr,'sources':sources[family],'calculation':reason,'program':{'code':good,'stdout':value+'\n','exit_code':0},
    'calculation_contract':copy.deepcopy(case),'verification':'Python calculation oracle, complete answer/input reconstruction, source proof and target execution'}
  if category=='debugging':row['debug_program']={'code':program(bad),'stdout':bad_value+'\n','exit_code':0}
  return row


def validate_contract(row):
  c=row['calculation_contract'];rebuilt=make_row(c,row['mode'],{c['family']:row['sources']})
  for key in ('question','answer','expression','program','debug_program','calculation','category'):
    if row.get(key)!=rebuilt.get(key):raise ValueError('Calculation/complete-answer contract differs: '+key)


def build(args):
  import torch
  import qos_ai as ai
  if args.output.exists():raise ValueError('Choose a fresh curriculum directory')
  old=read_rows(args.previous/'verified_train.jsonl');base=json.loads((args.previous/'splits.json').read_text())
  benchmark=read_rows(args.previous/'benchmark.jsonl');diagnostic=read_rows(args.previous/'coding_diagnostic.jsonl')
  qa=read_rows(args.qa);parent=torch.load(args.parent,map_location='cpu',weights_only=True)
  tokenizer=ai.StudentTokenizer(parent['qwen_ids'])
  def fits(rows):return all((lambda ids:0 not in ids and len(ids)<=192)(tokenizer.encode(ai.chat_text(r['question'],r['answer']))) for r in rows)
  sources={f:[source_for_builtin(args.snapshot,old,n) for n in ns] for f,ns in NEEDLES.items()}
  from qos_ai_pure_sol import Parser,Unsupported
  existing_code=set()
  for row in old+benchmark+diagnostic:
    for field in ('program','debug_program'):
      if field not in row:continue
      try:existing_code.add(Parser(row[field]['code']).program())
      except Unsupported:pass
  candidates=[];rejected=[]
  for pair in contrast_groups(args.seed):
    group=[make_row(case,mode,sources) for case in pair for mode in MODES]
    if not fits(group):rejected.append({'group':pair[0]['group'],'reason':'unknown token or complete answer exceeds 192 tokens'});continue
    if any(Parser(r[field]['code']).program() in existing_code for r in group for field in ('program','debug_program') if field in r):
      rejected.append({'group':pair[0]['group'],'reason':'complete reference/broken program overlaps previous data or benchmarks'});continue
    candidates.extend(group)
  kept,screened,_=screen_extension(base,old,candidates,qa,benchmark,[r['question'] for r in diagnostic])
  accepted={r['id'] for r in kept};groups=defaultdict(list)
  for row in candidates:groups[row['contrast_group']].append(row)
  selected_new=[];counts=Counter()
  for group in groups.values():
    family=group[0]['family']
    if counts[family]<4 and all(r['id'] in accepted for r in group):selected_new.extend(group);counts[family]+=1
  if any(counts[f]!=4 for f in FAMILIES):raise ValueError(f'Incomplete protected/vocabulary-safe contrast coverage: {counts}')
  canonical,_,manifest=screen_extension(base,old,selected_new,qa,benchmark,[r['question'] for r in diagnostic])
  facts=[r for r in old if not r.get('program') and fits([r])]
  generation=[r for r in old if r.get('category')=='code_generation' and fits([r])]
  random.Random(args.seed).shuffle(facts);random.Random(args.seed+1).shuffle(generation)
  if min(len(facts),len(generation))<96:raise ValueError('Insufficient replay')
  selected=selected_new+facts[:96]+generation[:96]
  if lexical_audit(selected,diagnostic):raise ValueError('Training question overlaps frozen coding diagnostic')
  if manifest['auxiliary_diagnostics']!=base['auxiliary_diagnostics']:raise ValueError('Diagnostic assignment changed')
  selection={'schema':1,'canonical_hash':rows_fingerprint(canonical),'split_hash':manifest['hash'],
    'policy':'576 new paired calculation/complete-answer examples plus 96 generation and 96 factual replay',
    'ids':[r['id'] for r in selected]};selection['hash']=digest(selection)
  args.output.mkdir(parents=True)
  for name,rows in (('verified_train.jsonl',canonical),('balanced_train_ready.jsonl',selected),('rejected.jsonl',rejected+screened)):
    write_rows(args.output/name,rows)
  for name in ('benchmark.jsonl','coding_diagnostic.jsonl','coding_verification.json'):
    (args.output/name).write_bytes((args.previous/name).read_bytes())
  for name,data in (('splits.json',manifest),('training_selection.json',selection)):
    (args.output/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
  summary={'new_examples':len(selected_new),'training_examples':len(selected),'canonical_examples':len(canonical),
    'new_cases':len(selected_new)//len(MODES),'contrast_pairs':dict(counts),'training_categories':dict(Counter(r.get('category','facts') for r in selected)),
    'training_modes':dict(Counter(r.get('mode','replay') for r in selected_new)), 'factual_replay':96,'generation_replay':96,
    'parent_sha256':file_hash(args.parent),'base_split_hash':base['hash'],'split_hash':manifest['hash'],
    'benchmark_sha256':file_hash(args.output/'benchmark.jsonl'),'diagnostic_hash':rows_fingerprint(diagnostic),
    'diagnostic_sha256':file_hash(args.output/'coding_diagnostic.jsonl'),'generator_sha256':file_hash(Path(__file__)),
    'seed':args.seed,'context_cap':192,'unknown_tokens':0,'note':'Both benchmarks are development diagnostics; legacy checkpoint exposure remains unknown.'}
  (args.output/'build.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


def verify(args):
  from qos_ai_balanced import validate_contract as validate_balanced
  from qos_ai_inputs_curriculum import validate_input_contract
  from qos_ai_finetune import selected_verified_rows
  rows=read_rows(args.data/'verified_train.jsonl')
  for row in rows:
    if 'calculation_contract' in row:validate_contract(row)
    elif 'balanced_contract' in row:validate_balanced(row)
    elif 'input_contract' in row:validate_input_contract(row)
  groups=defaultdict(list)
  for row in rows:
    if 'calculation_contract' in row and row['mode']=='output_only':groups[row['contrast_group']].append(row)
  for group in groups.values():
    if len(group)!=2 or {r['contrast_variant'] for r in group}!={0,1} or group[0]['answer']==group[1]['answer']:
      raise ValueError('Incomplete or identical-output contrast')
  verify_programs(args)
  manifest=json.loads((args.data/'splits.json').read_text());selection=json.loads((args.data/'training_selection.json').read_text())
  selected=selected_verified_rows(rows,manifest,selection);write_rows(args.data/'balanced_train_ready.jsonl',selected)
  from qos_ai_coding_diag import diagnostic
  diagnostic(args.data,args.snapshot,args.fpr)
  print(json.dumps({'complete_calculation_contracts':sum('calculation_contract' in r for r in rows),'contrast_pairs':len(groups),
    'prepared_examples':len(selected),'diagnostic_bytes_preserved':True},indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
  b=s.add_parser('build');b.set_defaults(fn=build)
  for name in ('previous','parent','snapshot','qa','output'):b.add_argument('--'+name,type=Path,required=True)
  b.add_argument('--seed',type=int,default=331)
  v=s.add_parser('verify');v.set_defaults(fn=verify)
  for name in ('data','snapshot','fpr'):v.add_argument('--'+name,type=Path,required=True)
  a=p.parse_args();a.fn(a)
