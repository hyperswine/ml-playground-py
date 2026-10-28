"""Build compiler-verified steps-first curricula and token-matched, ordered pilots.

Never modifies frozen evaluation files. Historical replay excludes matching development
programs/cases; new rows use only the previous training families (0-5), validation 6.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
from qos_ai_qwen_repair import read,write,digest,normal,make_row,FAMILIES


def trace_spec(family,inputs):
  xs,k,limit=inputs;st=[]
  def stage(name,value,expr):
    st.append({'name':name,'value':value,'expression':expr});return value
  if family==0:
    keep=stage('kept',[x for x in xs if x>limit],'List.filter (fn x -> x > limit) xs')
    mapped=stage('adjusted',[x-k for x in keep],'List.map (fn x -> x - k) (List.filter (fn x -> x > limit) xs)')
    result=sum(mapped)
  elif family==1:
    keep=stage('kept',[x for x in xs if x<=limit],'List.filter (fn x -> x <= limit) xs')
    rev=stage('reversed',keep[::-1],'List.rev (List.filter (fn x -> x <= limit) xs)')
    take=stage('prefix',rev[:2],'List.take 2 (List.rev (List.filter (fn x -> x <= limit) xs))');result=sum(take)
  elif family==2:
    drop=stage('remaining',xs[2:],'List.drop 2 xs')
    vals=stage('adjusted',[x-limit for x in drop],'List.map (fn x -> x - limit) (List.drop 2 xs)')
    total=stage('total',sum(vals),'List.sum (List.map (fn x -> x - limit) (List.drop 2 xs))');result=abs(total)
  elif family==3:
    stage('positive_k',int(k>0),'if k > 0 then 1 else 0')
    expr='List.map (fn x -> x * k) xs' if k>0 else 'List.map (fn x -> x - limit) xs'
    vals=stage('adjusted',[x*k if k>0 else x-limit for x in xs],expr);result=sum(vals)
  elif family==4:
    vals=stage('shifted',[x+2 for x in xs],'List.map (fn x -> x + 2) xs')
    keep=stage('kept',[x for x in vals if x!=k],'List.filter (fn x -> x != k) (List.map (fn x -> x + 2) xs)')
    total=stage('total',sum(keep),'List.sum (List.filter (fn x -> x != k) (List.map (fn x -> x + 2) xs))');result=max(limit,total)
  elif family==5:
    rev=stage('reversed',xs[::-1],'List.rev xs')
    stage('shifted',[x+k for x in rev],'List.map (fn x -> x + k) (List.rev xs)')
    vals=stage('bounded',[min(7,max(0,x+k)) for x in rev],'List.map (fn x -> Numeric.clamp 0 7 (x + k)) (List.rev xs)');result=sum(vals)
  elif family==6:
    take=stage('prefix',xs[:3],'List.take 3 xs')
    stage('adjusted',[x-k for x in take],'List.map (fn x -> x - k) (List.take 3 xs)')
    vals=stage('absolute',[abs(x-k) for x in take],'List.map (fn x -> Numeric.abs (x - k)) (List.take 3 xs)');result=sum(vals)
  else:raise ValueError('only training and validation families may be traced')
  stage('result',result,FAMILIES[family][0])
  assert result==FAMILIES[family][3](*inputs)
  return st


def trace_text(traces):
  return 'Steps: '+'; '.join(t['name']+'='+json.dumps(t['value']) for t in traces)+'.'


def teach(row):
  row=dict(row);traces=trace_spec(row['family'],row['inputs'])
  # The supplied bindings and complete reference program are retained byte-for-byte.
  row['question']+=' Show short name=value calculation steps before the final output.'
  prefix='No change needed.\n' if row['kind']=='nochange' else ''
  code=f"```sol\n{row['code']}\n```\n" if row['kind']!='output' else ''
  row.update(answer=prefix+code+trace_text(traces)+f"\nIt prints {row['expected']}.",traces=traces,origin='new')
  return row


def execute(code,fpr):
  # Only trusted, rebuilt training references enter this helper, never model answers.
  with tempfile.TemporaryDirectory(prefix='qos-curriculum-') as d:
    p=Path(d)/'reference.sol';p.write_text(code+'\n')
    env={**os.environ,'SOL_JIT':'0','SOL_GPU':'0','SOL_CACHE':'0','SOL_CACHE_DIR':d+'/cache'}
    r=subprocess.run([str(Path(fpr).resolve()),'sol',str(p)],cwd=d,env=env,capture_output=True,text=True,timeout=8)
    if r.returncode:raise ValueError(r.stderr[:500])
    return r.stdout.rstrip('\n')


def verify_new(row,fpr):
  xs,k,limit=row['inputs'];bindings=f'xs = {xs}.\nk = {k}.\nlimit = {limit}.\n'
  for t in row['traces']:
    got=execute(bindings+f"> print (str ({t['expression']})).",fpr)
    if got!=str(t['value']):raise ValueError((row['id'],t,got))
  if execute(row['code'],fpr)!=row['expected']:raise ValueError(row['id'])


def ordered_schedule(data,tokenizer,budgets,seed,mixed):
  """Complete-example budgets matched to within 1%; no target truncation/padding loss."""
  rng=random.Random(seed);kinds=('output','repair','nochange','generate')
  pools={};lengths={}
  for row in data:
    lengths[row['id']]=len(tokenizer.encode(row['answer']+'<|im_end|>',add_special_tokens=False))
    pools.setdefault((row['origin'],row['kind']),[]).append(row)
  for pool in pools.values():rng.shuffle(pool)
  # Draw without replacement. Final pair is length-selected to match the token budget.
  result=[];used=set();total=0;milestones=[];pair=0
  for budget in budgets:
    while total<budget:
      origins=('replay','new') if mixed else ('replay','replay')
      keys=[(origin,kinds[(2*pair+j)%4]) for j,origin in enumerate(origins)]
      # Rotate which categories receive new examples on alternating four-row blocks.
      if mixed and pair%4>=2: keys=[('new',keys[0][1]),('replay',keys[1][1])]
      choices=[[r for r in pools[key] if r['id'] not in used] for key in keys]
      if any(not c for c in choices):raise ValueError('not enough unique rows for schedule')
      remaining=budget-total
      if remaining<500:
        pairs=[(abs(lengths[a['id']]+lengths[b['id']]-remaining),i+j,a,b)
               for i,a in enumerate(choices[0]) for j,b in enumerate(choices[1]) if a['id']!=b['id']]
        _,_,a,b=min(pairs,key=lambda x:(x[0],x[1]))
      else:a,b=choices[0][0],choices[1][0]
      delta=lengths[a['id']]+lengths[b['id']]
      if abs(total-budget)<abs(total+delta-budget) and total>budget*.99:break
      result.extend([a,b]);used.update((a['id'],b['id']));total+=delta;pair+=1
      if abs(total-budget)<=budget*.01:break
    if abs(total-budget)>budget*.01:raise ValueError(('unmatched budget',budget,total))
    milestones.append({'budget':budget,'actual_tokens':total,'micro_batches':len(result),'optimizer_updates':len(result)//2,
      'unique_rows':len(used),'by_origin':dict(Counter(r['origin'] for r in result)),
      'by_kind':dict(Counter(r['kind'] for r in result))})
  return result,milestones


def build(a):
  from transformers import AutoTokenizer
  from qos_ai_generalization import validate_contract
  from qos_ai_qwen_sft import steps_first
  out=Path(a.output)
  if out.exists():raise ValueError('refusing to overwrite a frozen build')
  out.mkdir(parents=True)
  old=Path(a.previous);historical=Path(a.historical)
  dev=read(old/'prepared_v3/dev_rows.jsonl')
  # No final answers are generated or inspected. The existing final row IDs/questions
  # are included only in deterministic overlap exclusion.
  reserved=dev+read(old/'prepared_v3/final_rows.jsonl')
  frozen=read(historical/'generalization_diagnostic.jsonl')+read(historical/'coding_diagnostic.jsonl')+read(historical/'benchmark.jsonl')
  forbidden_questions={normal(r['question']) for r in reserved+frozen}
  forbidden_codes={normal(r.get('code',r.get('program',{}).get('code',''))) for r in reserved+frozen}
  forbidden_cases={r.get('case_id') for r in frozen if r.get('case_id')}
  rows=read(historical/'verified_train.jsonl');random.Random(a.seed).shuffle(rows)
  replay=[];rejected=[];counts=Counter()
  for row in rows:
    if 'generalization_contract' not in row:continue
    kind={'output_only':'output','output_trace':'output','repair_expr':'repair','repair_program':'repair','no_change':'nochange','generate':'generate'}[row['mode']]
    if counts[kind]>=60:continue
    if normal(row['question']) in forbidden_questions or normal(row['program']['code']) in forbidden_codes or row['case_id'] in forbidden_cases:continue
    validate_contract(row)
    try:
      # Existing scalar-by-scalar trace program also verifies list lengths.
      for key in ('program','trace_program'):
        if execute(row[key]['code'],a.fpr)!=row[key]['stdout'].rstrip('\n'):raise ValueError(key)
    except (ValueError,subprocess.TimeoutExpired) as e:
      rejected.append({'id':row['id'],'reason':str(e)[:300]});continue
    answer=steps_first(row['answer']) or row['answer']
    replay.append({**row,'id':'replay/'+row['id'],'answer':answer,'kind':kind,'origin':'replay','split':'train'})
    counts[kind]+=1
    if len(replay)==240:break
  if len(replay)!=240:raise ValueError(('insufficient verified replay',counts))
  new=[];valid=[];rng=random.Random(a.seed)
  for split,families,n in [('train',range(6),48),('valid',[6],16)]:
    for family in families:
      for i in range(n):
        # make_row's validation mode cycle provides equal output/repair/no-change/generate.
        row=make_row('valid',family,i,rng)
        row.update(id=f'{split}-trace-{family:02}-{i:03}',split=split)
        row=teach(row);verify_new(row,a.fpr)
        (new if split=='train' else valid).append(row)
  alltrain=replay+new
  for row in alltrain:
    if normal(row['question']) in forbidden_questions:raise ValueError('evaluation question overlap')
  write(out/'replay_pool.jsonl',replay);write(out/'new_pool.jsonl',new);write(out/'valid.jsonl',valid)
  tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
  report={'seed':a.seed,'compiler_sha256':digest(a.fpr),'historical_train_sha256':digest(historical/'verified_train.jsonl'),
    'policy':'coding only; all intermediate calculations compiler verified; steps before output; exact question/program and historical case exclusions; no final answers',
    'replay_counts':dict(counts),'new_counts':dict(Counter(r['kind'] for r in new)),
    'verified_rows':len(alltrain)+len(valid),'replay_rejections':rejected,'schedules':{}}
  for label,data,mixed in [('replay',replay,False),('mixed',alltrain,True)]:
    schedule,milestones=ordered_schedule(data,tok,[2048,4096,8192],a.seed,mixed)
    write(out/f'{label}_schedule.jsonl',schedule);report['schedules'][label]=milestones
  report['sha256']={p.name:digest(p) for p in out.glob('*.jsonl')}
  (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('previous','historical','fpr','tokenizer','output'):p.add_argument('--'+name,required=True)
  p.add_argument('--seed',type=int,default=20261028)
  build(p.parse_args())
