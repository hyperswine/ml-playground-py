"""Fresh Qwen diagnostic and compiler-verified repair curriculum (standard library only).

Split composition families BEFORE materializing inputs. Development and final families never
enter training. Old replay rows must come from a named training file and pass fresh verification.
Reference solutions are executed with a pinned compiler and checked against independent Python
oracles. Model programs use the existing pure-Sol admission check; grading never repairs answers.
"""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

from qos_ai_execute import BLOCK, run_sol, stated

# Repeated boundary probes share exact code/compiler pairs.
run_sol = lru_cache(maxsize=8192)(run_sol)

# (expression, erroneous expression, meaning, independent oracle)
FAMILIES = [
 ('List.sum (List.map (fn x -> x - k) (List.filter (fn x -> x > limit) xs))',
  'List.sum (List.map (fn x -> x + k) (List.filter (fn x -> x > limit) xs))',
  'keep values strictly above limit, subtract k from each remaining value, then sum them',
  lambda xs,k,l: sum(x-k for x in xs if x>l)),
 ('List.sum (List.take 2 (List.rev (List.filter (fn x -> x <= limit) xs)))',
  'List.sum (List.take 2 (List.filter (fn x -> x <= limit) xs))',
  'keep values at most limit, reverse their order, then sum the first two',
  lambda xs,k,l: sum(list(reversed([x for x in xs if x<=l]))[:2])),
 ('Numeric.abs (List.sum (List.map (fn x -> x - limit) (List.drop 2 xs)))',
  'List.sum (List.map (fn x -> x - limit) (List.drop 2 xs))',
  'drop the first two values, subtract limit from every remaining value, sum, then take the absolute value',
  lambda xs,k,l: abs(sum(x-l for x in xs[2:]))),
 ('if k > 0 then List.sum (List.map (fn x -> x * k) xs) else List.sum (List.map (fn x -> x - limit) xs)',
  'if k > 0 then List.sum (List.map (fn x -> x * k) xs) else List.sum xs',
  'if k is positive sum the values multiplied by k; otherwise sum the values after subtracting limit from each',
  lambda xs,k,l: sum(x*k for x in xs) if k>0 else sum(x-l for x in xs)),
 ('Numeric.max limit (List.sum (List.filter (fn x -> x != k) (List.map (fn x -> x + 2) xs)))',
  'Numeric.min limit (List.sum (List.filter (fn x -> x != k) (List.map (fn x -> x + 2) xs)))',
  'add two to each value, discard resulting values equal to k, sum, then return the larger of that sum and limit',
  lambda xs,k,l: max(l,sum(x+2 for x in xs if x+2!=k))),
 ('List.sum (List.map (fn x -> Numeric.clamp 0 7 (x + k)) (List.rev xs))',
  'List.sum (List.map (fn x -> Numeric.clamp 0 7 x) (List.rev xs))',
  'reverse xs, add k to each value, clamp each to zero through seven, and sum',
  lambda xs,k,l: sum(min(7,max(0,x+k)) for x in reversed(xs))),
 # Validation: distinct composition, not used for benchmark selection.
 ('List.sum (List.map (fn x -> Numeric.abs (x - k)) (List.take 3 xs))',
  'Numeric.abs (List.sum (List.map (fn x -> x - k) (List.take 3 xs)))',
  'take the first three values, subtract k from each, take each absolute value, and sum',
  lambda xs,k,l: sum(abs(x-k) for x in xs[:3])),
 # Development families.
 ('List.sum (List.take 3 (List.map (fn x -> x * k) (List.filter (fn x -> x >= limit) xs)))',
  'List.sum (List.take 3 (List.map (fn x -> x * k) (List.filter (fn x -> x > limit) xs)))',
  'keep values at least limit, multiply each by k, then sum the first three results',
  lambda xs,k,l: sum([x*k for x in xs if x>=l][:3])),
 ('if List.sum xs < limit then Numeric.abs (List.sum (List.map (fn x -> x - k) xs)) else Numeric.max k (List.sum (List.filter (fn x -> x <= 0) xs))',
  'if List.sum xs < limit then Numeric.abs (List.sum (List.map (fn x -> x + k) xs)) else Numeric.min k (List.sum (List.filter (fn x -> x <= 0) xs))',
  'if the sum of xs is below limit, subtract k from each value and take the absolute value of their sum; otherwise return the larger of k and the sum of nonpositive values in xs',
  lambda xs,k,l: abs(sum(x-k for x in xs)) if sum(xs)<l else max(k,sum(x for x in xs if x<=0))),
 ('List.sum (List.map (fn x -> x + limit) (List.drop 1 (List.take 4 (List.rev xs))))',
  'List.sum (List.map (fn x -> x + limit) (List.take 4 (List.drop 1 (List.rev xs))))',
  'reverse xs, take the first four, drop the first of those, add limit to each remaining value, and sum',
  lambda xs,k,l: sum(x+l for x in list(reversed(xs))[:4][1:])),
 # Final holdout families: do not inspect until candidate selection.
 ('Numeric.min limit (Numeric.abs (List.sum (List.map (fn x -> x * k) (List.filter (fn x -> x != 0) (List.drop 2 xs)))))',
  'Numeric.max limit (Numeric.abs (List.sum (List.map (fn x -> x * k) (List.filter (fn x -> x != 0) (List.drop 2 xs)))))',
  'drop the first two values, remove zeros, multiply each remaining value by k, take the absolute value of their sum, and return the smaller of that value and limit',
  lambda xs,k,l: min(l,abs(sum(x*k for x in xs[2:] if x!=0)))),
 ('List.sum (List.take 2 (List.rev (List.map (fn x -> Numeric.min limit x) (List.filter (fn x -> x >= k) xs))))',
  'List.sum (List.take 2 (List.map (fn x -> Numeric.min limit x) (List.filter (fn x -> x >= k) xs)))',
  'keep values at least k, cap each at limit, reverse, and sum the first two',
  lambda xs,k,l: sum(list(reversed([min(l,x) for x in xs if x>=k]))[:2])),
]


def read(path):
  return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write(path, rows):
  Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal(text):
  return re.sub(r'\s+', '', text)


def program(expr, case):
  xs,k,limit=case
  return f'xs = {xs}.\nk = {k}.\nlimit = {limit}.\nanswer = {expr}.\n> print (str answer).'


def bindings(code, case=None):
  """Extract/replace literal inputs without imposing one declaration per line.

  Accept parenthesized integer literals, but never evaluate expressions or calls.
  Mask strings/comments so text inside a string cannot impersonate a declaration.
  """
  found=[]
  for j,name in enumerate(('xs','k','limit')):
    masked=re.sub(r'"(?:[^"\\]|\\.)*"|#[^\n]*',lambda m:' '*len(m[0]),code)
    prefix=rf'(?m)(?:^|(?<=\.))\s*{name}\s*='
    pat=prefix+r'\s*([\[\]()\d, \t\r\n+\-]+?)\s*\.'
    matches=list(re.finditer(pat,masked))
    if len(matches)!=1 or len(re.findall(prefix,masked))!=1:
      raise ValueError(f'expected exactly one literal {name} binding')
    try: value=ast.literal_eval(matches[0][1])
    except (ValueError,SyntaxError): raise ValueError(f'invalid literal {name}') from None
    valid=(isinstance(value,list) and all(type(x) is int for x in value)) if name=='xs' else type(value) is int
    if not valid: raise ValueError(f'invalid input type for {name}')
    found.append(value)
    if case is not None:
      m=matches[0];code=code[:m.start()]+f'\n{name} = {case[j]}.'+code[m.end():]
  return code if case is not None else found


def make_row(split, family, i, rng):
  expr,wrong,meaning,oracle=FAMILIES[family]
  xs=[rng.randrange(-13,18) for _ in range(rng.randrange(0,9))]
  k,limit=rng.randrange(-5,7),rng.randrange(-5,9)
  if family==7 and i%2==0: xs=[limit,limit+1,limit-1,limit+3,limit]
  case=[xs,k,limit]
  probes=[case,[[],0,0],[[-7,0,3,3,11],-2,3],[[5,-2,0,8,5,1],4,5],[[1],0,-1],[[-5,-4,-3],-3,2],[[9,2,9,7,4,8],2,9]]
  kind=(('repair','repair','nochange','output') if split=='train' else ('repair','nochange','output','generate'))[i%4]
  good=program(expr,case); bad=program(wrong,case)
  # A repair must actually be wrong on at least one probe, even if its displayed output coincides.
  expected=str(oracle(*case))
  if kind=='output':
    question=f'Calculate the exact output of this Sol program. End with "It prints VALUE."\n```sol\n{good}\n```'
    answer=f'It prints {expected}.'
  else:
    requirement=f'Using the supplied xs, k and limit, {meaning}. Keep the supplied bindings unchanged. Return one complete Sol code block and finish with "It prints VALUE." The same computation must work for other input values.'
    if kind=='generate':
      question=f'Write a Sol program. {requirement}\nInputs: xs = {xs}; k = {k}; limit = {limit}.'
    else:
      question=f'Check this Sol implementation against the requirement: {requirement} If it already satisfies the requirement, say "No change needed" and return the unchanged complete program. Otherwise repair it.\n```sol\n{good if kind=="nochange" else bad}\n```'
    answer=('No change needed.\n' if kind=='nochange' else '')+f'```sol\n{good}\n```\nIt prints {expected}.'
  return {'id':f'{split}-{family:02}-{i:03}', 'suite':'fresh','split':split,'kind':kind,'family':family,
          'question':question,'answer':answer,'code':good,'input_code':good if kind=='nochange' else bad,
          'inputs':case,'expected':expected,'cases':[{'inputs':c,'expected':str(oracle(*c))} for c in probes],
          'max_new_tokens':320 if kind!='output' else 96}


def build(args):
  out=Path(args.output)
  if out.exists(): raise ValueError('refusing to overwrite a frozen build')
  out.mkdir(parents=True)
  rng=random.Random(2026102701)
  partitions={}
  for split,families,n in [('train',range(6),48),('valid',[6],16),('dev',range(7,10),16),('final',range(10,12),8)]:
    rows=[make_row(split,f,i,rng) for f in families for i in range(n)]
    for row in rows:
      for c in row['cases']:
        got,why=run_sol(bindings(row['code'],c['inputs']),args.fpr)
        if got!=c['expected']: raise ValueError((row['id'],c,got,why))
      if row['kind']=='repair':
        if all(run_sol(bindings(row['input_code'],c['inputs']),args.fpr)[0]==c['expected'] for c in row['cases']):
          raise ValueError(('ineffective repair',row['id']))
    partitions[split]=rows
  forbidden={normal(r['question']) for split,rs in partitions.items() if split!='train' for r in rs}
  forbidden_codes={normal(r['code']) for split,rs in partitions.items() if split!='train' for r in rs}
  replay=[]
  old=read(args.replay); rng.shuffle(old)
  for row in old:
    if normal(row['question']) in forbidden or normal(row.get('code','')) in forbidden_codes: continue
    # Only executable historical rows, reverified against this compiler, with a correct output claim.
    code=row.get('code')
    if not code or stated(row['answer'])!=row.get('expected'): continue
    got,why=run_sol(code,args.fpr)
    if got!=row['expected']: continue
    replay.append({**row,'id':'replay/'+row['id'],'split':'train','origin':'verified historical training replay'})
    if len(replay)==144: break
  if len(replay)!=144: raise ValueError('not enough verified replay')
  partitions['train']+=replay
  seen=set()
  for split,rows in partitions.items():
    questions={normal(r['question']) for r in rows}
    if seen & questions: raise ValueError('cross-split question overlap')
    seen|=questions
    write(out/f'{split}.jsonl',rows)
  manifest={'seed':2026102701,'compiler':str(Path(args.fpr).resolve()),'compiler_sha256':digest(args.fpr),
            'replay_source':str(args.replay),'replay_sha256':digest(args.replay),
            'split_policy':'disjoint new composition families; historical training replay reverified; final unopened until candidate selection',
            'counts':{s:dict(Counter(r['kind'] for r in rs)) for s,rs in partitions.items()},
            'sha256':{s:digest(out/f'{s}.jsonl') for s in partitions}}
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps(manifest,indent=2))


def score_row(row, answer, fpr):
  claim=stated(answer)
  result={'id':row['id'],'kind':row['kind'],'claim_correct':claim==row['expected'] if 'expected' in row else None}
  if row['kind']=='docs':
    # Both repository-relative and repository-prefixed file citations are valid.
    patterns=[p.replace('fprisc/', '(?:fprisc/)?', 1) if p.startswith('fprisc/') else p for p in row['required']]
    requirements=[bool(re.search(p,answer,re.I|re.S)) for p in patterns]
    forbidden=[p for p in row.get('forbidden',[]) if re.search(p,answer,re.I|re.S)]
    result.update(required_checks=requirements, forbidden_matches=forbidden,
                  rubric_proxy_pass=all(requirements) and not forbidden, correct=None, manual_review_required=True)
    return result
  if row['kind']=='output':
    result['correct']=result['claim_correct'];return result
  blocks=BLOCK.findall(answer)
  result.update(code_correct=False,input_preserved=False,case_passes=0,cases=len(row['cases']),nochange_correct=None)
  if len(blocks)!=1:
    result.update(correct=False,reason='expected one complete Sol block');return result
  code=blocks[0]
  try: result['input_preserved']=bindings(code)==row['inputs']
  except ValueError as e:
    result.update(correct=False,reason=str(e));return result
  for c in row['cases']:
    got,why=run_sol(bindings(code,c['inputs']),fpr)
    result['case_passes']+=got==c['expected']
  result['code_correct']=result['case_passes']==len(row['cases']) and result['input_preserved']
  if row['kind']=='nochange':
    result['nochange_correct']=bool(re.search(r'no change needed',answer,re.I)) and normal(code)==normal(row['code'])
  result['repair_decision_correct']=not bool(re.search(r'no change (?:is )?(?:needed|required)',answer,re.I)) if row['kind']=='repair' else None
  result['correct']=result['claim_correct'] and result['code_correct'] and result['nochange_correct'] is not False and result['repair_decision_correct'] is not False
  return result


def score(args):
  rows=read(args.rows); answers=read(args.answers)
  ids=[r['id'] for r in answers]
  if len(ids)!=len(set(ids)): raise ValueError('duplicate answers')
  by_id={r['id']:r['answer'] for r in answers}
  if set(by_id)!={r['id'] for r in rows}: raise ValueError('missing or unexpected answer ids')
  scored=[score_row(r,by_id[r['id']],args.fpr) for r in rows]
  code=[r for r in scored if r['kind']!='docs']; docs=[r for r in scored if r['kind']=='docs']
  report={'schema':2,'n':len(rows),'code_n':len(code),'code_complete':sum(r['correct'] for r in code),
          'code_programs_correct':sum(r.get('code_correct',False) for r in code),
          'code_claims_correct':sum(r['claim_correct'] for r in code),
          'documentation_n':len(docs),'documentation_rubric_proxy_pass':sum(r['rubric_proxy_pass'] for r in docs),
          'documentation_requires_manual_review':bool(docs),
          'by_kind':{k:{'n':sum(r['kind']==k for r in code),'correct':sum(r['correct'] for r in code if r['kind']==k)} for k in sorted({r['kind'] for r in code})},
          'scorer_sha256':digest(__file__),'compiler_sha256':digest(args.fpr),'rows_sha256':digest(args.rows),
          'answers_sha256':digest(args.answers),'rows':scored}
  Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
  b=sub.add_parser('build');b.add_argument('--replay',required=True)
  s=sub.add_parser('score');s.add_argument('--rows',required=True);s.add_argument('--answers',required=True)
  for c in (b,s): c.add_argument('--fpr',required=True);c.add_argument('--output',required=True)
  a=p.parse_args();{'build':build,'score':score}[a.command](a)
