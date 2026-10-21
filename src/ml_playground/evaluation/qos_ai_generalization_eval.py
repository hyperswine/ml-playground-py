"""Frozen generalization evaluation and checkpoint learning-curve diagnostics."""
import argparse
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

from qos_ai_data import file_hash,read_rows,rows_fingerprint,source_check
from qos_ai_compare import stated_output
from qos_ai_coding_diag import grade_answer
from qos_ai_generalization import trace_grade,check_row


def checked_diagnostic(data,snapshot):
  rows=read_rows(data/'generalization_diagnostic.jsonl');build=json.loads((data/'build.json').read_text())
  proof=json.loads((data/'generalization_verification.json').read_text())
  if rows_fingerprint(rows)!=build['generalization_hash'] or proof['diagnostic_hash']!=build['generalization_hash']:raise ValueError('Frozen diagnostic differs')
  expected={(r['id'],field) for r in rows for field in ('program','trace_program','debug_program') if field in r}
  passed={(r['id'],r['fixture']) for r in proof['results'] if r['passed']}
  if not expected.issubset(passed) or proof['passed']!=proof['total']:raise ValueError('Missing diagnostic reference/trace execution')
  if file_hash(snapshot/'fprisc/fpr')!=proof['compiler_sha256']:raise ValueError('Compiler differs')
  roots={n:str((snapshot/n).resolve()) for n in ('fprisc','qos')}
  for row in rows:check_row(row);source_check(row,roots)
  return rows


def run(args):
  import torch
  import qos_ai as ai
  rows=checked_diagnostic(args.data,args.snapshot)
  if args.output.exists():raise ValueError('Choose a fresh evaluation path')
  device=ai.choose_device('mps');torch.manual_seed(7)
  model,parent=ai.load_student(args.load,device);tok=ai.StudentTokenizer(parent['qwen_ids'])
  settings={'checkpoint_sha256':file_hash(args.load),'diagnostic_hash':rows_fingerprint(rows),
    'model_code_sha256':file_hash(Path(ai.__file__)),'evaluator_sha256':file_hash(Path(__file__)),
    'contract_code_sha256':file_hash(Path('qos_ai_generalization.py')),'seed':7,'temperature':0,'top_k':20,
    'no_repeat':3,'max_new_tokens':128,'device':str(device),'host':platform.node(),'torch':str(torch.__version__),
    'ancestry':parent.get('benchmark_ancestry','legacy_or_unknown')}
  args.output.mkdir(parents=True);(args.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
  answers=[]
  for row in rows:
    prompt=tok.encode(ai.chat_text(row['question']));budget=min(128,model.config.context_length-len(prompt))
    if 0 in prompt or budget!=128:raise ValueError('Unknown/truncated prompt')
    start=time.perf_counter()
    with torch.inference_mode():ids,finished=model.generate_batch([prompt],budget,parent['im_end_id'],0,20,3)[0]
    a={'id':row['id'],'question':row['question'],'answer':tok.decode(ids).strip(),'finished':finished,
       'tokens':len(ids),'effective_token_budget':budget,'severe_loop':ai.loops(ids,phrase=4,times=4),'seconds':time.perf_counter()-start}
    answers.append(a)
    with (args.output/'answers.jsonl').open('a') as out:out.write(json.dumps(a)+'\n')
    if len(answers)%12==0:print(len(answers),'/',len(rows),flush=True)
  (args.output/'summary.json').write_text(json.dumps({**settings,'count':len(rows),'finished':sum(a['finished'] for a in answers),
    'severe_loops':sum(a['severe_loop'] for a in answers)},indent=2)+'\n')


def grade(args):
  rows=checked_diagnostic(args.data,args.snapshot);answers=read_rows(args.output/'answers.jsonl')
  settings=json.loads((args.output/'summary.json').read_text())
  if settings['diagnostic_hash']!=rows_fingerprint(rows) or [(r['id'],r['question']) for r in rows]!=[(a['id'],a['question']) for a in answers]:raise ValueError('Answer identities/coverage differ')
  results=[];cache={}
  for r,a in zip(rows,answers):
    if r['category']=='output_prediction':
      g={'expected_stdout':r['program']['stdout'].strip(),'stated_output':stated_output(a['answer']),'behavior_pass':False,'executed':False}
      g['stated_output_match']=g['stated_output']==g['expected_stdout'];g['complete_pass']=g['stated_output_match']
    else:
      g=grade_answer(r,a['answer'],args.snapshot/'fprisc/fpr',args.snapshot,cache)
      # These generation prompts additionally request the output; unlike the old
      # coding diagnostic, correct code alone does not complete the new task.
      if r['category']=='code_generation':g['complete_pass']=g['behavior_pass'] and g['stated_output_match']
    g.update(id=r['id'],family=r['family'],category=r['category'],mode=r['mode'],stratum=r['stratum'],**trace_grade(r,a['answer']))
    g['complete_with_canonical_trace']=g['complete_pass'] and (r['mode']=='output_only' or g['canonical_trace_match'])
    results.append(g)
  def stats(rs):
    return {'count':len(rs),**{k:sum(bool(r[k]) for r in rs) for k in ('complete_pass','behavior_pass','stated_output_match','canonical_trace_match','complete_with_canonical_trace')}}
  report={'checkpoint_sha256':settings['checkpoint_sha256'],'diagnostic_hash':rows_fingerprint(rows),'answers_sha256':file_hash(args.output/'answers.jsonl'),
    'compiler_sha256':file_hash(args.snapshot/'fprisc/fpr'),'grader_sha256':file_hash(Path(__file__)),
    'scope':'Exact input/operation structure plus execution; canonical trace-field matching is strict and does not judge all free prose. Wording items deliberately share training cases; remaining strata exclude known canonical programs, not unknown legacy exposure.',
    'overall':stats(results),'strata':{s:stats([r for r in results if r['stratum']==s]) for s in sorted({r['stratum'] for r in results})},
    'categories':{s:stats([r for r in results if r['category']==s]) for s in sorted({r['category'] for r in results})},'results':results}
  (args.output/'grading.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)


def loss(args):
  import math
  import qos_ai as ai
  from qos_ai_finetune import encode_examples,pad_examples,training_loss
  diag=checked_diagnostic(args.data,args.snapshot);train=read_rows(args.data/'balanced_train_ready.jsonl')
  # Fixed probes, independent of training seed and unchanged across checkpoints.
  rng=random.Random(20261019);rng.shuffle(train);train=train[:96]
  model,parent=ai.load_student(args.load,ai.choose_device('mps'));tok=ai.StudentTokenizer(parent['qwen_ids'])
  result={'checkpoint_sha256':file_hash(args.load),'diagnostic_hash':rows_fingerprint(diag),'train_probe_hash':rows_fingerprint(train),'loss_code_sha256':file_hash(Path(__file__))}
  for name,rows in [('train_probe',train)]+[(s,[r for r in diag if r['stratum']==s]) for s in ('wording','inputs','combinations')]:
    examples,stats=encode_examples(rows,tok,model.config.context_length)
    examples=pad_examples(examples,parent['pad_id'],192)
    result[name]={'examples':len(rows),'answer_tokens':stats['answer_tokens'],'answer_cross_entropy':training_loss(model,examples,2,parent['pad_id'],'mps',32)}
  args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
  for command,fn in [('run',run),('grade',grade),('loss',loss)]:
    s=subs.add_parser(command);s.set_defaults(fn=fn)
    s.add_argument('--data',type=Path,required=True);s.add_argument('--snapshot',type=Path,default=Path('source_snapshot'));s.add_argument('--output',type=Path,required=True)
    if command!='grade':s.add_argument('--load',type=Path,required=True)
  a=p.parse_args();a.fn(a)
