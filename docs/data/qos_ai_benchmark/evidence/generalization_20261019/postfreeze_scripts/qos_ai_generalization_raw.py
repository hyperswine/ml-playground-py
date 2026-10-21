"""Supplemental no-guard pass; frozen prompts, model code and primary grader reused unchanged.

Added after the authored-answer feasibility audit found the standard repetition
blocker excludes repeated token trigrams in legitimate calculation answers.
"""
import argparse
import json
import platform
import time
from pathlib import Path
from qos_ai_data import file_hash,rows_fingerprint
from qos_ai_generalization_eval import checked_diagnostic,grade


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
    'no_repeat':0,'max_new_tokens':128,'device':str(device),'host':platform.node(),'torch':str(torch.__version__),
    'ancestry':parent.get('benchmark_ancestry','legacy_or_unknown')}
  args.output.mkdir(parents=True);(args.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
  answers=[]
  for row in rows:
    prompt=tok.encode(ai.chat_text(row['question']));budget=min(128,model.config.context_length-len(prompt))
    if 0 in prompt or budget!=128:raise ValueError('Unknown/truncated prompt')
    start=time.perf_counter()
    with torch.inference_mode():ids,finished=model.generate_batch([prompt],budget,parent['im_end_id'],0,20,0)[0]
    a={'id':row['id'],'question':row['question'],'answer':tok.decode(ids).strip(),'finished':finished,
       'tokens':len(ids),'effective_token_budget':budget,'severe_loop':ai.loops(ids,phrase=4,times=4),'seconds':time.perf_counter()-start}
    answers.append(a)
    with (args.output/'answers.jsonl').open('a') as out:out.write(json.dumps(a)+'\n')
    if len(answers)%12==0:print(len(answers),'/',len(rows),flush=True)
  (args.output/'summary.json').write_text(json.dumps({**settings,'count':len(rows),'finished':sum(a['finished'] for a in answers),
    'severe_loops':sum(a['severe_loop'] for a in answers)},indent=2)+'\n')


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
  for command,fn in [('run',run),('grade',grade)]:
    s=subs.add_parser(command);s.set_defaults(fn=fn)
    s.add_argument('--data',type=Path,required=True);s.add_argument('--snapshot',type=Path,default=Path('source_snapshot'));s.add_argument('--output',type=Path,required=True)
    if command=='run':s.add_argument('--load',type=Path,required=True)
  a=p.parse_args();a.fn(a)
