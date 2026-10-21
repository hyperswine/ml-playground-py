"""Generate unedited demonstration answers without changing training or benchmarks."""
import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from qos_ai_data import file_hash,read_rows,source_check
from qos_ai_compare import stated_output
from qos_ai_coding_diag import grade_answer


def generate(args):
  import torch
  import qos_ai as ai
  if args.output.exists():raise ValueError('Choose a fresh sample directory')
  expected='5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767'
  if file_hash(args.load)!=expected:raise ValueError('Gallery requires the selected stronger coding checkpoint')
  rows=read_rows(args.prompts)
  benchmark=read_rows(Path('qos_ai_benchmark/benchmark.jsonl'))
  if {r['question'] for r in rows}&{r['question'] for r in benchmark}:raise ValueError('A demonstration copies a benchmark prompt')
  roots={n:str(args.snapshot/n) for n in ('fprisc','qos')};fpr=args.snapshot/'fprisc/fpr'
  if file_hash(fpr)!='c41c6ab9377e59be4d9bccedf1d6dd33f188bad0453046660650c6982fbc2f36':raise ValueError('Compiler differs')
  proof=[]
  for row in rows:
    if row.get('sources'):source_check(row,roots)
    if not row.get('program'):continue
    with tempfile.TemporaryDirectory(prefix='qos-ai-gallery-reference-') as tmp:
      path=Path(tmp)/'reference.sol';path.write_text(row['program']['code'])
      run=subprocess.run([str(fpr.resolve()),'sol',str(path)],cwd=tmp,env=dict(os.environ,SOL_JIT='0',SOL_GPU='0',SOL_CACHE='0',FPR_HOME=str((args.snapshot/'fprisc').resolve())),capture_output=True,text=True,timeout=20)
      ok=run.returncode==0 and run.stdout.strip()==row['program']['stdout'].strip()
      proof.append({'id':row['id'],'passed':ok,'stdout':run.stdout,'stderr':run.stderr,'returncode':run.returncode})
      if not ok:raise ValueError('Authored reference failed: '+row['id'])
  args.output.mkdir(parents=True)
  (args.output/'reference_execution.json').write_text(json.dumps({'compiler_sha256':file_hash(fpr),'results':proof},indent=2)+'\n')
  torch.manual_seed(7);device=ai.choose_device('mps');model,parent=ai.load_student(args.load,device);model.eval()
  tokenizer=ai.StudentTokenizer(parent['qwen_ids'])
  settings={'checkpoint':str(args.load),'checkpoint_sha256':expected,'checkpoint_label':'balanced coding checkpoint, preceding calculation follow-up',
    'seed':7,'temperature':0,'top_k':20,'no_repeat':3,'max_new_tokens':160,'device':str(device),'host':platform.node(),
    'torch':str(torch.__version__),'model_code_sha256':file_hash(Path(ai.__file__)),'generation_code_sha256':file_hash(Path(__file__)),
    'prompts_sha256':file_hash(args.prompts),'scope':'demonstrations, not a new clean benchmark; related training concepts and inputs may be familiar'}
  (args.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n');results=[];cache={}
  for row in rows:
    prompt=tokenizer.encode(ai.chat_text(row['question']));budget=min(160,model.config.context_length-len(prompt))
    if budget<=0:raise ValueError('Prompt exceeds context')
    start=time.perf_counter()
    with torch.inference_mode():ids,finished=model.generate_batch([prompt],budget,parent['im_end_id'],0,20,3)[0]
    answer=tokenizer.decode(ids).strip()
    result={'id':row['id'],'kind':row['kind'],'question':row['question'],'answer':answer,'finished':finished,'tokens':len(ids),
      'effective_token_budget':budget,'unknown_prompt_tokens':prompt.count(0),'seconds':time.perf_counter()-start}
    if row['kind'] in ('code','repair','check'):
      category={'code':'code_generation','repair':'debugging','check':'no_change'}[row['kind']]
      review=grade_answer({**row,'category':category},answer,fpr,args.snapshot,cache)
      if review['complete_pass'] and review['stated_output'] is not None and not review['stated_output_match']:status='partial: working code with a wrong output claim'
      elif review['complete_pass']:status='passed the code/output checks'
      elif review['behavior_pass']:status='partial: correct code; requested result or judgment missing/wrong'
      else:status='unmatched: supplied code/answer does not pass the restricted checks'
      result['review']={'status':status,**review}
    elif row['kind'] in ('output','arithmetic'):
      expected_value=row['program']['stdout'].strip() if row.get('program') else row['expected']
      value=stated_output(answer);result['review']={'status':'correct stated value' if value==expected_value else 'wrong or missing stated value','expected':expected_value,'stated':value}
    else:result['review']={'status':'not automatically judged'}
    results.append(result)
    with (args.output/'answers.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(f'{len(results)}/{len(rows)} {row["id"]}: {result["review"]["status"]}',flush=True)
  (args.output/'summary.json').write_text(json.dumps({**settings,'count':len(results),'finished':sum(r['finished'] for r in results),'answers_sha256':file_hash(args.output/'answers.jsonl')},indent=2)+'\n')
  if file_hash(args.load)!=expected:raise ValueError('Checkpoint changed')


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('load','prompts','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--snapshot',type=Path,default=Path('source_snapshot'));generate(p.parse_args())
