"""Shorter operational MPS chunks, preserving the frozen optimizer treatment."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import qos_ai as ai
import qos_ai_finetune as base
from qos_ai_data import file_hash,digest
from qos_ai_resume import optimizer_plan,learning_rate,load_progress,save_progress


def apply_update(model,optimizer,examples,group,settings,step,total_steps,pad_id,device):
  model.train();accumulated=[base.make_batch([examples[i] for i in ids],pad_id,device,settings['pad_multiple']) for ids in group['batches']]
  count=sum(sum(examples[i][1][1:]) for ids in group['batches'] for i in ids)
  if count<=0:raise ValueError('No supervised targets')
  optimizer.zero_grad(set_to_none=True);lr=learning_rate(settings['learning_rate'],step,total_steps)
  for state in optimizer.param_groups:state['lr']=lr
  numerator=0.;seen=0
  for (tokens,mask),ids in zip(accumulated,group['batches']):
    loss,device_count=base.answer_loss(model,tokens,mask)
    if device_count.item()!=sum(sum(examples[i][1][1:]) for i in ids):raise ValueError('Device target count mismatch; recovery update rejected')
    (loss/count).backward();numerator+=loss.item();seen+=tokens.shape[0]
  norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
  if not math.isfinite(numerator) or not math.isfinite(norm):raise ValueError('Nonfinite recovery loss/gradient')
  optimizer.step()
  if str(device)=='mps' and step%settings['mps_cache_interval']==0:torch.mps.synchronize();torch.mps.empty_cache()
  return {'loss':numerator/count,'learning_rate':lr,'supervised_tokens':count,'examples':seen}


def main(args):
  experiment=args.experiment
  if (experiment/'student.pt').exists():raise ValueError('Final checkpoint already published')
  settings=json.loads((experiment/'settings.json').read_text());data=args.data
  if settings['training_code_sha256']!=file_hash(Path(base.__file__)) or settings['model_code_sha256']!=file_hash(Path(ai.__file__)):raise ValueError('Frozen training/model code changed')
  parent_path=Path(settings['parent'])
  if file_hash(parent_path)!=settings['parent_sha256']:raise ValueError('Parent changed')
  manifest=json.loads((data/'splits.json').read_text());selection=json.loads((data/'training_selection.json').read_text())
  rows=base.verified_rows(data/'balanced_train_ready.jsonl',manifest,{'fprisc':'source_snapshot/fprisc','qos':'source_snapshot/qos'},data/'verified_train.jsonl',selection)
  if file_hash(data/'balanced_train_ready.jsonl')!=settings['data_sha256']:raise ValueError('Training rows changed')
  torch.manual_seed(settings['seed']);device=ai.choose_device('mps');model,parent=ai.load_student(parent_path,device)
  examples,stats=base.encode_examples(rows,ai.StudentTokenizer(parent['qwen_ids']),model.config.context_length)
  examples=base.pad_examples(examples,parent['pad_id'],settings['padding_width'])
  plan=optimizer_plan(len(examples),settings['batch_size'],settings['grad_accumulation'],settings['epochs_requested'],settings['seed'])
  if digest(plan)!=settings['optimizer_plan_hash']:raise ValueError('Optimizer plan changed')
  optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],betas=(0.9,0.95),weight_decay=settings['weight_decay'])
  prior=load_progress(experiment/'progress_latest.pt',model,optimizer,settings);first=prior['step'];started=time.perf_counter()
  step=first;epochs=prior['epochs_completed'];seen=prior['examples_seen'];supervised=prior['supervised_tokens']
  for group in plan[first:first+args.chunk_steps]:
    step+=1;r=apply_update(model,optimizer,examples,group,settings,step,len(plan),parent['pad_id'],device)
    seen+=r['examples'];supervised+=r['supervised_tokens']
    if group['ends_epoch']:epochs=group['epoch']
    record={'step':step,'epoch':group['epoch'],'loss':r['loss'],'learning_rate':r['learning_rate'],
      'seconds':prior['elapsed_seconds']+time.perf_counter()-started,'supervised_tokens':supervised,'recovery_chunk':args.chunk_steps}
    with (experiment/'training.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
    if step==first+1 or step%5==0:print(json.dumps(record),flush=True)
  elapsed=prior['elapsed_seconds']+time.perf_counter()-started
  state={'step':step,'epochs_completed':epochs,'examples_seen':seen,'supervised_tokens':supervised,'initial_training_loss':prior['initial_training_loss'],
    'elapsed_seconds':elapsed,'segments_completed':prior['segments_completed']+1}
  if elapsed>settings['max_minutes']*60:raise ValueError('Recovery exceeds original training-time ceiling')
  if step<len(plan):
    torch.mps.synchronize();save_progress(experiment/'progress_latest.pt',model,optimizer,settings,state)
    with (experiment/'segments.jsonl').open('a') as out:out.write(json.dumps({'start_step':first,'end_step':step,'seconds':elapsed-prior['elapsed_seconds'],'operational_chunk_steps':args.chunk_steps,'recovery_code_sha256':file_hash(Path(__file__))})+'\n')
    print(json.dumps({'recovery_saved_step':step}),flush=True);return
  if seen!=2784 or supervised!=97693 or step!=696 or epochs!=1:raise ValueError('Incomplete canonical treatment')
  result={**settings,'optimizer_steps':step,'epochs_completed':epochs,'examples_seen':seen,'supervised_tokens':supervised,
    'initial_training_loss':prior['initial_training_loss'],'final_training_loss':None,'final_training_loss_deferred':True,
    'elapsed_seconds':elapsed,'budget_stop':False,'segments_completed':state['segments_completed'],'resumed_final_segment_from_step':first,
    'recovery_code_sha256':file_hash(Path(__file__)),'recovery_operational_chunk_steps':args.chunk_steps}
  payload=base.checkpoint_payload(parent,model,result,manifest['hash'],step);tmp=experiment/'student.recovery.partial.pt'
  torch.save(payload,tmp);torch.load(tmp,map_location='cpu',weights_only=True,mmap=True);tmp.replace(experiment/'student.pt')
  result['checkpoint_sha256']=file_hash(experiment/'student.pt');(experiment/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--chunk-steps',type=int,default=16);main(p.parse_args())
