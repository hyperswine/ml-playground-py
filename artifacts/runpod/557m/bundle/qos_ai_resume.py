"""Atomic, restricted-reader optimizer progress for deterministic short SFT segments."""
import json
import math
import random
import torch
from qos_ai_data import digest


def optimizer_plan(count,batch_size,accumulation,epochs,seed):
  plan=[]
  for epoch in range(epochs):
    order=list(range(count));random.Random(seed+epoch).shuffle(order)
    batches=[order[i:i+batch_size] for i in range(0,count,batch_size)]
    for start in range(0,len(batches),accumulation):
      plan.append({'epoch':epoch+1,'batches':batches[start:start+accumulation],
                   'ends_epoch':start+accumulation>=len(batches)})
  return plan


def learning_rate(base,step,total):
  progress=min(1.0,max(0.0,(step-5)/max(1,total-5)))
  return base*min(1.0,step/5)*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))


def cpu_tree(value):
  if isinstance(value,torch.Tensor):return value.detach().cpu()
  if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
  if isinstance(value,list):return [cpu_tree(v) for v in value]
  if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
  return value


def save_progress(path,model,optimizer,settings,progress):
  state={k:v.detach().cpu() for k,v in model.state_dict().items() if k!='language_model_head.weight'}
  if any(not torch.isfinite(v).all().item() for v in state.values()):raise ValueError('Nonfinite model progress')
  state['language_model_head.weight']=state['token_embedding.weight']
  payload={'schema':1,'settings_hash':digest(settings),'settings':json.loads(json.dumps(settings)),
           'progress':json.loads(json.dumps(progress)),'model_state':state,'optimizer_state':cpu_tree(optimizer.state_dict()),
           'cpu_rng_state':torch.get_rng_state()}
  temporary=path.with_name(path.name+'.partial')
  torch.save(payload,temporary)
  check=torch.load(temporary,map_location='cpu',weights_only=True)
  if check['settings_hash']!=digest(settings) or check['progress']['step']!=progress['step']:
    raise ValueError('Progress verification failed')
  temporary.replace(path)


def load_progress(path,model,optimizer,settings):
  payload=torch.load(path,map_location='cpu',weights_only=True)
  if payload.get('schema')!=1 or payload.get('settings_hash')!=digest(settings) or payload.get('settings')!=settings:
    raise ValueError('Resume settings/data/parent/code differ from the saved progress')
  model.load_state_dict(payload['model_state']);optimizer.load_state_dict(payload['optimizer_state'])
  torch.set_rng_state(payload['cpu_rng_state'])
  return payload['progress']
