"""M1 repair pilot: exact historical chat format, answer-only QLoRA, auditable greedy answers.

Use the pinned mlx-lm 0.32.2 environment. This driver deliberately bypasses the
model's changing chat-template defaults and refuses sequence truncation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def sha(path):
  h=hashlib.sha256()
  with open(path,'rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  return h.hexdigest()


def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def chat(q):return f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def identity(path):
  p=Path(path)
  return {f.name:sha(f) for f in sorted(p.iterdir()) if f.suffix in ('.json','.safetensors')}


class ExactDataset:
  def __init__(self, data, tok, maximum):
    self.data=data;self.encoded=[]
    for row in data:
      prefix=tok.encode(chat(row.get('prompt',row['question'])),add_special_tokens=False)
      target=tok.encode(row['answer']+'<|im_end|>',add_special_tokens=False)
      if len(prefix)+len(target)>maximum:
        raise ValueError(f"{row['id']}: {len(prefix)+len(target)} tokens exceeds {maximum}; refusing truncation")
      self.encoded.append((prefix+target,len(prefix)))
  def __len__(self):return len(self.data)
  def __getitem__(self,i):return {'index':i}
  def process(self,r):return self.encoded[r['index']]


def answer_only_loss(model, batch, lengths):
  """Offsets are inclusive, sequence lengths exclusive: learn completion plus EOS.

  Define the mask here: mlx-lm 0.32.2's default loss includes one padding token.
  This implementation does not depend on that version-specific end convention.
  """
  import mlx.core as mx
  import mlx.nn as nn
  inputs,targets=batch[:,:-1],batch[:,1:]
  logits=model(inputs)
  positions=mx.arange(1,targets.shape[1]+1)
  mask=mx.logical_and(positions>=lengths[:,0:1],positions<lengths[:,1:])
  ce=nn.losses.cross_entropy(logits,targets)*mask
  count=mask.sum()
  return ce.astype(mx.float32).sum()/count,count


def ordered_batches(dataset,batch_size,max_seq_length,loop=False,seed=None,comm_group=None):
  """Explicit row order; finite validation and exactly recorded training exposure."""
  import mlx.core as mx
  if batch_size!=1 or (comm_group is not None and comm_group.size()!=1):
    raise ValueError('ordered pilot supports one example and one device')
  while True:
    for i in range(len(dataset)):
      tokens,offset=dataset[i]
      if len(tokens)>max_seq_length:raise ValueError('refusing truncation')
      padded=min(max_seq_length,1+32*((len(tokens)+31)//32))
      yield mx.array([tokens+[0]*(padded-len(tokens))]),mx.array([[offset,len(tokens)]])
    if not loop:break


def train(a):
  import mlx.core as mx
  import numpy as np
  np.random.seed(a.seed)
  from mlx_lm import load
  from mlx_lm.lora import CONFIG_DEFAULTS
  from mlx_lm.tuner.datasets import CacheDataset
  from mlx_lm.tuner.trainer import TrainingArgs, train as fit
  from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
  import mlx.optimizers as optim
  dest=Path(a.output)
  if dest.exists():raise ValueError('choose a new adapter output directory')
  model,tok=load(a.model)
  tr=ExactDataset(rows(a.train),tok,a.max_seq_length)
  checkpoints=sorted(set(int(x) for x in a.checkpoints.split(',') if x))
  if any(x<2 or x%2 or x>a.iters for x in checkpoints):raise ValueError('checkpoints must be even and within training')
  if a.ordered and len(tr)!=a.iters:raise ValueError('ordered training requires exactly one row per micro-batch')
  va=ExactDataset(rows(a.valid),tok,a.max_seq_length)
  cfg={**CONFIG_DEFAULTS,'model':a.model,'data':str(Path(a.train).parent),'train':True,'fine_tune_type':'lora','num_layers':4,
       'lora_parameters':{'rank':8,'scale':16.,'dropout':0.},'batch_size':1,'iters':a.iters,
       'grad_accumulation_steps':2,'val_batches':4,'steps_per_eval':a.iters,
       'steps_per_report':5,'save_every':a.iters,'learning_rate':a.learning_rate,
       'optimizer':'adamw','adapter_path':str(dest),'max_seq_length':a.max_seq_length,
       'grad_checkpoint':True,'seed':a.seed,'resume_adapter_file':None,'report_to':None}
  if a.iters < 2 or a.iters % 2: raise ValueError('iters must be a positive multiple of two for accumulation')
  meta={'optimizer_updates':a.iters//2,'micro_batches':a.iters,'model':a.model,'model_files':identity(a.model),'train_sha256':sha(a.train),'valid_sha256':sha(a.valid),
        'train_rows':len(tr),'valid_rows':len(va),'max_tokens':max(len(x[0]) for x in tr.encoded),
        'script_sha256':sha(__file__),'ordered':a.ordered,'checkpoints':checkpoints,
        'loss_mask':'prompt excluded; EOS included; padding excluded; strict end-exclusive mask',
        'target_tokens_in_dataset':sum(len(x)-n for x,n in tr.encoded),'format':'manual historical no-thinking chat, answer-only',
        'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm'),'config':cfg}
  dest.mkdir();(dest/'run_input.json').write_text(json.dumps(meta,indent=2)+'\n')
  start=time.monotonic()
  class Record:
    def record(self, kind, data):
      with (dest/'metrics.jsonl').open('a') as f: f.write(json.dumps({'kind':kind,**data})+'\n')
    def on_train_loss_report(self, data):
      self.record('train',data)
      if data['iteration'] in checkpoints:
        from mlx.utils import tree_flatten
        cp=dest/f"step_{data['iteration']:04d}";cp.mkdir()
        mx.save_safetensors(str(cp/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
        (cp/'adapter_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        exposure=tr.data[:data['iteration']]
        (cp/'receipt.json').write_text(json.dumps({'iteration':data['iteration'],'trained_tokens':data['trained_tokens'],
          'adapter_sha256':sha(cp/'adapters.safetensors'),'ordered':a.ordered,
          'ids':[r['id'] for r in exposure] if a.ordered else None},indent=2)+'\n')
    def on_val_loss_report(self, data): self.record('validation',data)
  mx.random.seed(cfg['seed'])
  model.freeze()
  linear_to_lora_layers(model, cfg['num_layers'], cfg['lora_parameters'])
  print_trainable_parameters(model)
  (dest/'adapter_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
  optimizer=optim.AdamW(learning_rate=a.learning_rate,**cfg['optimizer_config'].get('adamw',{}))
  training_args=TrainingArgs(batch_size=1,iters=a.iters,val_batches=4,steps_per_report=5,
    steps_per_eval=a.iters,steps_per_save=a.iters,adapter_file=dest/'adapters.safetensors',
    max_seq_length=a.max_seq_length,grad_checkpoint=True,grad_accumulation_steps=2)
  if checkpoints:training_args.steps_per_report=1
  fit(model=model,args=training_args,optimizer=optimizer,train_dataset=CacheDataset(tr),
      val_dataset=CacheDataset(va),loss=answer_only_loss,training_callback=Record(),
      **({'iterate_batches':ordered_batches} if a.ordered else {}))
  meta.update(seconds=time.monotonic()-start,peak_memory_gb=mx.get_peak_memory()/1e9,adapter_sha256=sha(dest/'adapters.safetensors'))
  (dest/'result.json').write_text(json.dumps(meta,indent=2)+'\n')
  print(json.dumps({'TRAIN_COMPLETE':str(dest),'seconds':meta['seconds'],'peak_memory_gb':meta['peak_memory_gb']}),flush=True)


def answer(a):
  import mlx.core as mx
  from mlx_lm import load,stream_generate
  from mlx_lm.sample_utils import make_sampler
  dest=Path(a.output)
  inputs=rows(a.prompts)
  meta={'model':str(a.model),'model_files':identity(a.model),'adapter':a.adapter,
        'adapter_files':identity(a.adapter) if a.adapter else None,'prompts_sha256':sha(a.prompts),
        'sampler':'greedy temperature=0','format':'manual historical no-thinking chat','script_sha256':sha(__file__),
        'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm')}
  receipt=Path(str(dest)+'.json')
  if dest.exists() or receipt.exists():raise ValueError('choose a new answer output file')
  receipt.write_text(json.dumps(meta,indent=2)+'\n')
  model,tok=load(a.model,adapter_path=a.adapter)
  stop={tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')}
  start=time.monotonic()
  with dest.open('w') as f:
    for i,row in enumerate(inputs):
      out=[];finished=False
      prompt=tok.encode(chat(row.get('prompt',row['question'])),add_special_tokens=False)
      for resp in stream_generate(model,tok,prompt=prompt,max_tokens=row['max_new_tokens'],sampler=make_sampler(temp=0.)):
        if resp.token in stop:finished=True;break
        out.append(resp.token)
      f.write(json.dumps({'id':row['id'],'answer':tok.decode(out).strip(),'finished':finished,'tokens':len(out),'references':row.get('references',[])})+'\n');f.flush()
      if (i+1)%8==0:print(json.dumps({'completed':i+1,'total':len(inputs),'seconds':round(time.monotonic()-start,1)}),flush=True)
  meta.update(seconds=time.monotonic()-start,peak_memory_gb=mx.get_peak_memory()/1e9,answers_sha256=sha(dest),count=len(inputs))
  receipt.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'ANSWER_COMPLETE':str(dest),'seconds':meta['seconds']}),flush=True)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
  t=s.add_parser('train');t.add_argument('--train',required=True);t.add_argument('--valid',required=True)
  t.add_argument('--iters',type=int,default=120);t.add_argument('--learning-rate',type=float,default=1e-5)
  t.add_argument('--max-seq-length',type=int,default=1024)
  t.add_argument('--seed',type=int,default=20261027);t.add_argument('--ordered',action='store_true')
  t.add_argument('--checkpoints',default='',help='Comma-separated even micro-batch counts')
  g=s.add_parser('answer');g.add_argument('--prompts',required=True);g.add_argument('--adapter')
  for c in (t,g):c.add_argument('--model',required=True);c.add_argument('--output',required=True)
  a=p.parse_args();{'train':train,'answer':answer}[a.command](a)
