"""Read-only audit of the pinned M1 Qwen checkpoint and a bounded inference probe.

No training, downloads, default changes, or frozen benchmark writes. Model and
runtime files are identified by hashes. Run in the existing M1 MLX environment.
"""
import argparse,hashlib,importlib.metadata,json,struct,time
from collections import Counter
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def header(p):
 with Path(p).open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];return json.loads(f.read(size))


def inspect(a):
 import mlx_lm.models.qwen3_5 as code
 import mlx_lm.models.qwen3_next as next_code
 import mlx_lm.models.gated_delta as delta_code
 import mlx_lm.tuner.utils as lora_code
 from mlx_lm import load
 from mlx_lm.utils import get_total_parameters
 model,tok=load(a.model)
 cfg=json.loads((Path(a.model)/'config.json').read_text());text=cfg.get('text_config',cfg)
 actual={};total=get_total_parameters(model)
 actual['embedding']=text['vocab_size']*text['hidden_size']
 actual['final_norm']=model.model.norm.weight.size
 for i,layer in enumerate(model.layers):
  actual[f'layer_{i:02d}']={'kind':'linear_attention' if layer.is_linear else 'full_attention','parameters':get_total_parameters(layer)}
 assert actual['embedding']+actual['final_norm']+sum(v['parameters'] for k,v in actual.items() if k.startswith('layer_'))==total
 adapter=header(a.adapter) if a.adapter else {}
 by_layer=Counter();by_module=Counter();shapes={}
 for k,v in adapter.items():
  if k=='__metadata__':continue
  n=1
  for d in v['shape']:n*=d
  bits=k.split('.');by_layer[bits[bits.index('layers')+1]]+=n
  by_module['.'.join(bits[bits.index('layers')+2:-1])]+=n;shapes[k]=v['shape']
 q='What does this Sol program print?\n```sol\n> print (str (List.sum [1, 2, 3])).\n```'
 historical=f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
 template={}
 for mode in (False,True):
  template[str(mode)]=tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=mode)
 files={str(p):sha(p) for p in (Path(code.__file__),Path(next_code.__file__),Path(delta_code.__file__),Path(lora_code.__file__))}
 h=header(Path(a.model)/'model.safetensors');keys=[k for k in h if k!='__metadata__']
 result={'model':a.model,'config':cfg,'config_sha256':sha(Path(a.model)/'config.json'),'logical_text_parameters':total,
  'parameter_breakdown':actual,'adapter':a.adapter,'adapter_parameters':sum(by_layer.values()),'adapter_by_layer':dict(by_layer),'adapter_by_module':dict(by_module),'adapter_shapes':shapes,
  'adapter_sha256':sha(a.adapter) if a.adapter else None,'runtime_sources_sha256':files,
  'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm'),
  'historical_prefix_matches_current_no_thinking_template':template['False']==historical,'chat_prefix_examples':template,
  'mtp_tensors_in_converted_weights':sum('mtp.' in k for k in keys),'vision_tensors_in_converted_weights':sum('visual' in k or 'vision_tower' in k for k in keys)}
 save(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('config','parameter_breakdown','adapter_shapes','chat_prefix_examples')},indent=2))


def save(path,data):
 p=Path(path)
 if p.exists():raise ValueError('refusing to replace existing evidence')
 p.write_text(json.dumps(data,indent=2)+'\n')


def probe(a):
 from mlx_lm import load,stream_generate
 from mlx_lm.sample_utils import make_sampler
 selected=('dev-07-002','dev-08-006','dev-07-003','dev-09-000')
 data={r['id']:r for r in map(json.loads,Path(a.rows).read_text().splitlines())}
 assert all(i in data for i in selected)
 result={'policy':'Exploratory four-question development probe, not promotion evidence. Questions fixed before generation. Greedy across all modes; no final holdout.',
  'rows_sha256':sha(a.rows),'model':a.model,'config_sha256':sha(Path(a.model)/'config.json'),'conditions':[
   {'name':'historical_budget','thinking':False,'budget':'original per-question'},
   {'name':'longer_no_thinking','thinking':False,'budget':512},
   {'name':'thinking','thinking':True,'budget':512}], 'selected':selected,'answers':[]}
 save(str(a.output)+'.plan.json',{k:v for k,v in result.items() if k!='answers'})
 model,tok=load(a.model);stops={tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')}
 start=time.monotonic()
 for cond in result['conditions']:
  for ident in selected:
   row=data[ident];q=row['question'];budget=row['max_new_tokens'] if cond['budget']=='original per-question' else cond['budget']
   prompt=tok.apply_chat_template([{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,enable_thinking=cond['thinking'])
   out=[];finished=False;t=time.monotonic()
   for resp in stream_generate(model,tok,prompt=prompt,max_tokens=budget,sampler=make_sampler(temp=0.)):
    if resp.token in stops:finished=True;break
    out.append(resp.token)
   raw=tok.decode(out).strip();closed='</think>' in raw
   final=raw.split('</think>',1)[1].strip() if closed else ('' if cond['thinking'] else raw)
   result['answers'].append({'id':ident,'condition':cond['name'],'thinking':cond['thinking'],'max_new_tokens':budget,
    'finished':finished,'generated_tokens':len(out),'thinking_closed':closed,'answer':final,'raw':raw,'seconds':time.monotonic()-t})
   print(json.dumps({'id':ident,'condition':cond['name'],'tokens':len(out),'finished':finished,'has_final':bool(final)}),flush=True)
 result['seconds']=time.monotonic()-start;save(a.output,result)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 i=s.add_parser('inspect');i.add_argument('--adapter')
 q=s.add_parser('probe');q.add_argument('--rows',required=True)
 for c in (i,q):c.add_argument('--model',required=True);c.add_argument('--output',required=True)
 a=p.parse_args();{'inspect':inspect,'probe':probe}[a.command](a)
