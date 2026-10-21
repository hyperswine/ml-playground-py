import copy
import tempfile
import unittest
import argparse
import contextlib
import io
import json
from unittest.mock import patch
from pathlib import Path
import torch
from qos_ai import QosAi,StudentConfig
from qos_ai_finetune import answer_loss,make_batch
from qos_ai_finetune import fine_tune
from qos_ai_resume import optimizer_plan,learning_rate,save_progress,load_progress


class ResumeTests(unittest.TestCase):
  def test_complete_trainer_segments_match_continuous_run(self):
    import qos_ai as ai
    torch.manual_seed(7);original=self.model()
    class Tokenizer:
      def __init__(self,ids):pass
      def encode(self,text):return [1,2,3,4] if 'answer' in text else [1,2]
    def load(path,device):return copy.deepcopy(original).to(device),{'qwen_ids':[], 'pad_id':0,'step':0}
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);parent=root/'parent.pt';parent.write_bytes(b'fixed-parent')
      splits=root/'splits.json';splits.write_text(json.dumps({'hash':'fixed-split'}))
      data=root/'data.jsonl';data.write_text('fixed-data');bench=root/'bench.jsonl';bench.write_text('fixed-benchmark')
      args=argparse.Namespace(epochs=2,batch_size=2,grad_accumulation=2,learning_rate=.01,max_minutes=10,pad_multiple=1,
        mps_cache_interval=0,output=root/'full',resume=True,load=parent,segment_steps=0,root=[],splits=splits,
        selection=None,data=data,curriculum=None,seed=7,device='cpu',fixed_padding=False,dry_run=False,defer_final_loss=True)
      rows=[{'id':str(i),'question':'question','answer':'answer'} for i in range(7)]
      with patch('qos_ai_finetune.verified_rows',return_value=rows),patch.object(ai,'load_student',side_effect=load),\
        patch.object(ai,'StudentTokenizer',Tokenizer),patch.object(ai,'BENCHMARK_FILE',bench),contextlib.redirect_stdout(io.StringIO()):
        fine_tune(args);args.output=root/'segmented';args.segment_steps=2
        fine_tune(args);self.assertFalse((args.output/'student.pt').exists());fine_tune(args)
      full=torch.load(root/'full/student.pt',weights_only=True);segmented=torch.load(root/'segmented/student.pt',weights_only=True)
      for key,value in full['model_state'].items():self.assertTrue(torch.equal(value,segmented['model_state'][key]),key)
      self.assertEqual(segmented['verified_sft']['optimizer_steps'],4)
      self.assertEqual(segmented['verified_sft']['examples_seen'],14)

  def model(self):return QosAi(StudentConfig(vocabulary_size=16,context_length=8,embedding_size=16,head_count=2,layer_count=1))
  def advance(self,model,opt,plan,examples,start,end):
    for step in range(start,end):
      opt.zero_grad();opt.param_groups[0]['lr']=learning_rate(.01,step+1,len(plan))
      batches=plan[step]['batches'];count=sum(sum(examples[i][1][1:]) for ids in batches for i in ids)
      for ids in batches:
        loss,_=answer_loss(model,*make_batch([examples[i] for i in ids],0,'cpu'));(loss/count).backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  def test_resume_matches_uninterrupted_weights_adam_moments_schedule_and_rng(self):
    torch.manual_seed(7);base=self.model();a=copy.deepcopy(base);b=copy.deepcopy(base)
    oa=torch.optim.AdamW(a.parameters(),lr=.01,betas=(.9,.95));ob=torch.optim.AdamW(b.parameters(),lr=.01,betas=(.9,.95))
    examples=[([1,2,3,i],[False,False,True,True]) for i in range(4,11)]
    plan=optimizer_plan(7,2,2,2,7);self.advance(a,oa,plan,examples,0,len(plan));self.advance(b,ob,plan,examples,0,2)
    settings={'parent':'frozen','data':'frozen','code':'frozen','seed':7}
    with tempfile.TemporaryDirectory() as tmp:
      path=Path(tmp)/'progress.pt';save_progress(path,b,ob,settings,{'step':2});expected_rng=torch.rand(3)
      c=self.model();oc=torch.optim.AdamW(c.parameters(),lr=.01,betas=(.9,.95));torch.manual_seed(100)
      self.assertEqual(load_progress(path,c,oc,settings),{'step':2});self.assertTrue(torch.equal(torch.rand(3),expected_rng))
      payload=torch.load(path,weights_only=True)
      self.assertEqual(payload['model_state']['token_embedding.weight'].data_ptr(),payload['model_state']['language_model_head.weight'].data_ptr())
      self.advance(c,oc,plan,examples,2,len(plan))
      for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,c.state_dict()[key]),key)
      for left,right in zip(oa.state.values(),oc.state.values()):
        for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)
  def test_stale_settings_refused_before_model_mutation(self):
    model=self.model();opt=torch.optim.AdamW(model.parameters());settings={'parent':'a','data':'b','code':'c'}
    with tempfile.TemporaryDirectory() as tmp:
      path=Path(tmp)/'p.pt';save_progress(path,model,opt,settings,{'step':0});original=copy.deepcopy(model.state_dict())
      for key in settings:
        changed={**settings,key:'tampered'}
        with self.assertRaisesRegex(ValueError,'differ'):load_progress(path,model,opt,changed)
      for key,value in original.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
  def test_plan_covers_each_epoch_once_including_partial_batch(self):
    plan=optimizer_plan(7,2,2,3,7)
    for epoch in (1,2,3):
      groups=[g for g in plan if g['epoch']==epoch]
      self.assertEqual(sorted(i for g in groups for batch in g['batches'] for i in batch),list(range(7)))
      self.assertEqual(sum(g['ends_epoch'] for g in groups),1)
    self.assertEqual(plan,optimizer_plan(7,2,2,3,7))


if __name__=='__main__':unittest.main()
