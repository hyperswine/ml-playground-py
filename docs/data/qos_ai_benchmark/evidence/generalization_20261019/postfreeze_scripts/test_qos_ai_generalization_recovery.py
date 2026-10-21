import copy
import unittest
import torch
import qos_ai as ai
from qos_ai_resume import optimizer_plan
from qos_ai_generalization_recovery_step import apply_update


class RecoveryMathTests(unittest.TestCase):
  def test_updates_match_frozen_training_math_across_restored_chunks(self):
    import qos_ai_finetune as base
    torch.manual_seed(51);config=ai.StudentConfig(32,16,8,2,1);initial=ai.QosAi(config)
    a=copy.deepcopy(initial);b=copy.deepcopy(initial)
    opta=torch.optim.AdamW(a.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
    optb=torch.optim.AdamW(b.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
    examples=[([1,2,3+i,4],[False,False,True,True]) for i in range(8)]
    settings={'pad_multiple':1,'learning_rate':1e-5,'mps_cache_interval':12};plan=optimizer_plan(8,2,2,3,17)
    for step,g in enumerate(plan,1):
      opta.zero_grad(set_to_none=True);count=sum(sum(examples[i][1][1:]) for batch in g['batches'] for i in batch)
      from qos_ai_resume import learning_rate
      for p in opta.param_groups:p['lr']=learning_rate(1e-5,step,len(plan))
      for batch in g['batches']:
        loss,n=base.answer_loss(a,*base.make_batch([examples[i] for i in batch],0,'cpu'))
        (loss/count).backward()
      torch.nn.utils.clip_grad_norm_(a.parameters(),1.);opta.step()
      apply_update(b,optb,examples,g,settings,step,len(plan),0,'cpu')
      if step%2==0:
        state=copy.deepcopy(b.state_dict());os=copy.deepcopy(optb.state_dict());b=ai.QosAi(config);b.load_state_dict(state)
        optb=torch.optim.AdamW(b.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01);optb.load_state_dict(os)
    for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
    for x,y in zip(opta.state.values(),optb.state.values()):
      for k in x:self.assertTrue(torch.equal(x[k],y[k]))

if __name__=='__main__':unittest.main()
