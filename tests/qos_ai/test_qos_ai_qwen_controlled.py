import random
import unittest
from qos_ai_qwen_controlled import trace_spec,teach,ordered_schedule
from qos_ai_qwen_repair import make_row,FAMILIES

class ControlledTest(unittest.TestCase):
  def test_training_trace_oracles_cover_boundaries(self):
    for f in range(7):
      for case in [([],0,0),([-5,0,2,8],-2,2),([0,1,7,12],3,-1)]:
        traces=trace_spec(f,case)
        self.assertEqual(traces[-1]['value'],FAMILIES[f][3](*case))
        self.assertEqual(traces[-1]['name'],'result')
    with self.assertRaises(ValueError):trace_spec(7,([],0,0))

  def test_steps_precede_claim_and_input_program_preserved(self):
    for i in range(4):
      row=make_row('valid',0,i,random.Random(1));trained=teach(row)
      self.assertLess(trained['answer'].index('Steps:'),trained['answer'].index('It prints'))
      self.assertEqual(trained['code'],row['code'])
      if row['kind']!='output':self.assertIn(row['code'],trained['answer'])
      if row['kind']=='nochange':self.assertTrue(trained['answer'].startswith('No change needed.'))

  def test_schedule_is_complete_unique_balanced_and_repeatable(self):
    class Tok:
      def encode(self,text,**kwargs):return list(text)
    rows=[{'id':f'{origin}/{kind}/{i}','origin':origin,'kind':kind,'answer':'x'*(30+i)}
          for origin in ('replay','new') for kind in ('output','repair','nochange','generate') for i in range(40)]
    a,m=ordered_schedule(rows,Tok(),[2048,4096],7,True)
    b,n=ordered_schedule(rows,Tok(),[2048,4096],7,True)
    self.assertEqual((a,m),(b,n));self.assertEqual(len({r['id'] for r in a}),len(a))
    for point in m:
      self.assertLessEqual(abs(point['actual_tokens']-point['budget']),point['budget']*.01)
      self.assertEqual(point['micro_batches']%2,0)
      self.assertEqual(point['by_origin']['replay'],point['by_origin']['new'])

if __name__=='__main__':unittest.main()
