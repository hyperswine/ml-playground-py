import importlib.util
import math
import os
from pathlib import Path
import unittest

path=Path(os.environ.get('QOS_MLX_DRIVER',Path(__file__).resolve().parents[2]/'scripts/mac/mlx_repair.py'))
spec=importlib.util.spec_from_file_location('repair_driver',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class Tokenizer:
  def encode(self,text,**kwargs):return list(map(ord,text))


class DatasetTest(unittest.TestCase):
  def test_target_boundary_and_eos(self):
    ds=driver.ExactDataset([{'id':'x','question':'Q','answer':'A'}],Tokenizer(),1000)
    tokens,offset=ds.encoded[0]
    self.assertEqual(''.join(map(chr,tokens[:offset])),driver.chat('Q'))
    self.assertEqual(''.join(map(chr,tokens[offset:])),'A<|im_end|>')

  def test_no_silent_truncation(self):
    with self.assertRaisesRegex(ValueError,'refusing truncation'):
      driver.ExactDataset([{'id':'x','question':'Q','answer':'A'}],Tokenizer(),10)


@unittest.skipUnless(importlib.util.find_spec('mlx'),'requires the M1 MLX environment')
class MaskTest(unittest.TestCase):
  def test_ordered_batch_exposure_and_padding(self):
    batches=list(driver.ordered_batches([([1,2,3],1),([4,5],1)],1,64))
    self.assertEqual(len(batches),2)
    self.assertEqual(batches[0][0][0,:3].tolist(),[1,2,3])
    self.assertEqual(batches[1][0][0,:2].tolist(),[4,5])
    self.assertEqual(batches[0][1].tolist(),[[1,3]])
    with self.assertRaisesRegex(ValueError,'refusing truncation'):
      list(driver.ordered_batches([([1,2,3],1)],1,2))

  def test_prompt_and_padding_excluded_eos_included(self):
    import mlx.core as mx
    class Toy:
      def __call__(self,x):
        # Every target has uniform logits except EOS, whose loss is about 20.
        logits=mx.zeros((*x.shape,8))
        return logits.at[0,2,7].add(20.)
    batch=mx.array([[1,2,3,4,0,0]])
    loss,count=driver.answer_only_loss(Toy(),batch,mx.array([[2,4]]))
    self.assertEqual(count.item(),2)
    self.assertAlmostEqual(loss.item(),(math.log(8)+20)/2,places=5)


if __name__=='__main__':unittest.main()
