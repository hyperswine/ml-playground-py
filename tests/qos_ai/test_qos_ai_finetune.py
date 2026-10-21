import json
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
import torch
from qos_ai import QosAi, StudentConfig
from qos_ai_finetune import answer_loss, make_batch, encode_examples, verified_rows, checkpoint_payload, selected_verified_rows, pad_examples
from qos_ai_data import file_hash, make_manifest, rows_fingerprint, digest, question_id
import qos_ai as ai


class FineTuneTests(unittest.TestCase):
  def test_fixed_padding_preserves_real_causal_student_loss(self):
    torch.manual_seed(7)
    model=QosAi(StudentConfig(vocabulary_size=16,context_length=8,embedding_size=16,head_count=2,layer_count=1))
    examples=[([1,2,3,4],[False,False,True,True])]
    padded=pad_examples(examples,0,8)
    a,ac=answer_loss(model,*make_batch(examples,0,'cpu'))
    b,bc=answer_loss(model,*make_batch(padded,0,'cpu'))
    self.assertEqual(ac,bc);self.assertAlmostEqual(a.item(),b.item(),places=5)
    self.assertEqual(sum(padded[0][1]),2)

  def test_bucket_padding_preserves_real_tokens_and_answer_loss(self):
    class Model:
      def __call__(self,tokens):return torch.zeros(*tokens.shape,8)
    examples=[([1,2,3,4,5],[False,False,True,True,True])]
    plain=make_batch(examples,0,'cpu');bucket=make_batch(examples,0,'cpu',8)
    self.assertEqual(bucket[0].shape,(1,8))
    self.assertFalse(bucket[1][0,-3:].any())
    a,ac=answer_loss(Model(),*plain);b,bc=answer_loss(Model(),*bucket)
    self.assertEqual(ac,bc);self.assertEqual(a,b)

  @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
  def test_real_mps_step_has_correct_target_count_and_updates(self):
    torch.manual_seed(7)
    model=QosAi(StudentConfig(vocabulary_size=16,context_length=8,embedding_size=16,head_count=2,layer_count=1)).to('mps')
    tokens,mask=make_batch([([1,2,3,4],[False,False,True,True])],0,'mps',8)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    loss,count=answer_loss(model,tokens,mask)
    self.assertEqual(count.item(),2);self.assertGreater(loss.item(),0)
    original=model.token_embedding.weight.detach().cpu().clone()
    optimizer.zero_grad();(loss/count).backward();optimizer.step();torch.mps.synchronize()
    self.assertFalse(torch.equal(original,model.token_embedding.weight.detach().cpu()))

  def test_prompt_padding_and_unknown_targets_are_not_learned(self):
    class Tokenizer:
      def encode(self, text):
        return [1,2,3,4,0,5,6] if 'answer' in text else [1,2,3]
    examples, stats=encode_examples([{'id':'a','question':'question','answer':'answer'}],Tokenizer(),16)
    self.assertEqual(examples[0][1],[False,False,False,True,False,True,True])
    self.assertEqual(stats['unknown_answer_tokens_skipped'],1)
    tokens,mask=make_batch(examples+[([1,2,6],[False,False,True])],0,'cpu')
    self.assertEqual(mask[1].tolist(),[False,True,False,False,False,False])

  def test_shifted_loss_matches_only_the_answer_predictions(self):
    class Model:
      def __call__(self,tokens):return torch.zeros(*tokens.shape,8)
    tokens=torch.tensor([[1,2,3,4]])
    total,count=answer_loss(Model(),tokens,torch.tensor([[False,True,True]]))
    self.assertEqual(count,2)
    self.assertAlmostEqual(total.item(),2*torch.log(torch.tensor(8.)).item(),places=6)

  def test_oversized_verified_example_is_rejected(self):
    class Tokenizer:
      def encode(self,text):return list(range(20))
    with self.assertRaisesRegex(ValueError,'exceeds model context'):
      encode_examples([{'id':'a','question':'q','answer':'a'}],Tokenizer(),8)

  def test_checkpoint_roundtrip_uses_plain_metadata_and_preserves_tied_weights(self):
    model=QosAi(StudentConfig(vocabulary_size=16,context_length=8,embedding_size=16,head_count=2,layer_count=1))
    payload=checkpoint_payload({'step':5},model,{'parent_sha256':'parent','torch':torch.__version__},'split',3)
    with tempfile.TemporaryDirectory() as d:
      path=Path(d)/'student.pt';torch.save(payload,path)
      loaded=torch.load(path,weights_only=True)
      self.assertEqual(type(loaded['verified_sft']['torch']),str)
      self.assertEqual(loaded['step'],8)
      self.assertEqual(loaded['model_state']['language_model_head.weight'].data_ptr(),loaded['model_state']['token_embedding.weight'].data_ptr())
      for key,value in model.state_dict().items():self.assertTrue(torch.equal(loaded['model_state'][key],value))

  def test_real_tiny_student_updates_on_answer_tokens(self):
    torch.manual_seed(7)
    model=QosAi(StudentConfig(vocabulary_size=16,context_length=8,embedding_size=16,head_count=2,layer_count=1))
    tokens,mask=make_batch([([1,2,3,4],[False,False,True,True])],0,'cpu')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    initial,count=answer_loss(model,tokens,mask)
    optimizer.zero_grad();(initial/count).backward();optimizer.step()
    final,_=answer_loss(model,tokens,mask)
    self.assertLess(final.item(),initial.item())


class VerifiedInputTests(unittest.TestCase):
  def selection(self, rows, manifest, ids):
    result={'schema':1,'canonical_hash':rows_fingerprint(rows),'split_hash':manifest['hash'],'ids':ids,'policy':'fixed test selection'}
    result['hash']=digest(result)
    return result

  def test_fixed_verified_selection_is_accepted(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,manifest,roots=self.fixture(d)
      rows=[json.loads(canonical.read_text())];selection=self.selection(rows,manifest,['a'])
      with patch.object(ai,'BENCHMARK_FILE',benchmark):
        self.assertEqual(verified_rows(data,manifest,roots,canonical,selection),rows)

  def test_duplicate_and_unknown_selected_ids_are_refused(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,_,_,manifest,_=self.fixture(d);rows=[json.loads(canonical.read_text())]
      for ids in (['a','a'],['missing'],[]):
        with self.assertRaisesRegex(ValueError,'IDs'):selected_verified_rows(rows,manifest,self.selection(rows,manifest,ids))

  def test_tampered_selection_hash_is_refused(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,_,_,manifest,_=self.fixture(d);rows=[json.loads(canonical.read_text())]
      selection=self.selection(rows,manifest,['a']);selection['ids']=[]
      with self.assertRaisesRegex(ValueError,'hash/schema'):selected_verified_rows(rows,manifest,selection)

  def test_stale_selection_data_and_split_are_refused(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,_,_,manifest,_=self.fixture(d);rows=[json.loads(canonical.read_text())]
      for field in ('canonical_hash','split_hash'):
        selection=self.selection(rows,manifest,['a']);selection[field]='stale';selection.pop('hash');selection['hash']=digest(selection)
        with self.assertRaisesRegex(ValueError,'stale'):selected_verified_rows(rows,manifest,selection)

  def test_selected_holdout_is_refused(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,_,_,manifest,_=self.fixture(d);rows=[json.loads(canonical.read_text())]
      manifest['files'][rows[0]['path']]='dev';manifest['questions'][question_id(rows[0]['question'])]='dev'
      manifest.pop('hash');manifest['hash']=digest(manifest)
      with self.assertRaisesRegex(ValueError,'non-training'):selected_verified_rows(rows,manifest,self.selection(rows,manifest,['a']))

  def test_prepared_data_must_match_selected_subset_exactly(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,_,roots=self.fixture(d);a=json.loads(canonical.read_text())
      b={**a,'id':'b','question':'Another source grounded factual question.'};rows=[a,b]
      canonical.write_text(''.join(json.dumps(r)+'\n' for r in rows));data.write_text(canonical.read_text())
      (Path(d)/'verification.json').write_text(json.dumps({'verified_data_hash':rows_fingerprint(rows),'results':[]}))
      manifest=make_manifest([],[],rows,[]);selection=self.selection(rows,manifest,['a'])
      with patch.object(ai,'BENCHMARK_FILE',benchmark):
        with self.assertRaisesRegex(ValueError,'exactly the prepared'):verified_rows(data,manifest,roots,canonical,selection)

  def test_explicit_curriculum_does_not_require_global_path_override(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,manifest,roots=self.fixture(d)
      with patch.object(ai,'BENCHMARK_FILE',benchmark):
        self.assertEqual(len(verified_rows(data,manifest,roots,canonical)),1)

  def test_debugging_examples_require_passing_broken_fixture(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,manifest,roots=self.fixture(d)
      row=json.loads(data.read_text());row['debug_program']={'code':'bad','exit_code':1,'stdout':''}
      canonical.write_text(json.dumps(row)+'\n');data.write_text(canonical.read_text())
      manifest=make_manifest([],[],[row],[])
      (Path(d)/'verification.json').write_text(json.dumps({'verified_data_hash':rows_fingerprint([row]),'results':[]}))
      with patch.object(ai,'BENCHMARK_FILE',benchmark):
        with self.assertRaisesRegex(ValueError,'broken-program'):verified_rows(data,manifest,roots,canonical)

  def fixture(self, directory):
    root=Path(directory)
    source=root/'compiler/Sol/Preamble.hs';source.parent.mkdir(parents=True);source.write_text('a = 1.\n')
    row={'id':'a','path':'fprisc/compiler/Sol/Preamble.hs','question':'Explain this unique verified teaching contract.',
         'answer':'A verified answer.', 'sources':[{'path':'fprisc/compiler/Sol/Preamble.hs','sha256':file_hash(source),'evidence':'a = 1.'}]}
    canonical=root/'verified_train.jsonl';canonical.write_text(json.dumps(row)+'\n')
    benchmark=root/'benchmark.jsonl';benchmark.write_text('')
    data=root/'ready.jsonl';data.write_text(canonical.read_text())
    (root/'verification.json').write_text(json.dumps({'verified_data_hash':rows_fingerprint([row]),'results':[]}))
    manifest=make_manifest([],[],[row],[])
    return canonical,benchmark,data,manifest,{'fprisc':str(root)}

  def test_exact_verified_training_export_is_accepted(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,manifest,roots=self.fixture(d)
      with patch.object(ai,'VERIFIED_FILE',canonical),patch.object(ai,'BENCHMARK_FILE',benchmark):
        self.assertEqual(len(verified_rows(data,manifest,roots)),1)

  def test_modified_question_cannot_enter_as_verified_training(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,manifest,roots=self.fixture(d)
      row=json.loads(data.read_text());row['question']='An injected evaluation question.'
      data.write_text(json.dumps(row)+'\n')
      with patch.object(ai,'VERIFIED_FILE',canonical),patch.object(ai,'BENCHMARK_FILE',benchmark):
        with self.assertRaisesRegex(ValueError,'exactly the prepared'):verified_rows(data,manifest,roots)

  def test_stale_verification_proof_is_refused(self):
    with tempfile.TemporaryDirectory() as d:
      canonical,benchmark,data,manifest,roots=self.fixture(d)
      (Path(d)/'verification.json').write_text(json.dumps({'verified_data_hash':'stale','results':[]}))
      with patch.object(ai,'VERIFIED_FILE',canonical),patch.object(ai,'BENCHMARK_FILE',benchmark):
        with self.assertRaisesRegex(ValueError,'evidence is stale'):verified_rows(data,manifest,roots)


if __name__=='__main__':unittest.main()
