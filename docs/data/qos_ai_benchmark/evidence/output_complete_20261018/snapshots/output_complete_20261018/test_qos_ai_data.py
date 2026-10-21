import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qos_ai_data import (digest, lexical_score, make_manifest, pack_examples, question_id,
                         row_split, similar_pairs, source_check, validate_manifest)


def example(q='Explain the semantics of sending a message to an actor.', path='fprisc/actor.fpr'):
  return {'path': path, 'chunk': 0, 'round': 1, 'style': 'explain', 'question': q, 'answer': 'A sufficiently long source-grounded answer.'}


def benchmark(q='How does an unrelated operation work?', path='fprisc/held.fpr', split='dev', track='generalization'):
  return {'id': 'b', 'group':'b', 'question':q,'answer':'Reference answer.', 'split':split, 'track':track,
          'category':'api','sources':[{'path':path}], 'rubric':{'required':[['Reference']]}}


class SplitTests(unittest.TestCase):
  def data(self):
    rows=[example(), example('Describe the behavior of a transaction commit.', 'qos/txn.fpr')]
    corpus=[{'path':r['path'],'chunk':0,'text':'source text'} for r in rows]
    corpus.append({'path':'fprisc/held.fpr','chunk':0,'text':'held-out source'})
    return corpus,rows,[],[benchmark()]

  def test_frozen_manifest_is_order_independent(self):
    c,q,v,b=self.data()
    a=make_manifest(c,q,v,b)
    self.assertEqual(a,make_manifest(list(reversed(c)),list(reversed(q)),v,b))

  def test_whole_file_groups_rounds_and_chunks(self):
    c,q,v,b=self.data()
    q.append(dict(q[0],chunk=99,round=5,question='What exactly happens when actor messaging succeeds?'))
    m=make_manifest(c,q,v,b)
    self.assertEqual(row_split(q[0],m),row_split(q[-1],m))

  def test_exact_question_cross_partition_is_excluded_everywhere(self):
    c,q,v,b=self.data()
    q.append(dict(q[0],path='fprisc/held.fpr'))
    m=make_manifest(c,q,v,b)
    self.assertEqual(row_split(q[0],m),'exclude')
    self.assertEqual(row_split(q[-1],m),'exclude')

  def test_heldout_file_raw_docs_and_qa_are_both_reserved(self):
    c,q,v,b=self.data()
    q.append(example('Explain the API in the held file.', 'fprisc/held.fpr'))
    m=make_manifest(c,q,v,b)
    self.assertEqual(m['files']['fprisc/held.fpr'],'dev')
    self.assertEqual(row_split(q[-1],m),'dev')

  def test_benchmark_and_lexical_variants_are_excluded(self):
    c,q,v,b=self.data()
    b[0]['question']=q[0]['question']
    q.append(dict(q[0],question=q[0]['question']+' Please.'))
    m=make_manifest(c,q,v,b)
    self.assertEqual(row_split(q[0],m),'exclude')
    self.assertEqual(row_split(q[-1],m),'exclude')

  def test_selfplay_inherits_original_question_partition(self):
    c,q,v,b=self.data()
    m=make_manifest(c,q,v,b)
    self.assertEqual(row_split({'question':q[0]['question']},m),row_split(q[0],m))

  def test_unknown_selfplay_fails_closed(self):
    m=make_manifest(*self.data())
    with self.assertRaises(ValueError):row_split({'question':'Unknown new question.'},m)

  def test_content_change_same_length_invalidates_manifest(self):
    c,q,v,b=self.data()
    m=make_manifest(c,q,v,b)
    c[0]['text']='SOURCE TEXT'
    with self.assertRaisesRegex(ValueError,'corpus changed'):validate_manifest(m,c,q,v,b)

  def test_copied_source_chunk_cannot_leak_into_docs_or_qa(self):
    c,q,v,b=self.data()
    c[0]['text']='identical source definition for the actor mailbox API'
    c[-1]['text']=c[0]['text']
    m=make_manifest(c,q,v,b)
    self.assertIn(digest([c[0]['path'],c[0]['chunk']]),m['excluded_chunks'])
    self.assertEqual(row_split(q[0],m),'exclude')

  def test_manifest_tampering_is_refused(self):
    m=make_manifest(*self.data())
    m['files']['fprisc/held.fpr']='train'
    with self.assertRaises(ValueError):validate_manifest(m)

  def test_conflicting_generalization_file_splits_are_refused(self):
    c,q,v,b=self.data()
    b.append(dict(b[0],id='c',question='Another independent question.',split='test'))
    with self.assertRaisesRegex(ValueError,'overlap'):make_manifest(c,q,v,b)

  def test_near_duplicate_prefix_index_matches_bruteforce(self):
    from qos_ai_data import shingles
    random.seed(31)
    qs=[' '.join(random.choices(['actor','reply','call','send','wait','failure','data','token'],k=random.randrange(3,14))) for _ in range(90)]
    got={tuple(sorted(pair)) for pair in similar_pairs(qs)}
    expected={(i,j) for i in range(len(qs)) for j in range(i+1,len(qs)) if len(shingles(qs[i])&shingles(qs[j]))/len(shingles(qs[i])|shingles(qs[j]))>=.75}
    self.assertEqual(got,expected)

  def test_pack_does_not_cut_answer_from_question(self):
    items=[[(1,False),(2,True),(3,True)],[(4,False),(5,True),(6,True)]]
    self.assertEqual(pack_examples(items,4),[([1,2,3],[True,True,False]),([4,5,6],[True,True,False])])

  def test_oversized_example_fails_instead_of_silent_truncation(self):
    with self.assertRaisesRegex(ValueError,'exceeds context'):pack_examples([[(1,True)]*10],8)

  def test_source_mutation_is_detected(self):
    from qos_ai_data import file_hash
    with tempfile.TemporaryDirectory() as d:
      path=Path(d)/'api.fpr';path.write_text('a = 1.\n')
      row={'id':'a','sources':[{'path':'fprisc/api.fpr','sha256':file_hash(path),'evidence':'a = 1.'}]}
      source_check(row,{'fprisc':d})
      path.write_text('a = 2.\n')
      with self.assertRaisesRegex(ValueError,'Stale source'):source_check(row,{'fprisc':d})

  def test_proxy_is_not_factual_accuracy(self):
    row={'rubric':{'required':[['Option'],['absence']], 'forbidden':['always fails']}}
    score=lexical_score(row,'Option means absence.')
    self.assertTrue(score['rubric_proxy_pass'])
    self.assertIsNone(score['factual_accuracy'])
    self.assertFalse(lexical_score(row,'Option means absence and always fails')['rubric_proxy_pass'])


class ReviewTests(unittest.TestCase):
  def test_unreviewed_answers_never_become_measured_accuracy(self):
    from qos_ai_bench import reviewed_scores
    answer = {'id':'a','answer':'Some response'}
    scores=reviewed_scores([answer],[])
    self.assertIsNone(scores['factual_correct']['rate_on_reviewed'])
    self.assertEqual(scores['factual_correct']['coverage'],0)

  def test_review_requires_correct_answer_hash_and_evidence(self):
    from qos_ai_bench import reviewed_scores
    answers=[{'id':'a','answer':'A reviewed response'},{'id':'b','answer':'Unreviewed'}]
    row={'id':'a','answer_hash':digest(answers[0]['answer']),'reviewer':'source reviewer','evidence':'definition api.fpr:12','factual_correct':True}
    score=reviewed_scores(answers,[row])
    self.assertEqual(score['factual_correct']['coverage'],.5)
    row['answer_hash']='wrong'
    with self.assertRaises(ValueError):reviewed_scores(answers,[row])



class PipelineTests(unittest.TestCase):
  def test_test_partition_never_reaches_optimizer_or_development_loss(self):
    import torch
    import qos_ai as ai
    lessons={'tokens':torch.zeros(6,8),'partition':torch.tensor([0,1,2,0,1,2]),
             'is_qa':torch.tensor([False,False,False,True,True,True]),
             'is_self':torch.zeros(6,dtype=torch.bool),'is_repair':torch.zeros(6,dtype=torch.bool)}
    docs,qa,own,repair,dev=ai.split_windows(lessons)
    self.assertEqual(docs.tolist(),[0]);self.assertEqual(qa.tolist(),[3]);self.assertEqual(dev.tolist(),[1,4])
    self.assertFalse(set([2,5])&set(docs.tolist()+qa.tolist()+dev.tolist()))

  def test_legacy_lessons_are_rejected_before_tokenizer_or_teacher_load(self):
    import torch
    import qos_ai as ai
    with tempfile.TemporaryDirectory() as d:
      torch.save({'signature':'old'},Path(d)/'legacy.pt')
      with patch.object(ai,'LESSONS_DIR',Path(d)),patch.object(ai,'current_splits',return_value={'hash':'new'}),patch.object(ai,'load_tokenizer',side_effect=AssertionError('must not load')):
        with self.assertRaisesRegex(ValueError,'Legacy or stale'):ai.load_lessons()

  def test_teacher_shards_roundtrip_with_explicit_partitions_and_train_only_vocab(self):
    import torch
    import qos_ai as ai
    from argparse import Namespace
    from types import SimpleNamespace
    class Tokenizer:
      def __len__(self):return 128
      def convert_tokens_to_ids(self, token):
        return {ai.IM_START:1,ai.IM_END:2,ai.END_OF_TEXT:3,'<think>':4,'</think>':5}[token]
    class Teacher:
      def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=torch.arange(128,dtype=torch.float32).expand(*input_ids.shape,128))
    sources={'docs_train':('train-signature',[([6,7,2],[True,True,False])]),
             'qa_dev_r001':('dev-signature',[([50,2],[True,False])]),
             'qa_test_r001':('test-signature',[([99,2],[True,False])])}
    args=Namespace(device='cpu',context_length=8,top_k=2,limit=0,shard_size=2,batch_size=1)
    with tempfile.TemporaryDirectory() as d:
      with patch.object(ai,'LESSONS_DIR',Path(d)),patch.object(ai,'current_splits',return_value={'hash':'frozen'}),patch.object(ai,'lesson_sources',return_value=sources),patch.object(ai,'load_tokenizer',return_value=Tokenizer()),patch.object(ai,'load_teacher',return_value=Teacher()) as teacher:
        ai.teach(args)
        ai.teach(args)
        self.assertEqual(teacher.call_count,1)
        lessons=ai.load_lessons()
        docs,qa,own,repair,dev=ai.split_windows(lessons)
        self.assertEqual(docs.tolist(),[0]);self.assertEqual(dev.tolist(),[1])
        self.assertEqual(lessons['partition'].tolist(),[0,1,2])
        self.assertNotIn(99,lessons['qwen_ids'].tolist())
        self.assertNotIn(50,lessons['qwen_ids'].tolist())
        self.assertEqual(lessons['split_hash'],'frozen')

  def test_legacy_window_split_is_refused(self):
    import torch
    import qos_ai as ai
    with self.assertRaisesRegex(ValueError,'explicit partition'):ai.split_windows({'tokens':torch.zeros(2,8)})


if __name__=='__main__':unittest.main()
