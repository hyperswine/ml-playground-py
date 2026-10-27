import json
import random
import unittest
from unittest.mock import patch
from qos_ai_qwen_repair import bindings,make_row,score_row
from qos_ai_source_retrieval import SourceBM25,is_documentation_question


class RepairTests(unittest.TestCase):
  def test_binding_replacement_and_duplicates(self):
    row=make_row('dev',7,0,random.Random(1))
    self.assertEqual(bindings(row['code']),row['inputs'])
    self.assertEqual(bindings(bindings(row['code'],[[],0,-4])),[[],0,-4])
    with self.assertRaises(ValueError):bindings(row['code']+'\nxs = [0].')

  def test_valid_single_line_and_parenthesized_inputs(self):
    code='xs = [(-1), 2]. k = (-3). limit = 4.\n> print (str k).'
    self.assertEqual(bindings(code),[[-1,2],-3,4])
    self.assertEqual(bindings(bindings(code,[[9],1,0])),[[9],1,0])
    with self.assertRaises(ValueError):bindings('message = "xs = [1]. k = 2. limit = 3.".')
    with self.assertRaises(ValueError):bindings('xs = [1]. k = 1 + 2. limit = 3.')

  def test_correct_displayed_output_is_not_enough(self):
    row=make_row('dev',7,0,random.Random(1))
    with patch('qos_ai_qwen_repair.run_sol',return_value=(row['expected'],None)):
      result=score_row(row,row['answer'],'unused')
    self.assertTrue(result['claim_correct'])
    self.assertFalse(result['correct'])

  def test_nochange_must_preserve_code(self):
    row=make_row('dev',7,1,random.Random(1))
    answer=row['answer'].replace('No change needed.','Repaired.')
    with patch('qos_ai_qwen_repair.run_sol',side_effect=[(c['expected'],None) for c in row['cases']]):
      self.assertFalse(score_row(row,answer,'unused')['correct'])

  def test_repair_cannot_claim_no_change(self):
    row=make_row('dev',7,0,random.Random(1))
    with patch('qos_ai_qwen_repair.run_sol',side_effect=[(c['expected'],None) for c in row['cases']]):
      result=score_row(row,'No change needed.\n'+row['answer'],'unused')
    self.assertTrue(result['code_correct'])
    self.assertFalse(result['correct'])

  def test_repository_relative_citation_is_valid(self):
    row={'id':'doc','kind':'docs','required':['Ok','Err','fprisc/sol/lib/json.sol']}
    self.assertTrue(score_row(row,'Ok or Err. Source: sol/lib/json.sol','unused')['rubric_proxy_pass'])

  def test_doc_rubric_requires_all_facts_and_citation(self):
    row={'id':'doc','kind':'docs','required':['Ok','Err','sol/lib/json.sol'],'forbidden':['undefined']}
    self.assertTrue(score_row(row,'Ok or Err, sol/lib/json.sol','unused')['rubric_proxy_pass'])
    self.assertFalse(score_row(row,'Ok or Err; returns undefined','unused')['rubric_proxy_pass'])


class SourceTests(unittest.TestCase):
  def setUp(self):
    self.docs=[{'path':'fprisc/docs/STD-PLAN.md','text':'JSON parse JSON parse API proposed streaming','authority':'historical'},
               {'path':'fprisc/sol/lib/json.sol','text':'JSON parse API returns Result Json; whole-document parser','authority':'implementation'},
               {'path':'fprisc/sol/lib/proc.sol','text':'spec argv builds ProcessSpec','authority':'implementation'}]
  def test_implementation_over_plan(self):
    self.assertEqual(SourceBM25(self.docs).search('JSON parse API')[0]['path'],'fprisc/sol/lib/json.sol')
  def test_explicit_historical_request_still_works(self):
    self.assertEqual(SourceBM25(self.docs).search('fprisc/docs/STD-PLAN.md proposed JSON')[0]['path'],'fprisc/docs/STD-PLAN.md')
  def test_exact_module_path(self):
    self.assertEqual(SourceBM25(self.docs).search('What is in sol/lib/proc.sol?')[0]['path'],'fprisc/sol/lib/proc.sol')
  def test_docs_request_not_excluded_by_write(self):
    self.assertTrue(is_documentation_question('Write an explanation of sol/lib/json.sol.'))
    self.assertFalse(is_documentation_question('Write a program to add two numbers.'))
  def test_runtime_profile_for_query_without_path(self):
    docs=[{'path':'fprisc/std/json.fpr','text':'JSON parse API returns a value','authority':'implementation'},
          {'path':'fprisc/sol/lib/json.sol','text':'JSON parse API returns a value','authority':'implementation'}]
    self.assertEqual(SourceBM25(docs).search('How does Sol parse JSON?')[0]['path'],'fprisc/sol/lib/json.sol')
    self.assertEqual(SourceBM25(docs).search('How does native FP-RISC parse JSON?')[0]['path'],'fprisc/std/json.fpr')

  def test_empty_index_rejected(self):
    with self.assertRaises(ValueError):SourceBM25([])


if __name__=='__main__':unittest.main()
