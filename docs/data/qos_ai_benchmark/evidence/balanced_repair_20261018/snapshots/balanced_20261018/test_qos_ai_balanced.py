import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from qos_ai_balanced import FAMILIES,MODES,TRAIN_WORDS,DIAG_WORDS,case_pool,make_row,validate_contract,render
from qos_ai_pure_sol import Parser,Unsupported,extract
from qos_ai_coding_diag import grade_answer


class BalancedTests(unittest.TestCase):
  def test_all_modes_keep_authored_inputs_and_broken_results_differ(self):
    for case in case_pool(137,TRAIN_WORDS):
      for mode in MODES:
        row=make_row(case,mode,{case['family']:[]});validate_contract(row)
        Parser(row['program']['code']).program()
        if 'debug_program' in row:self.assertNotEqual(row['debug_program']['stdout'],row['program']['stdout'])
        if mode=='output_only':self.assertEqual(row['answer'],row['program']['stdout'].strip())
  def test_changed_contract_question_answer_and_program_are_rejected(self):
    case=case_pool(137,TRAIN_WORDS)[0];row=make_row(case,'repair_inline',{case['family']:[]})
    for field in ('question','answer','expression','program'):
      changed=copy.deepcopy(row);changed[field]='tampered'
      with self.assertRaisesRegex(ValueError,'contract'):validate_contract(changed)
  def test_generation_pools_cover_all_families(self):
    self.assertEqual({c['family'] for c in case_pool(137,TRAIN_WORDS)},set(FAMILIES))
    self.assertFalse(set(TRAIN_WORDS)&set(DIAG_WORDS))


class PureSolTests(unittest.TestCase):
  def test_bound_aliases_lambda_renaming_and_numeric_commutation_match(self):
    expected=Parser('List.sum (List.map (fn x -> x * 3) [2, 4])').expression()
    actual=Parser('values = [2,4].\nscaled = List.map (fn n -> 3 * n) values.\n> print (str (List.sum scaled)).').program()
    self.assertEqual(actual,expected)
  def test_raw_program_inline_and_fenced_expression(self):
    for answer in ('`Str.at "ABC" 0`','```sol\nStr.at "ABC" 0\n```','> print (str (Str.at "ABC" 0)).'):
      _,ast,_=extract(answer);self.assertEqual(ast,Parser('Str.at "ABC" 0').expression())
  def test_unsafe_unbounded_unused_and_recursive_code_rejected(self):
    for code in ('> print (Proc.run "id").','@/etc/passwd.','> print "{Proc.run x}".',
      'x = x.\n> print x.','unused = List.range 0 999.\n> print 2.',
      '> print (List.map (fn x -> List.range 0 50) [1,2]).',
      '> print (List.fold (fn a x -> a*a) 2 [1,2]).'):
      with self.subTest(code=code),self.assertRaises(Unsupported):Parser(code).program()
  def test_constant_answer_and_wrong_input_do_not_satisfy_contract(self):
    expected=Parser('Numeric.clamp 1 8 20').expression()
    self.assertNotEqual(Parser('8').expression(),expected)
    self.assertNotEqual(Parser('Numeric.clamp 0 8 20').expression(),expected)
  def test_grader_does_not_execute_wrong_contract(self):
    row={'id':'a','category':'code_generation','program':{'code':'> print (str (Numeric.min 3 7)).','stdout':'3\n'}}
    with patch('qos_ai_coding_diag.subprocess.run') as run:
      result=grade_answer(row,'```sol\n> print 3.\n```',Path('/compiler'),Path('/snapshot'),{})
      self.assertFalse(result['complete_pass']);run.assert_not_called()
  def test_correct_check_requires_no_change_and_explicit_correct_output(self):
    row={'id':'a','category':'no_change','program':{'code':'> print 3.','stdout':'3\n'}}
    for answer,expected in (('No change needed. It prints 3.',True),('It prints 3.',False),('No change needed. It prints 4.',False)):
      self.assertEqual(grade_answer(row,answer,Path('/compiler'),Path('/snapshot'),{})['complete_pass'],expected)


if __name__=='__main__':unittest.main()
