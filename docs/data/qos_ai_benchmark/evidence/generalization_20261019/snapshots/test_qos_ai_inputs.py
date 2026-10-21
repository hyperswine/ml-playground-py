import copy
import unittest
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
from qos_ai_inputs_curriculum import contrast_cases,render,candidates,validate_input_contract

class InputCurriculumTests(unittest.TestCase):
  def rows(self):
    source={'path':'fprisc/compiler/Sol/Preamble.hs','sha256':'source','evidence':'builtin'}
    with patch('qos_ai_inputs_curriculum.source_for_builtin',return_value=source):
      return candidates(Path('/unused'),[],83)

  def test_contrasts_change_only_one_input_and_change_the_result(self):
    groups=defaultdict(list)
    for case in contrast_cases():groups[case['group']].append(case)
    self.assertEqual(len(groups),64)
    for a,b in groups.values():
      self.assertEqual(sum(a['inputs'][k]!=b['inputs'][k] for k in a['inputs']),1)
      self.assertNotEqual(render(a['family'],a['inputs'])[1],render(b['family'],b['inputs'])[1])

  def test_all_512_targets_use_the_supplied_inputs(self):
    rows=self.rows();self.assertEqual(len(rows),512)
    for row in rows:validate_input_contract(row)
    self.assertEqual(sum(r['category']=='debugging' for r in rows),256)

  def test_substituted_multiplier_is_rejected_even_if_stdout_is_forged(self):
    row=next(r for r in self.rows() if r['family']=='map')
    row['program']['code']=row['program']['code'].replace('x * 2','x * 3')
    with self.assertRaisesRegex(ValueError,'supplied inputs'):validate_input_contract(row)

  def test_changed_question_input_is_refused(self):
    row=next(r for r in self.rows() if r['family']=='byte')
    row['question']=row['question'].replace(row['input_contract']['inputs']['text'],'DIFFERENT')
    with self.assertRaisesRegex(ValueError,'Question'):validate_input_contract(row)

  def test_conflicting_answer_code_is_refused(self):
    row=self.rows()[0];row['answer']=row['answer'].replace('result =','result = 0.\nunused =')
    with self.assertRaisesRegex(ValueError,'Answer program'):validate_input_contract(row)

  def test_debugging_fixture_cannot_be_silently_changed(self):
    row=next(r for r in self.rows() if r.get('debug_program'))
    row['debug_program']=copy.deepcopy(row['program'])
    with self.assertRaisesRegex(ValueError,'Broken example'):validate_input_contract(row)

if __name__=='__main__':unittest.main()
