import copy
import unittest
from qos_ai_output_curriculum import contrast_groups,make_row,validate_contract,specification,MODES,FAMILIES
from qos_ai_compare import stated_output
from qos_ai_pure_sol import Parser


class OutputCurriculumTests(unittest.TestCase):
  def test_pairs_change_one_field_and_one_result(self):
    seen=set()
    for a,b in contrast_groups(331):
      seen.add(a['family']);self.assertEqual(sum(a['inputs'][k]!=b['inputs'][k] for k in a['inputs']),1)
      self.assertNotEqual(specification(a['family'],a['inputs'])[1],specification(b['family'],b['inputs'])[1])
    self.assertEqual(seen,set(FAMILIES))

  def test_all_answers_bind_calculation_result_and_complete_repair(self):
    for pair in contrast_groups(331):
      for case in pair:
        for mode in MODES:
          row=make_row(case,mode,{case['family']:[]});validate_contract(row)
          expected=row['program']['stdout'].strip();parsed=stated_output(row['answer'])
          if parsed is None:
            # The frozen benchmark prose parser rejects stopword-like strings;
            # exact authored-answer reconstruction remains the training oracle.
            self.assertIn(expected.casefold(),('a','an','the'))
            self.assertIn('It prints '+expected+'.',row['answer'])
          else:self.assertEqual(parsed,expected)
          Parser(row['program']['code']).program()
          if mode!='output_only':self.assertIn(row['calculation'],row['answer'])
          if mode.startswith('repair_'):
            self.assertIn('It prints ',row['answer']);self.assertIn(row['expression'],row['answer'])
            self.assertNotEqual(row['debug_program']['stdout'],row['program']['stdout'])

  def test_modified_inputs_explanation_code_or_claim_are_refused(self):
    case=next(contrast_groups(331))[0];row=make_row(case,'repair_program',{case['family']:[]})
    for field in ('question','answer','calculation','expression','program','debug_program'):
      changed=copy.deepcopy(row);changed[field]='tampered'
      with self.assertRaisesRegex(ValueError,'contract'):validate_contract(changed)
    changed=copy.deepcopy(row);changed['calculation_contract']['inputs']['factor']+=1
    with self.assertRaisesRegex(ValueError,'contract'):validate_contract(changed)

  def test_byte_offsets_and_present_absent_search_have_independent_results(self):
    s=specification('byte',{'text':'Koala','index':2});self.assertEqual(s[1],'97');self.assertIn('"a" has ASCII code 97',s[3])
    self.assertEqual(specification('missing',{'text':'Koala','needle':97})[1],'2')
    self.assertEqual(specification('missing',{'text':'Koala','needle':126})[1],'-1')

  def test_range_and_filter_calculations_use_inclusive_boundaries(self):
    r=specification('range',{'lower':20,'upper':23});self.assertEqual(r[1],'86');self.assertIn('20 + 21 + 22 + 23 = 86',r[3])
    r=specification('filter',{'values':[3,5,8],'threshold':5});self.assertEqual(r[1],'13');self.assertIn('[5, 8]',r[3])


if __name__=='__main__':unittest.main()
