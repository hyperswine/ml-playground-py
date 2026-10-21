import copy
import unittest
from qos_ai_generalization import TRAIN_FAMILIES,HELD_COMBINATIONS,MODES,case_pool,make_row,check_row,spec,trace_grade
from qos_ai_pure_sol import Parser
from qos_ai_compare import stated_output


class GeneralizationTests(unittest.TestCase):
  def test_reference_structure_and_answers_cover_all_families_and_modes(self):
    for family in TRAIN_FAMILIES+HELD_COMBINATIONS:
      for case in list_case_pool(family,18):
        for mode in MODES:
          r=make_row(case,mode,{family:[]})
          check_row(r)
          Parser(r['program']['code']).program()
          if mode!='output_only':self.assertTrue(trace_grade(r,r['answer'])['canonical_trace_match'])
          self.assertEqual(stated_output(r['answer']),r['program']['stdout'].strip())
          if 'debug_program' in r:
            Parser(r['debug_program']['code']).program()
            self.assertNotEqual(r['debug_program']['stdout'],r['program']['stdout'])

  def test_tampered_calculations_inputs_or_programs_fail(self):
    row=make_row(next(case_pool('map_take',419)),'repair_program',{'map_take':[]})
    for key in ('question','answer','traces','program','trace_program'):
      bad=copy.deepcopy(row);bad[key]='changed'
      with self.assertRaises(ValueError):check_row(bad)
    bad=copy.deepcopy(row);bad['generalization_contract']['case']['inputs']['factor']+=1
    with self.assertRaises(ValueError):check_row(bad)

  def test_trace_parser_rejects_wrong_or_duplicate_fields_and_ignores_code(self):
    row=make_row(next(case_pool('map',419)),'output_trace',{'map':[]})
    self.assertTrue(trace_grade(row,row['answer'])['canonical_trace_match'])
    self.assertFalse(trace_grade(row,row['answer']+' result=777')['canonical_trace_match'])
    self.assertFalse(trace_grade(row,'```sol\n'+row['answer']+'\n```')['canonical_trace_match'])

  def test_wording_stratum_changes_words_only(self):
    case=next(case_pool('sum',419));a=make_row(case,'generate',{'sum':[]});b=make_row(case,'generate',{'sum':[]},1,'wording')
    for field in ('answer','program','trace_program','traces'):self.assertEqual(a[field],b[field])
    self.assertNotEqual(a['question'],b['question'])

  def test_python_oracles_include_boundaries_negatives_and_empty_prefixes(self):
    p=next(case_pool('sum',419))['inputs'];p.update(values=[-3,0,7],factor=-2,count=0,threshold=0,lower=-2,upper=4,value=10)
    self.assertEqual(spec('map',p)['value'],-8)
    self.assertEqual(spec('take',p)['value'],0)
    self.assertEqual(spec('filter',p)['value'],7)
    self.assertEqual(spec('clamp',p)['value'],4)
    self.assertEqual(spec('map_filter',p)['value'],6)
    self.assertEqual(spec('clamp_sum',p)['value'],2)


def list_case_pool(f,n):
  from itertools import islice
  return list(islice(case_pool(f,419),n))

if __name__=='__main__':unittest.main()
