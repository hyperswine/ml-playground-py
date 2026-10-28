import unittest
from qos_ai_qwen_trace_values import audit

class TraceValuesTest(unittest.TestCase):
  row={'traces':[{'name':'products','value':[2,4]},{'name':'result','value':6}]}
  def test_renamed_correct_values_are_separate_from_strict_fields(self):
    r=audit(self.row,'Steps: adjusted=[2, 4]; result=6.\nIt prints 6.')
    self.assertTrue(r['ordered_values_match']);self.assertTrue(r['renamed_fields_only'])
  def test_wrong_value_not_hidden_by_correct_final(self):
    self.assertFalse(audit(self.row,'Steps: products=[2, 5]; result=6.\nIt prints 6.')['ordered_values_match'])
  def test_trailing_junk_and_extra_fields_rejected(self):
    for answer in ('Steps: products=[2,4]; result=6oops.','Steps: products=[2,4]; result=6; result=7.'):
      self.assertFalse(audit(self.row,answer)['ordered_values_match'])
  def test_order_matters(self):
    self.assertFalse(audit(self.row,'Steps: result=6; products=[2,4].')['ordered_values_match'])
  def test_code_block_cannot_supply_trace(self):
    self.assertFalse(audit(self.row,'```sol\nSteps: products=[2,4]; result=6.\n```')['ordered_values_match'])

if __name__=='__main__':unittest.main()
