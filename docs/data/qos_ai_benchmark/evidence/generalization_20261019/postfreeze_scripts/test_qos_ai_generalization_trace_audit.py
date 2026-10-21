import unittest
from qos_ai_generalization_trace_audit import strict_trace


class StrictTraceTests(unittest.TestCase):
  def setUp(self):self.row={'traces':[{'name':'input','value':4},{'name':'result','value':4}]}
  def test_valid_fields(self):self.assertTrue(strict_trace(self.row,'Steps: input=4; result=4.')['strict_canonical_trace_match'])
  def test_extra_integer_is_not_a_valid_prefix_match(self):self.assertFalse(strict_trace(self.row,'Steps: input=4 4; result=4.')['strict_canonical_trace_match'])
  def test_decimal_not_integer_prefix(self):self.assertFalse(strict_trace(self.row,'Steps: input=4.5; result=4.')['strict_canonical_trace_match'])
  def test_wrong_type_or_duplicate(self):
    for answer in ('input="4"; result=4.','input=4; result=4; input=4.'):
      self.assertFalse(strict_trace(self.row,answer)['strict_canonical_trace_match'])
  def test_list_is_fully_parsed(self):
    r={'traces':[{'name':'products','value':[-2,4]}]}
    self.assertTrue(strict_trace(r,'products=[-2, 4].')['strict_canonical_trace_match'])
    self.assertFalse(strict_trace(r,'products=[-2, 4] 8.')['strict_canonical_trace_match'])
  def test_nested_list_values_keep_their_integer_types(self):
    r={'traces':[{'name':'values','value':[1,4]}]}
    for answer in ('values=[true,4].','values=[1,4.0].'):
      self.assertFalse(strict_trace(r,answer)['strict_canonical_trace_match'])
  def test_code_does_not_supply_prose_fields(self):self.assertFalse(strict_trace(self.row,'```sol\ninput=4; result=4.\n```')['strict_canonical_trace_match'])

if __name__=='__main__':unittest.main()
