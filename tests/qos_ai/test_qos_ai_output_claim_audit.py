import unittest
from qos_ai_output_claim_audit import claim_status


class ClaimAuditTests(unittest.TestCase):
  def test_claim_states_do_not_change_the_primary_code_score(self):
    code='```sol\n> print 6.\n```'
    self.assertEqual(claim_status(code,'6'),('absent',None))
    self.assertEqual(claim_status(code+'\nIt prints 6.','6'),('correct','6'))
    self.assertEqual(claim_status(code+'\nIt prints 7.','6'),('incorrect','7'))
    self.assertEqual(claim_status(code+'\nIt prints 6. It prints 7.','6'),('ambiguous',None))


if __name__=='__main__':unittest.main()
