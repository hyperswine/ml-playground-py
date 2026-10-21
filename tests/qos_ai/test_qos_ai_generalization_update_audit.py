import unittest
from qos_ai_generalization_update_audit import accepted_records


class UpdateAuditTests(unittest.TestCase):
  def test_retry_records_are_preserved_but_not_double_counted(self):
    rows=[{'step':i,'attempt':0} for i in (1,2,3)]+[{'step':i,'attempt':1} for i in (3,4,5,6)]
    accepted,meta=accepted_records(rows,[2,4],6)
    self.assertEqual([r['step'] for r in accepted],list(range(1,7)))
    self.assertEqual(accepted[2]['attempt'],1);self.assertEqual(meta['discarded_retry_records'],1)
    self.assertEqual(len(rows),7)
  def test_cannot_roll_back_a_successful_save(self):
    with self.assertRaises(ValueError):accepted_records([{'step':i} for i in (1,2,3,4,5,3,4,5,6)],[2,4],6)
  def test_unknown_resume_boundary_is_rejected(self):
    with self.assertRaises(ValueError):accepted_records([{'step':i} for i in (1,2,3,2,3,4,5,6)],[2,4],6)
  def test_missing_update_is_rejected(self):
    with self.assertRaises(ValueError):accepted_records([{'step':i} for i in (1,2,4,5,6)],[2,4],6)

if __name__=='__main__':unittest.main()
