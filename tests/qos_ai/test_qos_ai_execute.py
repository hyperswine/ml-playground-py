import os
import unittest
from pathlib import Path

from qos_ai_execute import admit, augment, stated

FPR = Path(os.environ.get('QOS_FPR', '/nonexistent/fpr'))


class AdmissionTest(unittest.TestCase):
  def test_pure_programs_are_admitted(self):
    for code in ['> print (str (List.sum [1, 2])).',
                 'xs = [1, 2].\ntotal = List.sum (List.map (fn x -> x * 2) xs).\n> print (str total).',
                 '> print (Str.upper "sol").',
                 '> print (str (if 3 > 2 then List.any (fn x -> x > 1) [2] else False)).']:
      self.assertIsNone(admit(code), code)

  def test_effects_and_unknown_names_are_refused(self):
    for code in ['> print (readPath "/etc/passwd").',
                 'p = use "x".\n> print (str 1).',
                 '> print ("{Proc.runNow 1}").',
                 '> Proc.runNow (Proc.spec "rm" []).',
                 '> print (str (List.foldl 0 [1])).',
                 '> print (str (Str.split "," "a,b")).']:
      self.assertIsNotNone(admit(code), code)


class AugmentTest(unittest.TestCase):
  def test_claim_replacement_keeps_steps(self):
    self.assertEqual(stated('It prints 5. Steps: a=1; sum=5.'), '5')

  @unittest.skipUnless(FPR.exists(), 'set QOS_FPR to the Sol compiler')
  def test_wrong_claim_is_corrected_from_answer_code(self):
    answer = '```sol\n> print (str (List.sum (List.map (fn x -> x * 5) [2, 7, 4]))).\n```\nIt prints 105. Steps: sum=105.'
    result = augment('Write it.', answer, FPR)
    self.assertEqual(result['executed'], '65')
    self.assertEqual(stated(result['answer']), '65')
    self.assertIn('Steps:', result['answer'])

  @unittest.skipUnless(FPR.exists(), 'set QOS_FPR to the Sol compiler')
  def test_question_program_is_run_when_answer_has_no_code(self):
    q = 'What does this print?\n```sol\n> print (str (List.last (List.range 1 7))).\n```'
    self.assertEqual(stated(augment(q, '6', FPR)['answer']), '7')


if __name__ == '__main__':
  unittest.main()
