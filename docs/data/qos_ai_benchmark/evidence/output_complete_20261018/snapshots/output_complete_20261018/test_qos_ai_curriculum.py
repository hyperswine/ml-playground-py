import copy
import unittest
from qos_ai_curriculum import screen_extension, validate_extension
from qos_ai_data import digest, make_manifest, question_id

class CurriculumTests(unittest.TestCase):
  def fixture(self):
    old=[{'id':'old','path':'fprisc/compiler/Sol/Preamble.hs','question':'Original teaching question about numeric operations.','answer':'Original answer.'}]
    bench=[{'id':'b','group':'g','split':'dev','track':'familiar','category':'api','question':'Reserve this exact benchmark prompt for evaluation.', 'answer':'Reference.','sources':[{'path':old[0]['path']}]}]
    base=make_manifest([],[],old,bench)
    self.assertEqual(base['files'][old[0]['path']],'train')
    candidate={**old[0],'id':'new','question':'Compute a triple transformation of these selected integers.'}
    return old,bench,base,candidate

  def test_expansion_retains_all_original_assignments(self):
    old,bench,base,c=self.fixture()
    kept,rejected,new=screen_extension(base,old,[c],[],bench)
    self.assertEqual(kept,old+[c]);self.assertEqual(rejected,[])
    validate_extension(base,new)
    self.assertTrue(all(new['questions'][k]==v for k,v in base['questions'].items()))

  def test_protected_question_and_lexical_variant_are_rejected(self):
    old,bench,base,c=self.fixture()
    c['question']=bench[0]['question']+' Please.'
    kept,rejected,_=screen_extension(base,old,[c],[],bench)
    self.assertEqual(kept,old);self.assertEqual(len(rejected),1)

  def test_reference_program_overlap_is_rejected(self):
    old,bench,base,c=self.fixture()
    bench[0]['question']='Trace the printed result.\n```sol\n> print (str (List.sum [4, 7])).\n```'
    base=make_manifest([],[],old,bench)
    c['expression']='List.sum [4,7]'
    kept,rejected,_=screen_extension(base,old,[c],[],bench)
    self.assertEqual(kept,old);self.assertEqual(rejected[0]['reason'],'benchmark program input overlap')

  def test_changed_existing_assignment_or_nontrain_addition_is_refused(self):
    old,bench,base,c=self.fixture()
    _,_,new=screen_extension(base,old,[c],[],bench)
    for key in (question_id(old[0]['question']),question_id(c['question'])):
      invalid=copy.deepcopy(new);invalid['questions'][key]='dev'
      invalid.pop('hash');invalid['hash']=digest(invalid)
      with self.assertRaises(ValueError):validate_extension(base,invalid)

  def test_changed_historical_fingerprint_is_refused(self):
    old,bench,base,c=self.fixture()
    _,_,new=screen_extension(base,old,[c],[],bench)
    new['inputs']['qa']='altered';new.pop('hash');new['hash']=digest(new)
    with self.assertRaisesRegex(ValueError,'historical data'):validate_extension(base,new)

if __name__=='__main__':unittest.main()
