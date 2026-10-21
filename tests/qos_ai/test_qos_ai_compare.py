import json
import tempfile
import unittest
from pathlib import Path
from qos_ai_compare import pair_runs, stated_output


class PairedComparisonTests(unittest.TestCase):
  def test_multigeneration_manifest_chain_checks_intermediate_assignments(self):
    import copy
    from qos_ai_curriculum import screen_extension
    from qos_ai_data import make_manifest,digest
    old=[{'id':'a','path':'fprisc/compiler/Sol/Preamble.hs','question':'Original unique teaching topic.','answer':'Answer.'}]
    base=make_manifest([],[],old,[])
    rows,_,middle=screen_extension(base,old,[{**old[0],'id':'b','question':'Explain another independent library function.'}],[],[])
    _,_,new=screen_extension(middle,rows,[{**old[0],'id':'c','question':'Why are the execution backend constants fixed?'}],[],[])
    with tempfile.TemporaryDirectory() as d:
      a,b,bench=self.fixtures(d)
      for path,manifest in ((a,base),(b,new)):
        s=json.loads((path/'summary.json').read_text());s['split_manifest_hash']=manifest['hash'];(path/'summary.json').write_text(json.dumps(s))
      self.assertEqual(len(pair_runs(a,b,bench,[base,middle,new])[2]),1)
      bad=copy.deepcopy(middle);bad['files'][old[0]['path']]='test';bad.pop('hash');bad['hash']=digest(bad)
      with self.assertRaises(ValueError):pair_runs(a,b,bench,[base,bad,new])

  def test_manifest_extension_allows_only_proven_split_hash_change(self):
    from qos_ai_curriculum import screen_extension
    from qos_ai_data import make_manifest
    old=[{'id':'old','path':'fprisc/compiler/Sol/Preamble.hs','question':'Original numeric teaching topic.','answer':'Answer.'}]
    base=make_manifest([],[],old,[])
    _,_,new=screen_extension(base,old,[{**old[0],'id':'new','question':'Compute a different list transformation.'}],[],[])
    with tempfile.TemporaryDirectory() as d:
      a,b,bench=self.fixtures(d)
      for path,manifest in ((a,base),(b,new)):
        s=json.loads((path/'summary.json').read_text());s['split_manifest_hash']=manifest['hash']
        (path/'summary.json').write_text(json.dumps(s))
      with self.assertRaisesRegex(ValueError,'settings'):pair_runs(a,b,bench)
      self.assertEqual(len(pair_runs(a,b,bench,(base,new))[2]),1)
      s['evaluator_sha256']='changed';(b/'summary.json').write_text(json.dumps(s))
      with self.assertRaisesRegex(ValueError,'settings'):pair_runs(a,b,bench,(base,new))

  def fixtures(self, directory):
    root=Path(directory)
    a,b=root/'parent',root/'tuned';a.mkdir();b.mkdir()
    summary={'benchmark_sha256':'same','split_manifest_hash':'same','split':'dev','seed':7,'temperature':0,
             'top_k':20,'no_repeat':0,'requested_max_new_tokens':200,'device':'mps',
             'code_sha256':'same','evaluator_sha256':'same','split_policy_code_sha256':'same'}
    rows=[{'id':'dev-a','group':'a','category':'api','question':'A question?','effective_token_budget':200,
           'reference_answer':'Reference','answer':'Answer','rubric_proxy_pass':False,'finished':True,'severe_loop':False}]
    for path in (a,b):
      (path/'summary.json').write_text(json.dumps(summary))
      (path/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    benchmark=[{'id':'dev-a','question':'A question?','split':'dev'},{'id':'test-a','question':'Test?','split':'test'}]
    return a,b,benchmark

  def test_same_complete_development_prompts_are_paired(self):
    with tempfile.TemporaryDirectory() as d:
      a,b,bench=self.fixtures(d)
      _,_,pairs=pair_runs(a,b,bench)
      self.assertEqual([p['id'] for p in pairs],['dev-a'])

  def test_changed_decoding_settings_are_refused(self):
    with tempfile.TemporaryDirectory() as d:
      a,b,bench=self.fixtures(d)
      s=json.loads((b/'summary.json').read_text());s['seed']=99
      (b/'summary.json').write_text(json.dumps(s))
      with self.assertRaisesRegex(ValueError,'settings'):pair_runs(a,b,bench)

  def test_missing_development_questions_are_refused(self):
    with tempfile.TemporaryDirectory() as d:
      a,b,bench=self.fixtures(d)
      bench.append({'id':'dev-b','question':'Missing?','split':'dev'})
      with self.assertRaisesRegex(ValueError,'complete development'):pair_runs(a,b,bench)

  def test_same_stale_prompt_in_both_runs_is_refused(self):
    with tempfile.TemporaryDirectory() as d:
      a,b,bench=self.fixtures(d)
      bench[0]['question']='Different current question?'
      with self.assertRaisesRegex(ValueError,'differs from the benchmark'):pair_runs(a,b,bench)


class OutputDiagnosticTests(unittest.TestCase):
  def test_bare_stdout_literal_is_parsed_without_accepting_arbitrary_prose(self):
    self.assertEqual(stated_output('10'),'10')
    self.assertEqual(stated_output('"bcd"'),'bcd')
    self.assertIsNone(stated_output('Maybe 10 or 15'))

  def test_code_echo_is_not_a_stated_output(self):
    self.assertIsNone(stated_output('```sol\n> print (Numeric.clamp 0 10 15).\n```'))

  def test_wrong_claim_is_not_rescued_by_number_in_code(self):
    self.assertEqual(stated_output('```sol\n> print (Numeric.clamp 0 10 15).\n```\nIt prints 15.'),'15')

  def test_plain_and_quoted_values_are_parsed(self):
    self.assertEqual(stated_output('It prints -1.'),'-1')
    self.assertEqual(stated_output('This Sol program prints the string `"bcd"`.'),'bcd')

  def test_conflicting_stated_outputs_are_ambiguous(self):
    self.assertIsNone(stated_output('It prints 10.\nIt prints 15.'))


if __name__=='__main__':unittest.main()
