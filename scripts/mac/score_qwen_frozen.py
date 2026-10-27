"""Assemble MLX answers and invoke the preserved benchmark scorers with their own compilers."""
import argparse,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--answers',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root).resolve();answer_path=Path(a.answers).resolve();out=Path(a.output).resolve();here=Path(__file__).resolve().parent
sys.path.append(str(root));os.chdir(root)
import qos_ai_qwen_sft as sft
import qos_ai_noguard as ng
import qos_ai_stage4_eval as multi
import qos_ai_solgen_score as sol
from qos_ai_pure_sol import Unsupported
from qos_ai_execute import admit
# Existing original coding/generalization fixtures and compiler remain in place.
data=root/'curricula/generalization_20261019'
for idx,name in ((2,'benchmark.jsonl'),(3,'coding_diagnostic.jsonl'),(4,'generalization_diagnostic.jsonl')):
 suite,_,split,budget=sft.SOURCES[idx];sft.SOURCES[idx]=(suite,str(data/name),split,budget)
sft.assemble(SimpleNamespace(answers=answer_path,output=out,label=None))
receipt_path=Path(str(answer_path)+'.json')
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
for d in (out/'bench').iterdir():
 path=d/'summary.json';r=json.loads(path.read_text());r['device']='mlx';r['generation_receipt']=str(receipt_path) if receipt else None;r['cached_responses_without_generation_receipt']=receipt is None;path.write_text(json.dumps(r,indent=2)+'\n')
ng.grade(SimpleNamespace(data=data,output=out/'bench'))
fpr=root/'experiments/stage4_20261021/preparation/toolchain/fpr'
suite=json.loads((here/'multi_input.json').read_text())
m=multi.score(suite,json.loads((out/'multi_input_answers.json').read_text()),fpr)
(out/'multi_scores.json').write_text(json.dumps(m,indent=2)+'\n')
# Keep scoring logic intact, but only send the conservative pure expression grammar to the compiler.
original=sol.run;declined=[]
def checked_run(fpr,code):
 try: multi.ExpressionParser(code).program()
 except (Unsupported,RecursionError) as e:
  if admit(code) is not None:
   declined.append(str(e));return None
 return original(fpr,code)
sol.run=checked_run
tests=[json.loads(l) for name in ('fresh','held_pairs','big_numbers') for l in (here/f'test_{name}.jsonl').open()]
answers={r['id']:r['answer'] for r in json.loads((out/'solgen_answers.json').read_text())}
scored=sol.score(tests,answers,str(fpr))
(out/'solgen_scores.json').write_text(json.dumps(scored,indent=2)+'\n')
summary={'answers_sha256':hashlib.sha256(answer_path.read_bytes()).hexdigest(),'generation_receipt':receipt,'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'solgen_compiler_sha256':multi.sha(fpr),'solgen_correct':sum(r['correct'] for r in scored),'solgen_n':len(scored),'program_correct':sum(r.get('program_ok',False) for r in scored),'pure_admission_declines':declined,'multi_correct':m['correct'],'multi_total':m['total'],'bench':json.loads((out/'bench/scores.json').read_text())}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
