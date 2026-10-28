import json,sys
from pathlib import Path
root=Path('/Users/jq/qos_ai_eval');work=root/'experiments/qwen2b_controlled_20261028';old=root/'experiments/qwen2b_repair_20261027'
sys.path[:0]=[str(work),str(old),str(root)]
from qos_ai_qwen_repair import read,bindings,run_sol,digest
from mlx_repair import ExactDataset,chat
from transformers import AutoTokenizer
fpr=str(root/'experiments/stage4_20261021/preparation/toolchain/fpr')
rows=read(work/'prepared/new_pool.jsonl')+read(work/'prepared/valid.jsonl')
checked=0;repairs=0
for row in rows:
 for c in row['cases']:
  got,why=run_sol(bindings(row['code'],c['inputs']),fpr)
  assert got==c['expected'],(row['id'],got,why,c)
  checked+=1
 if row['kind']=='repair':
  assert any(run_sol(bindings(row['input_code'],c['inputs']),fpr)[0]!=c['expected'] for c in row['cases']),row['id']
  repairs+=1
 if checked%140==0:print('Verified probes',checked,flush=True)
tok=AutoTokenizer.from_pretrained(root/'experiments/mlx_20261027/mlx_2b_round4_q8',local_files_only=True)
encoded={}
for name in ('replay_pool','new_pool','valid'):
 data=read(work/f'prepared/{name}.jsonl');ds=ExactDataset(data,tok,1024)
 for r,(tokens,n) in zip(data,ds.encoded):
  assert tokens==tok.encode(chat(r['question'])+r['answer']+'<|im_end|>',add_special_tokens=False),r['id']
 encoded[name]={'rows':len(data),'max_sequence':max(len(t) for t,n in ds.encoded),'target_tokens':sum(len(t)-n for t,n in ds.encoded)}
result={'new_and_validation_case_executions':checked,'effective_repairs':repairs,'all_passed':True,'tokenization':encoded,'compiler_sha256':digest(fpr)}
(work/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
