import hashlib,json,shutil
from pathlib import Path
import torch
from qos_ai_data import file_hash

root=Path('experiments/verified_sft5')
checkpoint=root/'student.pt'
old_hash=file_hash(checkpoint)
result=json.loads((root/'result.json').read_text())
assert old_hash==result['checkpoint_sha256']
# This is our own newly created artifact; allow only the exact version-metadata
# class identified by the restricted loader, never disable weights_only.
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
 payload=torch.load(checkpoint,map_location='cpu',weights_only=True)

def weights_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  h.update(k.encode());h.update(v.numpy().tobytes())
 return h.hexdigest()

before=weights_hash(payload['model_state'])
payload['verified_sft']=json.loads(json.dumps(payload['verified_sft']))
assert torch.equal(payload['model_state']['language_model_head.weight'],payload['model_state']['token_embedding.weight'])
payload['model_state']['language_model_head.weight']=payload['model_state']['token_embedding.weight']
temporary=root/'student.repaired.pt'
torch.save(payload,temporary)
loaded=torch.load(temporary,map_location='cpu',weights_only=True)
after=weights_hash(loaded['model_state'])
assert before==after
checkpoint.rename(root/'student.original-format.pt')
temporary.rename(checkpoint)
result['checkpoint_sha256']=file_hash(checkpoint)
result['metadata_repair']={'reason':'plain version string for restricted loader; shared tied-weight storage',
 'previous_file_sha256':old_hash,'weights_before_sha256':before,'weights_after_sha256':after,
 'repair_code_sha256':file_hash(Path(__file__))}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['metadata_repair'],indent=2));print('checkpoint',result['checkpoint_sha256'])
failed=Path('reports/sft5_tuned_raw')
if failed.exists():
 assert not (failed/'answers.jsonl').exists()
 failed.rename('reports/sft5_tuned_raw_load_failure')
