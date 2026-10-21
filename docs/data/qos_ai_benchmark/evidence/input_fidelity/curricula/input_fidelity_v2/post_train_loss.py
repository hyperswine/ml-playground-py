import json,math,platform,time
from pathlib import Path
import qos_ai as ai
from qos_ai_data import file_hash
from qos_ai_finetune import verified_rows,encode_examples,pad_examples,training_loss
root=Path('/Users/jq/qos_ai_eval');data=root/'curricula/input_fidelity_v2';experiment=root/'experiments/input_fidelity_sft1'
result=json.loads((experiment/'result.json').read_text())
checkpoint=experiment/'student.pt'
if file_hash(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Checkpoint identity differs')
manifest=json.loads((data/'splits.json').read_text());selection=json.loads((data/'training_selection.json').read_text())
rows=verified_rows(data/'focused_train_ready.jsonl',manifest,{'fprisc':str(root/'source_snapshot/fprisc'),'qos':str(root/'source_snapshot/qos')},data/'verified_train.jsonl',selection)
started=time.perf_counter();device=ai.choose_device('mps');model,saved=ai.load_student(checkpoint,device)
examples,stats=encode_examples(rows,ai.StudentTokenizer(saved['qwen_ids']),model.config.context_length)
examples=pad_examples(examples,saved['pad_id'],192)
loss=training_loss(model,examples,2,saved['pad_id'],device,32)
if not math.isfinite(loss):raise ValueError('Nonfinite post-training loss')
measured={'checkpoint_sha256':result['checkpoint_sha256'],'data_sha256':file_hash(data/'focused_train_ready.jsonl'),
          'final_training_loss':loss,'elapsed_seconds':time.perf_counter()-started,'device':str(device),'host':platform.node(),
          'scope':'training-set diagnostic in a fresh process; no optimizer updates or benchmark prompts',**stats}
(experiment/'post_training_loss.json').write_text(json.dumps(measured,indent=2)+'\n')
print(json.dumps(measured,indent=2))
