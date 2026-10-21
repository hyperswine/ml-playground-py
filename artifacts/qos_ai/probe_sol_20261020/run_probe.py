"""Ask each checkpoint the fresh Sol questions; greedy, no-repeat 0, 200 new tokens."""
import json, sys, torch
from pathlib import Path
import qos_ai as ai
Q = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for name, path in [a.split('=') for a in sys.argv[3:]]:
  device = ai.choose_device('mps'); model, ck = ai.load_student(Path(path), device); tok = ai.StudentTokenizer(ck['qwen_ids'])
  rows = []
  for q in Q:
    torch.manual_seed(7); prompt = tok.encode(ai.chat_text(q['question']))
    with torch.inference_mode():
      ids, finished = model.generate_batch([prompt], 200, ck['im_end_id'], 0, 20, 0)[0]
    rows.append({'id': q['id'], 'answer': tok.decode(ids).strip(), 'finished': finished, 'tokens': len(ids)})
  (out / f'{name}.json').write_text(json.dumps(rows, indent=1)); print(name, 'done', flush=True)
  del model
