"""Stage-3 SFT data: the 60k executed Sol programs plus the 4,709 verified examples, with the 4B Q&A replay pool.

Same payload shape as qos_ai_stage2_data.py (`verified` = examples trained every epoch, `qa_pool` = replay), so the
existing `qos_ai_stage2_train.py sft` trains on it unchanged.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from tokenizers import Tokenizer
import qos_ai as ai
from qos_ai_data import digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--solgen', type=Path, default=Path('qos_ai/solgen_20261020/train.jsonl'))
p.add_argument('--stage2', type=Path, default=Path('qos_ai/stage2_20261020/sft_mix.pt'))
p.add_argument('--tokenizer', type=Path, default=Path('runpod/557m/bundle/tokenizer'))
p.add_argument('--vocabulary', type=Path, default=Path('runpod/557m/bundle/vocabulary.pt'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
  raise SystemExit('Choose a new output file')
v = torch.load(a.vocabulary, weights_only=True); qwen = Tokenizer.from_file(str(a.tokenizer / 'tokenizer.json'))
lookup = np.zeros(max(qwen.get_vocab_size(), int(v['qwen_ids'].max()) + 1), dtype=np.int64)
lookup[v['qwen_ids'][1:].numpy()] = np.arange(1, len(v['qwen_ids']))
enc = lambda t: lookup[np.asarray(qwen.encode(t, add_special_tokens=False).ids, dtype=np.int64)].tolist()
base = torch.load(a.stage2, weights_only=True)
sol, dropped = [], 0
for line in open(a.solgen):
  r = json.loads(line)
  prompt = enc(ai.chat_text(r['question'])); ids = enc(ai.chat_text(r['question'], r['answer']))
  if 0 in ids or len(ids) > 512:
    dropped += 1; continue
  sol.append((ids, [i >= len(prompt) for i in range(len(ids))]))
payload = {'verified': sol + base['verified'], 'qa_pool': base['qa_pool'], 'pad_id': base['pad_id'],
           'sources': {'solgen': len(sol), 'verified_curricula': len(base['verified'])}, 'solgen_dropped': dropped,
           'splits_hash': base['splits_hash'], 'vocabulary_sha256': base['vocabulary_sha256']}
payload['hash'] = digest({'verified': payload['verified'], 'qa_pool': payload['qa_pool']})
torch.save(payload, a.output)
lens = [len(i) for i, _ in sol]
print(json.dumps({'solgen': len(sol), 'dropped': dropped, 'verified_total': len(payload['verified']),
                  'solgen_tokens': sum(lens), 'solgen_answer_tokens': sum(sum(l[1:]) for _, l in sol),
                  'mean_len': sum(lens) / len(lens), 'max_len': max(lens), 'hash': payload['hash']}, indent=1))
