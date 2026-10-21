"""Fixed loss probes for Stage 4 candidates: the Stage 1 probe set plus arithmetic-validation answer loss.

Writes one JSON per checkpoint. The English, documentation and Q&A probes are the same windows used for the
domain parent in Stage 2 (`m557_1b/result.json` domain_probes), so the 1.25x retention gate compares like for like.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import qos_ai as ai
from qos_ai_finetune import answer_loss, make_batch
from qos_ai_stage1_train import all_probes, pilot_windows
from qos_ai_stage4_eval import sha


@torch.no_grad()
def arithmetic_loss(model, examples, pad_id, batch=16):
  model.eval(); device = next(model.parameters()).device; total = count = 0.
  for start in range(0, len(examples), batch):
    tokens, mask = make_batch(examples[start:start + batch], pad_id, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
      loss, n = answer_loss(model, tokens, mask)
    total += loss.item(); count += n.item()
  return {'answer_cross_entropy': total / count, 'answer_perplexity': math.exp(total / count), 'target_tokens': int(count)}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint', type=Path, required=True)
  p.add_argument('--pretraining-data', type=Path, required=True)
  p.add_argument('--stage1-data', type=Path, required=True)
  p.add_argument('--english-probe', type=Path, required=True)
  p.add_argument('--arithmetic-validation', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--device', default='cuda')
  a = p.parse_args()
  manifest = json.loads((a.pretraining_data / 'dataset.json').read_text())
  streams = {k: np.memmap(a.pretraining_data / r['files']['validation']['path'], mode='r', dtype=np.uint32)
             for k, r in manifest['results'].items()}
  load = lambda name: np.load(a.stage1_data / f'{name}.npy', mmap_mode='r')
  probes = {'pilot_english_validation': (pilot_windows(streams['english']), None),
            'pilot_python_validation': (pilot_windows(streams['python']), None),
            'fineweb013_english': (np.load(a.english_probe), None),
            'dev_docs': (load('probe_docs'), None),
            'dev_qa_answers': (load('probe_qa_rows'), load('probe_qa_mask'))}
  model, ck = ai.load_student(a.checkpoint, torch.device(a.device))
  arith = torch.load(a.arithmetic_validation, map_location='cpu', weights_only=True)
  result = {'checkpoint': str(a.checkpoint), 'checkpoint_sha256': sha(a.checkpoint),
            'probes': all_probes(model, probes),
            'arithmetic_validation': arithmetic_loss(model, arith['examples'], arith['pad_id'])}
  a.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({k: (round(v['perplexity'], 2) if 'perplexity' in v else v) for k, v in result['probes'].items()}),
        json.dumps(result['arithmetic_validation']), flush=True)


if __name__ == '__main__':
  main()
