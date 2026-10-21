"""Measure deferred answer-token training loss in a fresh device process."""
import argparse
import json
import math
from pathlib import Path
import qos_ai as ai
from qos_ai_data import file_hash
from qos_ai_finetune import verified_rows,encode_examples,pad_examples,training_loss


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('load','data','output'):p.add_argument('--'+name,type=Path,required=True)
  a=p.parse_args();model,parent=ai.load_student(a.load,ai.choose_device('mps'))
  manifest=json.loads((a.data/'splits.json').read_text());selection=json.loads((a.data/'training_selection.json').read_text())
  rows=verified_rows(a.data/'balanced_train_ready.jsonl',manifest,{'fprisc':'source_snapshot/fprisc','qos':'source_snapshot/qos'},a.data/'verified_train.jsonl',selection)
  examples,stats=encode_examples(rows,ai.StudentTokenizer(parent['qwen_ids']),model.config.context_length)
  examples=pad_examples(examples,parent['pad_id'],math.ceil(stats['longest_example']/32)*32)
  loss=training_loss(model,examples,2,parent['pad_id'],'mps',32)
  result={'checkpoint_sha256':file_hash(a.load),'data_sha256':file_hash(a.data/'balanced_train_ready.jsonl'),
    'final_training_loss':loss,'examples':len(rows),'code_sha256':file_hash(Path(__file__))}
  a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
