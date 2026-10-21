"""Reconstruct accepted updates from append-only retry logs and successful save boundaries."""
import json
from pathlib import Path
from qos_ai_data import read_rows,file_hash
from qos_ai_curriculum import write_rows


def accepted_records(records,boundaries,total):
  boundaries=sorted({0,total,*boundaries})
  resets=[]
  for a,b in zip(records,records[1:]):
    if b['step']==a['step']+1:continue
    saved=b['step']-1
    if b['step']>a['step'] or saved not in boundaries:raise ValueError('Retry does not start at a saved boundary')
    next_boundary=next(x for x in boundaries if x>saved)
    if a['step']>=next_boundary:raise ValueError('Retry crosses a successful save boundary')
    resets.append({'discarded_through_step':a['step'],'resumed_from_saved_step':saved})
  # Failed attempts append their updates before the resume reader rolls the
  # model/Adam state back. The final occurrence is the accepted successful
  # attempt for each update in that saved interval.
  lookup={r['step']:r for r in records}
  if sorted(lookup)!=list(range(1,total+1)) or records[-1]['step']!=total:raise ValueError('Missing/extra/incomplete update records')
  accepted=[lookup[i] for i in range(1,total+1)]
  return accepted,{'raw_records':len(records),'accepted_records':len(accepted),'discarded_retry_records':len(records)-len(accepted),'resets':resets}


def audit(experiment,total=696):
  import torch
  records=read_rows(experiment/'training.jsonl');segments=read_rows(experiment/'segments.jsonl')
  end=0
  for segment in segments:
    if segment['start_step']!=end or segment['end_step']<=end:raise ValueError('Successful segment chain differs')
    end=segment['end_step']
  accepted,metadata=accepted_records(records,[s['end_step'] for s in segments],total)
  progress=torch.load(experiment/'progress_latest.pt',map_location='cpu',weights_only=True,mmap=True)['progress']
  row=accepted[progress['step']-1]
  if row['supervised_tokens']!=progress['supervised_tokens'] or end!=progress['step']:raise ValueError('Accepted log differs from saved optimizer progress')
  if progress['examples_seen']!=progress['step']*4:raise ValueError('Saved example count differs')
  output=experiment/'accepted_training.jsonl';write_rows(output,accepted)
  metadata.update(raw_log_sha256=file_hash(experiment/'training.jsonl'),accepted_log_sha256=file_hash(output),
    successful_segments_sha256=file_hash(experiment/'segments.jsonl'),saved_optimizer_step=progress['step'],
    code_sha256=file_hash(Path(__file__)),policy='Keep raw logs unchanged; retain final attempt per update, only for resets inside a documented successful save interval; bind to saved optimizer token counts.')
  (experiment/'accepted_update_audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
  return accepted,metadata
