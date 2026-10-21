"""Compare a continued checkpoint with its saved parent on the frozen dev benchmark."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from qos_ai_compare import pair_runs,output_diagnostic,metrics
from qos_ai_curriculum import validate_extension
from qos_ai_data import file_hash,read_rows


def compare(args):
  benchmark=read_rows(args.benchmark)
  base=json.loads(args.base_splits.read_text());expanded=json.loads(args.splits.read_text())
  validate_extension(base,expanded)
  training=json.loads(args.training.read_text())
  result={'training':training,'benchmark_sha256':file_hash(args.benchmark),
          'base_split_hash':base['hash'],'split_hash':expanded['hash'],
          'scope':'complete development partition; no model evaluation of test',
          'control_reuse':'Saved parent outputs from the same M1, with matching prompt order/budgets, decoding, model/evaluator/split code hashes, host and PyTorch version. Only the proven training-only manifest extension differs.',
          'accuracy_note':'Lexical rubric and output-literal matching are diagnostics, not factual accuracy. Generated-code behavior requires separate execution.',
          'comparison_code_sha256':file_hash(Path(__file__)),
          'diagnostic_code_sha256':file_hash(Path(__file__).with_name('qos_ai_compare.py')),'runs':{}}
  for condition in ('raw','guard'):
    before,after,pairs=pair_runs(args.reports/f'{args.parent_stem}_{condition}',args.reports/f'{args.tuned_stem}_{condition}',benchmark,(base,expanded))
    if before['checkpoint_sha256']!=training['parent_sha256'] or after['checkpoint_sha256']!=training['checkpoint_sha256']:
      raise ValueError('Reports differ from recorded training lineage')
    if before['benchmark_sha256']!=result['benchmark_sha256'] or training['split_hash']!=expanded['hash']:
      raise ValueError('Benchmark or training split identity differs')
    if any(before[k]!=after[k] for k in ('host','torch','machine')):raise ValueError('Execution platform differs')
    grouped={field:defaultdict(list) for field in ('category','track','group')}
    for p in pairs:
      for field in grouped:grouped[field][p[field]].append(p)
    counts=lambda rows:{'count':len(rows),'parent_proxy':sum(r['parent_proxy'] for r in rows),'tuned_proxy':sum(r['tuned_proxy'] for r in rows)}
    details={'parent':metrics(before),'tuned':metrics(after),'no_repeat':before['no_repeat'],
             'max_new_tokens':before['requested_max_new_tokens'],'output_diagnostic':output_diagnostic(pairs),
             'proxy_improved':sum(not p['parent_proxy'] and p['tuned_proxy'] for p in pairs),
             'proxy_regressed':sum(p['parent_proxy'] and not p['tuned_proxy'] for p in pairs),
             **{field:{k:counts(v) for k,v in rows.items()} for field,rows in grouped.items()}}
    result['runs'][condition]=details
    (args.reports/f'{args.prefix}_paired_{condition}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
  args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({condition:{k:v for k,v in details.items() if k not in ('output_diagnostic','group')} | {'output_matches':{k:v for k,v in details['output_diagnostic'].items() if k!='details'}} for condition,details in result['runs'].items()},indent=2))

if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('reports','base-splits','splits','training','output'):p.add_argument('--'+name,required=True,type=Path)
  p.add_argument('--benchmark',type=Path,default=Path('qos_ai_benchmark/benchmark.jsonl'))
  p.add_argument('--parent-stem',required=True);p.add_argument('--tuned-stem',required=True);p.add_argument('--prefix',required=True)
  compare(p.parse_args())
