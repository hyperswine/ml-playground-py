"""Compare continued expanded SFT with preserved controls and frozen dev prompts."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from qos_ai_compare import pair_runs, output_diagnostic, metrics
from qos_ai_curriculum import validate_extension
from qos_ai_data import file_hash, read_rows


def compare(args):
  benchmark=read_rows(args.benchmark)
  base=json.loads(args.base_splits.read_text());expanded=json.loads(args.splits.read_text())
  validate_extension(base,expanded)
  training=json.loads(args.training.read_text())
  result={'benchmark_sha256':file_hash(args.benchmark),'base_split_hash':base['hash'],'expanded_split_hash':expanded['hash'],
          'training':training,'scope':'all 90 development prompts; no model evaluation of test',
          'control_reuse':'Previously recorded original and SFT5 outputs on the same M1; prompt order, budget, decoding, evaluator and model code hashes checked. Only the proven append-only training manifest extension differs.',
          'comparison_code_sha256':file_hash(Path(__file__)),'diagnostic_code_sha256':file_hash(Path(__file__).with_name('qos_ai_compare.py')),
          'accuracy_note':'Lexical proxy and output-literal diagnostics are not full factual accuracy. One continued run cannot isolate added data from additional optimization.',
          'comparisons':{}}
  for control,stem in (('original','sft5_parent'),('previous_sft','sft5_tuned')):
    for condition in ('raw','guard'):
      before,after,pairs=pair_runs(args.reports/f'{stem}_{condition}',args.reports/f'expanded_tuned_{condition}',benchmark,(base,expanded))
      if before['benchmark_sha256'] != result['benchmark_sha256'] or after['checkpoint_sha256'] != training['checkpoint_sha256']:
        raise ValueError('Benchmark/checkpoint identity mismatch')
      if control=='previous_sft' and before['checkpoint_sha256'] != training['parent_sha256']:
        raise ValueError('Previous SFT checkpoint is not this run\'s parent')
      groups=defaultdict(list)
      for p in pairs:
        for field in ('category','track'):groups[(field,p[field])].append(p)
      counts=lambda rows:{'count':len(rows),'before_proxy':sum(p['parent_proxy'] for p in rows),'after_proxy':sum(p['tuned_proxy'] for p in rows)}
      key=f'{control}_{condition}'
      result['comparisons'][key]={'before':metrics(before),'after':metrics(after),
         'no_repeat':before['no_repeat'],'max_new_tokens':before['requested_max_new_tokens'],
         'proxy_improved':sum(not p['parent_proxy'] and p['tuned_proxy'] for p in pairs),
         'proxy_regressed':sum(p['parent_proxy'] and not p['tuned_proxy'] for p in pairs),
         'output_diagnostic':output_diagnostic(pairs),
         'categories':{v:counts(rows) for (field,v),rows in groups.items() if field=='category'},
         'tracks':{v:counts(rows) for (field,v),rows in groups.items() if field=='track'}}
      (args.reports/f'expanded_paired_{key}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
  args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:{f:v for f,v in r.items() if f!='output_diagnostic'} | {'output_matches':{f:v for f,v in r['output_diagnostic'].items() if f!='details'}} for k,r in result['comparisons'].items()},indent=2))

if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('reports','benchmark','base-splits','splits','training','output'):p.add_argument('--'+name,required=True,type=Path)
  compare(p.parse_args())
