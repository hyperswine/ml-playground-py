"""Grade backed-up RunPod answers on the M1 using its frozen Sol toolchain.

This transfers generated responses only. It never starts GPU resources or
loads cloud credentials. The training controller calls it after backup.
"""
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).absolute().parent
EVIDENCE = ROOT / 'qos_ai_benchmark/evidence/runpod_557m_20261019'
REMOTE_ROOT = '/Users/jq/qos_ai_eval'
REMOTE_REPORT = 'reports/runpod_557m_20261019'
HOST = 'jq@100.107.153.43'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new']

FROZEN = {
  'qos_ai_coding_diag.py': 'e4e1cf111bc85249a721df58d942de5ebbddc92d857ded48bafa5e6ccc28d348',
  'qos_ai_generalization_eval.py': 'e9cceff554e8687429f14c04c47614881b9c25f4a29deec88f3fed2bc30faed6',
  'qos_ai_generalization.py': '2f3dbb990496415f5b37849936302d7b2f1947f8bc5d26bb071604e89178be1b',
  'qos_ai_generalization_trace_audit.py': 'b7eb266a59e34ea4d23db4aeb8e104fbe6d36e3a7ae060895125ee55ff20e2ce',
  'qos_ai_pure_sol.py': 'b64fa21c1553e92fa5f482a4a67dd4cbdd08a24da33adb7274cf0981ed76b187',
  'qos_ai_compare.py': '5b36d9f618c7ff643c4b1b45205bd4abc7738bd1c40d34dfc691d5866da279ff',
  'qos_ai_data.py': '37eb78c7661f52aa277a18889c7c88fb312feadc808f0315eb3636ffe5a0d0de',
  'source_snapshot/fprisc/fpr': 'c41c6ab9377e59be4d9bccedf1d6dd33f188bad0453046660650c6982fbc2f36',
  'curricula/generalization_20261019/benchmark.jsonl': '1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7',
  'curricula/generalization_20261019/coding_diagnostic.jsonl': '5328ff3c3cf01b132322349ef1830959b325a302926b437c78d32edaef8d59ce',
  'curricula/generalization_20261019/generalization_diagnostic.jsonl': '67b2155a65688599b10d79d77413fec8061bc2c3a73948fee13f0dd9fc4c4cda',
  'reports/generalization_20261019/comparison.json': '5a40b58dffc02d3b936080d0ec6aac440ed9660f7d8b70963f569dc9eb287fd6',
}

REMOTE_GRADE = r'''
import json, subprocess, sys
from pathlib import Path
from qos_ai_data import file_hash, read_rows, lexical_score
from qos_ai_compare import stated_output
from qos_ai_generalization_trace_audit import strict_trace
root=Path(globals().get('REPORT_ROOT','reports/runpod_557m_20261019')); data=Path(globals().get('DATA_ROOT','curricula/generalization_20261019'))
benchmark=[r for r in read_rows(data/'benchmark.jsonl') if r['split']=='dev']
diag={r['id']:r for r in read_rows(data/'generalization_diagnostic.jsonl')}
results={}
for label in ('pretrained','sft'):
 p=root/'evaluation'/label
 if not globals().get('SKIP_EXECUTION',False):
  subprocess.run([sys.executable,'qos_ai_coding_diag.py','grade','--data',str(data),'--reports',str(p/'coding')],check=True)
  subprocess.run([sys.executable,'qos_ai_generalization_eval.py','grade','--data',str(data),'--output',str(p/'generalization')],check=True)
 summary=json.loads((p/'development/summary.json').read_text()); answers=read_rows(p/'development/answers.jsonl')
 assert [(r['id'],r['question']) for r in benchmark]==[(a['id'],a['question']) for a in answers]
 assert summary['benchmark_sha256']==file_hash(data/'benchmark.jsonl')
 assert (summary['temperature'],summary['top_k'],summary['no_repeat'],summary['max_new_tokens'])==(0,20,3,200)
 outputs=[(r,a) for r,a in zip(benchmark,answers) if r['category']=='output_prediction']
 assert all(stated_output(r['answer']) is not None for r,a in outputs)
 development={'count':len(answers),'lexical_matches':sum(lexical_score(r,a['answer'])['rubric_proxy_pass'] for r,a in zip(benchmark,answers)),
  'output_count':len(outputs),'output_matches':sum(stated_output(r['answer'])==stated_output(a['answer']) for r,a in outputs),
  'finished':sum(a['finished'] for a in answers),'severe_loops':sum(a['severe_loop'] for a in answers)}
 coding=json.loads((p/'coding/grading.json').read_text()); general=json.loads((p/'generalization/grading.json').read_text())
 for name,grading in [('coding',coding),('generalization',general)]:
  settings=json.loads((p/name/'summary.json').read_text())
  assert settings['checkpoint_sha256']==summary['checkpoint_sha256']==grading['checkpoint_sha256']
  assert (settings['temperature'],settings['top_k'],settings['no_repeat'],settings['max_new_tokens'])==(0,20,3,128)
  assert grading['answers_sha256']==file_hash(p/name/'answers.jsonl')
 answers_by_id={a['id']:a['answer'] for a in read_rows(p/'generalization/answers.jsonl')}
 traces=[]
 for g in general['results']:
  row=diag[g['id']]
  if row['mode']=='output_only':continue
  audit=strict_trace(row,answers_by_id[g['id']]); traces.append({'id':g['id'],'answer_pass':g['complete_pass'],**audit})
 strict={'count':len(traces),'strict_trace_passes':sum(t['strict_canonical_trace_match'] for t in traces),
  'answer_and_trace_passes':sum(t['answer_pass'] and t['strict_canonical_trace_match'] for t in traces),'details':traces}
 (p/'generalization/strict_trace.json').write_text(json.dumps(strict,indent=2)+'\n')
 results[label]={'checkpoint_sha256':summary['checkpoint_sha256'],'development':development,
  'coding':{k:v for k,v in coding.items() if k!='results'},'generalization':{k:v for k,v in general.items() if k!='results'},
  'strict_trace':{k:v for k,v in strict.items() if k!='details'}}
baseline=Path(globals().get('BASELINE_FILE','reports/generalization_20261019/comparison.json'))
prior=json.loads(baseline.read_text())
historical={}
for key in ('parent','seed7_final'):
 r=prior['runs'][key];historical[key]={'checkpoint_sha256':r['checkpoint_sha256'],'development':r['development']['guard'],'coding':r['coding'],'generalization':r['generalization'],
  'strict_trace':prior['supplemental_strict_trace_audit']['runs']['guard'][key]}
report={'runs':results,'historical_168m':historical,'historical_report_sha256':file_hash(baseline),
 'scope':'Same frozen development prompts and decoding, CUDA fp32 generations with TF32 allowed and M1 executed grading. Historical 168M generations used M1 fp32. Larger architecture, new random lineage, public pretraining and SFT differ together. This budget pilot is not an isolated parameter-count comparison or a clean unseen-exposure claim.'}
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
'''


def command(args, **kwargs):
  return subprocess.run(args, check=True, timeout=1800, text=True, **kwargs)


def ssh(code):
  return command(['ssh', *OPTIONS, HOST, 'cd ' + shlex.quote(REMOTE_ROOT) + ' && .venv/bin/python -'], input=code, capture_output=True)


def check_m1():
  code = 'import hashlib,json\nfrom pathlib import Path\nexpected=' + repr(FROZEN) + '\n'
  code += "for p,h in expected.items():\n assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p\nprint(json.dumps({'frozen_m1_inputs_verified':True,'files':expected}))\n"
  result = ssh(code)
  EVIDENCE.mkdir(parents=True, exist_ok=True)
  (EVIDENCE / 'm1_frozen_inputs.json').write_text(result.stdout)
  return result.stdout


def main():
  import argparse
  p = argparse.ArgumentParser(); p.add_argument('--check-m1', action='store_true')
  p.add_argument('--grade-only', action='store_true'); p.add_argument('--reuse-grading', action='store_true')
  p.add_argument('--from-local-grades', action='store_true', help='Assemble already completed M1 grading evidence without any network or program execution')
  args = p.parse_args()
  if not args.from_local_grades:
    print(check_m1(), flush=True)
  if args.check_m1:
    return
  source = EVIDENCE / 'remote/evaluation'
  if not (source / 'sft/generalization/summary.json').exists():
    raise ValueError('Complete backed-up generations are required before grading')
  target = EVIDENCE / 'graded'; target.mkdir(exist_ok=True)
  if args.from_local_grades:
    from qos_ai_data import file_hash
    for name, sha in FROZEN.items():
      if name.endswith('.py') and file_hash(ROOT / name) != sha:
        raise ValueError('Local report parser differs from frozen M1 version: ' + name)
    for label in ('pretrained','sft'):
      for track, grader in [('coding','qos_ai_coding_diag.py'),('generalization','qos_ai_generalization_eval.py')]:
        g=json.loads((target/'evaluation'/label/track/'grading.json').read_text())
        if g['compiler_sha256'] != FROZEN['source_snapshot/fprisc/fpr'] or g['grader_sha256'] != FROZEN[grader]:
          raise ValueError('Recovered M1 grader/compiler identity differs')
    baseline = ROOT / 'qos_ai_benchmark/evidence/generalization_20261019/reports/generalization_20261019/comparison.json'
    if file_hash(baseline) != FROZEN['reports/generalization_20261019/comparison.json']:
      raise ValueError('Historical baseline identity differs')
    exec(compile(REMOTE_GRADE,'assemble_completed_m1_grades','exec'),
         {'REPORT_ROOT':str(target),'DATA_ROOT':str(ROOT/'runpod/557m/bundle/benchmarks'),
          'BASELINE_FILE':str(baseline),'SKIP_EXECUTION':True})
    (EVIDENCE/'comparison.json').write_bytes((target/'comparison.json').read_bytes())
  elif not args.reuse_grading:
    ssh("from pathlib import Path\nPath('" + REMOTE_REPORT + "').mkdir(parents=True,exist_ok=True)\n")
    command(['scp', *OPTIONS, '-r', str(source), HOST + ':' + REMOTE_ROOT + '/' + REMOTE_REPORT + '/'])
    result = ssh(REMOTE_GRADE)
    (EVIDENCE / 'm1_grading.log').write_text(result.stdout + result.stderr)
    command(['scp', *OPTIONS, '-r', HOST + ':' + REMOTE_ROOT + '/' + REMOTE_REPORT + '/evaluation', str(target) + '/'])
    command(['scp', *OPTIONS, HOST + ':' + REMOTE_ROOT + '/' + REMOTE_REPORT + '/comparison.json', str(EVIDENCE / 'comparison.json')])
  report = json.loads((EVIDENCE / 'comparison.json').read_text())
  from qos_ai_data import file_hash
  for label in ('pretrained', 'sft'):
    for track in ('development', 'coding', 'generalization'):
      original = source / label / track; graded = target / 'evaluation' / label / track
      for name in ('answers.jsonl', 'summary.json'):
        if file_hash(original / name) != file_hash(graded / name):
          raise ValueError('Grading inputs differ from recovered generation: ' + label + '/' + track)
      settings = json.loads((original / 'summary.json').read_text())
      if settings['checkpoint_sha256'] != report['runs'][label]['checkpoint_sha256']:
        raise ValueError('Grading checkpoint identity differs')
  if args.grade_only:
    print(json.dumps({'m1_grading_complete': True, 'weight_backup_verification_pending': True, 'runs': report['runs']}), flush=True)
    return
  # Bind copied grading to the already verified local model identities.
  checksums = json.loads((ROOT / 'qos_ai/runpod_557m_20261019/checksums.json').read_text())
  for label, filename in [('pretrained', 'pretrained.pt'), ('sft', 'student.pt')]:
    if report['runs'][label]['checkpoint_sha256'] != checksums[filename]:
      raise ValueError('M1 grades are for a different checkpoint')
  metadata = EVIDENCE / 'remote'
  report['training'] = json.loads((metadata / 'pretraining_result.json').read_text())
  report['initial_validation'] = json.loads((metadata / 'initial_validation.json').read_text())
  report['sft'] = json.loads((metadata / 'sft_result.json').read_text())
  report['backed_up_checksums'] = checksums
  (EVIDENCE / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
  table = ['\n## Completed GPU experiment and M1 grading\n',
           '| Checkpoint | Development lexical /90 | Stated outputs /33 | Working generation /24 | Applied repairs /48 | Complete repairs /48 | No-change + output /24 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
  display = [('168M coding parent', report['historical_168m']['parent']),
             ('168M broader SFT, seed 7', report['historical_168m']['seed7_final']), *report['runs'].items()]
  for label, row in display:
    d = row['development']; c = row['coding']['categories']
    table.append(f"| {label} | {d['lexical_matches']} | {d['output_matches']} | {c['code_generation']['behavior_pass']} | {c['debugging']['behavior_pass']} | {c['debugging']['complete_pass']} | {c['no_change']['complete_pass']} |")
  table.extend(['', '| Checkpoint | Wording /48 | Inputs /48 | Combinations /48 | Strict trace /120 |',
                '| --- | ---: | ---: | ---: | ---: |'])
  for label, row in display:
    g = row['generalization']['strata']; trace = row['strict_trace']
    table.append(f"| {label} | {g['wording']['complete_pass']} | {g['inputs']['complete_pass']} | {g['combinations']['complete_pass']} | {trace['strict_trace_passes']} |")
  table.extend(['', report['scope'], '', 'Full results: [comparison.json](evidence/runpod_557m_20261019/comparison.json). The default and existing checkpoints are preserved.'])
  t = report['training']
  table.extend(['', f"Pretraining processed {t['processed_tokens']:,} tokens in {t['accepted_steps']:,} optimizer updates; frozen target {t['target_tokens']:,} tokens. Budget stop: {t['budget_stop']}. Verified SFT processed {report['sft']['answer_tokens']:,} answer tokens."])
  for kind in ('english', 'python'):
    initial = report['initial_validation'][kind]['cross_entropy']; final = t['validation'][kind]['cross_entropy']
    table.append(f"Fixed {kind} validation cross-entropy: {initial:.4f} to {final:.4f}.")
  path = ROOT / 'qos_ai_benchmark/RUNPOD_557M_2026-10-19.md'
  text = path.read_text(); marker = '\n## Completed GPU experiment and M1 grading\n'
  text = re.sub(r'Status: \*\*[^\n]*\*\*\.', 'Status: **training, verified backup and M1 grading complete**.', text, count=1)
  text = text.replace('# 557M RunPod training preparation', '# 557M RunPod training pilot', 1)
  path.write_text(text.split(marker)[0] + '\n'.join(table) + '\n')
  print(json.dumps({'m1_grading_complete': True, 'runs': report['runs']}), flush=True)


if __name__ == '__main__':
  main()
