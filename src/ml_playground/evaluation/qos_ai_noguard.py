"""Rerun the frozen dev/coding/generalization prompts with a chosen no-repeat setting, then grade.

Same prompts, seed, greedy decoding, top-k and token budgets as the original
evaluations; only `--no-repeat` changes. Single-prompt generation on the M1,
matching the 168M protocol. Graders are the existing frozen ones.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from qos_ai_data import file_hash, lexical_score, read_rows
from qos_ai_compare import stated_output

TRACKS = [('development', 'benchmark.jsonl', 200), ('coding', 'coding_diagnostic.jsonl', 128),
          ('generalization', 'generalization_diagnostic.jsonl', 128)]


def generate(args):
  import torch
  import qos_ai as ai
  device = ai.choose_device('mps')
  model, parent = ai.load_student(args.load, device)
  tok = ai.StudentTokenizer(parent['qwen_ids'])
  for name, filename, budget in TRACKS:
    rows = read_rows(args.data / filename)
    if name == 'development':
      rows = [r for r in rows if r['split'] == 'dev']
    out = args.output / name
    if (out / 'summary.json').exists():
      continue
    out.mkdir(parents=True, exist_ok=True)
    (out / 'answers.jsonl').unlink(missing_ok=True)
    torch.manual_seed(7)
    answers = []
    for row in rows:
      prompt = tok.encode(ai.chat_text(row['question']))
      if 0 in prompt or len(prompt) + budget > model.config.context_length:
        raise ValueError('Unknown or truncated prompt: ' + row['id'])
      with torch.inference_mode():
        ids, finished = model.generate_batch([prompt], budget, parent['im_end_id'], 0, 20, args.no_repeat)[0]
      answer = tok.decode(ids).strip()
      record = {'id': row['id'], 'question': row['question'], 'answer': answer, 'finished': finished,
                'tokens': len(ids), 'severe_loop': ai.loops(ids, phrase=4, times=4)}
      if name == 'development':
        record.update(lexical_score(row, answer))
      answers.append(record)
      with (out / 'answers.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    summary = {'checkpoint_sha256': file_hash(args.load), 'diagnostic_hash': None, 'seed': 7, 'temperature': 0,
               'top_k': 20, 'no_repeat': args.no_repeat, 'max_new_tokens': budget, 'device': str(device),
               'host': platform.node(), 'torch': str(torch.__version__), 'count': len(answers),
               'finished': sum(a['finished'] for a in answers), 'severe_loops': sum(a['severe_loop'] for a in answers),
               'mean_tokens': sum(a['tokens'] for a in answers) / len(answers)}
    from qos_ai_data import rows_fingerprint
    summary['diagnostic_hash'] = rows_fingerprint(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(name, 'done', summary['finished'], '/', summary['count'], 'finished', flush=True)


def grade(args):
  from qos_ai_generalization_trace_audit import strict_trace
  subprocess.run([sys.executable, 'qos_ai_coding_diag.py', 'grade', '--data', str(args.data),
                  '--reports', str(args.output / 'coding')], check=True, capture_output=True)
  subprocess.run([sys.executable, 'qos_ai_generalization_eval.py', 'grade', '--data', str(args.data),
                  '--output', str(args.output / 'generalization')], check=True, capture_output=True)
  bench = [r for r in read_rows(args.data / 'benchmark.jsonl') if r['split'] == 'dev']
  dev = read_rows(args.output / 'development/answers.jsonl')
  outputs = [(r, a) for r, a in zip(bench, dev) if r['category'] == 'output_prediction']
  coding = json.loads((args.output / 'coding/grading.json').read_text())
  general = json.loads((args.output / 'generalization/grading.json').read_text())
  diag = {r['id']: r for r in read_rows(args.data / 'generalization_diagnostic.jsonl')}
  gen_answers = {a['id']: a['answer'] for a in read_rows(args.output / 'generalization/answers.jsonl')}
  traces = [strict_trace(diag[g['id']], gen_answers[g['id']]) for g in general['results']
            if diag[g['id']]['mode'] != 'output_only']
  c = coding['categories']; s = general['strata']
  summary = {
    'checkpoint_sha256': coding['checkpoint_sha256'],
    'no_repeat': json.loads((args.output / 'coding/summary.json').read_text())['no_repeat'],
    'dev_lexical_90': sum(a['rubric_proxy_pass'] for a in dev),
    'dev_outputs_33': sum(stated_output(r['answer']) == stated_output(a['answer']) for r, a in outputs),
    'working_generation_24': c['code_generation']['behavior_pass'],
    'applied_repairs_48': c['debugging']['behavior_pass'],
    'repair_plus_output_48': c['debugging']['complete_pass'],
    'no_change_plus_output_24': c['no_change']['complete_pass'],
    'gen_wording_48': s['wording']['complete_pass'], 'gen_inputs_48': s['inputs']['complete_pass'],
    'gen_combinations_48': s['combinations']['complete_pass'],
    'strict_trace_120': sum(t['strict_canonical_trace_match'] for t in traces),
    'finished': {t: json.loads((args.output / t / 'summary.json').read_text())['finished'] for t, _, _ in TRACKS},
    'severe_loops': {t: json.loads((args.output / t / 'summary.json').read_text())['severe_loops'] for t, _, _ in TRACKS},
  }
  (args.output / 'scores.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('command', choices=('generate', 'grade', 'both'))
  p.add_argument('--load', type=Path)
  p.add_argument('--data', type=Path, default=Path('curricula/generalization_20261019'))
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--no-repeat', type=int, default=0)
  a = p.parse_args()
  started = time.time()
  if a.command in ('generate', 'both'):
    generate(a)
  if a.command in ('grade', 'both'):
    grade(a)
  print('seconds', round(time.time() - started), flush=True)
