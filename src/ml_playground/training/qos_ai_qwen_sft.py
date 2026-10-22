"""Reference comparison: fine-tune Qwen3.5-0.8B on exactly the Stage 5 mixture and evaluate it like the 557M.

The Stage 5 mixture is stored in student token ids. The student vocabulary is a subset of Qwen's tokenizer and uses
Qwen's chat format (thinking disabled), so `qwen_ids[student_id]` recovers the exact Qwen token sequence: the same
examples, targets and masks, losslessly. Training uses the Stage 4/5 plan (one shuffled pass, seed 7, batch 8 x 4,
answer-only loss, cosine with 20 warmup updates, AdamW (0.9, 0.95)); only the learning rate differs (1e-5, for a
pretrained instruction model). Logits are computed only at target positions to keep the 248k-word head affordable.
The vision tower is frozen and unused.

  prompts   [local] every frozen evaluation prompt (Sol tests + probe, multi-input suite, original benchmarks)
  convert   [pod]   Stage 5 sft.pt -> Qwen token ids
  train     [pod]   full fine-tune, bf16 autocast, gradient checkpointing
  generate  [pod]   greedy answers for the prompt file (base or fine-tuned model)
  assemble  [local] split answers into the inputs of the unchanged scorers
"""
import argparse
import json
import math
import time
from pathlib import Path

QWEN = 'Qwen/Qwen3.5-0.8B'
SOURCES = [  # (suite, file, filter, budget)
  ('solgen', 'qos_ai/solgen_20261020/eval_questions.json', None, 200),
  ('multi', 'qos_ai_benchmark/stage4_20261021/multi_input.json', None, 200),
  ('dev', 'qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/benchmark.jsonl', 'dev', 200),
  ('coding', 'qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/coding_diagnostic.jsonl', None, 128),
  ('gen', 'qos_ai_benchmark/evidence/generalization_20261019/curricula/generalization_20261019/generalization_diagnostic.jsonl', None, 128),
]


def chat(question):
  import qos_ai as ai
  return ai.chat_text(question)


def prompts(args):
  from qos_ai_data import read_rows
  out = []
  for suite, path, split, budget in SOURCES:
    if path.endswith('.json'):
      data = json.load(open(path)); rows = data['rows'] if isinstance(data, dict) else data
    else:
      rows = read_rows(path)
    for r in rows:
      if split and r.get('split') != split: continue
      out.append({'id': f'{suite}/{r["id"]}', 'suite': suite, 'question': r['question'], 'max_new_tokens': budget})
  Path(args.output).write_text('\n'.join(json.dumps(r) for r in out) + '\n')
  print(json.dumps({s: sum(r['suite'] == s for r in out) for s, *_ in SOURCES}))


def convert(args):
  import torch
  from qos_ai_stage4_data import checked_load
  data = checked_load(args.sft)
  qwen_ids = torch.load(args.vocabulary, weights_only=True)['qwen_ids'].tolist()
  examples = []
  for ids, mask in data['verified']:
    if 0 in ids: raise ValueError('unknown token in mixture')
    examples.append(([qwen_ids[i] for i in ids], list(mask)))
  torch.save({'examples': examples, 'source_sft': str(args.sft)}, args.output)
  print(json.dumps({'examples': len(examples), 'target_tokens': sum(sum(m[1:]) for _, m in examples)}))


def merge(args):
  """Append round-2 rows (question/answer text) to a converted mixture, tokenized with Qwen's tokenizer."""
  import torch
  import qos_ai as ai
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(args.tokenizer)
  data = torch.load(args.sft, weights_only=True); examples = list(data['examples']); base = len(examples)
  report = {'base_examples': base, 'sources': {}, 'skipped_too_long': 0, 'skipped_boundary': 0}
  for path in args.rows:
    for r in map(json.loads, open(path)):
      prompt = tok.encode(ai.chat_text(r['question']), add_special_tokens=False)
      ids = tok.encode(ai.chat_text(r['question'], r['answer']), add_special_tokens=False)
      if len(ids) > args.max_length: report['skipped_too_long'] += 1; continue
      if ids[:len(prompt)] != prompt: report['skipped_boundary'] += 1; continue
      mask = [i >= len(prompt) for i in range(len(ids))]
      examples.append((ids, mask))
      s = report['sources'].setdefault(r.get('source', 'unknown'), {'examples': 0, 'target_tokens': 0})
      s['examples'] += 1; s['target_tokens'] += sum(mask[1:])
  report['examples'] = len(examples)
  report['base_target_tokens'] = sum(sum(m[1:]) for _, m in examples[:base])
  report['added_target_tokens'] = sum(sum(m[1:]) for _, m in examples[base:])
  report['max_length'] = max(len(ids) for ids, _ in examples)
  torch.save({'examples': examples, 'source_sft': data.get('source_sft'), 'merge': report}, args.output)
  print(json.dumps(report))


def micro_batches(members, examples, token_budget=2048, target_budget=1024):
  """Split one update's examples, sorted by length, into micro-batches of at most token_budget padded tokens and
  target_budget answer tokens (a single example always fits). Memory then never exceeds the Stage 5 worst case
  (4 x 512 tokens), however long the retrieval examples are, and short examples pack more densely."""
  out, cur, cur_targets = [], [], 0
  for i in sorted(members, key=lambda i: len(examples[i][0])):
    n, t = len(examples[i][0]), sum(examples[i][1][1:])
    if cur and ((len(cur) + 1) * n > token_budget or cur_targets + t > target_budget):
      out.append(cur); cur, cur_targets = [], 0
    cur.append(i); cur_targets += t
  return out + ([cur] if cur else [])


def rate(base, step, total, warmup):
  if step <= warmup: return base * step / max(1, warmup)
  return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup)))))


def train(args):
  import torch
  from torch.nn import functional as F
  from transformers import Qwen3_5ForConditionalGeneration
  from qos_ai_resume import optimizer_plan
  torch.manual_seed(args.seed)
  data = torch.load(args.data, weights_only=True); examples = data['examples']
  plan = optimizer_plan(len(examples), args.batch_size, args.accumulation, 1, args.seed)
  model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).cuda()
  for p in model.model.visual.parameters(): p.requires_grad_(False)
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
  model.config.use_cache = False; model.train()
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.AdamW(params, lr=args.learning_rate, betas=(.9, .95), weight_decay=.1, fused=True)
  pad = args.pad_id; lm = model.model.language_model; head = model.lm_head
  status = Path(args.output) / 'status.json'; Path(args.output).mkdir(parents=True, exist_ok=True)
  history = []; started = time.perf_counter(); tokens_seen = 0
  for index, group in enumerate(plan):
    step = index + 1; lr = rate(args.learning_rate, step, len(plan), 20)
    for g in opt.param_groups: g['lr'] = lr
    known = sum(sum(examples[i][1][1:]) for b in group['batches'] for i in b)
    total = 0.
    # Same examples and summed loss per update; packing them by length only changes how they are split.
    members = [i for b in group['batches'] for i in b]
    for indexes in micro_batches(members, examples, args.token_budget, args.target_budget):
      batch = [examples[i] for i in indexes]; width = max(len(ids) for ids, _ in batch)
      ids = torch.full((len(batch), width), pad, dtype=torch.long); att = torch.zeros_like(ids); tgt = torch.zeros_like(ids, dtype=torch.bool)
      for j, (x, m) in enumerate(batch):
        ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1; tgt[j, :len(m)] = torch.tensor(m)
      ids, att, tgt = ids.cuda(), att.cuda(), tgt.cuda()
      with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = lm(input_ids=ids, attention_mask=att).last_hidden_state
        sel = tgt[:, 1:]
        logits = head(hidden[:, :-1][sel])
      loss = F.cross_entropy(logits.float(), ids[:, 1:][sel], reduction='sum')
      if not torch.isfinite(loss): raise ValueError('nonfinite loss')
      (loss / known).backward(); total += loss.item(); tokens_seen += int(att.sum())
    norm = torch.nn.utils.clip_grad_norm_(params, 1.).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    history.append({'step': step, 'loss': total / known, 'learning_rate': lr, 'gradient_norm': norm, 'target_tokens': known})
    if step % 50 == 0 or step == len(plan):
      row = {**history[-1], 'total_steps': len(plan), 'tokens_per_second': tokens_seen / (time.perf_counter() - started)}
      status.write_text(json.dumps(row)); print(json.dumps(row), flush=True)
  model.config.use_cache = True
  model.to(torch.bfloat16)  # generation runs in bf16; halves the artifact to pull
  model.save_pretrained(Path(args.output) / 'model', safe_serialization=True)
  (Path(args.output) / 'result.json').write_text(json.dumps({'steps': len(plan), 'seed': args.seed, 'learning_rate': args.learning_rate,
    'batch_size': args.batch_size, 'accumulation': args.accumulation, 'token_budget': args.token_budget,
    'target_budget': args.target_budget, 'history': history}) + '\n')


def generate(args):
  import torch
  from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
  tok = AutoTokenizer.from_pretrained(args.tokenizer)
  model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
  stop = [tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')]
  rows = [json.loads(l) for l in open(args.prompts)]
  rows.sort(key=lambda r: (r['max_new_tokens'], len(r['question'])))
  out = []
  for start in range(0, len(rows), args.batch_size):
    part = rows[start:start + args.batch_size]
    encoded = [tok.encode(chat(r['question']), add_special_tokens=False) for r in part]
    width = max(map(len, encoded))
    ids = torch.full((len(part), width), tok.pad_token_id or stop[1], dtype=torch.long); att = torch.zeros_like(ids)
    for j, e in enumerate(encoded): ids[j, width - len(e):] = torch.tensor(e); att[j, width - len(e):] = 1
    with torch.inference_mode():
      gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=max(r['max_new_tokens'] for r in part),
                           do_sample=False, eos_token_id=stop, pad_token_id=ids[0, 0].item())
    for r, g in zip(part, gen[:, width:].tolist()):
      g = g[:r['max_new_tokens']]
      end = next((i for i, t in enumerate(g) if t in stop), None)
      body = g if end is None else g[:end]
      out.append({'id': r['id'], 'answer': tok.decode(body, skip_special_tokens=True).strip(), 'finished': end is not None, 'tokens': len(body)})
    print(f'{min(start + args.batch_size, len(rows))}/{len(rows)}', flush=True)
  Path(args.output).write_text('\n'.join(json.dumps(r) for r in out) + '\n')


def assemble(args):
  """Write scorer inputs: run_probe-style lists, the multi-input list and qos_ai_noguard-style track directories."""
  import qos_ai as ai
  from qos_ai_data import lexical_score, read_rows, rows_fingerprint
  ans = {json.loads(l)['id']: json.loads(l) for l in open(args.answers)}
  o = Path(args.output); o.mkdir(parents=True, exist_ok=True)
  pick = lambda suite: [{**v, 'id': k.split('/', 1)[1]} for k, v in ans.items() if k.startswith(suite + '/')]
  (o / 'solgen_answers.json').write_text(json.dumps(pick('solgen'), indent=1))
  multi = pick('multi')
  for a in multi: a['severe_loop'] = False
  (o / 'multi_input_answers.json').write_text(json.dumps(multi, indent=1))
  for track, suite, path, split, budget in (('development', 'dev', SOURCES[2][1], 'dev', 200),
                                             ('coding', 'coding', SOURCES[3][1], None, 128),
                                             ('generalization', 'gen', SOURCES[4][1], None, 128)):
    rows = [r for r in read_rows(path) if not split or r.get('split') == split]
    d = o / 'bench' / track; d.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in rows:
      a = ans[f'{suite}/{r["id"]}']
      rec = {'id': r['id'], 'question': r['question'], 'answer': a['answer'], 'finished': a['finished'],
             'tokens': a['tokens'], 'severe_loop': False}
      if track == 'development': rec.update(lexical_score(r, a['answer']))
      recs.append(rec)
    (d / 'answers.jsonl').write_text('\n'.join(json.dumps(x) for x in recs) + '\n')
    (d / 'summary.json').write_text(json.dumps({'checkpoint_sha256': args.label, 'diagnostic_hash': rows_fingerprint(rows),
      'seed': 7, 'temperature': 0, 'top_k': 0, 'no_repeat': 0, 'max_new_tokens': budget, 'device': 'cuda',
      'count': len(recs), 'finished': sum(x['finished'] for x in recs), 'severe_loops': 0, 'mean_tokens': 0}, indent=2))
  print('assembled', o)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  s = p.add_subparsers(dest='command', required=True)
  c = s.add_parser('prompts'); c.add_argument('--output', required=True)
  c = s.add_parser('convert'); c.add_argument('--sft', required=True); c.add_argument('--vocabulary', required=True); c.add_argument('--output', required=True)
  c = s.add_parser('train'); c.add_argument('--data', required=True); c.add_argument('--model', default=QWEN); c.add_argument('--output', required=True)
  c.add_argument('--learning-rate', type=float, default=1e-5); c.add_argument('--batch-size', type=int, default=4)
  c.add_argument('--accumulation', type=int, default=8); c.add_argument('--pad-id', type=int, default=248044)
  c.add_argument('--seed', type=int, default=7, help='data order and initialization of the training run')
  c.add_argument('--token-budget', type=int, default=2048, help='padded tokens per micro-batch')
  c.add_argument('--target-budget', type=int, default=1024, help='answer tokens per micro-batch')
  c = s.add_parser('generate'); c.add_argument('--model', required=True); c.add_argument('--tokenizer', default=QWEN)
  c.add_argument('--prompts', required=True); c.add_argument('--output', required=True); c.add_argument('--batch-size', type=int, default=32)
  c = s.add_parser('assemble'); c.add_argument('--answers', required=True); c.add_argument('--output', required=True); c.add_argument('--label', required=True)
  c = s.add_parser('merge'); c.add_argument('--sft', required=True); c.add_argument('--rows', nargs='+', required=True)
  c.add_argument('--output', required=True); c.add_argument('--tokenizer', default=QWEN); c.add_argument('--max-length', type=int, default=2048)
  a = p.parse_args()
  {'prompts': prompts, 'convert': convert, 'train': train, 'generate': generate, 'assemble': assemble, 'merge': merge}[a.command](a)
