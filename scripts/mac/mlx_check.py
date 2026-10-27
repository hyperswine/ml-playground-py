"""MLX parity and speed: answer a fixed sample of the frozen prompts greedily and compare with the GPU-pod answers."""
import json, random, sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
model_dir, prompts_path, reference_path, n, out_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
rows = [json.loads(l) for l in open(prompts_path)]
reference = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(reference_path)}
random.Random(20261022).shuffle(rows); rows = rows[:n]
model, tok = load(model_dir)
stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
results, gen_tokens, gen_time, prompt_tps = [], 0, 0.0, []
started = time.perf_counter()
for r in rows:
  ids = tok.encode(chat(r["question"]), add_special_tokens=False)
  out, t0, first = [], None, None
  for resp in stream_generate(model, tok, prompt=ids, max_tokens=r["max_new_tokens"]):
    if resp.token in stop: break
    out.append(resp.token)
    if first is None: first = time.perf_counter(); prompt_tps.append(resp.prompt_tps)
  if out: gen_tokens += len(out) - 1; gen_time += time.perf_counter() - first
  a = tok.decode(out).strip(); ref = reference[r["id"]]
  results.append({"id": r["id"], "local": a, "pod": ref, "identical": a == ref, "same_first_line": a.split("\n")[0] == ref.split("\n")[0]})
total = time.perf_counter() - started
report = {"engine": "mlx", "model": model_dir, "n": len(results), "identical": sum(r["identical"] for r in results),
  "same_first_line": sum(r["same_first_line"] for r in results), "seconds": round(total, 1),
  "seconds_per_answer": round(total / len(results), 2), "decode_tokens_per_s": round(gen_tokens / gen_time, 1),
  "median_prompt_tokens_per_s": sorted(prompt_tps)[len(prompt_tps) // 2], "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}
json.dump({"report": report, "answers": results}, open(out_path, "w"), indent=1); print(json.dumps(report, indent=1))
