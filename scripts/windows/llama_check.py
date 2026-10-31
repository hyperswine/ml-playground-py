"""llama.cpp parity and speed: answer a fixed sample of the frozen prompts through llama-server's /completion
endpoint (greedy) and compare with the GPU-pod answers. Same sample (seed 20261022) as the MPS and MLX checks.

  python3 llama_check.py URL prompts.jsonl reference_answers.jsonl N out.json
"""
import json, random, sys, time, urllib.request

url, prompts_path, reference_path, n, out_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
rows = [json.loads(l) for l in open(prompts_path)]
reference = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(reference_path)}
random.Random(20261022).shuffle(rows); rows = rows[:n]


def complete(prompt, budget):
  body = json.dumps({"prompt": prompt, "n_predict": budget, "temperature": 0, "top_k": 1, "seed": 0,
                     "cache_prompt": False, "stop": ["<|im_end|>", "<|endoftext|>"]}).encode()
  req = urllib.request.Request(url.rstrip("/") + "/completion", data=body, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=600) as r:
    return json.loads(r.read())


results, decode_tps, prompt_tps, tokens = [], [], [], 0
started = time.perf_counter()
for r in rows:
  out = complete(chat(r["question"]), r["max_new_tokens"])
  a, ref = out["content"].strip(), reference[r["id"]]
  t = out.get("timings", {})
  if t.get("predicted_n", 0) > 1: decode_tps.append(t["predicted_per_second"])
  if t.get("prompt_n", 0) > 1: prompt_tps.append(t["prompt_per_second"])
  tokens += out.get("tokens_predicted", 0)
  results.append({"id": r["id"], "local": a, "pod": ref, "identical": a == ref,
                  "same_first_line": a.split("\n")[0] == ref.split("\n")[0], "stop_type": out.get("stop_type")})
total = time.perf_counter() - started
median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
report = {"engine": "llama.cpp", "url": url, "n": len(results), "identical": sum(x["identical"] for x in results),
          "same_first_line": sum(x["same_first_line"] for x in results), "seconds": round(total, 1),
          "seconds_per_answer": round(total / len(results), 2), "generated_tokens": tokens,
          "median_decode_tokens_per_s": median(decode_tps), "median_prompt_tokens_per_s": median(prompt_tps)}
json.dump({"report": report, "answers": results}, open(out_path, "w"), indent=1)
print(json.dumps(report, indent=1))
