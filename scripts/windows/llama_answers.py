"""Greedy llama-server answers for every frozen prompt, in the pod tuned_answers.jsonl format."""
import json, sys, time, urllib.request
url, prompts_path, out_path = sys.argv[1:4]
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
t = time.perf_counter()
with open(out_path, "w") as f:
  for r in map(json.loads, open(prompts_path)):
    body = json.dumps({"prompt": chat(r["question"]), "n_predict": r["max_new_tokens"], "temperature": 0, "top_k": 1,
                       "cache_prompt": False, "stop": ["<|im_end|>", "<|endoftext|>"]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    f.write(json.dumps({"id": r["id"], "answer": out["content"].strip(), "finished": out.get("stop_type") in ("eos", "word"),
                        "tokens": out.get("tokens_predicted", 0)}) + "\n")
print(json.dumps({"answers_file": out_path, "seconds": round(time.perf_counter() - t)}))
