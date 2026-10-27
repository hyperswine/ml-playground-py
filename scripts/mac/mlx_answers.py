"""Greedy MLX answers for every frozen prompt, in the pod tuned_answers.jsonl format."""
import json, sys, time
from mlx_lm import load, stream_generate
model_dir, prompts_path, out_path = sys.argv[1:4]
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
model, tok = load(model_dir)
stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
t = time.perf_counter()
with open(out_path, "w") as f:
  for r in map(json.loads, open(prompts_path)):
    out, finished = [], False
    for resp in stream_generate(model, tok, prompt=tok.encode(chat(r["question"]), add_special_tokens=False), max_tokens=r["max_new_tokens"]):
      if resp.token in stop: finished = True; break
      out.append(resp.token)
    f.write(json.dumps({"id": r["id"], "answer": tok.decode(out).strip(), "finished": finished, "tokens": len(out)}) + "\n")
print(json.dumps({"model": model_dir, "seconds": round(time.perf_counter() - t)}))
