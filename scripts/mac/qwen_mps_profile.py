"""Where does Qwen3.5 inference time go on MPS? Prefill vs decode, and per module type."""
import collections, json, sys, time, warnings
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
warnings.simplefilter("always")
M = sys.argv[1]; dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[sys.argv[2]]
dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
m = Qwen3_5ForConditionalGeneration.from_pretrained(M, dtype=dtype).to(dev).eval()
q = "<|im_start|>user\nWhat does this Sol program print?\n```sol\nxs = [3, 8, 12, 5].\n> print (str (List.sum (List.map (fn x -> x * 3) (List.filter (fn x -> x > 4) xs)))).\n```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok(q, return_tensors="pt").input_ids.to(dev)
long = tok(q * 12, return_tensors="pt").input_ids.to(dev)
def sync(): torch.mps.synchronize()
def timed(f, n=3):
  f(); sync(); t = time.perf_counter()
  for _ in range(n): f()
  sync(); return (time.perf_counter() - t) / n
out = {"model": M, "dtype": sys.argv[2], "prompt_tokens": ids.shape[1], "long_prompt_tokens": long.shape[1]}
with torch.inference_mode():
  out["prefill_s"] = timed(lambda: m(input_ids=ids))
  out["prefill_long_s"] = timed(lambda: m(input_ids=long))
  for bs in (1, 8):
    b = ids.repeat(bs, 1)
    g = lambda: m.generate(input_ids=b, attention_mask=torch.ones_like(b), max_new_tokens=64, min_new_tokens=64, do_sample=False)
    t = timed(g, 2); out[f"generate64_bs{bs}_s"] = t; out[f"decode_ms_per_step_bs{bs}"] = round((t - out["prefill_s"]) / 64 * 1000, 1)
  # Per-module-type time for one decode-like forward with cache: time each leaf-ish module with syncs.
  times = collections.defaultdict(float); counts = collections.Counter()
  def pre(mod, inp): sync(); mod._t = time.perf_counter()
  def post(mod, inp, outp): sync(); times[type(mod).__name__] += time.perf_counter() - mod._t; counts[type(mod).__name__] += 1
  hooks = []
  lm = m.model.language_model
  for layer in lm.layers:
    for name, sub in layer.named_children():
      hooks += [sub.register_forward_pre_hook(pre), sub.register_forward_hook(post)]
  b = ids
  g = lambda: m.generate(input_ids=b, attention_mask=torch.ones_like(b), max_new_tokens=32, min_new_tokens=32, do_sample=False)
  g(); times.clear(); counts.clear(); sync(); t = time.perf_counter(); g(); sync(); total = time.perf_counter() - t
  for h in hooks: h.remove()
  out["hooked_generate32_s"] = round(total, 3)
  out["by_module_s"] = {k: round(v, 3) for k, v in sorted(times.items(), key=lambda kv: -kv[1])}
  out["calls"] = dict(counts)
print(json.dumps(out, indent=1))
