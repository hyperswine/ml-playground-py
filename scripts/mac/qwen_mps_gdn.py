"""Time the pieces of Qwen3.5 GatedDeltaNet during cached decoding on MPS."""
import collections, json, sys, time
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
M = sys.argv[1]; dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
m = Qwen3_5ForConditionalGeneration.from_pretrained(M, dtype=torch.float32).to(dev).eval()
q = "<|im_start|>user\nWhat does this Sol program print?\n```sol\n> print (str (List.sum [1, 2, 3])).\n```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok(q, return_tensors="pt").input_ids.to(dev)
T = collections.defaultdict(float); C = collections.Counter()
def sync(): torch.mps.synchronize()
def wrap(name):
  f = getattr(mq, name)
  def g(*a, **k):
    sync(); t = time.perf_counter(); r = f(*a, **k); sync(); T[name] += time.perf_counter() - t; C[name] += 1; return r
  setattr(mq, name, g)
for n in ("causal_conv1d_update", "causal_conv1d_fn", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"): wrap(n)
def pre(mod, inp): sync(); mod._t = time.perf_counter()
def post(mod, inp, out): sync(); k = mod._k; T[k] += time.perf_counter() - mod._t; C[k] += 1
for layer in m.model.language_model.layers:
  la = getattr(layer, "linear_attn", None)
  if la is None: continue
  for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "norm", "out_proj"):
    sub = getattr(la, nm); sub._k = nm; sub.register_forward_pre_hook(pre); sub.register_forward_hook(post)
  la._k = "GatedDeltaNet_total"; la.register_forward_pre_hook(pre); la.register_forward_hook(post)
with torch.inference_mode():
  gen = lambda n: m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=False, return_dict_in_generate=True)
  gen(4); T.clear(); C.clear()
  out = gen(33)
  cache = out.past_key_values
  try: rp = [getattr(l, "record_past", None) for l in cache.layers][:4]
  except Exception as e: rp = str(e)
print(json.dumps({"record_past_first_layers": rp, "cache_type": type(cache).__name__,
  "seconds": {k: round(v, 3) for k, v in sorted(T.items(), key=lambda kv: -kv[1])}, "calls": dict(C)}, indent=1))
