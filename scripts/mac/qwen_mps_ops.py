"""Count aten ops per decode step and compare wall time with CPU time (dispatch-bound or GPU-bound?)."""
import collections, json, sys, time, resource
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
M = sys.argv[1]; dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
m = Qwen3_5ForConditionalGeneration.from_pretrained(M, dtype=torch.float32).to(dev).eval()
ids = tok("<|im_start|>user\nWhat is 2 + 3 in Sol?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", return_tensors="pt").input_ids.to(dev)
class Count(TorchDispatchMode):
  def __init__(s): super().__init__(); s.c = collections.Counter()
  def __torch_dispatch__(s, func, types, args=(), kwargs=None):
    s.c[str(func.overloadpacket.__name__)] += 1; return func(*args, **(kwargs or {}))
with torch.inference_mode():
  gen = lambda n: m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=False)
  gen(4)
  with Count() as c1: gen(1)
  with Count() as c9: gen(9)
  per_step = {k: (c9.c[k] - c1.c[k]) / 8 for k in c9.c if c9.c[k] - c1.c[k] > 0}
  torch.mps.synchronize(); r0 = resource.getrusage(resource.RUSAGE_SELF); t0 = time.perf_counter()
  gen(64); torch.mps.synchronize()
  wall = time.perf_counter() - t0; r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
print(json.dumps({"aten_ops_per_decode_step": round(sum(per_step.values())), "top_ops": dict(sorted(per_step.items(), key=lambda kv: -kv[1])[:15]),
  "generate64_wall_s": round(wall, 2), "process_cpu_s": round(cpu, 2), "cpu_share_of_wall": round(cpu / wall, 2)}, indent=1))
