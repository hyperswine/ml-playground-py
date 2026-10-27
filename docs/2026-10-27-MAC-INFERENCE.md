# Mac inference: why Qwen3.5 was slow on the M1, and MLX as the fix

Created: 2026-10-27T13:05:00+11:00 (Australia/Sydney). Kind: investigation report. Machine: M1 Mac mini, 16 GB
(about 68 GB/s memory bandwidth), torch 2.13.0, transformers 5.18.0.dev0, mlx 0.32.4, mlx-lm 0.32.2. Models: the
round-4 Qwen3.5-0.8B and Qwen3.5-2B fine-tunes ([round 4](2026-10-26-QWEN-ROUND4.md), [2B](2026-10-27-QWEN-2B.md)).
No cloud compute. Scripts: `scripts/mac/`; raw data: `data/qos_ai_benchmark/mac_inference_20261027/`.

## Symptom

Transformers warns on the Mac that `causal_conv1d` and the `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`
kernels "fall back to the reference PyTorch implementation". Qwen3.5 has 18 Gated DeltaNet (linear attention) layers
and 6 full-attention layers. The optimized kernels (`flash-linear-attention`, `causal-conv1d`) are Triton/CUDA and
cannot run on Apple GPUs, so the fallback is the only PyTorch option there.

## Measurements (PyTorch on MPS, 0.8B in fp32)

- **Decoding is the problem**: 160 ms per generated token at batch size 1 (about 6 tokens/s); prefill is acceptable
  (0.54 s for 75 prompt tokens, 3.9 s for 900).
- **Gated DeltaNet dominates**: in a synchronised per-module breakdown it takes 4.2 s of 7.1 s (7.3 ms per call, versus
  1.8 ms per MLP and 2.9 ms per full-attention call). The fast single-token path *is* used (`causal_conv1d_update` +
  `torch_recurrent_gated_delta_rule`; `record_past` is false), so this is not a wrong-branch bug.
- **It is launch overhead, not arithmetic**: `in_proj_b` and `in_proj_a` (1,024 → 16 projections) cost about 0.5 ms per
  call, the same order as the large projections. One generated token runs **2,829 PyTorch operations**, mostly tiny
  elementwise ones (414 `mul`, 401 `to`, 275 `add`, 187 `linear`, 115 `rsqrt`...). The process uses CPU for only 36% of
  the wall time, so Python is not the bottleneck: the GPU is waiting on a long chain of small kernels, each with a fixed
  cost of tens of microseconds on the M1. On CUDA, fused kernels collapse dozens of these into one.

## The fix: MLX

mlx-lm 0.32.2 implements Qwen3.5 with Metal kernels for the gated delta rule and reads these fine-tuned checkpoints
directly (it drops the vision tower and renames `model.language_model.*`). `qos_ai_qwen_chat.py mlx-convert` adds the
unchanged base tokenizer (the fine-tuned runs save weights only) and converts to bf16 or 8-bit.

Parity on 120 frozen prompts (same deterministic sample as the earlier MPS check; answers compared with the GPU pod's):

| On the M1 | Identical / 120 | Same first line | s / answer | Decode tokens/s | Peak memory |
|---|---:|---:|---:|---:|---:|
| 0.8B PyTorch fp32 (earlier check, batches of 8) | 108 | 114 | 2.76 | about 6 | about 3.5 GB |
| 0.8B MLX bf16 | 108 | 114 | 1.10 | 34.5 | 1.75 GB |
| **0.8B MLX 8-bit** | 110 | 115 | **0.69** | **57.7** | 1.26 GB |
| 2B MLX bf16 | 112 | 117 | 2.52 | 14.7 | 3.97 GB |
| **2B MLX 8-bit** | 109 | 114 | **1.50** | **26.1** | 2.6 GB |

Differences are the same kind as between any two backends: invented example inputs in multi-input answers, prose
wording, and occasionally a stated value; no program differed.

MLX decoding is now close to the hardware limit: each token reads every weight once, so at about 68 GB/s the 2B in
bf16 (4.4 GB) cannot exceed about 15 tokens/s (measured 14.7) and the 0.8B (1.5 GB) about 45 (measured 34.5). 8-bit
weights halve the bytes read, hence the further gain.

**Full evaluation of the 8-bit models** (all 740 frozen prompts, answered on the M1, scored with the unchanged scorers):

| | 0.8B pod (bf16, CUDA) | 0.8B MLX 8-bit | 2B pod | 2B MLX 8-bit |
|---|---:|---:|---:|---:|
| Frozen Sol tests /360 | 324 | 325 | 339 | 341 |
| Programs that run correctly /111 | 107 | 107 | 110 | 111 |
| Multi-input /18 (cases /108) | 16 (96) | 16 (96) | 18 (108) | 18 (108) |
| Dev lexical /90 | 43 | 41 | 52 | 51 |
| Time for all 740 prompts on the M1 | | 9 min | | 19 min |

8-bit quantization costs nothing measurable. End to end, `ask --engine mlx` with the 2B 8-bit model answered the eight
fresh questions from the earlier demo (execution and retrieval on) in 24 seconds including model load; all six
runnable programs were stated correctly without needing correction
([transcript](data/qos_ai_benchmark/mac_inference_20261027/fresh_ask_2b_mlx.txt)).

## Changes

- `qos_ai_qwen_chat.py`: `--engine mlx` for `ask` and `answer`, `generate_mlx` (same stop tokens, sampling and
  repetition guard as the PyTorch path), and `mlx-convert` (`--q-bits 8` for 8-bit). MLX is imported only when used;
  it is macOS-only and not in the lockfile, so it lives in a separate environment (on the M1:
  `~/qos_ai_eval/mlx_venv` with `mlx-lm==0.32.2`).
- `scripts/mac/`: the three PyTorch profiling scripts and the MLX parity and full-answer scripts.
- MLX models on the M1: `~/qos_ai_eval/experiments/mlx_20261027/mlx_{round4,2b_round4}{,_q8}` (and
  `mlx_2b_round4_q8_cli` from the new command).

PyTorch remains the engine for `rag-loss` and `mps-check` (prompt processing and scoring, which the fallback handles
far better than decoding).

## Recommendation

For local use, run the **2B in 8-bit on MLX**: the best model, 1.5 s per answer, 2.6 GB of memory. The 0.8B in 8-bit
(0.7 s per answer) is the faster option.
