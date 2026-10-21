# Reference: Qwen3.5-0.8B fine-tuned on the Stage 5 data

Created: 2026-10-22T04:55:00+11:00 (Australia/Sydney). Kind: experiment report. GPU run 2026-10-21 16:33-17:35 UTC on
one Secure RTX 4090 (pod `pj3dj7cgwgjgz0`, terminated). Expected cost about $1.00 ($0.33 posted at the last read).
Code: `src/ml_playground/training/qos_ai_qwen_sft.py`, `scripts/runpod/qwen_sft/job.sh`.

## Setup: same data, same evaluation

- **Data.** The pod rebuilt the Stage 5 mixture and refused to continue unless its file hash matched Stage 5's
  (`937452d1...`, 83,033 examples, 4.89M target tokens, 65/14/14/7). The student vocabulary is a subset of Qwen's
  tokenizer and uses Qwen's chat format with thinking disabled, so `qwen_ids[student_id]` recovers the exact Qwen token
  sequence: identical examples, targets and masks.
- **Training.** Full fine-tune of Qwen3.5-0.8B (853M parameters; vision tower frozen), fp32 master weights with bf16
  autocast, gradient checkpointing, logits only at target positions. Same plan as Stage 5: one shuffled pass, seed 7,
  32 examples per update (4 x 8), 2,595 updates, cosine with 20 warmup updates, AdamW (0.9, 0.95), weight decay 0.1.
  Learning rate 1e-5 (Stage 5 used 2e-5 from random-initialized weights). Loss 2.79 → 0.71 (last 100 updates).
  About 3,700 tokens/s; 51 minutes of training.
- **Evaluation.** Base and fine-tuned Qwen answered all 740 frozen prompts on the pod with greedy decoding and the same
  token budgets (Sol tests and probe 200, multi-input 200, development 200, coding and generalization 128). Batched
  left-padded generation was checked locally to match single-prompt output. Scoring used the unchanged scorers: the
  pinned compiler `9987da49...` here, and the M1's frozen compiler for the original coding/generalization benchmarks.
  Fine-tuned weights: M1 `/Users/jq/qos_ai_eval/experiments/qwen_sft_20261022/model/` (bf16, hash-verified).

## Results

| Measurement | Qwen3.5-0.8B base | **Qwen3.5-0.8B + Stage 5 data** | 557M Stage 5 | 557M Stage 3 |
|---|---:|---:|---:|---:|
| Frozen Sol tests /360 | 0 | **289** | 142 | 179 |
| ... with execution (`qos execute`) | 155 | **358** | 347 | 352 |
| Generated programs that run correctly /111 | 10 | **109** | 98 | 103 |
| Multi-input suite /18 (cases /108) | 0 (0) | **13 (93)** | 8 (82) | 0 |
| Fresh 32-question probe | 17 | **21** | 13 | 13 |
| Dev lexical /90 | 33 | **42** | 17 | 24 |
| Stated outputs /33 | 3 | **27** | 10 | 12 |
| Working generation /24 | 0 | **24** | 18 | 18 |
| Applied repairs /48 | 3 | **46** | 32 | 36 |
| Repair + stated output /48 | 0 | **34** | 8 | 11 |
| No-change + output /24 | 0 | **19** | 0 | 6 |
| Wording / Inputs / Combinations (/48 each) | 0 / 1 / 1 | **38 / 34 / 29** | 15 / 15 / 6 | 17 / 18 / 10 |
| Strict trace /120 | 0 | **72** | 24 | 29 |

Frozen Sol tests by stratum (fine-tuned Qwen): fresh 97/120, held-out operation pairs 95/120, large numbers 97/120.
Multi-input families: map_sum 3/3, clamp_sum 3/3, signed_branch 3/3, filter_sum 2/3, filter_map_sum 2/3, branch_max 0/3.

Base Qwen knows no Sol: it invents Scala-like syntax (`def f(...) = xs.filter(_ > limit)`), and its zero on the frozen
tests partly reflects not using the "It prints X." form. Its 33/90 development lexical score and 17/32 probe come from
general reasoning and prose. Loss probes are not comparable across tokenizers and are not reported.

## Assessment

- **With identical data, the pretrained 0.8B model is far ahead of the 557M on every measure**: 289 vs 142-179 on the
  frozen Sol tests, 13/18 vs 8/18 on the multi-input suite, and large margins on the original benchmarks (stated
  outputs 27 vs 10-12, strict traces 72 vs 24-29, "no change needed" 19/24 vs 0-6).
- It also generalizes across the held-out strata (operation pairs and large numbers) about as well as on fresh
  programs, and solves the conditional `signed_branch` family that every 557M model failed.
- The gap is what trillions of pretraining tokens buy over the 557M's 0.9 billion. The curriculum, verification and
  evaluation built for the 557M transfer directly: the same data teaches the pretrained model Sol well.
- Practical implication: for a usable Sol assistant, fine-tuning a pretrained small model is the stronger path; the
  557M remains useful as a from-scratch research lineage.

## Next

1. `qos execute` with the fine-tuned Qwen (358/360 already with execution on the frozen tests).
2. Conditional `branch_max` and retrieval-context answering with the fine-tuned Qwen, which has a much longer context.
3. A second seed and an M1/MPS inference check before using it as a default.
