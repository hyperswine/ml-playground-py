# Fine-tuned Qwen3.5-0.8B: second seed, local M1 use, execution and retrieval

Created: 2026-10-22T07:30:00+11:00 (Australia/Sydney). Kind: experiment report. Follows
[the Qwen reference run](2026-10-22-QWEN-SFT-REFERENCE.md). GPU run 2026-10-21 17:50-19:25 UTC on one Secure RTX PRO
4500 Blackwell (pod `nd7k1xdgc2tkhq`, terminated; about $1.15 at $0.72/hr), plus a CPU read pod for two minutes.
Everything else ran on the M1 Mac mini (16 GB, MPS). Code: `src/ml_playground/model/qos_ai_qwen_chat.py`;
results: `artifacts/qos_ai/qwen_sft_20261022_seed17/`, `docs/data/qos_ai_benchmark/qwen_local_20261022/`.

## 1. Second seed

Same job (`scripts/runpod/qwen_sft/job.sh` with `SEED=17 SKIP_BASE=1 RUN_SUFFIX=_seed17`): the mixture hash again
matched Stage 5 exactly; only the seed, and with it the data order, changed. Training took
89 minutes on this GPU (51 on the 4090). Mean loss over the last 500 updates: 0.849 (seed 7) and 0.869 (seed 17).

| Measurement | Seed 7 | Seed 17 |
|---|---:|---:|
| Frozen Sol tests /360 | 289 | 279 |
| ... with execution | 358 | 357 |
| Generated programs that run correctly /111 | 109 | 108 |
| Multi-input suite /18 (cases /108) | 13 (93) | **16 (101)** |
| Fresh 32-question probe | 21 | 22 |
| Dev lexical /90 | 42 | 41 |
| Stated outputs /33 | 27 | 24 |
| Working generation /24 | 24 | 24 |
| Applied repairs /48 | 46 | 48 |
| Repair + stated output /48 | 34 | 34 |
| No-change + output /24 | 19 | 18 |
| Wording / Inputs / Combinations (/48 each) | 38 / 34 / 29 | 34 / 37 / 34 |
| Strict trace /120 | 72 | 81 |

Seed 17 multi-input families: map_sum 3/3, clamp_sum 3/3, signed_branch 3/3, filter_map_sum 3/3, filter_sum 2/3,
branch_max 2/3 (seed 7 had 0/3 on branch_max). The result reproduces: every measure is within a few points, and both
seeds are far above the 557M (frozen tests 142-179, multi-input 0-8).

**Lost weights.** The seed-17 weights were not kept. The M1 puller read the checksum list in a `while read` loop, and
the `ssh` inside `pod_pull.sh` consumed the rest of that list, so only `config.json` was copied before the pod was
told the pull was complete; stopping the pod cleared its disk. The answers and receipts were already on the network
volume. `pod_pull.sh` on the M1 now starts with `exec < /dev/null`. Seed 7's weights on the M1 are complete
(hash-checked).

## 2. M1/MPS inference check

`qos_ai_qwen_chat.py mps-check` re-answered a fixed random sample of 120 of the 740 frozen prompts on the M1 and
compared them with the pod's answers (bf16 on CUDA, batches of 32).

| Local setting | Identical answers | Same first line | Seconds per answer |
|---|---:|---:|---:|
| fp32 on MPS, batches of 8 | 108/120 | 114/120 | 2.8 |
| bf16 on MPS, batches of 8 | 107/120 | 113/120 | 3.8 |

All differences are near-ties: example inputs the model invents in multi-input answers, prose wording, and one trailing
period after an otherwise identical expression. None changes a program. fp32 is both closer and faster on the M1,
so it is the default. Transformers falls back to its reference kernels for the linear-attention layers on the Mac
(no `flash-linear-attention`); outputs are correct, only slower.

## 3. Answer, then execute (`ask`)

`qos_ai_qwen_chat.py ask --model DIR --fpr FPR` answers with the fine-tuned Qwen, then runs the Sol code from the
answer (or from the question, when the answer has none) through the existing `qos_ai_execute` allow-list and the
compiler, and replaces the stated output with the real one. A corrected answer also loses its "Steps:" trace, which
led to the wrong value (`augment(..., drop_wrong_steps=True)`; scoring keeps the trace so trace metrics stay
comparable).

Demo ([questions](data/qos_ai_benchmark/qwen_local_20261022/demo_questions.txt),
[output](data/qos_ai_benchmark/qwen_local_20261022/demo_ask_rag.txt)): of four program questions, the model wrote correct
code every time but stated the right output once; execution corrected 16 → 20, 15 → 55 and 409 → 415 and confirmed
`[FP, RISC]`. This matches the frozen tests: 289 correct as stated, 358 with execution.

## 4. Retrieval with a longer context

`qos_ai_qwen_chat.py rag-loss` uses the same 300 held-out dev questions, BM25 index and user-turn format as the 557M
Stage 5 measurement, and reports answer-only perplexity under five conditions. Perplexities are per Qwen token and are
not comparable in absolute terms with the 557M's; the relative change is.

| Reference in the user turn | Base Qwen | Fine-tuned Qwen |
|---|---:|---:|
| None | 12.55 | 5.77 |
| Top-1 chunk, cut so the whole example fits 512 tokens | 7.07 | 4.04 |
| Top-1 chunk, up to 1,500 tokens | 6.19 | 3.68 |
| **Top-3 chunks, up to 1,500 tokens** | **5.40** | **3.39** |
| Random chunk of the question's own source file | 8.43 | 4.84 |

- Retrieved references lower the fine-tuned model's answer perplexity by 41% (top 3), against 11% for the 557M Stage 5
  model (base Qwen: 57%). Going beyond 512 tokens lowers it a further 16% (4.04 → 3.39).
- The fine-tuned model without a reference (5.77) is already close to the base model with three references (5.40);
  with references it is lower still.
- BM25 finds the question's source file first only 56% of the time; better retrieval is the obvious next lever.

`ask --retrieve 3` applies this to answering. References are added only to questions without code that do not ask for
a program or expression: in the first demo run they turned a correct sum-of-squares program into a string-interpolation
one. With references, the command-line-argument question was answered from `docs/SCRIPTING.md` and `tests/args.sol`
(`args Unit`, correct), and the JSON question correctly ("strict; trailing text is an error") but then fell into a
repetition loop. BM25 found nothing relevant for "How do I print a value in Sol?", and the answer (`print 42`) is wrong.

## Assessment

- The fine-tuned Qwen result holds across two seeds and runs on the M1 with the same answers as the GPU pod. It is a
  practical local Sol assistant when paired with execution (357-358/360 on the frozen tests) and, for documentation
  questions, retrieval.
- No default model has been changed. Seed 7 is the copy with weights; seed 17 confirms its results.

## Next

Follow-up on 2026-10-22: [round 2](2026-10-22-QWEN-ROUND2.md) (hybrid retrieval, long-reference training, repetition guard).

1. Better retrieval for documentation questions (dense or hybrid ranking; the top-1 source-file rate is 56%).
2. A repetition guard for `ask` (the JSON answer loop).
3. If the Qwen model becomes the default: decide where its weights live (M1 only today, 1.7 GB bf16).
