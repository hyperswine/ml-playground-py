# Qwen 2B: fresh diagnostic, source retrieval, and M1 repair pilot

Updated: 2026-10-27T17:11:49+11:00. Status: complete. **Keep the existing round-4 checkpoint.**
The corrected short adapter did not improve coding and failed the promotion gate.
It produced more concise, supported documentation answers on a small development
sample, which is useful evidence for further data work, not a general upgrade.
No RunPod compute was started. The parent and both experimental adapters are retained.

## Matched fresh comparison

All three models used the M1, MLX 8-bit, the same tokenizer, historical no-thinking
chat prefix, greedy decoding and token budgets. Documentation used the same
current-source references and grounding instruction. Code prompts were identical
across retrieval conditions. The untuned public base revision is
`15852e8c16360a2fea060d615a32b45270f8a8fc`.

| Measure | Untuned 2B | Round-4 parent | Corrected adapter |
| --- | ---: | ---: | ---: |
| Complete code answers /48 | 0 | 4 | 3 |
| Programs passing all seven inputs /36 | 1 | 21 | 20 |
| Correct stated outputs /48 | 1 | 4 | 8 |
| Complete, supported documentation answers /8, manual review | 3 | 1 | 4 |

The new diagnostic exposes a real gap hidden by the easier older tests: the parent
can often write or repair the computation, but omit the requested output, calculate
it incorrectly, or claim no change while actually changing the program. The adapter
reduced some omissions but did not improve complete solutions. In particular,
output-only accuracy fell from 3/12 to 1/12, exceeding the prespecified limit of one
lost case in any code category. **The 24-question final holdout remains unused.**

Documentation review is small, unblinded development evidence from the implementing
assistant. The corrected model still confuses `parse` with its cursor-returning
helper, sometimes omits the requested hosted-Sol library, and invents a nesting-depth
example. It should not be treated as a reliable documentation authority.
[All 24 primary answers and review reasons](data/qos_ai_benchmark/qwen2b_repair_20261027/documentation_manual_review.json)
are retained alongside the superseded provisional run.

## Original benchmark regression check

The parent's saved MLX responses were regraded with their frozen compilers and
fixtures; they reproduce the earlier results. The corrected adapter generated all
740 prompts afresh in 13.8 minutes.
Both used identical prompt text, decoding and per-question budgets. The parent
responses are cached evidence, not a new full generation run. Execution admission
checks declined no program in either set, so they did not change the historical
score comparison.

| Measure | Round-4 parent | Corrected adapter |
| --- | ---: | ---: |
| Frozen Sol tests /360 | 341 | 253 |
| Executable programs /111 | 111 | 111 |
| Multi-input tasks /18 | 18 | 18 |
| Original dev lexical proxy /90 | 51 | 47 |
| Original dev outputs /33 | 30 | 24 |
| Working generation /24 | 24 | 24 |
| Applied repairs /48 | 46 | 42 |
| Repair plus output /48 | 35 | 31 |
| No-change plus output /24 | 19 | 18 |
| Generalization wording /48 | 43 | 38 |
| Generalization inputs /48 | 42 | 40 |
| Generalization combinations /48 | 41 | 16 |
| Strict trace /120 | 108 | 2 |

The sharpest regression is the requested intermediate calculation format: strict
trace matches fell from 108/120 to 2/120. This checks canonical fields, not every
valid prose explanation; inspected failures frequently omit those calculations
entirely. Composition accuracy also fell from 41/48 to 16/48, so the regression
extends beyond presentation.

The lexical rubric is a proxy, not factual accuracy. The new factual audit also
found that regex matches can accept unsupported extra claims and reject valid
paraphrases. The final fresh scorer therefore reports documentation proxies
separately from executable correctness and requires manual review.

## What was implemented

- A fresh 56-question development diagnostic: 12 each for output calculation,
  generation, repair and no-change, plus eight documentation questions. Generated
  programs preserve the supplied bindings and are checked on seven inputs, including
  empty lists, negative values, branches and threshold equality. Complete success
  also requires a correct stated output and a consistent repair/no-change decision.
- Disjoint new composition families for train, validation, development and final
  splits, assigned before input generation. The
  [split audit](data/qos_ai_benchmark/qwen2b_repair_20261027/split_audit.json) found no
  duplicate IDs/questions or cross-split question overlaps. This does not prove
  absence of related compositions in the original pretraining or earlier curricula.
- Opt-in `--retrieval-profile source`: exact path/identifier ranking, explicit source
  authority, runtime-profile preference for ambiguous queries, and concise grounded
  answers with source paths. Historical path requests remain available. Legacy
  retrieval behavior remains available unchanged.
- A pinned index of six current library files plus historical proposal distractors.
  It contains source excerpts, not question/answer keys. Current hosted-Sol JSON is
  `sol/lib/json.sol`; the boundary comment in `std/json.fpr` warns about native UTF-8
  byte strings versus hosted Unicode strings. This index is a bounded library pilot,
  not a replacement for the entire QOS/FP-RISC documentation corpus.
- MLX adapter loading in the normal `qwen_chat` interface and a training/generation
  driver that records inputs, model hashes, versions, runtime and memory.

## Verified curriculum and actual training exposure

Prepared **444 examples**: 144 new repairs, 72 no-change answers, 72 output
calculations, 144 historical training examples replayed, and 12 source-verified
library answers. Validation uses 16 questions from another composition family.
All 432 executable targets/question programs were reexecuted on the M1 and passed;
the 12 library answers were checked against source. Tokenization boundaries match
the historical chat format, and no sequence was truncated (maximum 692 tokens).

The corrected pilot used the round-4 q8 parent, four final decoder layers, rank 8,
scale 16, dropout zero, seed 20261027, AdamW at 5e-5, batch one and accumulation two.
It updated approximately 1.402 million adapter parameters, not all model weights.
**48 micro-batches / 24 optimizer updates** took **147.9 seconds**, with
**11.62 GB peak MLX memory**. This is not total system RAM.
Only **2,452 actual answer tokens** were trained, including EOS and
excluding padding. The full prepared dataset contains 22,695
answer tokens; the pilot was not a full pass through it.

The reconstructed seeded sampler matches every logged token total: 17 output,
13 repair, 12 no-change, four generation and two documentation rows; 16 of the 48
were replay. See the [exposure audit](data/qos_ai_benchmark/qwen2b_repair_20261027/training_exposure_audit.json).

## Loss-mask correction and validation

A synthetic test showed that mlx-lm 0.32.2's default loss includes one padding token
when given an exclusive sequence length: it counted three supervised tokens for a
two-token completion. The final loss explicitly uses
`prompt_length <= target_position < sequence_length`, retaining EOS and excluding
prompt/padding. The first adapter is retained as provisional evidence; the corrected
pilot restarted from the untouched parent, with the same examples, seed and recipe.
The final self-contained loss is mathematically equivalent to the end correction
used by that repeat. [Reproduction fixture](data/qos_ai_benchmark/qwen2b_repair_20261027/loss_mask_audit.json).

Validation passed: 23 Qwen regression tests, five repository-layout checks, and all
three MLX dataset/mask tests on the M1. The final source/adapter CLI smoke test also
passed. The saved inputs reproduce byte-for-byte on the M1. Benchmarks retain their
own compiler identities: the new and Stage-4 checks use SHA-256
`9987da491a791af853300f38ad3e73c52a243dbfd1422379a8dbe437369b776e`; the older coding and
generalization scorers retain their original snapshot. Current library documentation
has separate [content hashes](data/qos_ai_benchmark/qwen2b_repair_20261027/source_manifest.json).

## Reproduction and retained outputs

Source: `qwen_repair` builds/scores fixtures; `source_retrieval` supplies ranking;
`scripts/mac/prepare_qwen_repair.py` freezes prompts;
`scripts/mac/mlx_repair.py` trains and answers; `scripts/mac/score_qwen_frozen.py`
invokes the preserved scorers. Use `uv run --locked qos test --pattern 'test_qos_ai_qwen*.py'`
for the focused local checks. MLX stays in the separate M1 environment
`~/qos_ai_eval/mlx_venv` (mlx-lm 0.32.2, mlx 0.32.4).

The M1 experiment is `~/qos_ai_eval/experiments/qwen2b_repair_20261027`:
`dataset_v2` and `prepared_v3` are the frozen inputs; `adapter_corrected` is the final
experimental adapter. A hash-verified backup is in local ignored
`artifacts/qos_ai/qwen2b_repair_20261027/adapter_corrected`. The parent remains
`../mlx_20261027/mlx_2b_round4_q8`. Nothing was promoted automatically.

To inspect the candidate explicitly on the M1:

```sh
cd ~/qos_ai_eval/experiments/qwen2b_repair_20261027
PYTHONPATH=/Users/jq/qos_ai_eval ../../mlx_venv/bin/python qos_ai_qwen_chat.py ask \
  --engine mlx --model ../mlx_20261027/mlx_2b_round4_q8 \
  --adapter adapter_corrected --retrieval-profile source --retrieve 3 \
  --reference-tokens 900 --corpus source_corpus.jsonl \
  --question 'What does get do with duplicate object keys in sol/lib/json.sol?'
```

## Next experiment

Keep the parent for general use. Before another update, preserve round-4's
steps-before-output format in replay and new arithmetic examples, add verified
intermediate calculations and generation examples with explicit supplied bindings,
and separate the documentation-style objective from computational repairs. The
current replay was drawn from an older verified pool and retained its older output
ordering, a plausible source of interference that this experiment does not isolate.
Test a conservative update on the development set before opening the final holdout.
These results point to curriculum, answer format and evaluation work; they do not
establish that another parameter increase is needed.

References: [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) and
[MLX-LM adapter training](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).
