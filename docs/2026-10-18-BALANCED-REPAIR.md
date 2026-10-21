# Balanced repair curriculum and two-parent M1 comparison

Date: 2026-10-18 (Australia/Sydney). Both fixed trials and all evaluations
completed on the M1. No default model was replaced.

## Dataset and diagnostic

The selected training set contains **672 examples**: 216 repairs, 72 checks of
already-correct programs, 144 output predictions, 144 complete-program requests,
and 96 source-grounded factual replay examples. There are 72 new complete input
cases across 12 families: map, clamp, first-byte index, reversed-list index,
string slice, filter, sum, take, string length, numeric minimum, inclusive range,
and missing-byte search. Every case has eight response modes. Repairs include
inline expressions, short answers and complete files. Output-only requests have
literal answers; correct-program checks explicitly say no change is needed.

The canonical collection preserves the previous 1,160 rows and adds 576, for
1,736 total. Training uses only the frozen 672-row selection. All complete
question/answer encodings fit 192 tokens in the existing vocabulary, with zero
unknown tokens. Each row carries exact source evidence and an executable
reference; repairs also carry the original broken behavior. Python result
oracles and reconstruction checks bind questions, inputs, answers and programs.

Reference verification on the M1 passed **2,204/2,204** canonical fixture checks
and **144/144** diagnostic fixture checks. Repeated fixtures share execution:
866 distinct canonical executions and 48 distinct diagnostic executions.

The separate **96-prompt diagnostic** contains 48 repair requests, 24 complete
program requests and 24 already-correct checks. It covers **24 complete input
cases, two per family**; four prompts share each case, so these are correlated
observations. Its data and settings were frozen before model generations and
before either fine-tune. Different complete inputs and question wording were
screened against the verified history, historical Q&A question identities and
both correct/broken program expressions. This checks familiar-source input use,
not generalization to new libraries or a clean legacy pretraining history.

## Fixed training protocol

Both trials ran on **JQs-Mac-mini.local, Apple M1**, through Tailscale SSH as
`jq@100.107.153.43`, in `/Users/jq/qos_ai_eval`. The two parents are the broader
expanded SFT checkpoint and its input-focused descendant. Starting histories
therefore differ; the follow-up treatment is identical.

Each trial requests exactly one epoch: **168 optimizer updates**, seed 7,
AdamW learning rate **0.00002**, weight decay 0.01, five warmup steps followed by
the same cosine schedule, microbatch two and gradient accumulation two. Only
answer tokens contribute to cross-entropy. Both use the same data, selected IDs,
shuffle order, vocabulary, architecture and fixed padding width 192.

To avoid the earlier long-process Metal failures, training is split into
64/64/40-update processes. Atomic progress files preserve full Adam moments,
model weights, CPU RNG state, schedule position and counts. The current student
has no stochastic training layers. Resume refuses changes
to the recorded parent, data, code or settings. CPU tests establish that the
complete segmented trainer reproduces uninterrupted weights and optimizer
moments. Fresh-process training loss is a fit diagnostic, not model quality.
The time budget is ten minutes per trial; settings were not adapted to results.

## Evaluation contract

The original **90-prompt development benchmark is byte-for-byte unchanged**.
Both new checkpoints use the same original evaluator, model code, seed 7,
temperature zero, top-k 20 and 200-token requested limit, individually evaluated
with repetition blocking off and with no-repeat 3. Saved starting-checkpoint
outputs from the same M1 are reused only after checking decoding, prompt/order,
effective budgets, platform and code hashes. Manifest extension chains prove
that the intervening changes add training questions without moving old holdouts.
No model inference uses the 42 reserved test prompts.

All four checkpoints additionally answer the same 96 diagnostic prompts, with
seed 7, temperature zero, top-k 20, no-repeat 3 and a 128-token requested limit.
The grader was checked against all 96 authored responses before baseline
inference. It separates correct repair structure, actual executed stdout,
explicit output claims, complete runnable code and no-change judgments.

Generated code is parsed into a restricted pure Sol subset. Imports, process or
file APIs, interpolation, recursion, unbounded ranges and unsupported syntax are
rejected. Only matching reviewed operation/input structures are executed, with
an eight-second limit. Alias variables and lambda renaming are normalized;
constant-answer substitutions and changed inputs fail the contract. This is a
conservative structural diagnostic, not a general code judge or an OS sandbox.
Alternative correct algorithms outside that subset remain unmatched. Already-
correct prose answers use the independently executed reference proof; they are
not counted as generated code executions.

Lexical rubric matches and explicit output-literal matches remain separate
proxies. A matched keyword does not prove factual or coding correctness. The
four original development coding tasks receive a separate structural/execution
review; debugging the first-byte task also requires the requested byte value.

## Results and interpretation

Both runs completed one epoch, 168 updates and **13,806 supervised answer tokens**
without retries or budget stops. Training wall time including all process
segments and saves was **315.4 seconds** from the broader parent and **326.5
seconds** from the input-focused parent. Answer-token training loss fell from
0.8060 to 0.0661 and from 0.7805 to 0.0563 respectively, measured on the same
selected training data. These losses measure fit to teaching examples.

The original development benchmark has the following results. Counts are
identical with repetition blocking off and on.

| Checkpoint | Lexical matches /90 | Explicit output matches /33 | Executed generation tasks /2 | Complete debugging tasks /2 |
| --- | ---: | ---: | ---: | ---: |
| Broader parent | 43 | 11 | 0 | 0 |
| Balanced from broader parent | **51** | **16** | **2** | **1** |
| Input-focused parent | 40 | 1 | 2 | 0 |
| Balanced from input-focused parent | **54** | **18** | **2** | **1** |

The two branches gained eight and fourteen lexical matches respectively, with
no lexical losses against their own parents. This does not establish that all
newly matched answers are factually correct. All new development answers finish,
with no severe loops or context truncation in either condition. Mean answer
length is about 12.5 tokens for the broader branch and 13.0 for the input-focused
branch with repetition blocking. All gains occur in familiar-source prompts;
the 15 held-out-source development prompts still produce zero lexical matches.

Both new models generate working map/sum and clamp programs preserving the
specified inputs. Both original index repairs now change the index to zero.
The broader branch correctly reports 6 for the reversed list; the input-focused
branch supplies correct runnable code, sufficient for that question. Both fail
the first-byte task's requested value: `Str.at "ABC" 0` executes to 65, while
the broader branch claims **67** and the input-focused branch claims **70**.
Thus applied repair structure is 2/2, but complete original debugging is 1/2.

The separate 96-prompt diagnostic exposes a different tradeoff:

| Checkpoint | Runnable generation /24 | Applied repairs /48 | Repairs plus correct stated stdout /48 | Correct no-change judgment plus stdout /24 |
| --- | ---: | ---: | ---: | ---: |
| Broader parent | 13 | 2 | 0 | 0 |
| Balanced from broader parent | **19** | **24** | **5** | **6** |
| Input-focused parent | 14 | 18 | 0 | 0 |
| Balanced from input-focused parent | **21** | **27** | **1** | **4** |

Applied repairs require the reviewed operation/input structure and matching
executed stdout; the next column also requires the requested output claim.
Text-only no-change passes require the explicit judgment and correct literal.
They rely on the independently executed reference, not a model code execution.
Diagnostic complete-request totals are 13 to 30 out of 96 from the broader
parent, and 14 to 26 from the input-focused parent. A single total hides the
category tradeoff and should not choose the checkpoint alone.

The input-focused branch gives the strongest generation and original-development
results. The broader branch better completes repair and no-change requests on
the new diagnostic. **Neither dominates all objectives.** Keep both as separate
experiment candidates. Successful repairs still frequently omit or misstate
stdout, and both branches still alter some supplied inputs. For example, the
three remaining generation mismatches in the input-focused branch replace
`[8, 14, 16]` with `[8, 14, 0]`, replace the final filter-list value 16 with 0,
and replace `[24, 21, 22, 4]` with `[24, 21, 42]`. All generation failures in the
four runs were inspected; they change inputs/operations, omit required code or
contain malformed code, rather than merely using a different correct algorithm.

For another experiment, prioritize verified computation of stated outputs and
complete repair answers on varied inputs, then repeat with another seed. The
current results support these narrow gains; they do not justify a broad factual
accuracy or general coding claim.

## Frozen identities

| Artifact | SHA-256 / manifest fingerprint |
| --- | --- |
| Broader starting checkpoint | `0707da88d15ea674ebbf1395ad948e9dad5438a9f308e7611b07fde53e80d83d` |
| Input-focused starting checkpoint | `70e1fa8b26391b16156170ce121706c2abd4e187f2f12b7037630c10d43f42db` |
| Balanced from broader parent | `8290a83c43a5555dbe58ea1c79610df34c7fd7822bef97607c8c04ba7a74f58b` |
| Balanced from input-focused parent | `5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767` |
| Original benchmark bytes | `1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7` |
| Diagnostic row fingerprint | `97c4f5c14160362307c0a51b2693ddbb100345d45811f4d355913cf56bcd24df` |
| New split manifest | `600fa43a371aa1413079147f727f209b757440a5ec1fd63fcd15e92f0dc6dc74` |
| Matching arm64 compiler | `c41c6ab9377e59be4d9bccedf1d6dd33f188bad0453046660650c6982fbc2f36` |

The original default and both parents remain separate from experiment outputs.
Both new checkpoints inherit `legacy_or_unknown` benchmark ancestry. One seed
and one short run per parent cannot establish a robust gain. These small,
templated pure-expression tasks do not test multi-file programs, compiler-error
repair, actors, operating-system effects or hardware behavior.

## Reproduction and evidence

The implementation is `qos_ai_balanced.py` (build and reference verification),
`qos_ai_resume.py` plus `qos_ai_finetune.py` (full-state segmented SFT),
`qos_ai_coding_diag.py` plus `qos_ai_pure_sol.py` (frozen generation and restricted
grading), and `qos_ai_balanced_compare.py` (paired identity checks and summary).
`qos_ai_balanced_workflow.py` records the exact dated run; it refuses an existing
evidence directory. Use fresh output paths for another experiment.

The M1 regression suite passed **76 tests**, including a real MPS update,
answer/padding masks, split protection, multi-generation manifest chains,
input/answer contracts, rejected unsafe code, complete trainer equivalence and
restored Adam moments/RNG. The diagnostic grader accepted all 96 authored
responses before the first baseline generation.

To construct a new curriculum with the existing parents on the same snapshot:

```sh
.venv/bin/python qos_ai_balanced.py build \
  --previous curricula/input_fidelity_v2 --qa evaldata/qa.jsonl \
  --snapshot source_snapshot \
  --parent experiments/verified_expanded_sft2_retry/student.pt \
  --parent experiments/input_fidelity_sft1/student.pt \
  --output curricula/my_balanced_repair
.venv/bin/python qos_ai_balanced.py verify \
  --data curricula/my_balanced_repair --snapshot source_snapshot \
  --fpr source_snapshot/fprisc/fpr
```

Local evidence is under `evidence/balanced_repair_20261018/`: frozen curricula,
training/evaluator code snapshots, reference proofs, regression log, training settings/update
logs/segment metadata, all four diagnostic response sets and grading details,
all original development controls and four new development runs, paired outputs
and `comparison.json`. The M1 retains optimizer progress in the experiment
directories; those large transient state files are not copied into the report.

All evidence checksums and code identities are available in the saved JSON.
Models and optimizer states are excluded from Git. No cloud compute, default
model promotion, commit or push is part of this experiment.

Local checkpoints are `qos_ai/m1_balanced_broad/student.pt` and
`qos_ai/m1_balanced_codegen/student.pt`. Corresponding M1 checkpoints are
`experiments/balanced_from_broad_20261018/student.pt` and
`experiments/balanced_from_codegen_20261018/student.pt` under the workspace above.
Both local copies are verified against the M1 SHA-256 identities in the table.
