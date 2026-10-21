# Verified calculation and complete-repair follow-up

Date: 2026-10-18 (Australia/Sydney). The M1 run and both evaluations are complete.

**Recommendation: retain the previous checkpoint for coding.** The new run
applies more repairs but does not improve output prediction, loses two
generation tasks, and introduces false calculation explanations. The new
checkpoint remains a separate experiment.

## Objective and starting checkpoint

Continue the stronger coding checkpoint from the previous balanced experiment:
`experiments/balanced_from_codegen_20261018/student.pt` on the M1, SHA-256
`5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767`.
Its saved controls are 54/90 development lexical matches, 18/33 explicit output
matches, 21/24 diagnostic runnable programs, 27/48 applied repairs, 1/48 repairs
with the correct stated output, and 4/24 complete no-change judgments. The
current experiment measures whether verified calculations improve those weak
completion/output objectives while retaining generation behavior.

## Verified curriculum

Added **576 new examples** across **96 complete input cases** in **48 contrast
pairs**, four pairs in each of 12 operation families. Each pair changes one
input field and its expected result. The six response modes are output-only,
output with calculation, inline repair with calculation/output, complete-file
repair with calculation/output, correct-program checking, and full generation
with calculation/output. Every non-output-only target contains both the actual
result and a short checked derivation.

Calculations expose mapped products, retained filter values, sum terms, prefix
contents, reversed lists, byte offsets and ASCII codes, slice endpoints, byte
counts, minimum/clamp choices, inclusive range terms, and present/absent search
results. Index examples include offsets zero, one and two. String and numeric
inputs vary; targets use the supplied inputs rather than copying an old result.
Repair fixtures must demonstrate a different observed behavior before the fix.

The selected treatment is **768 examples**: 192 output predictions, 192 repairs,
96 correct-program checks, 192 generation examples (96 new and 96 replay), and
96 factual replay examples. The canonical history grows from 1,736 to 2,312
rows, but training uses only the frozen selected IDs. Complete question/answer
encodings fit the existing 192-token training width, with zero unknown tokens.

Source evidence, exact input/question/answer/program reconstruction and Python
result/calculation oracles are checked before target execution. The M1 passed
**2,972/2,972 canonical reference checks** in 1,037 distinct executions. This
includes corrected and broken fixtures. All 48 selected contrast groups retain
both variants and all six response modes. The full regression suite passed
**81 tests**, including the five new calculation/contrast/completeness tests.

Both existing benchmarks are protected during extension. Canonical printed
expression structures are screened against old references and both benchmarks,
including aliases and formatting differences. Exact historical question IDs,
protected lexical question components and source-file exclusions are preserved.
This does not establish independence from unknown legacy pretraining history.

## Fixed M1 training protocol

Host: **JQs-Mac-mini.local, Apple M1**, workspace `/Users/jq/qos_ai_eval`, accessed
through Tailscale SSH as `jq@100.107.153.43`. No cloud compute is used.

One epoch, **192 optimizer updates**, answer-only cross-entropy, AdamW learning
rate **0.00002**, weight decay 0.01, five warmup steps then cosine decay, seed 7,
microbatch two and gradient accumulation two. Fixed padding width is 192, with
MPS cache release every 20 steps. The ten-minute budget and all hyperparameters
were fixed before new evaluation. No teacher, architecture or vocabulary change
is part of this follow-up.

Training uses three fresh 64-update processes with the previously tested full
optimizer continuation. Model weights, Adam state, schedule position, counts and
CPU RNG state are saved atomically. The student has no stochastic training
layers. A fresh process measures final training loss. Training loss measures
fit to the selected examples; it does not establish benchmark quality.

The completed epoch contains **26,051 supervised answer tokens** in exactly
192 canonical optimizer records, with no budget stop. Initial answer-token
training loss was **1.8298**, falling to **0.07294** in the fresh-process final
measurement. The trainer's successful cumulative timer is **322.9 seconds**;
wall time including saves, failed work and recovery orchestration is **590.5
seconds**. These timings describe different scopes.

## GPU failure and continuation

The initial final segment failed while attempting update 146, after a Metal
command-buffer `SubmissionsIgnored` error caused the device target-count guard
to reject the batch. No final checkpoint was published. The latest atomic
progress file held model and optimizer state at update 128.

`qos_ai_output_recover.py` resumed the same settings and plan in a fresh process.
It passed the failed update and completed all 192 updates on the first retry.
The failed 1-145 update trace is retained as `failed_training_attempt1.jsonl`;
only its 1-128 prefix remains in the canonical training trace before continuing.
Unsaved updates 129-145 are discarded and replayed from the saved optimizer
state. This is an execution retry, with no data, hyperparameter or benchmark
adaptation. Recovery metadata records the exact saved step and discarded range.
Short process segments reduce exposure to the M1 failure but do not eliminate it.

## Unchanged evaluations

The original benchmark is still **90 development prompts**, with the same
source evidence, evaluator, model code and split exclusions. Seed 7,
temperature zero, top-k 20, requested 200 new tokens and one prompt per call
are unchanged. Both no-repeat zero and no-repeat three are evaluated. The
42 reserved test prompts receive no model inference.

The **same 96 coding prompts**, covering 24 input cases in 12 families, use the
same questions, references, compiler, grading code, seed, temperature, top-k,
no-repeat three and 128-token requested budget. Four prompts share each input
case, so these are correlated observations rather than 96 independent concepts.
This remains a familiar-source development diagnostic, not unseen-library proof.

Saved controls from the exact starting checkpoint are reused only after checking
checkpoint lineage, prompt order/content, effective budgets, decoding, code,
data, environment and compiler identities. The new manifest must directly
extend the old one without moving historical assignments. Both benchmark files
and the diagnostic reference proof are copied byte-for-byte and audited locally.

The existing restricted pure-Sol grader executes only supported, reviewed
operation/input structures. It normalizes aliases and lambda names and rejects
constant-answer substitutions or changed data. Different valid algorithms
outside its supported subset remain unmatched. Complete repairs require both
matching executed code and a correct stated result. Correct-program prose
answers rely on independently executed reference proof.

The primary coding-generation score counts working programs; it does not judge
an additional output claim. A separately labelled supplementary audit therefore
checks claims in those same answers, without changing the primary benchmark or
regenerating prompts. It distinguishes correct, incorrect, ambiguous and absent
claims. Neither the primary metrics nor this literal audit assess every sentence
of explanatory prose. Lexical matches remain proxies, not factual accuracy.

## Results

### Original 90-prompt development benchmark

| Measurement | Starting checkpoint | New checkpoint |
| --- | ---: | ---: |
| Lexical matches, no repetition blocking | 54/90 | 54/90 |
| Lexical matches, no-repeat three | 54/90 | 55/90 |
| Explicit output literals, either condition | 18/33 | 18/33 |
| Original executed generation tasks | 2/2 | 2/2 |
| Original complete debugging tasks | 1/2 | 1/2 |
| Finished answers, no blocking | 90/90 | 89/90 |
| Severe loops, no blocking | 0/90 | 1/90 |
| Finished answers / severe loops, no-repeat three | 90/90, 0/90 | 90/90, 0/90 |

Raw lexical passes are identical on every question. Guarding adds one lexical
match without losing another. However, that new held-out-source match is a
**false positive on review**: asked about `Option.withDefault`, the model says
“No. It prints 1. It tries "Some x" and then "None".” It does not give the
actual contract, which returns the default for `None` and the contained value
for `Some`. Therefore the held-out-source proxy increase from 0/15 to 1/15 with
guarding is not evidence of a factual generalization gain. Raw remains 0/15.

The raw repetition failure is the held-out binary-offset question. It begins
with offset zero but repeatedly emits invented byte/offset lists until the
200-token limit. Mean answer length rises from 13.17 to 15.83 tokens raw and
13.04 to 14.17 guarded; all effective prompt budgets remain identical.

Both original generation programs still execute correctly with the exact map
and clamp inputs. Both index repairs have correct code. The reversed-list
answer states the correct 6, but the first-byte answer now claims **111** for
`Str.at "ABC" 0`; actual execution gives **65**. The previous model claimed 70,
so that task remains incomplete rather than improving.

### Same 96-prompt coding diagnostic

| Measurement | Starting checkpoint | New checkpoint |
| --- | ---: | ---: |
| Runnable generation with correct supplied inputs | 21/24 | **19/24** |
| Applied repairs with matching executed stdout | 27/48 | **35/48** |
| Repair code plus correct stated output literal | 1/48 | **3/48** |
| No-change judgment plus correct stated output literal | 4/24 | **6/24** |
| Total primary complete-request passes | 26/96 | 28/96 |

The model applies eight more repairs, but result calculation remains weak.
Among applied repairs, missing/ambiguous output claims fall from 17 to 5,
while **incorrect claims rise from 9 to 27**; correct claims rise only from
1 to 3. It learned to append result claims much more often than it learned to
calculate those results correctly.

All five new generation misses were inspected. The map task substitutes
multiplier four for three; reverse and filter tasks alter supplied list entries;
one sum program is malformed and another drops the supplied 22. These are real
input/syntax failures, rather than merely unsupported correct algorithms. The
supplementary generation-claim audit finds no additional stdout claims in any
working generated program, before or after: the answers provide code only.
Consequently that audit does not alter the 21-to-19 generation finding.

### Review of the newly added explanations

The fixed primary diagnostic checks code and stated literals, not every sentence
of prose. A separate source/execution review covered **every primary-passing
repair and no-change answer** in both checkpoints:

| Primary-passing answers reviewed | Before: fully correct / reviewed | After: fully correct / reviewed |
| --- | ---: | ---: |
| Repairs | 1/1 | **1/3** |
| Correct-program judgments | 4/4 | **0/6** |

Two of the three new repair passes have false explanations. For `Str.len "Grape"`,
the model claims the bytes are `G, e, a, n` while stating length five. For
`Str.len "Maple"`, it lists `L, i, l, n`, again claiming five. Both code and the
output literal are correct, but the described bytes and arithmetic are not.
The one fully correct new repair clamps zero to the lower bound four and
correctly explains `0 < 4`.

All six new no-change passes also contain false calculations or contradicted
inputs: examples include `4 + 13 = 4`, choosing an upper bound of 11 from a
request whose upper bound is 12, and stating index zero is 17 after showing a
wrong reversed list. The before answers were short correct judgments/literals.
These review counts apply only to the stated primary-passing subset; they are
not a new full-benchmark factual accuracy estimate. Exact answer hashes, source
evidence and individual judgments are saved in `prose_review.json`.

The curriculum targets themselves passed source, calculation and execution
checks. The model fitted those targets but did not reliably transfer the
calculation steps to different inputs and question wording. This run supports a
narrow gain in applying edits, not a reliable gain in correct complete repair
answers. Keep the previous 21/24 coding checkpoint available as the stronger
coding candidate; preserve this new branch for analysis rather than replacing it.

## Frozen identities

| Artifact | SHA-256 or manifest fingerprint |
| --- | --- |
| Starting checkpoint | `5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767` |
| New checkpoint | `4b7d9bf013bcf357b6c04c50e0da672505cbb6fd491334bc91f311d0da4f793b` |
| Original benchmark bytes | `1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7` |
| Coding diagnostic bytes | `5328ff3c3cf01b132322349ef1830959b325a302926b437c78d32edaef8d59ce` |
| Coding diagnostic row fingerprint | `97c4f5c14160362307c0a51b2693ddbb100345d45811f4d355913cf56bcd24df` |
| New split manifest | `54346de3d687868ce0213138c5d5004e37f8bbcc2f6a396dd9ee065f01c75d2f` |
| Target arm64 compiler | `c41c6ab9377e59be4d9bccedf1d6dd33f188bad0453046660650c6982fbc2f36` |

All checkpoints retain `legacy_or_unknown` exposure ancestry. One fixed seed and
one short run do not establish robustness. The diagnostic is small and templated;
it does not cover multi-file programs, actors, OS effects or hardware behavior.

## Implementation and evidence

`qos_ai_output_curriculum.py` builds and verifies the appended examples.
`qos_ai_output_workflow.py` records the exact dated experiment.
`qos_ai_output_compare.py` checks paired identities, scores both unchanged
benchmarks and executes the four original coding/repair answers.
`qos_ai_output_claim_audit.py` adds the supplementary claim-consistency report.
The existing trainer, resume implementation, benchmark and grading modules are
unchanged from the preceding experiment.

Local evidence is under `evidence/output_complete_20261018/`: canonical and
selected data, split/selection hashes, reference proof, code snapshots,
regression log, training settings/update/segment records, all new responses,
paired comparisons, the supplementary audit and the separate prose review. The M1 retains the full Adam
progress files in the experiment directory; large state files are excluded from
the local evidence bundle and Git. Previous evidence remains separate.

To build a new copy on the same M1 snapshot, choose a fresh output directory:

```sh
.venv/bin/python qos_ai_output_curriculum.py build \
  --previous curricula/balanced_repair_20261018 \
  --parent experiments/balanced_from_codegen_20261018/student.pt \
  --snapshot source_snapshot --qa evaldata/qa.jsonl \
  --output curricula/my_calculation_curriculum
.venv/bin/python qos_ai_output_curriculum.py verify \
  --data curricula/my_calculation_curriculum --snapshot source_snapshot \
  --fpr source_snapshot/fprisc/fpr
```

The completed M1 checkpoint is
`/Users/jq/qos_ai_eval/experiments/output_complete_20261018/student.pt`.
The local backup is `qos_ai/m1_output_complete/student.pt`, verified against the
M1 SHA-256 in the identity table. Existing parents/defaults and benchmark files
are preserved. No commit, push or model promotion was made.
