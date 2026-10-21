# Input-preserving debugging and code-generation experiment

Date: 2026-10-16, Australia/Sydney.

Added **496 verified input-focused examples** and completed a **four-minute M1
fine-tune**. Actual code-generation correctness improved **0/2 → 2/2** on the
unchanged development tasks. Debugging remains **0/2**, lexical matches fell
**43/90 → 40/90**, and output-literal matches fell **11/33 → 1/33**. Keep the
checkpoint as a separate experiment.

## Verified curriculum

Added **496 new examples**: 248 debugging repairs and 248 code-generation tasks.
They form **62 contrast groups** across eight operation families. Each group has
two input variants, with exactly one input field changed and a different expected
result, and four teaching modes per variant:

- A complete runnable program.
- A brief restatement of supplied inputs followed by the program.
- Repairing an operation mistake.
- Repairing code that substituted, omitted or added input values.

The families are mapped sums with varied multipliers, three-argument clamping,
first-byte access, reversed-list indexing, half-open string slicing, filtered
sums with inclusive thresholds, exact-list summation and prefix summation.
Examples cover changed list contents/lengths, multipliers, bounds, values,
strings, thresholds, slice endpoints and prefix counts. Index repairs use the
current source snapshot's zero-based behavior.

Each new example has a structured input contract. Preparation recomputes its
expression and expected result from those inputs, checks that the question
contains the requested task, and requires the answer's sole code block to match
the input-preserving reference program exactly. Broken-program fixtures are also
checked against the authored mistake. M1 execution independently checks actual
stdout and exit status against the Python oracle.

The canonical verified collection retains all 664 previous examples and appends
the 496 new examples, for **1,160 examples**. The focused training pass selects
all 496 new tasks plus all **102 previous source-only factual examples**, for
**598 training examples**. Older executable templates remain preserved but are
not replayed in this pass, to focus learning on supplied inputs rather than their
previous narrow numeric patterns. This selection is frozen before optimization,
with a content hash, canonical-data fingerprint, split hash and exact ID list.
The trainer refuses duplicate/unknown IDs, non-training examples, stale hashes,
or a prepared file that differs from the selection.

## Vocabulary and verification

The first draft exposed an input-representation problem: several generated string
fragments were outside the existing reduced vocabulary. Those drafts were not
trained. The final builder checks the full question/answer encoding and removes
an entire contrast group if any token is unknown. Sixteen candidates in two
groups were removed this way. The selected training answers have **zero unknown
tokens**; the model architecture and vocabulary are unchanged.

Verification ran on `JQs-Mac-mini.local`, Apple M1/arm64, in
`/Users/jq/qos_ai_eval`, through Tailscale SSH as `jq`:

- **496/496 new input contracts passed**.
- **1,412/1,412 program checks passed**, covering **723 distinct executions**,
  including the preserved reference programs and broken debugging fixtures.
- **61 regression tests passed**, including a real MPS optimizer step, changed
  input/answer rejection, contrast-pair behavior, exact frozen selections,
  source proof and unchanged holdout assignments.
- Source files, snippets and reference compiler identity were checked against the
  preserved snapshot. Actor/OS factual replay remains source-only evidence, with
  no claims of executing actors or physical targets.

Reference compiler SHA-256:
`c41c6ab9377e59be4d9bccedf1d6dd33f188bad0453046660650c6982fbc2f36`.

## Frozen evaluation and training settings

The benchmark is unchanged, with SHA-256:
`1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7`.
Evaluation uses all **90 development prompts in 34 concept groups**, seed 7,
greedy decoding, top-k setting 20, a 200-token maximum, one prompt at a time and
MPS. Both raw decoding and the existing three-token repetition guard are tested.
No model evaluation uses the 42-prompt test partition.

The previous expanded manifest remains intact:
`10b314ad4e484922399e1a1b3980763786065da32838d501ceeb60edd69094b0`.
The new manifest is its verified training-only extension:
`2d2878212aae9f37cecb5feb40c74f7c3aae2c7ebef96869db36a4d142925f73`.
Every existing file/question assignment, historical corpus/Q&A fingerprint,
benchmark fingerprint and source exclusion remains unchanged. Binary, option and
order source holdouts remain reserved.

Continued training starts from the previous expanded checkpoint, SHA-256:
`0707da88d15ea674ebbf1395ad948e9dad5438a9f308e7611b07fde53e80d83d`.
The completed short run uses one epoch, learning rate 0.00003, AdamW, batch two with accumulation
two, seed 7, answer-only loss, five warmup steps, cosine decay, gradient clipping
1.0 and a ten-minute optimization budget. Sequence widths are bucketed by 32,
unused MPS cache is released every 20 steps, and GPU target counts are checked
against CPU masks. Complete examples fit within context; the longest is 185 tokens.

The saved parent development outputs are reused. The comparator verifies prompt
order/content, effective budgets, decoding settings, checkpoint lineage,
benchmark/model/evaluator/split-code hashes, host, machine and PyTorch version.
Only the proven training-only manifest extension is allowed to differ. Latency
is not compared between separately recorded runs.

## Execution recovery

The first optimization attempt recorded 220 successful steps, then the GPU
returned a Metal command-buffer error and a target count inconsistent with the
CPU mask. The safety check stopped the process before another optimizer update
and before any checkpoint was saved. No development evaluation used that attempt.
Its settings, training log, console output and exact script were preserved.

A fresh retry starts from the same preserved parent and frozen selection, with
all training/loss batches padded to one width of 192 tokens. A real causal-model
regression check verifies that this padding preserves supervised counts and loss.
Temporary `caffeinate` assertions keep the Mac awake only for the workflow's
lifetime. The GPU/CPU count checks remain enabled. The precise cause of the Metal
failure is unconfirmed. **62 regression tests pass** with the retry implementation.


The fixed-width two-epoch retry also stopped safely, after 246 successful steps,
with another Metal command-buffer error. It did not publish a checkpoint or run
development evaluation. Both failed attempts are retained separately. Fixed
padding and awake assertions therefore did not establish a complete GPU fix.

The final execution plan reduces optimization to **one epoch** and defers the
final training-set loss measurement to a **fresh process** after checkpoint
publication. Development evaluations also run in separate fresh processes. This
limits the lifetime of the optimizer's GPU context without changing the frozen
training selection or parent checkpoint. The saved result records the deferred
loss explicitly; its separate diagnostic is tied to the final checkpoint/data
hashes and performs no optimizer updates. The learning-rate schedule uses the
actual one-epoch budget. No development benchmark result was used to select this
execution adjustment.

## Completed short run

The final one-epoch run completed successfully: **150 optimizer steps**, 598
example presentations and 24,090 supervised tokens in **241.9 seconds** (about
four minutes). The time-budget guard was not reached. The normal restricted
checkpoint reader was checked before publication. A fresh MPS process measured
final training loss in another **40.4 seconds**, with no optimizer updates:
**1.4074 → 0.00931**. This is fitting to the training curriculum, not development
accuracy.

New checkpoint SHA-256:
`70e1fa8b26391b16156170ce121706c2abd4e187f2f12b7037630c10d43f42db`.
The M1 artifact is
`/Users/jq/qos_ai_eval/experiments/input_fidelity_sft1/student.pt`.
A local copy is `qos_ai/m1_input_fidelity/student.pt`, verified against the same
SHA-256 after transfer. The previous checkpoint remains unchanged.

## Development comparison

The new checkpoint improves the two code-generation tasks by preserving their
supplied inputs. It does not fix either debugging task, and output prediction
regresses substantially. Retain it as a separate experiment; these results do
not support replacing the default checkpoint.

| Measurement | Previous expanded SFT | Input-focused SFT |
| --- | ---: | ---: |
| Executed code-generation tasks correct | 0/2 | **2/2** |
| Debugging task behavior correct | 0/2 | 0/2 |
| Full-development lexical matches | 43/90 | **40/90** |
| Explicit output-literal matches | 11/33 | **1/33** |
| Held-out-source lexical matches | 0/15 | 0/15 |
| Finished within token budget | 90/90 | 90/90 |
| Severe repetition, raw | 0/90 | 2/90 |
| Severe repetition, guarded | 0/90 | 0/90 |
| Mean tokens, raw | 14.3 | 29.6 |
| Mean tokens, guarded | 14.3 | 28.8 |

The task correctness and match counts are the same for raw and guarded decoding.
Both comparisons have three proxy improvements and six proxy regressions, and
no context-limited prompts. All 40 lexical matches are in the familiar-source
track (40/75); held-out-source and insufficient-information performance remain
poor. The same 12-question source review remains **6/12**: correct clamp code
replaces a previously correct string-slice output answer. This review covers
12/90 prompts and is not an estimate of accuracy on the full set.

Lexical matching is not factual accuracy. For example, several output-prediction
answers match a number only because it appears in echoed code. The output-literal
diagnostic is unchanged from the previous comparison and excludes those echoes.
The new model frequently emits a program, with generic and sometimes incorrect
repair explanations, when asked to state stdout. This response-format drift
accounts for many lost output-literal matches; it does not establish that all
underlying computations became worse. Some answers also change actual inputs,
such as replacing a clamp upper bound of 10 with 15.

## Actual coding behavior and supplied inputs

All four development debugging/code-generation tasks were reviewed, covering the
entire targeted subset. Only manually inspected pure Sol snippets were executed.
Source references, answer hashes, extracted code, exit status, stdout and input
contract judgments are recorded. Seven distinct executions cover the repeated
parent/tuned snippets in both decoding conditions.

| Task | Required behavior | Previous behavior | New behavior |
| --- | --- | --- | --- |
| Double `[2, 3]`, sum and print | Preserve `[2, 3]` and multiplier 2; print 10 | Multiplier 3 and list `[2, 3, 2]`; prints 21 | **Correct inputs and operation; prints 10** |
| Clamp 20 into 1 through 8 | Supply lower 1, upper 8, value 20; print 8 | Missing argument; prints `<fn Numeric.clamp/1>` | **All three arguments in order; prints 8** |
| Repair first-byte index | Change index 1 to 0; byte code 65 | Prints 66 | Raw retains index 1 and prints **66**; guarded supplies no corrected code and incorrectly claims byte length 1 |
| Repair reversed-list index | Change index 1 to 0; print 6 | Prints 5 | Retains index 1; prints **5** |

The two generated programs compile and execute successfully, preserve the exact
requested values, and produce the required output. This is stronger evidence
than the previous map answer's keyword-only proxy pass. It is still only two
code-generation tasks, not a broad coding accuracy claim.

Debugging remains a concrete failure: the model describes zero-based indexing
or says to preserve inputs while copying the original incorrect index into its
answer. The new verified repairs were learned on training inputs, but did not
transfer to these inline development repair requests. The evidence supports a
narrow gain in generation input use, with continuing weakness in selecting and
applying the required edit.

## Artifacts and limitations

Evidence is under `qos_ai_benchmark/evidence/input_fidelity/` locally and the
corresponding `/Users/jq/qos_ai_eval` directories on the M1:

- `curricula/input_fidelity_v2/`: final canonical/prepared examples, 62 contrast
  groups, frozen selection, vocabulary/source/program proof, preflight, 62-test
  log, workflow and generated-code review script.
- `experiments/input_fidelity_sft1/`: completed training settings/log/result and
  the separate post-training loss diagnostic.
- `experiments/input_fidelity_sft2/` and `input_fidelity_sft2_retry/`: discarded
  failed-attempt logs, settings and exact scripts, without checkpoints.
- `reports/`: both complete development runs, paired comparison, fixed-sample
  source review and actual generated-program/input-contract verification.

Earlier dataset drafts are retained separately and were not trained. The final
correct and broken reference programs also have no copied benchmark reference
expressions in the literal-expression audit. The overlap checks are lexical,
not a proof of semantic independence; familiar-source concepts intentionally
share training sources.

Only one short checkpoint was successfully completed and benchmark-evaluated.
There was no benchmark-based selection among the failed runs. Both parent and
new checkpoint retain `legacy_or_unknown` prior-exposure ancestry. The benchmark
is used for development iteration; its new split cannot certify older training
history. No model evaluation used the test partition. The model architecture,
vocabulary, evaluator, original/default checkpoint and earlier checkpoints are
preserved. No cloud compute, default-model promotion, commit or push was made.
