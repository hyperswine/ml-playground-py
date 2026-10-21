# Expanded verified curriculum and M1 comparison

Date: 2026-10-16, Australia/Sydney.

Expanded to **664 verified examples** and completed a **7.3-minute M1 fine-tune**.
On the unchanged development benchmark, lexical matches rose **20/90 → 43/90**
and output-literal matches **7–8/33 → 11/33**. Factual recall improved; generated
code and debugging still fail the checked tasks. Keep this as an experimental
checkpoint.

## Dataset and verification

Expanded the verified curriculum from 65 to **664 examples**, retaining all
original examples and adding **599**:

| New category | Examples |
| --- | ---: |
| Source-backed domain contracts | 85 |
| Output prediction | 140 |
| Complete runnable code | 140 |
| Code explanation | 128 |
| Debugging and repairs | 106 |

The executable questions cover 18 builtin-operation families, plus missing-binding
debugging: strings and byte offsets, half-open slicing versus substring counts,
inclusive ranges, seeded folds, map/filter, list prefixes/reversal, lengths,
minimum/maximum, absolute values and clamping. Inputs include varied ASCII
strings, positive and negative numbers, boundary equality and missing searches.
Several phrasings exercise each operation; debugging pairs an observed mistake
with a tested repair. Domain questions cover repository ownership, profiles,
launch/build commands, dependency resolution/pinning, actor contracts, checked
integer parsing and JSON behavior.

All 65 original examples were retained. Thirty candidate questions were removed
as duplicates within the training set. The protected-question lexical and
benchmark-program-input screens found no further overlap in the retained new
examples. These screens do not prove semantic independence; familiar-concept
questions intentionally share training sources.

Verification ran on `JQs-Mac-mini.local`, Apple M1/arm64, in
`/Users/jq/qos_ai_eval`, through Tailscale SSH as `jq`:

- **668/668 executable fixture checks passed**, covering **393 distinct
  executions**. Both successful reference programs and deliberately broken
  debugging fixtures were checked against expected exit status and stdout.
- There are 562 examples with executable references; the remaining 102 examples
  have source-backed explanations. Actor and OS contracts were source reviewed,
  without claims of actor runtime or physical-target execution.
- Source bytes and evidence were checked against the immutable source snapshot.
  Reference compiler SHA-256:
  `c41c6ab9377e59be4d9bccedf1d6dd33f188bad0453046660650c6982fbc2f36`.
- **49 regression tests passed on the M1**, including a real GPU optimization
  step, answer masking, checkpoint loading, source proof, overlap rejection and
  preserved holdout assignments.

## Frozen benchmark and split extension

The benchmark remains byte-for-byte unchanged, with SHA-256:
`1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7`.

The original manifest remains intact:
`cd57f4a7b5c2b02b40c6b8679c39a82712535c33254385a9139cfe8504529be3`.
The expanded curriculum uses an explicitly linked append-only extension:
`10b314ad4e484922399e1a1b3980763786065da32838d501ceeb60edd69094b0`.
All original file/question assignments, historical corpus/Q&A fingerprints,
benchmark fingerprint and source exclusions are unchanged. New questions are
assigned only to training. The held-out binary/option/order source files remain
held out. No model evaluation uses the test partition.

Evaluation covers the same **90 development prompts in 34 concept groups**, on
MPS with seed 7, greedy decoding, top-k setting 20, maximum 200 new tokens, and
identical effective budgets. Both raw decoding and the existing three-token
repetition guard are measured.

Saved original and previous-SFT control outputs are reused from the earlier M1
comparison. The comparator requires identical prompts/order, budgets, decoding
settings, benchmark and evaluator/model/split-code hashes. The only permitted
manifest difference is the verified training-only extension described above.
Latency is not compared across these separately recorded runs.

## Training and execution recovery

Continued from the previous fine-tuned checkpoint, SHA-256
`f12076f64ad8c04336ddfca35e0ccc828e86ee4cd23608b0f0e026cd11eb107f`.
The planned run uses two epochs, learning rate 0.00003, AdamW, batch two with
accumulation two, seed 7, answer-only loss and a ten-minute optimization budget.
The architecture and vocabulary are unchanged. Only three answer-token positions
per pass are outside the existing vocabulary and are skipped.

The first attempt failed after 163 recorded optimizer steps with Metal command
buffer errors and an invalid GPU target count. No checkpoint was saved and no
development evaluation was performed for that attempt. Its logs/settings and
training script are preserved separately. A retry starts from the same preserved
parent, with sequence-width buckets of 32 and unused MPS cache released every
20 steps. CPU-derived target counts are checked against GPU results before an
update. Bucket padding does not change the supervised targets; a regression test
checks loss equivalence. The exact cause of the device failure is unconfirmed.


The retry completed **two epochs / 332 optimizer steps**, with 1,328 example
presentations and 33,022 supervised tokens, in **436.7 seconds**. It did not reach
the time-budget guard. Training-set loss decreased **2.1135 → 0.0482**; this
measures fitting to the curriculum, not development accuracy.

New checkpoint SHA-256:
`0707da88d15ea674ebbf1395ad948e9dad5438a9f308e7611b07fde53e80d83d`.
The M1 artifact is
`/Users/jq/qos_ai_eval/experiments/verified_expanded_sft2_retry/student.pt`;
a local copy is `qos_ai/m1_verified_expanded/student.pt`.
The normal restricted checkpoint reader was checked before publishing the saved
artifact. The original model and previous fine-tuned checkpoint were preserved.

## Development results

The expansion improves familiar factual recall and removes the remaining severe
repetition in raw decoding. It still does not solve reliable calculation,
debugging, runnable-code correctness or held-out-source generalization.

With the existing repetition guard:

| Measurement | Original | Previous 65-example SFT | Expanded SFT |
| --- | ---: | ---: | ---: |
| Lexical rubric matches | 12/90 | 20/90 | **43/90** |
| Explicit output-literal matches | 1/33 | 8/33 | **11/33** |
| Finished within budget | 74/90 | 90/90 | **90/90** |
| Severe repeated loops | 0/90 | 0/90 | 0/90 |
| Mean answer length | 109.8 tokens | 22.0 tokens | **14.3 tokens** |

Without the repetition guard:

| Measurement | Original | Previous 65-example SFT | Expanded SFT |
| --- | ---: | ---: | ---: |
| Lexical rubric matches | 7/90 | 20/90 | **43/90** |
| Explicit output-literal matches | 1/33 | 7/33 | **11/33** |
| Finished within budget | 51/90 | 90/90 | **90/90** |
| Severe repeated loops | 49/90 | 7/90 | **0/90** |
| Mean answer length | 135.0 tokens | 22.7 tokens | **14.3 tokens** |

There are no context-limited cases. Against the previous SFT, each decoding
condition has 27 proxy improvements and four proxy regressions. All 43 expanded
proxy matches are in the familiar-source track (43/75); the held-out-source track
remains **0/15**. Debugging remains **0/2** and insufficient-information handling
**0/2** by the lexical rubric.

Lexical matches are **not factual accuracy**. The printed-literal diagnostic
strips code echoes and now also accepts a bare stdout literal, because the new
model frequently answers with just the value. The same diagnostic is applied to
all three checkpoints' saved answers. Ambiguous claims are unmatched. This change
affects reporting only; training, evaluator, prompts and decoding are unchanged.
The earlier checkpoint diagnostic counts remain the same.

## Source review and actual generated-program execution

The same **12 preselected development questions** from the earlier experiment
were source reviewed again, using raw answers:

| Model | Correct on reviewed sample |
| --- | ---: |
| Original | 0/12 |
| Previous SFT | 2/12 |
| Expanded SFT | **6/12** |

Correct expanded answers cover repository ownership, the Base default profile,
compiler release pinning, FreeBSD host support, whole-document JSON validation
and the string-slice result. This is Codex source review with answer hashes and
source citations, not independent human review or an accuracy estimate on all
90 questions. Correlated actor calls, held-out binary/option contracts and
recognizing insufficient information remain failures in this sample.

All four debugging/code-generation tasks were additionally checked on the M1
using manually inspected pure Sol snippets. The complete generated programs were
executed unchanged; the reversed-list expression received only a print wrapper.
Raw and guarded answers were identical for these four tasks, so four distinct
executions establish the behavior in both conditions:

| Development task | Required result | Actual generated behavior |
| --- | --- | --- |
| Fix first-byte index | 65 | Keeps index 1; prints **66** |
| Fix first reversed-list element | 6 | Keeps index 1; prints **5** |
| Double `[2, 3]`, sum and print | 10 | Triples `[2, 3, 2]`; prints **21** |
| Clamp 20 to 1 through 8 | 8 | Omits an argument; prints **`<fn Numeric.clamp/1>`** |

All four snippets execute with exit status zero, but **none meets the requested
behavior**. In particular, code-generation lexical matches rise to 1/2 only
because the wrong map program includes the expected function names. Actual
code-generation correctness remains **0/2**, and debugging correctness **0/2**.
The generated-execution evidence records complete code, answer hashes, stdout,
stderr, expected behavior and extraction method.

The output diagnostic shows modest transfer on slicing and missing-byte searches.
Most new numeric/list inputs are still wrong; several answers reuse training-set
values such as 108 instead of evaluating the supplied program. The expanded model
also regresses on one previously correct minimum-output paraphrase. More varied
examples improved recall substantially, but this run still shows memorized
patterns in place of dependable code reasoning.

## Artifacts and limits

The new checkpoint is retained as a separate experiment. The original/default
and previous SFT checkpoints remain unchanged. The local checkpoint copy was
verified against its SHA-256 after transfer from the M1. No default-model
promotion, commit, push or cloud compute was performed.

Local evidence is in `qos_ai_benchmark/evidence/verified_expanded/`:

- `curricula/verified_expanded/`: all 664 canonical/prepared examples, duplicate
  rejections, extension manifest, source/execution proof, test logs, run script
  and console logs.
- `experiments/verified_expanded_sft2/`: failed-attempt settings/logs and original
  training script, without a checkpoint.
- `experiments/verified_expanded_sft2_retry/`: completed settings, training log and
  result with model/data/code hashes.
- `reports/`: both new full-development runs, comparisons against each saved
  control, sample source judgments and actual generated-program executions.

The M1 retains the same artifacts under `/Users/jq/qos_ai_eval`. The model and
evaluator source hashes match the previous comparison exactly. The parent and
new checkpoint retain `legacy_or_unknown` exposure ancestry; a new split cannot
retroactively certify their earlier training. The development set is used for
iteration, and the test partition was not model-evaluated. This is one completed
continued-training run and seed, with an execution-only retry before evaluation.
It does not isolate the effects of added examples from additional optimization,
or establish general-English or hardware-target capability.
