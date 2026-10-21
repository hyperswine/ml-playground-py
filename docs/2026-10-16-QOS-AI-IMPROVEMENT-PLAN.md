# qos_ai improvement plan

Date: 2026-10-16 (Australia/Sydney).

## Objective

Improve the custom student's factual accuracy, executable FP-RISC/Sol examples,
and ability to give short, complete answers. Continue the existing approach of
training our own student from scratch and distilling a teacher into it.

This document records recommendations and a proposed experiment sequence.
It does not record new training results or authorize cloud provisioning.

## Current baseline

The project report `qos_ai_report_2026-10-16.md` identifies the best student as
the 168M-parameter model in `qos_ai/pod_pt15/student.pt`, also copied to the
default `qos_ai/student.pt`. It received 1.5B tokens of English pretraining
and distillation using 118,193 Qwen3.5-4B question/answer pairs.

Recorded results, not rerun for this plan:

| Measurement | Recorded baseline |
| --- | --- |
| Documentation perplexity | 18.4 |
| Severe loops without a no-repeat rule | 60% |
| Answers finishing without a no-repeat rule | 79% |
| Severe loops with the 3-token no-repeat rule | 0% |
| Answers finishing with that rule | 97.5% |

These generation measurements used 200 questions and a 400-token limit.
The report describes fluent English but frequent incorrect domain facts and
invented APIs. Blocking repeated phrases improves usability but does not
establish factual correctness.

The 3B-token run slightly improved perplexity to 18.1 but worsened generation.
Self-correction and teacher-repair experiments also worsened generation.
These results do not establish that the student has reached its capacity limit.

## 1. Establish a clean correctness benchmark

Create approximately 100-200 reviewed questions covering:

- Language syntax, types, conversions, and standard-library APIs.
- QOS and Sol behavior, including transactions and actor communication.
- Predicting program output and explaining compiler diagnostics.
- Small programs with compilation and behavioral checks.
- False premises, unavailable APIs, and questions with insufficient information.

Record expected facts, prohibited invented claims, relevant source locations,
repository revisions, and executable checks where applicable. Score factual
accuracy, compilation, behavior, uncertainty, repetition, and completion
separately. Keep answer length and latency as secondary measurements.

Source inspection found two evaluation limitations:

- `is_eval_question()` reserves questions from self-play. However,
  `lesson_sources()` builds the original Q&A lessons from `usable_pairs()`
  without excluding those questions. Thus `evalloops` is not guaranteed to
  evaluate questions unseen during ordinary Q&A training.
- `split_windows()` holds out at most 32 documentation and 32 Q&A windows after
  packing. This does not isolate whole examples, related paraphrases, or source
  files. Related material can occur in both training and validation.

Split before packing and lesson scoring. Exclude benchmark questions and
near-duplicates from every training source. Maintain two distinct tests:
unseen questions about trained concepts, and generalization to held-out source
material. For the latter, exclude the corresponding raw documentation too.
Keep a development set for choosing settings and an untouched final test set.

## 2. Improve the teaching data

The current `clean_pair()` filter checks length, language, and obvious text
artifacts. It does not verify factual claims or compile generated examples.

Build a few thousand verified examples around real definitions and working
programs. Include function/type explanations, output prediction, compiler-error
repair, API contrasts, and minimal runnable examples. Track concept coverage
and remove exact and near-duplicate examples so volume is not mistaken for
diversity.

Preserve the source path, relevant definition or excerpt, repository commit,
teacher identity, generation settings, and validation outcome as metadata.
Compile and test self-contained code examples. Review factual explanations
against current source; a stronger teacher or a second model's agreement alone
does not prove correctness.

Research motivation: [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
supports testing curated educational/code data. Its results are not a prediction
of the gains this student will achieve.

## 3. Teach concise answers and stopping

Include short, complete answers with correct end-of-answer tokens. Teach the
student to correct false premises and express uncertainty when the supplied
information is insufficient. Keep enough explanation for the task; do not
optimize length alone.

The earlier repair runs produced longer answers and worse completion, making
this a useful hypothesis to test. Evaluate with the no-repeat rule both off
and on, and check whether phrase blocking damages legitimate code repetition.

Check training-window boundaries: `to_windows()` slices packed streams into
fixed windows, so an answer can be separated from its question. Prefer complete
examples where they fit, and deliberately handle oversized examples.

## 4. Compare training objectives using existing checkpoints

Run small, controlled experiments before buying more pretraining compute.
Use identical verified data and evaluation settings while comparing:

| Experiment | Base | Objective |
| --- | --- | --- |
| A | 1.5B-token pretrained checkpoint | Supervised next-token learning on verified answers |
| B | Same base | Same answers plus teacher distillation |
| C | 3B-token pretrained checkpoint | Best objective selected on the development set |

Also test a short supervised fine-tune of the existing best distilled student
as a cheap initial experiment. Treat that as a separate comparison because it
starts with additional domain training.

Use shorter training runs and a lower learning rate as initial candidates;
select checkpoints by development correctness and completion, not minimum
documentation perplexity alone. Repeat promising comparisons with another seed.

The current default distillation uses a 0.7 soft-loss weight, temperature 2.0,
and the teacher's top 32 logits renormalized over retained predictions. Compare
against hard-target-only learning before changing top-k or temperature. The
truncated distribution is an approximation, not evidence of a broken loss.

Research motivation: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

## 5. Test a better pretraining mixture

Compare English-only pretraining against a deliberate mixture of general
English, code, technical explanations, and domain material at the same token
budget. Preserve benchmark exclusions in all sources.

Test mixing earlier and new data rather than assuming this fixes the 3B run.
First verify the actual token sampling and data ranges used by prior runs;
the report's explanation of the regression remains a hypothesis. Track both
general-language quality and domain correctness to detect forgetting.

## 6. Consider architecture and capacity later

After the cheaper experiments, consider a larger student, longer context,
or alternative positional encoding. These require additional training and
should be justified by measured failure patterns. First resolve example
packing and data-quality limitations in the current 512-token setup.

## First implementation milestone

1. Add the benchmark format, reviewed questions, and explicit split manifests.
2. Exclude evaluation material before generating/scoring/packing training lessons.
3. Assemble a small verified, concise teaching set.
4. Evaluate the current best checkpoint as the baseline.
5. Run a short supervised fine-tune, then compare it against the baseline using
   identical prompts, sampling settings, token limits, and decoding rules.

Acceptance: improve correctness on the development set without a material
regression in completion or executable examples; confirm the selected result
once on the final test set. Define numerical targets after establishing the
clean baseline, rather than inventing a target from the current loop metrics.

For every experiment save the checkpoint lineage, data/split hashes, source
revisions, settings, seed, token counts, evaluation outputs, runtime, and cost.

## RunPod API access check

Checked at **2026-10-16 20:41:58 Australia/Sydney (09:41:58 UTC)** through the
connected RunPod MCP API. Authenticated account-resource reads succeeded:

| API read | Result | Pagination |
| --- | --- | --- |
| `list-pods`, including cluster pods | 0 pods | No further pages |
| `list-endpoints` | 0 serverless endpoints | No further pages |
| `list-network-volumes` | 0 network volumes | No further pages |

The API is accessible in this chat. No existing compute or network-volume
storage was returned by these reads. This verifies account read access; it
does not test provisioning permission, GPU availability, billing balance,
SSH/file transfer, or a training workload. No cloud resources were changed
or created, and no training jobs were submitted.

## Implementation update - M1 Mac mini

The benchmark, frozen source/example splits, duplicate exclusions, and verified
starter curriculum are implemented and tested in `/Users/jq/qos_ai_eval` on the
M1 over Tailscale SSH. The implementation and evidence were copied back here for
review. See [the implementation record](2026-10-16-IMPLEMENTATION.md)
and [usage instructions](2026-10-17-QOS-AI-BENCHMARK.md).

Delivered: 132 benchmark prompts in 52 concept groups, 65 verified teaching
examples, 22 passing regression tests and 104 passing reference-program checks.
Full-dataset packing retained all eligible examples. No weight training has run;
the existing checkpoint retains its legacy/unknown exposure label.


## Short supervised experiment completed

The five-epoch M1 fine-tune and full 90-prompt development comparison are complete.
The separate checkpoint improves stopping and some familiar-domain recall, but
held-out-source, debugging and code-generation performance remains poor. See
[the measured comparison](2026-10-16-FINETUNE.md). The default
checkpoint remains unchanged and the test partition remains reserved.

## Expanded curriculum experiment completed

Expanded the verified training set from 65 to 664 examples, including varied
source questions, output prediction, runnable code and 106 debugging cases.
All 668 executable fixture checks and 49 regression tests passed on the M1.
A separate two-epoch continued fine-tune completed in 7.3 minutes after an
execution-only GPU retry. The original benchmark and holdout assignments were
preserved.

Full development lexical matches improved from 20/90 to 43/90; the same source
review sample improved from 2/12 to 6/12. Actual generated-code behavior and
debugging remain 0/2 each, and held-out-source proxy matches remain 0/15. See
[the expanded-training comparison](2026-10-16-EXPANDED-FINETUNE.md)
for checkpoint hashes, paired results, direct program executions and limitations.
The new checkpoint remains separate from the default; the test partition remains
reserved.

## Input-preserving generation and repair experiment completed

Added 496 verified debugging/generation examples in 62 contrast groups. A frozen
598-example training selection combines them with 102 factual replay examples.
All 1,412 reference/broken-program checks and 62 regression tests passed on the
M1. Unknown input tokens are screened out before training.

Two longer GPU attempts failed safely without publishing weights. A one-epoch
run completed in four minutes, with final training loss measured in a fresh
process. On the same development benchmark, actual code-generation correctness
improved from 0/2 to 2/2, preserving the requested values. Debugging remains 0/2.
Overall lexical matches fell 43/90 to 40/90, and explicit output matches fell
11/33 to 1/33 as responses shifted toward code instead of requested values.

Keep the new checkpoint separate. See
[the input-focused comparison](2026-10-16-INPUTS-FINETUNE.md) for
actual program execution, failures/recovery, source review, hashes and artifacts.
The default checkpoint and reserved test partition remain unchanged.

## Balanced repair comparison completed, 2026-10-18

Built an identical 672-example treatment for both recent checkpoints, balancing
repairs with correct-program controls, output-only answers, generation and
factual replay. Added a separately frozen 96-prompt coding diagnostic across
24 complete input cases and 12 operation families. All 2,204 canonical fixture
checks, 144 diagnostic fixture checks and 76 regression tests passed on the M1.

Both one-epoch fine-tunes completed 168 updates without GPU retries, using full
optimizer continuation across short processes. On the unchanged development
benchmark, broader-parent lexical matches improved 43/90 to 51/90 and output
literals 11/33 to 16/33. The input-focused branch improved 40/90 to 54/90 and
1/33 to 18/33. Both now pass two original generation tasks and one of the two
complete debugging tasks; both fix the first-byte code but misstate its result.

The larger diagnostic favors the input-focused branch for generation, 21/24
versus 19/24, but the broader branch for repairs with correct stated outputs,
5/48 versus 1/48. Neither dominates all objectives. Stated computation and
response completeness remain concrete weaknesses; held-out-source lexical
matches remain 0/15. Keep both experimental checkpoints separate and repeat
promising treatments with another seed before choosing a default.

See the [balanced repair report](2026-10-18-BALANCED-REPAIR.md)
for paired scores, execution coverage, code/data identities and local/M1 paths.
The original benchmark, reserved test partition, default and parent checkpoints
are preserved.

## Calculation and complete-repair follow-up completed, 2026-10-18

Added 576 verified calculation/complete-answer examples in 48 input-contrast
pairs, plus 192 generation/factual replay examples. All 2,972 canonical fixture
checks and 81 regression tests passed. A one-epoch M1 fine-tune completed 192
updates from the stronger 21/24 coding checkpoint, recovering an execution-only
Metal failure from the full optimizer state at update 128.

On the unchanged benchmarks, applied repairs improved 27/48 to 35/48, but output
prediction stayed 18/33 and generation fell 21/24 to 19/24. Primary code-plus-
literal repair passes rose 1/48 to 3/48, but two of those three contain false
calculation explanations. All six new primary-passing no-change answers also
have false explanations. The small guarded lexical rise, 54/90 to 55/90, comes
from a reviewed false positive and does not establish a factual gain.

Retain the previous coding checkpoint; keep the new one separate for analysis.
Verified teaching targets and low training loss did not establish transfer of
correct calculations to new inputs. Evaluate explanatory claims as well as code
and final literals before considering another model promotion. Full results,
failure/recovery provenance and exact answer reviews are in the
[follow-up report](2026-10-18-OUTPUT-COMPLETE.md).
