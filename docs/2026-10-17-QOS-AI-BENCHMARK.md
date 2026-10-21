# qos_ai clean benchmark and verified teaching set

The current work is [Stage 4 preparation](2026-10-21-STAGE4-PREPARATION.md): stronger multi-input evaluation,
verified arithmetic, gentler SFT variants and recovery checks. No Stage 4 training has started.
The latest trained model is the [557M Stage 3 Sol specialist](2026-10-20-STAGE3-SOL-PROGRAMS.md), following
[Stage 2 continuation](2026-10-20-STAGE2-557M.md), [Stage 1 domain training](2026-10-20-STAGE1-557M.md),
the [budgeted RunPod pilot](2026-10-19-RUNPOD-557M.md) and the [557M M1 pilot](2026-10-19-SCALE-557M.md).
This is a separate lineage initialized from random weights. The preceding
[168M coverage experiment](2026-10-19-GENERALIZATION.md) improved repair application
and within-domain diagnostics with mixed results on the older benchmarks.
The [balanced repair comparison](2026-10-18-BALANCED-REPAIR.md) records its coding
parent and the fixed 96-prompt diagnostic. The original 90 development prompts
remain unchanged. Experimental checkpoints remain separate from the default.

The verified curriculum and reference grading were implemented and checked on **JQs-Mac-mini.local, Apple M1**, through SSH over
Tailscale as `jq@100.107.153.43`. Primary workspace: `/Users/jq/qos_ai_eval`.

## Contents and evidence

- `benchmark.jsonl`: 132 prompts in **52 concept groups**, partitioned into
  90 development prompts and 42 test prompts. Paraphrases within a concept stay
  together; these are not 132 independent concepts.
- `verified_train.jsonl`: 65 authored teaching examples, including 48 complete
  executable Sol programs with different inputs from the benchmark fixtures.
- `sources.json`: inventory of nine source files; each example carries its own
  exact evidence snippet, line numbers, full-file SHA-256 and repository revision.
- `verification.json`: 104 passing reference-program checks on the M1, covering
  the 48 teaching programs and 56 benchmark reference fixtures. Reference
  execution does not establish that generated model code works.

Explanations were reviewed against source by Codex. This is an initial curated
set, not a claim of independent human review or a finished, exhaustive benchmark.
It covers architecture, profiles, modules, tooling, actors, binary operations,
options, ordering, parsing, output prediction, false premises, debugging, code
requests and insufficient-information questions. Add more independent concepts
and human review before using it as a release gate.

The M1's sibling source repositories were older. Verification therefore uses
an isolated `source_snapshot/` of the current referenced source files and the
matching supplied macOS/arm64 `fpr` executable, whose hash is recorded. It does
not overwrite the M1's sibling repositories or claim that the compiler was built
on the M1. Required libgit2/libarchive libraries were installed on the M1.

## Split contract

`qos_ai_data.py` assigns whole source files deterministically before packing.
All chunks, styles and rounds for a file inherit its partition. Files named by
source-generalization benchmark questions are reserved in that benchmark's
partition; familiar-concept questions permit the source material in training.
This holds out source files, not every related concept mentioned elsewhere.

Exact normalized questions and lexical near duplicates (word-trigram Jaccard
similarity at least 0.75) cannot cross training/development/test boundaries.
Conflicting duplicate components are excluded from all partitions. Benchmark
questions and their lexical duplicate components are excluded from every lesson
source, including self-play and repair. This is a lexical safeguard, not a
promise to detect every semantic paraphrase. Copied source chunks crossing
partitions are also excluded, together with associated generated Q&A.

The manifest fingerprints the full corpus, usable Q&A, curated examples and
benchmark. It is immutable: changed inputs require a new experiment directory.
`teach` writes to `lessons_v2/`, preserving legacy lessons. Shards record the split
schema/hash and partition. `train` rejects missing/stale metadata and takes
optimizer samples only from training partitions. Development windows select
checkpoints; test windows do not. Scratch vocabulary construction uses training
shards only.

Q&A is packed as complete examples. An oversized example is rejected whole and
listed in `packing_rejections.jsonl`, rather than separating its answer from the
question. `lessonplan` checks this without loading a teacher.

Existing checkpoints have unknown prior benchmark exposure. They are labelled
`legacy_or_unknown`; applying these splits does not retroactively produce a
clean model. New training inherits its starting checkpoint's ancestry status.
The evaluator makes no automatic clean-test claim.

## M1 workflow

The full dataset is referenced through symlinks in `clean_benchmark_v2/`; the
original `evaldata/` files and checkpoint are unchanged. The manifest contains:

| Kind | Training | Development | Test | Excluded |
| --- | --- | --- | --- | --- |
| Source files | 444 | 34 | 19 | — |
| Historical teacher Q&A | 107,364 | 7,702 | 2,637 | 490 |
| Verified teaching examples | 65 | 0 | 0 | 0 |

Historical Q&A and raw corpus are preserved snapshots. Their correctness against
current source has not been verified by this work. The curated examples carry
current source evidence. A short supervised follow-up on the 65 verified examples
is now complete; see [the fine-tune comparison](2026-10-16-FINETUNE.md).

```sh
cd /Users/jq/qos_ai_eval
export QOS_AI_DIR=clean_benchmark_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.venv/bin/python -m unittest -v test_qos_ai_data

.venv/bin/python qos_ai_bench.py validate \
  --root fprisc=$PWD/source_snapshot/fprisc \
  --root qos=$PWD/source_snapshot/qos

.venv/bin/python qos_ai_bench.py references \
  --root fprisc=$PWD/source_snapshot/fprisc \
  --root qos=$PWD/source_snapshot/qos \
  --fpr source_snapshot/fprisc/fpr \
  --output qos_ai_benchmark/verification.json

# Freeze splits for a new dataset, or verify an unchanged existing manifest.
.venv/bin/python qos_ai.py splits
.venv/bin/python qos_ai.py lessonplan

.venv/bin/python qos_ai_bench.py prepare \
  --root fprisc=$PWD/source_snapshot/fprisc \
  --root qos=$PWD/source_snapshot/qos \
  --splits clean_benchmark_v2/splits.json \
  --output clean_benchmark_v2/verified_train_ready.jsonl

# Use a fresh output directory for each run. Test is reserved for final evaluation.
.venv/bin/python qos_ai_bench.py run \
  --load qos_ai/pod_pt15/student.pt --split dev \
  --device mps --temperature 0 --no-repeat 0 --max-new-tokens 200 \
  --output reports/dev_baseline
```

The benchmark saves every response, decoding settings, requested/effective token
budgets, checkpoint/data/code hashes, ancestry, repetition and stopping metrics.
Prompts run individually so changing batch composition cannot silently change
another prompt's available token budget. `--limit 12` selects a reproducible
sample across categories and concept groups for a smoke check.

Rubric matches are labelled **proxies**, not factual accuracy. Review answers
using `review_template.jsonl`; set factual/code judgments to true or false only
with a named reviewer and supporting evidence. Leave unassessed fields null.
Generated model code is not executed automatically. Only reviewed, authored
reference fixtures run during `references`.

The separate `qos_ai_coding_diag.py` workflow additionally admits a restricted
pure Sol subset of generated answers, checks their operation/input structure
against reviewed references, and executes matching snippets. Unsupported or
different algorithms remain unmatched; see the balanced experiment's coverage
limits. This does not change `qos_ai_bench.py` or its original scores.

```sh
.venv/bin/python qos_ai_bench.py score \
  --report reports/dev_baseline \
  --judgments reports/dev_baseline/review_template.jsonl
```

Scoring rejects stale answer hashes and reports review coverage separately from
accuracy on reviewed answers. An unreviewed benchmark has null factual accuracy.

The source workflow backup is in `backups/2026-10-16-clean-benchmark/`. Existing
RunPod deployment scripts have not been exercised or updated in this M1 task;
any future training environment must receive the new modules, benchmark files,
verification evidence and matching frozen manifest before teaching/training.


## Verified-example supervised fine-tuning

`qos_ai_finetune.py` trains only on the exact prepared verified training partition,
using answer-only loss. It validates the manifest, source hashes and reference
execution proof, preserves checkpoint ancestry/vocabulary, uses a time budget,
and refuses to overwrite a parent or an existing experiment directory.

The completed M1 experiment is `experiments/verified_sft5/student.pt`. Its local
backup is `qos_ai/m1_verified_sft5/student.pt`. Neither replaces the default model.

To run a separate follow-up experiment on the M1:

```sh
.venv/bin/python qos_ai_finetune.py \
  --load qos_ai/pod_pt15/student.pt \
  --data clean_benchmark_v2/verified_train_ready.jsonl \
  --splits clean_benchmark_v2/splits.json \
  --root fprisc=$PWD/source_snapshot/fprisc \
  --root qos=$PWD/source_snapshot/qos \
  --output experiments/verified_sft_next
```

Defaults are five epochs, learning rate 0.00005, microbatch two with accumulation
two, and a ten-minute training budget. Use `--dry-run` to validate inputs without
training. The completed run took about 104 seconds. The fixed-prompt development
comparison, source review, limitations and checkpoint hashes are in the linked
fine-tune report. Test prompts remain reserved.

## Expanded curriculum and continued M1 training

`qos_ai_curriculum.py` builds a separate curriculum while preserving the original
manifest's file assignments, question assignments, benchmark fingerprint, and
source exclusions. It appends training questions only. The overlap screen rejects
lexical duplicates of protected historical questions and benchmark prompts, and
rejects reuse of benchmark program inputs. This is a lexical screen, not a proof
of semantic independence.

The expanded set contains 664 examples: the original 65 plus 85 source-grounded
contract questions, 140 output predictions, 140 runnable code tasks, 128 code
explanations and 106 debugging repairs. Eighteen builtin operation families have
varied strings, numeric inputs and boundary cases. Debugging fixtures check both
the observed broken behavior and the repair. Source-only actor and OS examples
make no claim of runtime or hardware execution.

On the M1, build a fresh curriculum directory and verify it before training:

```sh
.venv/bin/python qos_ai_curriculum.py build \
  --base-splits clean_benchmark_v2/splits.json --qa evaldata/qa.jsonl \
  --snapshot source_snapshot --output curricula/my_expansion
.venv/bin/python qos_ai_curriculum.py verify --data curricula/my_expansion \
  --snapshot source_snapshot --fpr source_snapshot/fprisc/fpr
.venv/bin/python qos_ai_finetune.py \
  --load experiments/verified_sft5/student.pt \
  --curriculum curricula/my_expansion \
  --data curricula/my_expansion/verified_train_ready.jsonl \
  --splits curricula/my_expansion/splits.json \
  --root fprisc=$PWD/source_snapshot/fprisc --root qos=$PWD/source_snapshot/qos \
  --output experiments/my_expansion --epochs 2 --learning-rate 0.00003 \
  --pad-multiple 32 --mps-cache-interval 20
```

For evaluation, retain the original benchmark data directory and pass the
expanded manifest through `--splits`. Its separate verification file covers the
new training examples; the original verification file proves the unchanged
benchmark references. The completed workflow and comparison are recorded in
[EXPANDED_FINETUNE_2026-10-16.md](2026-10-16-EXPANDED-FINETUNE.md).

## Focused input-preservation curriculum

`qos_ai_inputs_curriculum.py` creates verified debugging/generation contrast pairs.
Changing one input field must change the expected result. It checks the structured
input contract, the exact reference code in the answer, source provenance and both
correct/broken behavior. Full question/answer encodings must fit the unchanged
student vocabulary. Whole groups are removed if a variant is screened out.

The focused pass uses a frozen `training_selection.json`: all new contract tasks
plus factual replay. `--selection` is optional for other experiments; when set,
the trainer requires that exact verified training subset and records its hash.
The canonical collection and all historical holdout assignments are preserved.

On the M1, using a fresh output directory:

```sh
.venv/bin/python qos_ai_inputs_curriculum.py build \
  --previous curricula/verified_expanded --qa evaldata/qa.jsonl \
  --snapshot source_snapshot \
  --tokenizer-checkpoint experiments/verified_expanded_sft2_retry/student.pt \
  --output curricula/my_input_curriculum
.venv/bin/python qos_ai_inputs_curriculum.py verify \
  --data curricula/my_input_curriculum --snapshot source_snapshot \
  --fpr source_snapshot/fprisc/fpr
.venv/bin/python qos_ai_finetune.py \
  --load experiments/verified_expanded_sft2_retry/student.pt \
  --curriculum curricula/my_input_curriculum \
  --selection curricula/my_input_curriculum/training_selection.json \
  --data curricula/my_input_curriculum/focused_train_ready.jsonl \
  --splits curricula/my_input_curriculum/splits.json \
  --root fprisc=$PWD/source_snapshot/fprisc --root qos=$PWD/source_snapshot/qos \
  --output experiments/my_input_curriculum --epochs 1 --learning-rate 0.00003 \
  --pad-multiple 32 --mps-cache-interval 20 --fixed-padding --defer-final-loss
```

The completed input-focused run uses one epoch after longer M1 GPU jobs failed.
`--defer-final-loss` records that the final training-loss diagnostic is measured
in a fresh process; it does not change optimization or the saved weights. The
completed workflow and diagnostic script are preserved with the experiment.

Retain the original benchmark directory for evaluation and pass the new manifest
with `--splits`. `qos_ai_compare_followup.py` checks paired full-development runs
against their recorded parent, accepting only the verified manifest extension.
See [INPUTS_FINETUNE_2026-10-16.md](2026-10-16-INPUTS-FINETUNE.md) for the completed
experiment and actual generated-code behavior.

## Broader verified calculations and generalization

The dated M1 experiment in [GENERALIZATION_2026-10-19.md](2026-10-19-GENERALIZATION.md)
adds 2,400 distinct executable cases and 384 replay examples without changing the
168M architecture or either existing development benchmark. It checks every
intermediate value against the pinned Sol compiler, freezes separate wording,
new-input and held-out-composition diagnostics, and compares two training seeds
at intermediate and final checkpoints.

`qos_ai_generalization.py` builds/verifies the data;
`qos_ai_generalization_workflow.py` records the primary fixed protocol.
`qos_ai_generalization_eval.py` retains the original guarded diagnostic, while
`qos_ai_generalization_raw.py` supplies a separately labelled no-guard control
because legitimate calculation answers repeat token sequences. A whole-field
trace audit rejects malformed numeric prefixes without changing primary scores.
Operational MPS recovery preserves the optimizer treatment and archives rejected
updates. See the dated report for completed checks, results, limits and artifact
identities. These remain development diagnostics with unknown legacy exposure.

## Custom 557M scaling pilot

The M1-only, randomly initialized 556,526,080-parameter experiment is documented in
[SCALE_557M_2026-10-19.md](2026-10-19-SCALE-557M.md). It introduces an isolated,
resumable runner and a bounded Adafactor learning pilot. This is infrastructure
and early-loss evidence; it does not establish a capability upgrade over the
pretrained 168M checkpoints.

The subsequent [RunPod experiment](2026-10-19-RUNPOD-557M.md) uses a Secure RTX 4090,
an eight-hour GPU ceiling and durable checkpoints. It measures pretraining
throughput, then adds verified SFT and uses the unchanged development benchmarks.
GPU generations are graded with the M1's frozen Sol compiler. It is a budget
training pilot; capability claims require the completed results.

## Next 557M training milestone

The [next training plan](2026-10-19-NEXT-STEPS-557M.md) records the completed
pilot's baseline and proposes continued pretraining to approximately 1B total
tokens, a more varied verified curriculum and controlled SFT comparisons. It
also records the required recovery/shutdown checks and the user's instruction
to retain the checkpoint volume. These are planned experiments, not new results
or authorization to restart paid compute.
