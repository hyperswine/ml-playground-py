# M1 benchmark and split implementation - 2026-10-16

Implemented and validated in `/Users/jq/qos_ai_eval` on `JQs-Mac-mini.local`
(Apple M1), connected as `jq@100.107.153.43` over Tailscale.

## Delivered

- A source-backed benchmark with 132 prompts in 52 concept groups: 90 development
  prompts and 42 test prompts. Includes false premises, insufficient information,
  debugging, code generation, output prediction and domain contracts.
- Frozen, file-grouped splits prepared before lesson packing, with exact/lexical
  near-duplicate exclusions and source-generalization holdouts.
- Explicit shard partition/hash metadata and rejection of stale/legacy lessons;
  training and checkpoint-selection paths exclude test windows.
- Complete-example Q&A packing, content-based cache signatures, train-only
  vocabulary construction for scratch students, and self-play/repair exclusions.
- Per-answer benchmark reports with checkpoint/data/code hashes, decoding and
  context-budget settings, stopping/repetition metrics, review templates and
  evidence-linked factual scoring. Rubric matches remain separately named proxies.
- A starter teaching set of 65 source-reviewed examples, including 48 executable
  Sol programs. Exported training-only rows are in
  `clean_benchmark_v2/verified_train_ready.jsonl` on the M1.

## Validation

- **22 regression tests passed on the M1**, including a real shard write/load/cache
  roundtrip using a tiny deterministic teacher.
- **104 reference-program checks passed on the M1.** These include teaching programs
  and benchmark reference variants; this count is not a count of independent
  concepts or of tested generated model programs.
- All curated source hashes and exact evidence snippets passed validation against
  the isolated current source snapshot.
- Full historical dataset packing passed at context length 512, with **zero
  oversized/rejected examples**, including zero rejected verified examples.
- The final evaluator passed an additional MPS generation check after source and
  reference-evidence enforcement was added.

## Frozen dataset

Manifest: `/Users/jq/qos_ai_eval/clean_benchmark_v2/splits.json`.

Hash: `cd57f4a7b5c2b02b40c6b8679c39a82712535c33254385a9139cfe8504529be3`.

| Kind | Training | Development | Test | Excluded |
| --- | --- | --- | --- | --- |
| Whole source files | 444 | 34 | 19 | 0 |
| Historical teacher Q&A | 107,364 | 7,702 | 2,637 | 490 |
| Verified teaching examples | 65 | 0 | 0 | 0 |

The lexical analysis found 4,034 near-duplicate links. There were no copied
source chunks crossing these partitions at the configured threshold.

## Existing-checkpoint diagnostic sample

Checkpoint: `qos_ai/pod_pt15/student.pt`.

SHA-256: `2db239f619ce87ac330b201c0c61c7aa76ea87b086a7326f86faf3b3bffe7a3d`.

Two M1/MPS smoke runs used the same 12 development prompts from 12 concept groups,
seed 7, greedy decoding, and a 128-token answer budget:

| No-repeat setting | Rubric proxy passes | Finished | Severe loops |
| --- | --- | --- | --- |
| Off | 0/12 | 6/12 | 5/12 |
| 3 tokens | 0/12 | 8/12 | 0/12 |

The unguarded answers were separately reviewed against their source references
by Codex: **0/12 were correct**. Judgments include specific reasons and source
locations. This is a small diagnostic sample and an agent source review, not
independent human review or a full-benchmark quality estimate. Generated model
code was not executed; compilation/behavior metrics remain unassessed.

No model evaluation was run on the reserved 42-prompt test partition. The model
is labelled `legacy_or_unknown` because old training exposure cannot be ruled
out. The new splits do not retroactively establish a clean test result.

## Preserved state and next use

The original checkpoint, `evaldata/`, old lessons and sibling repositories were
preserved. The original evaluation script is backed up under
`backups/2026-10-16-clean-benchmark/`. Current reference files and the matching
supplied arm64 compiler were isolated in `source_snapshot/`; the compiler was
executed on the M1, not built there. libgit2/libarchive runtime dependencies were
installed through Homebrew.

Historical teacher Q&A correctness against current source has not been certified.
The 65-example curriculum is a verified starter, not the proposed eventual
few-thousand-example set. Source-file isolation and lexical duplicate filtering
also do not guarantee complete separation of every semantically related concept.

No teacher scoring, weight training, cloud provisioning, commits or pushes were
performed. The next training experiment can consume the prepared curriculum and
frozen manifest, while retaining the starting checkpoint's exposure label.

See `qos_ai_benchmark/README.md` for exact commands and the review workflow.
