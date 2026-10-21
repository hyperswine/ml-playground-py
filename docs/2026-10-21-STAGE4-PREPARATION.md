# Stage 4 preparation: retain language while strengthening Sol behavior

Date: 2026-10-21. Status: preparation and local verification complete; no Stage 4 training started and no paid resources launched. “Next steps 1-44” was interpreted as steps 1–4 of the latest assessment.

The accumulated Stage 1–3 source, data and textual evidence were committed as `e293565d`. Large checkpoints and rebuildable tensor payloads remain outside Git. The default model is unchanged.

## 1. Stronger evaluation

[`stage4_20261021/multi_input.json`](data/qos_ai_benchmark/stage4_20261021/multi_input.json) freezes 18 requests: six computations with three wordings each. Every answer must supply one reusable expression over `xs`, `k` and `limit`. That same expression runs on six inputs, including zero, negative numbers, empty lists, equality boundaries and both conditional branches. There are 36 distinct reference executions and 108 checks when scoring all 18 answers. A request passes only when all six cases pass.

All references were executed with compiler SHA-256 `9987da491a791af853300f38ad3e73c52a243dbfd1422379a8dbe437369b776e`. Tests establish that constant answers, `>=` in place of `>`, and an incorrect otherwise-unused branch fail. The evaluator admits a bounded pure subset, reconstructs parsed expressions before executing them and reports unsupported answers separately. This is finite behavioral evidence, not a general Sol sandbox or proof over every possible input.

All 18 prompts were audited with the frozen tokenizer and vocabulary: zero unknown tokens, at most 69 prompt tokens, and room for the full 200-token answer budget. Generation loads this local tokenizer and records its identity.

The new diagnostic is a prospective **development** set. Shared concepts and three paraphrases are not 18 independent capabilities. The original 90-prompt development benchmark, 96-prompt coding diagnostic, 144-prompt generalization diagnostic and their historical graders remain unchanged. The Stage 3 360-task set has now been inspected and informs development; it should not be described as an untouched final test for future runs. The original reserved final test remains unused.

Generate baseline answers for the domain parent, Stage 3 specialist and 168M coding reference before the next training run. Use the same M1, greedy decoding, seed 7, 200 new tokens and no repetition guard for this new suite. Preserve the original token budgets for the old benchmarks. No new model scores are claimed in this preparation.

## 2. Gentler variants and token-accounted replay

All three variants start from the same 0.9B-token domain parent, SHA-256 `5879f9c385158e1b679e8fda9c6fc53f8f15919e109d25c05c7a332f5d6df730`, with a fresh optimizer. The Stage 3 specialist remains a separate comparison checkpoint.

| Variant | Duration | Peak learning rate |
| --- | --- | --- |
| `half_2e5` | Half of one shuffled pass, rounded to an update | 2e-5 |
| `one_2e5` | One pass | 2e-5 |
| `one_1e5` | One pass | 1e-5 |

Seed 7, effective batch 32, AdamW betas (0.9, 0.95), weight decay 0.1 and a 20-update warmup are fixed. [`experiments.json`](data/qos_ai_benchmark/stage4_20261021/experiments.json) records the plan. The runner records the actual update and target-token counts, so a half pass is not claimed to be exactly half the supervised tokens.

The proposed mixture is **65% verified targets, 14% English, 14% training-documentation and 7% training-Q&A targets**. Shares are measured by supervised tokens, excluding prompt-only positions. Whole examples are retained; the sampler reports overshoot and repeated pool cycles. This corrects the ambiguity of Stage 3's “10% replay,” which meant ten Q&A examples per hundred verified examples, about 9.1% of the combined example count.

Prepared locally:

| Pool | Examples | Supervised target tokens |
| --- | ---: | ---: |
| Verified Stage 3 plus arithmetic training | 65,017 | 2,158,102 |
| Frozen training-split Q&A pool | 20,000 | 2,636,899 |
| Training-documentation pieces | 2,302 | 762,183 |

Eight short or unknown-token documentation pieces were rejected. Accepted examples contain zero unknown tokens and are at most 512 tokens long. The 125 arithmetic validation rows are encoded separately and never enter the replay mixture. [`pools/preparation.json`](data/qos_ai_benchmark/stage4_20261021/pools/preparation.json) records source identities and audits. Tensor files are ignored and reproducible with the preparation command.

The actual English replay mixture is **pending retrieval of the original training stream**. The mixer requires its archived manifest and matching byte hash; it never substitutes an English validation probe. These are repeated training windows, not new pretraining documents. A synthetic-stream test covers mixing and hash rejection, but does not substitute for assembling the real stream.

Select candidates using the separate arithmetic validation and the fixed loss probes. Predeclared retention thresholds are English, documentation and Q&A perplexity no worse than 1.25 times the common domain parent's corresponding result. These are experimental thresholds, not measured successes. Report every variant on development diagnostics, inspect complete-answer correctness and regressions, and repeat a promising variant with another seed before considering promotion. No automatic default-model replacement is configured.

## 3. Verified arithmetic and phrasing

[`arithmetic/verification.json`](data/qos_ai_benchmark/stage4_20261021/arithmetic/verification.json) records 1,024 unique input groups: **899 training and 125 validation**. Families cover mapped products, strict filtering, absolute values and clamping, followed by sums. Answers include intermediate lists, running sums and individual products where relevant. All 3,854 intermediate expressions agreed with the real compiler.

Groups are assigned before selecting wording; training and validation use disjoint wording templates. Existing benchmark/probe programs and questions were checked for exact normalized overlap. This holds out input groups and wording, not entire mathematical concepts, and is not a claim of semantic decontamination. The supplement preserves Stage 3's broader operation coverage rather than replacing it with four arithmetic families.

## 4. Recovery and storage

On 2026-10-21, read-only SSH to `jq@100.107.153.43` confirmed `JQs-Mac-mini.local`. Full file hashes of the domain parent and Stage 3 specialist matched their recorded values. Both files are about 2.23 GB. The M1 filesystem reported 65,797,436 KiB free, about 62.75 GiB. This confirms byte identity and current local capacity; checkpoint structures were not reloaded during the backup check.

[`recovery_inventory.json`](data/qos_ai_benchmark/stage4_20261021/recovery_inventory.json) keeps recovery incomplete until the latest pretrained weights, matching optimizer and document ledger have verified off-volume copies. A targeted search of the M1 experiment directory found no copies of the original English stream, dataset manifest, ledger or Stage 2 optimizer under their expected names. The archived dataset manifest exists in this repository. The latest optimizer's expected digest still needs an authoritative read from the retained volume.

The new runner periodically writes full optimizer progress, checks space before each replacement, and preserves the previous checkpoint until the replacement has been read back and renamed. It requires 9 GiB of free write space plus a 2 GiB reserve before a progress save, or 3 GiB plus reserve before publishing weights. Existing files already consume capacity; their sizes are not incorrectly subtracted from the free-space requirement. A tiny CPU test verifies that interrupted and uninterrupted runs end with identical weights and Adam state. GPU/MPS resume equivalence and full-size save timing remain untested.

Do not infer that the retained 40 GB cloud volume has enough space from the M1's capacity. Budget the original artifacts, all variant outputs, a temporary checkpoint and safety reserve together, then measure the actual destination. No cleanup or volume deletion is part of this preparation.

Before any paid launch, finish off-volume recovery, assemble the real mixture, capture baseline evaluations, agree a cost ceiling and verify an independent persistent controller with a fixed deadline, bounded stop retries and stop-state read-back. A trainer deadline saves and exits; it does **not** release a paid GPU. The retained volume `xewpn3z8za` must remain. Cloud resource state and prices were not queried in this turn.

## Reproduction and validation

See the [Stage 4 runbook](2026-10-21-RUNPOD-STAGE4.md) for preparation, generation, scoring and future training commands. The new evaluator references, adversarial cases, split/encoding audits, replay accounting, disk-headroom failure and CPU resume path are covered by `test_qos_ai_stage4.py`. The full focused suite was run with the pinned compiler; the final validation receipt records the counts and tested source hashes.

The isolated M1 kit lives at `/Users/jq/qos_ai_eval/experiments/stage4_20261021/preparation`. It includes source,
prepared pools, the local tokenizer/vocabulary and a separate copy of the pinned compiler under `toolchain/`.
Its manifest hashes every included file. The separate staging receipt records transfer verification and the
M1 checks; it does not contain new model inference or training results. Existing M1 experiments are preserved.
