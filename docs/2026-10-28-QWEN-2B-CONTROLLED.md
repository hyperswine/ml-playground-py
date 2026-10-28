# Qwen 2B controlled coding curriculum

Updated: 2026-10-28T06:44:06+11:00. Status: complete; retain round 4.

The previous experiment is committed as `3abbaf6`. Its failed adapter remains experimental.
The reference remains the round-4 Qwen3.5-2B MLX q8 checkpoint. No cloud resources
are started and no default model is changed by this experiment.

## Full original regression and final decision

**Retain round 4.** The first seed improved working programs on the harder diagnostic,
and that program-count gain repeated. Its complete-answer improvement did not
reproduce strongly enough to satisfy the predeclared screen. No model was promoted.
The final holdout remains unused, and the failed repeat did not receive another
740-prompt run. All authorized pilot and confirmation work is finished.

The selected first seed passes the full regression floors.
The selected adapter generated all 740 original prompts in 19.8 minutes.
The parent comparison reuses the previously regraded saved responses, not a new
parent generation run. Prompt text, decoding budgets and scorer/compiler identities
remain frozen. No pure-program admission was declined in the selected Sol suite.

| Measure | Round 4 | Selected mixed 8K |
| --- | ---: | ---: |
| Original Sol tests /360 | 341 | 342 |
| Executable programs /111 | 111 | 111 |
| Multi-input tasks /18 | 18 | 18 |
| Original dev lexical proxy /90 | 51 | 53 |
| Original dev outputs /33 | 30 | 30 |
| Working generation /24 | 24 | 24 |
| Applied repairs /48 | 46 | 46 |
| Repair plus output /48 | 35 | 35 |
| No-change plus output /24 | 19 | 18 |
| Generalization wording /48 | 43 | 43 |
| Generalization inputs /48 | 42 | 41 |
| Generalization combinations /48 | 41 | 41 |
| Strict trace fields /120 | 108 | 109 |
| Ordered trace values /120, supplementary | 108 | 109 |

The lexical and calculation-field metrics are proxies, not broad factual accuracy.
The ordered-value audit is post-hoc and supplementary; it preserves the original
scores and decision thresholds. The full result is separate from the 24-item early
calculation guard. [Full regression summary](data/qos_ai_benchmark/qwen2b_controlled_20261028/selected_frozen_summary.json),
[calculation audit](data/qos_ai_benchmark/qwen2b_controlled_20261028/full_trace_value_audit.json),
and [confirmation decision](data/qos_ai_benchmark/qwen2b_controlled_20261028/confirmation_decision.json).

## First-seed results

The mixed 8K checkpoint passed the predeclared development screen and advanced
to confirmation. All other first-seed checkpoints failed that screen.

| Checkpoint | Complete /48 | Programs /36 | Stated outputs /48 | Strict fields /24 | Ordered values /24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round 4 | 4 | 21 | 4 | 20 | 20 |
| replay_2048 | 5 | 20 | 6 | 22 | 22 |
| replay_4096 | 6 | 20 | 7 | 22 | 22 |
| replay_8192 | 4 | 20 | 4 | 21 | 21 |
| mixed_2048 | 5 | 20 | 6 | 11 | 22 |
| mixed_4096 | 5 | 17 | 6 | 21 | 21 |
| mixed_8192 | 6 | 24 | 6 | 20 | 20 |

The mixed 8K candidate gains six working generation programs and loses three working
repairs compared with the parent, for a net gain of three. Complete generation
answers improve from 1/12 to 3/12; complete repairs and complete no-change answers
remain 0/12. This is a narrow development gain, not yet a reliable debugging upgrade.

The mixed 2K checkpoint's apparent strict-trace collapse largely reflects field
names: eleven correct ordered traces use labels such as `adjusted` instead of
`products`. A post-hoc supplementary audit distinguishes exact ordered values from
canonical field-name matches. Neither it nor the frozen scores prove arbitrary
prose equivalence or program correctness. The primary selection criteria remain
unchanged, and the selected 8K checkpoint passes them without this adjustment.
[Audit details](data/qos_ai_benchmark/qwen2b_controlled_20261028/trace_value_audit.json).

Replay training took 325.5 seconds (3.49 GB peak MLX memory); mixed training took
309.4 seconds (6.16 GB). Both adapters and all six intermediate checkpoints have
hash-verified local backups. These memory figures are MLX allocation peaks, not
system RAM. [Primary comparison](data/qos_ai_benchmark/qwen2b_controlled_20261028/comparison.json).

The confirmation uses seed 20261029, the same source pools and 8,192 answer tokens
(116 unique rows, 58 updates). The full-regression floors were frozen before its
outputs and before the selected model's 740-prompt evaluation. The final holdout
can run only if both seeds pass full regression and the repeat passes development.
[Confirmation plan](data/qos_ai_benchmark/qwen2b_controlled_20261028/confirmation_plan.json).

## Second-seed development confirmation

Seed 20261029 trained on exactly 8,192 supervised tokens in 300.7 seconds, with
5.61 GB peak MLX memory. It reproduced 24/36 working programs and 20/24 calculation
traces, but reached only 5/48 complete answers (parent 4/48; first seed 6/48).
Its complete generation score is 1/12 and output-only score 4/12, compared with
3/12 and 3/12 in the selected first seed. Complete repairs and no-change answers
remain zero. The complete-answer gain therefore did not reproduce strongly enough
to pass the predeclared screen. No model is promoted and the final holdout stays
unused. The selected first seed was measured on the full original
740-prompt suite to characterize retention; the repeat did not receive another
full run after failing its development screen.

[Repeat comparison](data/qos_ai_benchmark/qwen2b_controlled_20261028/repeat_comparison.json).

## Predeclared comparison

Two conditions start from the identical round-4 model at learning rate 1e-5,
seed 20261028, rank 8, last four layers, batch one and accumulation two.
The corrected completion-only loss excludes prompt/padding and includes EOS.
Replay alone is compared with replay plus new calculation/repair/generation
examples. Documentation examples are excluded from both conditions.

The replay pool has 240 historical, compiler-reverified examples (60 per category).
The new pool has 288 examples (72 each output, repair, no-change and generation).
Six training composition families are retained; validation uses a seventh family
with 16 examples. All 544 reference programs and their intermediate calculations
were executed successfully against the pinned compiler. New repair/no-change and
generation targets preserve the supplied bindings and return a complete program.
Historical replay preserves its requested response format, including output-only
and expression-only tasks. Steps precede the final stated output when requested.

Replay excludes exact normalized evaluation questions/programs and shared historical
case IDs. This does not establish semantic decontamination of the original pretraining.
The frozen development/final fixtures are unchanged. The final holdout has no answers.

| Condition | Target budget | Actual target tokens | Examples | Updates |
| --- | ---: | ---: | ---: | ---: |
| Replay | 2,048 | 2,048 | 50 | 25 |
| Mixed | 2,048 | 2,048 | 28 | 14 |
| Replay | 4,096 | 4,060 | 100 | 50 |
| Mixed | 4,096 | 4,096 | 58 | 29 |
| Replay | 8,192 | 8,191 | 208 | 104 |
| Mixed | 8,192 | 8,208 | 120 | 60 |

Rows are processed in a frozen, recorded order without replacement. Mixed schedules
contain equal numbers of new and replay examples and approximately balanced categories.
At the final mixed checkpoint, new examples supply 5,874/8,208 target tokens (71.6%).
The final pair at each milestone is length-selected to match token exposure within 1%.
No target is truncated. Budgets match tokens, not optimizer updates: the longer new
answers mean fewer examples/updates in the mixed condition. This is a practical
curriculum comparison, not an isolation of data effects from update count.

All three checkpoints are saved within each bounded 8K-token pilot and evaluated
before considering any further exposure. The screen requires at least two additional
complete coding answers, no reduction in working programs, at most one lost case
per category, and at most one lost strict-trace case on a fixed 24-item regression
guard. Parent scores are 4/48 complete, 21/36 programs and 20/24 trace guard.
A passing recipe must be repeated with another seed and checked against the full
original suite before the final holdout is considered. Nothing is automatically promoted.

The independent audit also passed 2,128 boundary-case checks (reusing identical verified executions) and confirmed
76 effective repairs across training and validation; all tokenizer boundaries match
and the longest new sequence is 388 tokens.
[Four training targets for review](data/qos_ai_benchmark/qwen2b_controlled_20261028/curriculum_samples.json)
are explicitly labeled as authored examples, not model outputs.

Evidence: [frozen plan](data/qos_ai_benchmark/qwen2b_controlled_20261028/plan.json),
[verified data and schedules](data/qos_ai_benchmark/qwen2b_controlled_20261028/prepared_manifest.json).

## Independent retrieval coverage

Added current implementation excerpts for `sol/lib/rand.sol`, `matrix.sol`,
`plot.sol`, `archive.sol` and `git.sol`, keeping the earlier index immutable.
The 20 new development queries cover these added libraries. The previous index
had none of them; its zero score measures missing coverage, not a fair ranking comparison.
On the expanded corpus, plain BM25 finds the target file in its top three for
19/20 queries and the specific evidence substring for 18/20. Source-aware ranking
finds the file for 20/20 and the evidence for 19/20. The remaining miss retrieves
the matrix module but omits the cell-reader definition. These are retrieval proxies,
not evidence of model factual accuracy. No retrieval query enters training.

[Retrieval report and excerpts](data/qos_ai_benchmark/qwen2b_controlled_20261028/retrieval/report.json).
The expanded corpus remains opt-in using `--retrieval-profile source --corpus PATH`.

## Implementation and validation

`qwen_controlled` builds the verified curriculum and schedules; `source_audit`
freezes and scores the expanded retrieval corpus. `scripts/mac/mlx_repair.py`
now accepts an explicit seed, ordered batches and intermediate checkpoint counts.
`scripts/mac/run_qwen_controlled.py` records the sequential experiment status,
exposure receipts and development selection. Model weights and prepared training
rows stay ignored under `artifacts/qos_ai/qwen2b_controlled_20261028`.

Passed locally: 31 focused Qwen tests and five layout tests. All four dataset/mask/
batching tests passed on the M1; the two MLX-specific tests are skipped locally,
where MLX is absent.

## Next data and evaluation work

Prioritize consistent repair decisions and complete answers. New training-only
counterexamples should distinguish “the program is already correct” from “I changed
the program,” including cases where an incorrect operation order happens to pass
the displayed input. Require the final output as well as runnable code, and verify
against additional inputs. Do not copy development or final answers into training.

Preserve calculation-field conventions or preregister an equivalence-aware audit
before selection; this run showed how correct renamed calculations can look like
forgetting under a canonical-name metric. The ordered-value proxy alone is not a
semantic judge. A future comparison should also control optimizer-update count or
explicitly separate its effect from token exposure. These small runs do not establish
a model-capacity limit or justify increasing parameter count yet.

The M1 experiment is `~/qos_ai_eval/experiments/qwen2b_controlled_20261028`.
`prepared/manifest.json` pins both first-seed schedules. `confirmation_plan.json`
pins the second seed and full-regression floors. Adapters and prepared rows are
backed up locally in ignored `artifacts/qos_ai/qwen2b_controlled_20261028`.
The original checkpoints, both first-seed adapters, all intermediate checkpoints,
and the repeat adapter are retained. No paid cloud compute was used.
