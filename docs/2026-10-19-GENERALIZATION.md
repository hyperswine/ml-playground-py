# Broader verified curriculum and generalization experiment

Date: 2026-10-19 (Australia/Sydney). Status: both M1 fine-tunes, all predeclared comparisons, supplemental decoding controls, verified local backups and identity audits are complete.

The experiment tests whether broader verified training at the current size improves supplied-input use, calculation, and repair completion. It changes coverage and intermediate-calculation teaching together; it is not a trace-only causal ablation or a parameter-scaling experiment.

## Results and recommendation

At the same 168M parameter count, the broader curriculum improved repair application and the new within-domain diagnostics in both seeds. It is not a uniform upgrade: both final checkpoints generate 20/24 working programs versus 21/24 for the parent, and score 49/90 development lexical matches versus 54/90. Keep the variants separate. These results demonstrate useful headroom on the measured tasks, not a proven capacity ceiling or broad coding competence.

### Unchanged development and coding benchmarks

All values below use the original guarded condition. Lexical matches are proxies, not factual accuracy. Working generation does not by itself assess every sentence of any accompanying explanation.

| Checkpoint | Lexical /90 | Stated outputs /33 | Working generation /24 | Applied repairs /48 | Repair + stated output /48 | No-change + output /24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent | 54 | 18 | 21 | 27 | 1 | 4 |
| Seed 7, update 336 | 45 | 13 | 20 | 34 | 5 | 6 |
| Seed 7, final | 49 | 15 | 20 | 36 | 7 | 7 |
| Seed 17, update 336 | 46 | 14 | 20 | 35 | 6 | 7 |
| Seed 17, final | 49 | 16 | 20 | 37 | 6 | 8 |

The new repair gains are consistent across the two seeds, while generation and original development regressions also recur. The broader training and calculation teaching changed together, so this experiment cannot assign the gain specifically to intermediate traces.

### Frozen generalization diagnostic

Each cell is passes out of 48 under the code/output/judgment checks. These checks do not require a correct complete explanation; calculation traces receive the separate audit below. Output literals use the frozen stated-value parser. Wording intentionally shares training cases, while the other groups exclude known canonical program identities.

| Checkpoint | Guard: wording | Guard: inputs | Guard: combinations | No guard: wording | No guard: inputs | No guard: combinations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent | 6 | 5 | 2 | 7 | 5 | 2 |
| Seed 7, update 336 | 13 | 11 | 2 | 17 | 12 | 2 |
| Seed 7, final | 16 | 13 | 5 | 20 | 14 | 5 |
| Seed 17, update 336 | 14 | 10 | 1 | 16 | 11 | 1 |
| Seed 17, final | 16 | 11 | 4 | 19 | 12 | 4 |

Novel compositions remain weak, despite gains on familiar operation structures. Shared-source and templated tasks cannot establish unseen-library or general conversational ability. No reserved test model inference was run.

### Whole-field calculation audit and repetition

These supplementary scores parse every required canonical value in full, rejecting malformed numeric prefixes. They cover the 120 prompts that request calculation steps. Different correct formats or field names remain unmatched, and other free prose is not completely judged.

| Checkpoint | Strict trace fields, guard /120 | Strict trace fields, no guard /120 | Answer + strict trace, no guard /120 | No-guard finished /144 | No-guard repetition flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent | 0 | 0 | 0 | 144 | 0 |
| Seed 7, update 336 | 6 | 17 | 17 | 143 | 1 |
| Seed 7, final | 7 | 20 | 20 | 142 | 2 |
| Seed 17, update 336 | 5 | 15 | 15 | 140 | 5 |
| Seed 17, final | 7 | 21 | 21 | 144 | 1 |

Removing the repetition blocker allows legitimate repeated lists and values, but some raw runs repeat until the output limit. Manual inspection of all nine detector flags found seven repetitive truncations and two finite answers with repeated token patterns; the flag count alone is not an observation of an endless loop. Calculation reliability remains substantially below code-structure reliability.

### Learning-curve probes

The same 96 training examples and three fixed 48-prompt diagnostic groups are used at each checkpoint. Answer cross-entropy measures target-token fit; it is not an exact-answer accuracy measure.

| Checkpoint | Training probe | Wording | Inputs | Combinations |
| --- | ---: | ---: | ---: | ---: |
| Parent | 1.9899 | 2.2292 | 2.2245 | 2.4882 |
| Seed 7, update 336 | 0.2091 | 0.1898 | 0.2316 | 0.6292 |
| Seed 7, final | 0.1703 | 0.1541 | 0.1982 | 0.5855 |
| Seed 17, update 336 | 0.2140 | 0.1932 | 0.2261 | 0.6481 |
| Seed 17, final | 0.1688 | 0.1518 | 0.1973 | 0.6143 |

### Validity, checkpoint preservation and operational limits

- All 8,917 reference checks passed across 6,870 distinct executions. The initial 87 M1 regressions and all 144 authored diagnostic controls passed. The recovery-math test, seven whole-field-audit tests and four retry-ledger tests also passed on the M1.
- Both accepted training logs contain exactly updates 1–696, the original learning-rate schedule, and 97,693 final supervised tokens. Seed 7 automatically retried one process and left 30 rejected records in its append-only raw log. Its accepted ledger is reconstructed against successful save boundaries, with the original log preserved. Seed 17 needed three failed 48-update attempts and two failed shorter attempts; all rejected work was rolled back and archived. It completed with 16-update operational processes after saved update 384. CPU equivalence does not imply bitwise deterministic MPS execution.
- Frozen input/code identities, diagnostic coverage, answer hashes, platform/decoding settings, checkpoint lineage and paired benchmark conditions passed the comparison audit.
- 28 inference/loss logs were checked with no detected Metal/target-count errors, and all loss probes were finite. Training-process errors are explicitly retained separately.
- All four trained checkpoints were copied locally, hash-checked against the M1 and checked for the unchanged architecture. The experiment does not replace the default or its parent checkpoint. No cloud compute, commit or push was performed.

| Checkpoint | SHA-256 | Local backup |
| --- | --- | --- |
| Seed 7, update 336 | `be58269af971f038d877e6f3ebdc099f3b6734544f66a773da3bea9618084137` | `qos_ai/m1_generalization_seed7/step_0336.pt` |
| Seed 7, final | `f51187bc5176b2711f22b4c1f631b75c134951fc12cd25c4441a6ea9a5b51de2` | `qos_ai/m1_generalization_seed7/student.pt` |
| Seed 17, update 336 | `ad4f7447bc611e470646d834acd45bef9e8a8e2f98fd2951943a4c3a069e6d3a` | `qos_ai/m1_generalization_seed17/step_0336.pt` |
| Seed 17, final | `ce30025d311c210b68d4f323f4a3c5a23d78904ed9d1e27bdd1ea67350e2d1cf` | `qos_ai/m1_generalization_seed17/student.pt` |

The complete machine-readable result is [comparison.json](data/qos_ai_benchmark/evidence/generalization_20261019/reports/generalization_20261019/comparison.json). Frozen data, raw responses, reference proofs, code snapshots, GPU-error logs and recovery records are retained under `evidence/generalization_20261019/`. Model weights and optimizer progress are excluded from Git; last saved full optimizer states remain on the M1.

## Fixed model and treatment

Both seeds start from the same balanced coding checkpoint, SHA-256 `5312f018cc482f8b209a5c09ffc6a5c77de8bb8695661075f5d511654e5b0767`. Its 168,331,264 unique parameters, eight layers, 1,024-wide embeddings, 65,536-token vocabulary and 512-token context remain unchanged. Parent and default checkpoints are preserved.

Training is one epoch over 2,784 examples: 2,400 distinct new program cases, 96 source-grounded factual replay examples, and 288 prior verified coding examples. New cases cover the previous 12 families plus map/take, filter/reverse, and map/clamp compositions. Lists have two through five elements and include negative and zero values. Prefixes include empty prefixes. Clamp examples include boundaries; search includes present/absent bytes. New variable names, ASCII strings and input combinations vary across cases. All complete examples fit the existing 192-token training width with zero unknown tokens. The fixed pass supplies 97,693 supervised answer tokens per seed.

New answer modes include literal output, output with calculation, expression repair, complete-file repair, already-correct checks, and complete program generation. Explanations use mechanically reconstructed name=value intermediate fields. Every list intermediate is checked element by element and for its length by authored Sol fixtures; scalar intermediates and final outputs are also executed. Incorrect repair fixtures must demonstrate the authored wrong behavior.

The predeclared protocol uses the same data, one epoch, AdamW learning rate 0.00001, batch size two, gradient accumulation two, answer-only loss, fixed padding, and 48-update processes with full optimizer-state resumes. Seed 17 later uses shorter operational processes as documented below; its learning treatment is preserved. Each seed has 696 updates; an inference-compatible checkpoint is also exported at update 336. The training time ceiling is 60 minutes per seed. A stopped treatment is incomplete, not a successful epoch.

## Frozen evaluation

The existing 90-prompt development benchmark and 96-prompt coding diagnostic remain byte-for-byte unchanged. The original benchmark SHA-256 remains `1394b9a12fd1d2de8d55c57437b7c5945b0e270a9850c01039b22c38b38c81e7`. Reserved test prompts are not used for model inference.

The new 144-prompt diagnostic has three separately reported 48-prompt groups. Every group contains eight prompts in each of the six response modes:

- Wording: exact new-training inputs, operations and response modes, with paraphrased instructions. This intentionally shares program cases with training and does not measure new-input generalization.
- Inputs: familiar training templates and operation structures, but complete programs differ from the known curriculum and benchmark programs. Shared task wording is intentional. Historical lexical protections remain active; the new-input diagnostic itself is protected by program identity rather than rejecting familiar templates.
- Combinations: map/filter, map/reverse and take/filter compositions omitted from the selected curriculum. Known complete-program and operation-structure overlap is audited separately. Unknown legacy pretraining exposure cannot be excluded.

All cases are frozen before model generations. Known historical development and coding diagnostic protections are retained. Both models inherit `legacy_or_unknown` ancestry. These remain small bounded, pure Sol tasks; they do not establish new-library, multi-file, operating-system or general conversational ability.

The parent, both intermediate checkpoints and both final checkpoints receive the same guarded development, coding and new generalization evaluations. Parent and final checkpoints also receive the original development benchmark without repetition blocking. New generation uses greedy decoding, seed seven, top-k 20, no-repeat three, and 128 new tokens. Old benchmark settings are preserved.

The new grader separates correct stated output, applied program behavior, completed code/output/judgment requests, and canonical intermediate-field prefix matches. The supplemental whole-field audit below corrects prefix-parser false positives. Code execution requires the existing conservative syntax and exact operation/input structure. Different correct algorithms outside that subset remain unmatched. Correct already-correct judgments rely on independently executed references. Canonical field matching is deliberately strict; missing or differently named fields do not establish incorrect arithmetic, and the metric does not judge every sentence of free prose.

A fixed 96-example training probe and all three diagnostic groups receive answer-token cross-entropy measurements at every evaluated checkpoint. Those losses measure answer fit, not standalone correctness. All 144 authored diagnostic answers must pass the grader before model inference.

## Interpretation and artifacts

Report each seed and checkpoint, including regressions, rather than choosing a favorable run after seeing the new diagnostic. Improvements should be assessed on final values and complete requests, separately from training fit and trace formatting. Two seeds and these small correlated task families provide exploratory evidence, not a robust estimate of broad capability or a proven capacity ceiling.

Implementation: `qos_ai_generalization.py`, `qos_ai_generalization_eval.py`, `qos_ai_generalization_controls.py`, `qos_ai_generalization_workflow.py` and `test_qos_ai_generalization.py`. The existing model, trainer, parser and benchmark evaluator are reused without changes.

M1 workspace: `/Users/jq/qos_ai_eval`. Frozen data: `curricula/generalization_20261019`. Run reports: `reports/generalization_20261019`. Seed checkpoints: `experiments/generalization_20261019_seed7` and `experiments/generalization_20261019_seed17`. Local evidence: `evidence/generalization_20261019/`.

## Verified preparation and starting checkpoint

The M1 passed 8,917/8,917 fixture checks across 6,870 distinct executions. The full regression run passed 87 tests. All 144 authored diagnostic answers passed the pre-inference grader controls, including required canonical trace fields. An independent local separation audit reproduced the M1 result: 48 intentional known-program overlaps for wording, zero known-program overlaps for inputs, and zero known-program or operation-structure overlaps for compositions.

The freshly evaluated parent reproduces 54/90 development lexical matches, 21/24 working coding-generation programs, 27/48 applied repairs, 1/48 repairs with the correct stated output, and 4/24 correct no-change judgments with output. Lexical matches are proxies, not factual accuracy.

| New diagnostic group | Complete code/output/judgment requests | Correct stated output literals | All canonical trace fields |
| --- | ---: | ---: | ---: |
| Wording | 6/48 | 9/48 | 0/48 |
| Inputs | 5/48 | 6/48 | 0/48 |
| Combinations | 2/48 | 2/48 | 0/48 |

These complete-request counts do not require the canonical trace format; that stricter metric is reported separately. Starting answer cross-entropy is 1.9899 on the fixed training probe, 2.2292 for wording, 2.2245 for inputs, and 2.4882 for compositions.

The new data remains procedurally authored. Its new repair fixtures focus on extraneous +1 or appended-string result adjustments; the replay retains earlier repair types. This experiment is therefore not evidence of broad compiler-error debugging, natural conversation, or arbitrary calculation ability. The wording group uses a fixed alternate instruction style per response mode, rather than a large independent paraphrase corpus.

## Supplemental decoding feasibility control

After freezing the primary protocol, an audit of the authored responses found that the existing no-repeat-three token guard makes 94/144 exact canonical answers impossible: 17 output-trace answers, 20 expression repairs, 21 complete-file repairs, 15 no-change answers and 21 generations. This was checked by feeding the exact encoded target continuation through the model's actual `NoRepeat` implementation; some alternate correct wording may avoid repeats. Repeating a list in code and its calculation is legitimate, so guarded canonical trace scores have a structural decoding limitation.

The primary experiment and its guarded scores remain unchanged. A separately labelled supplemental pass uses the same frozen 144 questions, all five checkpoint identities, model code, greedy seed, token budget and grader, with only `no_repeat=0`. It runs after the primary workflow to avoid concurrent MPS workloads. Its baseline checkpoint is unchanged even though this supplementary baseline inference happens later. The reason and settings are recorded before any supplemental inference in `raw_protocol.json`; exact blocked-answer evidence is in `repetition_feasibility.json`. This is an additional decoding control, not a change to training data or a replacement benchmark score.

## Supplemental whole-field trace audit

Review of first-seed responses found malformed fields such as `input=4 4` that the frozen primary trace parser accepts by matching the integer prefix. The primary field-match metric therefore remains a proxy. A separate post-freeze audit parses each entire semicolon/newline-delimited canonical value as JSON and rejects extra tokens, decimals used as integer prefixes, duplicate fields and wrong types. All 120 authored answers requesting traces pass this stricter control, and seven regression tests pass. The audit is applied consistently to every saved checkpoint under both decoding conditions. It preserves the primary scores and does not regenerate answers or change training. Different correct prose, field names or formats can still remain unmatched; it is not a full semantic explanation judge.

## Seed-17 operational recovery

The original 48-update MPS process starting at saved update 384 failed three times near updates 420–422 with Metal command-buffer errors and the device-target-count safety check. No affected segment checkpoint was published. Recovery archives the failed update records and retains the unique accepted log prefix through update 384.

`qos_ai_generalization_recovery_step.py` resumes the exact saved weights, Adam moments, CPU RNG state, data order, full 696-update learning-rate schedule, loss, accumulation, clipping, and MPS device. Its operational process boundary is reduced from 48 to 16 updates. A separate regression test matches the frozen training math and restored optimizer moments exactly on CPU; the same test passes on the M1 before recovery. This is not a claim of bitwise deterministic MPS execution. The recovery code hash and changed process boundaries are recorded explicitly; the original model/trainer/data/evaluator files remain frozen. The independent seed comparison therefore retains the same learning treatment but includes this operational difference.

Recovery completed all 696 accepted updates, all 2,784 examples and 97,693 supervised tokens. Two shorter-process attempts also required rollback/retry. Both final benchmark conditions and all supplemental decoding passes then completed.

## Retry-ledger reconstruction

The final audit identified one automatic retry in seed 7 that periodic progress checks missed. Its raw training log has 726 records: an unsaved attempt through update 270 restarts at update 241 from the successful save at 240. `qos_ai_generalization_update_audit.py` preserves that raw log and reconstructs 696 accepted records by selecting the final attempt inside each documented successful-save interval. It rejects rollbacks across successful save boundaries, missing steps and unknown resume boundaries, and binds the ledger to the last saved optimizer token/example counts. Four regression tests pass. Seed 17's rejected records were already archived during recovery; its accepted raw log is directly canonical. Both separate `accepted_training.jsonl` ledgers and reconstruction metadata are retained with the local model backups and experiment evidence. No weights or evaluation answers change during this audit.
