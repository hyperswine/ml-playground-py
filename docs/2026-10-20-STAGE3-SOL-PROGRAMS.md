# Stage 3: training on 60,000 executed Sol programs

Date: 2026-10-20 (GPU run 10:59-11:19 UTC). Status: complete; pod terminated. Follows
[Stage 2](2026-10-20-STAGE2-557M.md) and the [fresh-question probe](2026-10-20-SOL-PROBE.md).

2026-10-21 interpretation update: the program score below checks one supplied input per answer. It can accept
a constant answer or a program with a wrong unexecuted branch. It is not evidence that every generated program
uses inputs correctly across cases. [Stage 4 preparation](2026-10-21-STAGE4-PREPARATION.md) adds that diagnostic
separately without changing these historical scores.

## Data

`qos_ai_solgen.py` generates random, well-typed Sol programs over the builtins that exist today, including ones
never used in earlier training (`Str.upper`, `Str.trim`, `Str.join`, `List.any`, `List.all`, `List.zip`,
`List.drop`, `List.last`, `Numeric.max/min/abs`, `if`, `%`, `/`). Expression trees are 1-3 levels deep. Each program
is evaluated by a Python reference **and run through the real `fpr` compiler**; all 60,360 agreed (0 dropped).

Each program becomes one task: output prediction with `name=value` steps (71%), generation from an English
description (20%) or repair of a mutated program (9%). 591 were dropped at encoding for tokens outside the student
vocabulary, leaving 59,409 (5.5M tokens, 2.0M answer tokens).

A test set was built and frozen **before** training, 120 programs per stratum, none of whose programs or questions
appear in training:
- `fresh`: same distribution as training
- `held_pairs`: every program contains one of six parent/child operation pairs that never occur in training
  (`abs∘sum`, `any∘map`, `drop∘rev`, `filter∘range`, `max∘last`, `upper∘slice`)
- `big_numbers`: integer literals 61-150 (training literals are -40..60)

## Training

From the 0.9B-token 557M domain checkpoint (`m557_1b/domain.pt`): the 59,409 programs plus the 4,709 verified
examples, 10% training-split 4B Q&A replay, 2 epochs, lr 3e-5, batch 32, 4,410 updates, answer-only loss.
19 minutes on one RTX 4090. Answer loss 2.33 → 0.53.

## Results

Frozen generalization tests (correct = exact stated output; for generation/repair the program must also run under
`fpr` and print the right value):

| Model | fresh /120 | held_pairs /120 | big_numbers /120 | Generated programs that run correctly |
|---|---:|---:|---:|---:|
| **557M stage 3** | **52** | **67** | **60** | **103/111** |
| 557M 0.9B, pilot SFT | 5 | 0 | 7 | 7/111 |
| 168M coding parent | 4 | 3 | 5 | 12/111 |

Per kind for stage 3: output prediction 119/249, generation 33/72, repair 27/39. Thus **60/111** generation/repair
answers had both a passing program and correct stated output, compared with 103/111 passing programs alone.

Fresh 32-question probe (after fixing a scorer bug that rejected "It prints 14." because of the full stop; earlier
model totals change from 2/4/5/3 to 2/6/5/4):

| Group | n | 557M 0.9B pilot | 557M 0.9B mixed | 168M coding parent | 168M seed 7 | **557M stage 3** |
|---|---:|---:|---:|---:|---:|---:|
| Familiar operations, new inputs | 8 | 1 | 1 | 2 | 2 | 3 |
| Operations never taught before | 12 | 1 | 2 | 1 | 2 | **7** |
| New compositions | 4 | 0 | 1 | 2 | 0 | 2 |
| Concepts | 8 | 0 | 2 | 0 | 0 | 1 |
| **Total** | 32 | 2 | 6 | 5 | 4 | **13** |

Original M1 benchmarks (raw decoding, same graders as earlier reports):

| Model | Dev lexical /90 | Stated outputs /33 | Working programs /24 | Applied repairs /48 | Repair + output /48 | No-change + output /24 | Wording / Inputs / Combos | Strict trace /120 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| **557M stage 3** | 24 | 12 | 18 | 36 | **11** | 6 | 17 / 18 / **10** | **29** |
| 557M 0.9B, pilot SFT | 30 | 3 | 19 | 27 | 4 | 8 | 20 / 18 / 4 | 29 |
| 168M coding parent | **54** | **18** | **24** | 31 | 3 | 4 | 7 / 5 / 2 | 0 |
| 168M generalization seed 7 | 49 | 15 | 24 | **43** | 10 | 7 | **20** / 14 / 5 | 20 |

Stage 3 has the best complete repairs (11/48) and novel combinations (10/48) of any model, and stated outputs rise
from 3 to 12; dev lexical matches fall from 30 to 24 and remain well behind the chained 168M models.

Loss probes (perplexity): dev-partition docs 21.8 → 36.7, dev Q&A answers 9.36 → 12.9, FineWeb English 43.2 → 160.7.

## What changed and what did not

- **It improved substantially on the frozen generated tasks.** Complete-answer accuracy is 56% on held-out
  operation pairs, 43% in distribution and 50% on large literals. 93% of generated/repaired programs print the
  correct value for the supplied case; complete answers pass on 54% of those tasks. Broader input fidelity and
  unexecuted branches were not tested by this score.
- **Arithmetic is the weak point.** Wrong answers usually have the right steps and a wrong number, e.g.
  `map=[10, 25, 20]` for `x * 5` over `[2, 7, 4]`, or a correct `range=[3, 4, 5, 6]` summed to 10.
- **It overfits the generator's phrasing.** On the hand-written probe, `Str.upper "sol"` → "Sol", the trim answer
  loses its space after "It prints", and one generation request drops the doubling step.
- **It forgot general English and some domain prose.** 2 epochs on 64k short examples with only 10% replay pushed
  FineWeb perplexity from 43 to 161 and dev-doc perplexity from 22 to 37. More replay (docs and English, not only Q&A)
  is needed in the next mix.

## Operations and cost

- Upload into EU-RO-1 pods stalls from this Mac beyond small payloads; the 5 MB bundle went in 42 × 120 KB chunks,
  four connections at a time (3 minutes). The pod stopped itself at 11:19 UTC; it was restarted briefly to pull the
  checkpoint straight to the M1, then terminated.
- GPU: about 42 minutes at $0.89/hour ≈ $0.62 (not yet posted at the last billing read).

## Artifacts

| Item | Location |
|---|---|
| Generator, scorer, data encoder | `qos_ai_solgen.py`, `qos_ai_solgen_score.py`, `qos_ai_stage3_data.py` |
| Programs and frozen tests | `qos_ai/solgen_20261020/` (`train.jsonl`, `test_*.jsonl`, `build.json`) |
| Answers and scores | `qos_ai/solgen_20261020/answers/`, `qos_ai/probe_sol_20261020/answers/` |
| Checkpoint (SHA-256 `ea62b00586637abd2d16f70f9d45de9dcdd1f9b806ed6dfb25721b9c8f4619b0`) | M1 `experiments/stage3_20261020/m557_sol.pt`; volume `/workspace/qos_ai_stage3/m557_sol/student.pt` |

## Suggested next steps

1. Same data with 30-40% replay split across docs, English and Q&A, to keep the general and domain gains.
2. More arithmetic coverage: explicit element-by-element steps for `map`/`sum`, and paraphrased question templates
   (the hand-written probe shows template overfitting).
3. Grow the operation set (string functions, `Result`/`|>?`, records) with the same compiler-verified generator.
