# Next steps 1-6: execution, compiler-verified self-training, wording, Stage 4 evaluation, retrieval

Created: 2026-10-21T16:20:00+11:00 (Australia/Sydney). Kind: progress record. No paid compute was used for this
page; all work ran locally or on the M1 (`/Users/jq/qos_ai_eval`). Compiler: local `fprisc/fpr`, SHA-256
`9987da491a791af853300f38ad3e73c52a243dbfd1422379a8dbe437369b776e`.

## Step 4: Stage 4 evaluation (complete except one run)

Original benchmarks, raw decoding ([scores](data/qos_ai_benchmark/stage4_20261021/benchmarks/)):

| Model | Dev lexical /90 | Outputs /33 | Working /24 | Repairs /48 | Repair+out /48 | No-change /24 | Wording/Inputs/Combos | Trace /120 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Stage 3 specialist | 24 | 12 | 18 | 36 | 11 | 6 | 17/18/10 | 29 |
| Stage 4 `one_2e5` | 23 | 11 | 17 | 34 | 9 | 0 | 16/13/7 | 22 |
| Stage 4 `one_1e5` | 18 | 11 | 16 | 28 | 8 | 0 | 16/12/5 | 15 |
| Stage 4 `half_2e5` | 17 | 8 | 17 | 25 | 8 | 0 | 13/11/3 | 10 |

Frozen Stage 3 Sol tests (360 programs; `one_1e5` pending):

| Model | Correct /360 | With execution (step 1) | Generated programs that run correctly |
|---|---:|---:|---:|
| Stage 3 specialist | 179 | 352 | 103/111 |
| Stage 4 `one_2e5` | 155 | 350 | 101/111 |
| Stage 4 `half_2e5` | 123 | 341 | 92/111 |

Multi-input suite: `half_2e5` 3/18, `one_2e5` 2/18, `one_1e5` 1/18 (all baselines 0/18). All three pass the
1.25x retention gate. **`one_2e5` is the best all-round candidate**: close to the Stage 3 specialist on Sol programs
while keeping FineWeb English perplexity at 50.8 (Stage 3: 160.7). No variant states "no change needed" (0/24).

## Step 1: execution-augmented answers (`qos execute`)

`src/ml_playground/model/qos_ai_execute.py` runs the answer's Sol code (or the question's program when the answer
has none) and replaces the stated output with the real one. Only side-effect-free code is admitted: pure
`List`/`Str`/`Numeric` calls, local names and `> print`; `use`, process/file/network calls and string interpolation
are refused (`tests/qos_ai/test_qos_ai_execute.py`). Rescoring saved answers with the unchanged Stage 3 scorer
([data](data/qos_ai_benchmark/execute_rescore_20261021.json)):

| Model | Without | With execution | Generation /72 | Repair /39 |
|---|---:|---:|---:|---:|
| Stage 3 specialist | 179 | 352 | 33 → 65 | 27 → 38 |
| 557M 0.9B pilot SFT | 12 | 246 | 0 → 2 | 0 → 4 |
| 168M coding parent | 12 | 259 | 0 → 5 | 0 → 7 |

Output-prediction gains (249/249) come from running the program already in the question and say nothing about the
model. Generation and repair gains (60 → 103 of 111 for Stage 3) are real: its code is right more often than its
stated output.

## Step 3: varied wording (`qos stage5_param`)

`src/ml_playground/data/qos_ai_stage5_param.py` keeps Stage 4's generator, oracle, suite-skeleton exclusion and
runner verification, and varies only the English: several phrasings per operation, noun-phrase and step-by-step
forms ("starting from xs, keep only the elements above limit, then multiply each by k, and finally add them up"),
nested parts parenthesized when compound. Descriptions sharing six consecutive words with a suite description are
rejected (167) as are suite skeletons (159). 4,000 tasks, all verified on six input sets
([build](data/qos_ai_benchmark/stage5_20261021/param/build.json)). Merged into the Stage 4 pools with zero
vocabulary drops: 71,017 verified examples, 2.54M target tokens (`artifacts/qos_ai/stage5_20261021/pools/`).

## Step 2: compiler-verified self-training data (`qos rft`)

`src/ml_playground/workflows/qos_ai_rft.py`: 2,683 fresh tasks (1,183 parameterized, 1,027 generation, 473 repair),
disjoint from all training and test programs, references confirmed by the compiler. Sampling 8 answers per task
from `one_2e5` (temperature 0.8) is queued on the M1 after the last evaluation; `verify` keeps up to two distinct
answers per task whose code runs correctly and whose stated output is right. Training on them needs one short GPU
run (not yet authorized).

## Step 5: retrieval (`qos retrieval`)

BM25 over all 1,820 corpus chunks ([data](data/qos_ai_benchmark/retrieval_20261021.json)). On 500 held-out
dev-partition questions the question's own source file is the top hit 55% of the time, top 3 75%, top 5 81%.
With `one_2e5` and no retrieval training, placing the top chunk before 300 held-out questions lowers answer
perplexity from **12.19 to 10.84**; a random chunk of the correct file gives 11.63
([data](data/qos_ai_benchmark/retrieval_loss_one_2e5_20261021.json)). Next: training examples with retrieved
context, and a 1,024-token context so chunks need less trimming.

## Step 6: more pretraining

Deferred: about $5 per 500M tokens does not fit the remaining budget alongside steps 2, 3 and 5.

## Proposed paid run (awaiting approval)

One RTX 4090 job from the domain parent (or `one_2e5`): Stage 5 pools + verified RFT answers + retrieval-context
examples, mixed 65/14/14/7 as in Stage 4, one pass at 2e-5. About 20-30 minutes of training, roughly $0.60-0.90
including setup, with the same self-stop and M1 pull protocol. Uploads use `ssh -o IPQoS=none`.
