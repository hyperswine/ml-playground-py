# ML Playground (Python)

Custom QOS language models, verified Sol curricula and evaluation, and small neural-network experiments.
The QOS student is a custom architecture with experimental 168M and 557M checkpoint lineages.

Start with the [project overview](docs/2026-10-28-PROJECT-OVERVIEW.md),
the [dated documentation index](docs/README.md) and the
[running guide](docs/2026-10-21-RUNNING-EXPERIMENTS.md). Historical experiment reports describe
their recorded revision and machine; they are not current deployment or model-promotion claims.

## Project results so far

The project has produced custom 168M and 557M model lineages, compiler-verified
Sol curricula, reproducible evaluation, and a pretrained Qwen reference. The
strongest established reference is **Qwen3.5-2B round 4 on MLX 8-bit**.

| Milestone | Original Sol suite /360 | Older multi-input suite /18 |
| --- | ---: | ---: |
| Custom 557M Stage 3 specialist | 179 | 0 |
| Custom 557M Stage 5 with retention/wording/retrieval mixture | 142 | 8 |
| Qwen 0.8B round 4, CUDA | 324 | 16 |
| Qwen 2B round 4, MLX 8-bit | 341 | 18 |

These are recorded milestones with different recipes and runtimes, not an isolated
parameter-scaling experiment. The original suite now informs development; it is
not an untouched final test.

**What worked:** verified and balanced curricula, broad executable Sol examples,
checking answers by execution, the later steps-first Qwen recipe, pretrained
foundations, targeted source retrieval, and MLX for practical M1 use.

**What did not consistently work:** unverified self-correction, aggressive narrow
fine-tunes, assuming lower loss or more training means better answers, and dense
retrieval as an automatic improvement. Replay and specialization often traded
one skill for another. Repetition blocking improved stopping but did not establish
correctness.

**The main remaining gap is complete answers.** On the harder development set,
round 4 produces working programs for 21/36 code tasks but completes only 4/48
requests. The latest controlled curriculum reaches 24/36 working programs in both
seeds, with 6/48 and 5/48 complete answers. The selected first seed preserves the
original suite (342/360), but the complete-answer gain missed second-seed
confirmation. Round 4 remains the reference and the new final holdout is unused.

The custom 557M remains the research platform for a model of our own; Qwen provides
a strong comparison for curriculum and evaluation work. See the
[whole-project overview](docs/2026-10-28-PROJECT-OVERVIEW.md) for source-linked
results, failures and limitations, and the
[latest controlled experiment](docs/2026-10-28-QWEN-2B-CONTROLLED.md) for the full comparison.

## Qwen internals and the current bottleneck

The inspected Qwen3.5-2B is a dense hybrid: 18 Gated DeltaNet blocks and six full
attention blocks, with RMSNorm, gated MLPs and rotary positions. Our recent M1
adapters reached both mixer types and the MLPs, but changed only the last four
blocks using 1.40M adapter parameters and small partial passes through the data.
The evidence points first to curriculum, complete-answer contracts, adaptation
coverage and response-mode handling; it does not isolate an architecture defect
or establish that parameter count is the limiting factor.

See the [source-level internals audit and bounded inference probe](docs/2026-10-28-QWEN-INTERNALS.md)
for the architectural comparison, actual adapter coverage, thinking-mode findings
and experiments that could distinguish these causes.

## Running the project

```sh
uv sync --locked
uv run --locked qos list
uv run --locked qos model --help
uv run --locked qos stage4_eval --help
uv run --locked qos test
```

For executable Sol checks, supply the intended compiler explicitly:

```sh
QOS_FPR=/Users/jasonqin/Documents/GitHub/fprisc/fpr uv run --locked qos test
```

```text
src/ml_playground/
  model/          QOS architecture, tokenizer interface and main model CLI
  data/           split manifests, verified curricula and generators
  training/       fine-tuning, pretraining, checkpoint resume and scaling
  evaluation/     benchmarks, comparisons, executable grading and loss probes
  workflows/      experiment orchestration and recovery workflows
  cloud/          RunPod artifact/recovery helpers
  examples/       Qwen inference, tiny transformer and filename classifier
scripts/runpod/   shell launch, setup, transfer and staged job scripts
tests/            existing QOS regression suite and repository-layout checks
configs/          architecture configuration and launcher path catalog
docs/             dated reports, runbooks and chronological index
docs/data/        unchanged benchmark/evaluation evidence and migration receipts
artifacts/        local checkpoints, prepared datasets and frozen job bundles
.runtime/         generated, ignored compatibility workspace
```

Use the `qos` launcher instead of executing categorized source files directly. It
provides the old flat filenames and data paths through local links, preserving
existing imports, subprocess commands and source-hash checks. Source and data are
stored once in their categorized locations. The workspace contains no credential links.

Personal environment settings belong in ignored `.env`; optional loading is explicit:
`uv run --env-file .env --locked qos model ...`. The existing RunPod key file is in
ignored `.secrets/runpod_api_key`. Neither is loaded or displayed by `qos list`,
`qos paths`, `qos test` or the workspace builder. See `.env.example` for non-secret settings.

The M1 experiment checkout remains `/Users/jq/qos_ai_eval`, reached as `jq` through
Tailscale. Moving this local repository does not relocate remote experiments or
authorize paid cloud jobs. Retain RunPod volume `xewpn3z8za` unless the user explicitly
changes that instruction.
