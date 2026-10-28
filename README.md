# ML Playground (Python)

Custom QOS language models, verified Sol curricula and evaluation, and small neural-network experiments.
The QOS student is a custom architecture with experimental 168M and 557M checkpoint lineages.

Start with the [project overview](docs/2026-10-28-PROJECT-OVERVIEW.md),
the [dated documentation index](docs/README.md) and the
[running guide](docs/2026-10-21-RUNNING-EXPERIMENTS.md). Historical experiment reports describe
their recorded revision and machine; they are not current deployment or model-promotion claims.

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
