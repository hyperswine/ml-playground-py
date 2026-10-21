# Running experiments from the organized repository

Updated: 2026-10-21T13:39:07+11:00 (Australia/Sydney). Kind: living runbook.

## Environment

Run `uv sync --locked` from `ml-playground-py` to create its own environment. The
lockfile preserves the dependency resolutions used before the move where applicable,
including the resolved Transformers Git revision. The original general playground
retains its own environment and dependencies for unrelated work.

The default dependencies cover local models, evaluation and CPU/MPS tests. RunPod
jobs may additionally require the image's CUDA stack, PyArrow, vLLM or optional
kernels, as recorded by their frozen job/setup scripts. Those GPU installations
were not executed by the repository migration.

`.env` and `.secrets/runpod_api_key` are local, ignored and owner-readable only.
They were moved as opaque files without reading their values. The launcher does
not automatically load them. Optional `.env` loading is performed by
`uv run --env-file .env --locked qos ...`; never put credentials in command
arguments, versioned files or experiment logs. Non-secret choices are illustrated
in the root `.env.example`.

## Commands

| Purpose | Command |
| --- | --- |
| List scripts by category | `uv run --locked qos list` |
| Show repository locations | `uv run --locked qos paths` |
| Original model CLI | `uv run --locked qos model --help` |
| Data preparation | `uv run --locked qos stage4_data --help` |
| New multi-input diagnostic | `uv run --locked qos stage4_eval --help` |
| Controlled fine-tuning | `uv run --locked qos stage4_train --help` |
| Local Qwen example | `uv run --locked qos qwen08 --help` |
| Tiny transformer example | `uv run --locked qos neural-lite-transformer --help` |
| Regression and layout tests | `uv run --locked qos test` |
| One focused test file | `uv run --locked qos test --pattern test_qos_ai_stage4.py` |

Long names such as `qos_ai_stage4_eval` also work. Script argument interfaces remain
the same. For example, `python qos_ai_stage4_eval.py score ...` becomes
`uv run --locked qos stage4_eval score ...`.

The launcher runs in `.runtime/workspace`, a generated collection of local links.
Old paths such as `qos_ai/...`, `qos_ai_benchmark/...` and `runpod/557m/...` resolve
to the moved artifacts. New paths beginning with `artifacts/`, `docs/`, `configs/`,
`scripts/` and `src/` are also available there. Use absolute paths for inputs
outside this repository. `QOS_AI_DIR=qos_ai` keeps the existing model location;
an absolute `QOS_AI_DIR` can isolate another experiment.

Source files retain their historical basenames under their new categories. Their
flat imports are intentionally supported by the launcher. Do not edit generated
workspace links or use a direct `python src/.../qos_ai_*.py` invocation without the
compatibility environment. `configs/layout.json` is the explicit source-to-alias
catalog; update it when adding or renaming scripts.

## Evidence and remote workflows

Frozen job bundles and embedded source snapshots are retained byte-for-byte.
They are experimental inputs, not copies to refactor alongside active source.
Recorded absolute paths, source hashes and dates remain original provenance.

Use `QOS_FPR=/absolute/path/to/fpr uv run --locked qos test` for executable checks.
The Stage 4 suite requires the compiler digest recorded in its frozen JSON. Earlier
benchmarks may require another compiler; never replace one silently to make a score
run. The original reserved final test remains separate from development diagnostics.

The local move does not alter the M1 checkout `/Users/jq/qos_ai_eval` or its staged
experiments. Send future changes as an explicit new bundle, with source and artifact
hashes, before remote execution. Existing RunPod jobs expect their documented pod
filesystem and must not be run on the Mac just because they are now under `scripts/`.
For intentionally invoked host transfer scripts, use e.g.
`uv run --locked qos shell runpod/pull.sh HOST PORT`; this performs the action,
so supply a known owned destination and follow the existing authorization scope.

## Documentation

Add new narrative pages as `docs/YYYY-MM-DD-TOPIC.md`, with an ISO-8601 timestamp
and timezone for each substantive update. Keep the first-written date in the name
and add the page to `docs/README.md`. Store raw measurements in `docs/data/` and
large local model/dataset files in `artifacts/`. Distinguish preparation, executed
checks, training results and promotion decisions in every report.
