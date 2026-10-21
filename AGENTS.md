# Working in this repository

- Run categorized Python scripts through `uv run --locked qos SCRIPT ...`; use
  `qos list` for names and `qos test` for the regression suite.
- Active source is under `src/ml_playground/`. Update `configs/layout.json` when
  adding or moving modules. `.runtime/workspace` is generated compatibility
  plumbing; edit the source files, not its links.
- Keep frozen job bundles, compiler snapshots, benchmark inputs and experiment
  receipts unchanged. Their hashes and recorded paths are historical provenance.
- Narrative documentation belongs in dated `docs/YYYY-MM-DD-TOPIC.md` pages,
  indexed by `docs/README.md`. Record substantive updates with an ISO-8601 time
  and timezone. Raw measurements belong in `docs/data/`; weights and prepared
  model datasets belong in ignored files under `artifacts/`.
- `.env` and `.secrets/` are local and ignored. Never print credential values or
  include them in bundles, logs, Git changes or training corpora. Parent
  `AGENTS.md` instructions still apply.
- The M1 checkout remains `/Users/jq/qos_ai_eval` through Tailscale as `jq`.
  Verify the intended source/compiler identity before comparing experiments.
- Retain RunPod network volume `xewpn3z8za`. Repository maintenance does not
  authorize starting paid compute, deleting retained artifacts or changing the
  default model.
