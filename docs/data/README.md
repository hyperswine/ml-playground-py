# Raw data and evidence

- `qos_ai_benchmark/`: canonical benchmark inputs, frozen split manifests, teaching sets, dated stage results and historical source snapshots.
- `samples/`: original generated sample JSON/HTML and images. Narrative gallery pages are indexed in `docs/README.md`.
- `reports/`: outputs of future legacy workflow runners.
- `migration/2026-10-21/`: original-file SHA-256 inventory, source archive, link changes and validation receipts.

Original JSON/JSONL, compiler snapshots, answers and measurements retain their bytes and
recorded paths. Those paths describe the original experiment, not a new execution in
this repository. Do not rewrite manifests to make old hashes agree with new code.
Small links at legacy report locations point to the dated narrative pages.
Model checkpoints and large tensor files live under `artifacts/` and remain Git-ignored.
