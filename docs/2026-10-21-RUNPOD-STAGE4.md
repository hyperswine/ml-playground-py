# Stage 4 runbook

This directory describes a prepared experiment, not a running pod. Read [the preparation report](2026-10-21-STAGE4-PREPARATION.md) and complete its recovery, baseline, storage and budget prerequisites before a paid launch. Keep volume `xewpn3z8za`; do not delete retained artifacts to make these commands fit.

Run commands from `python-playground-UV`. Use `.venv/bin/python` locally, or the corresponding prepared environment on the M1/GPU host. No command below provisions cloud resources. New output directories are required; reuse only a matching interrupted trainer output when explicitly resuming it.

The isolated M1 kit is `/Users/jq/qos_ai_eval/experiments/stage4_20261021/preparation`; from there use
`/Users/jq/qos_ai_eval/.venv/bin/python`. Its already prepared pools are at
`qos_ai_benchmark/stage4_20261021/pools/pools.pt`. The reference compiler is `toolchain/fpr`, separate from
the compiler used for old benchmarks. Rebuilding pools from scratch also requires the historical source
payloads named in `preparation.json`; they are retained in their original locations, not all duplicated in the kit.

## Local preparation

The generated JSONL and frozen diagnostic are checked in. To audit/rebuild the token pools:

```sh
.venv/bin/python qos_ai_stage4_data.py prepare --output /tmp/qos-stage4-pools
QOS_FPR=/Users/jasonqin/Documents/GitHub/fprisc/fpr .venv/bin/python -m unittest discover -p 'test_qos_ai*.py'
```

`prepare` fails on stale verification, partition, vocabulary, unknown tokens or an oversized complete arithmetic answer. It produces `pools.pt`, separate `arithmetic_validation.pt` and `preparation.json`. Preserve the manifest next to ignored tensor files.

## Complete recovery and English replay

The retained volume's expected sources, based on the previous run, are:

- `/workspace/qos_ai_stage2/m557_1b_pretrain/pretrained.pt`
- `/workspace/qos_ai_stage2/m557_1b_pretrain/progress_stage2.pt`
- `/workspace/qos_ai_557m/pretraining_data/documents.jsonl.gz`
- `/workspace/qos_ai_557m/pretraining_data/dataset.json` and `english_train.bin`

Use the existing authenticated managed connection when retrieval is authorized. Read the optimizer's digest from the source, copy it and the other recovery files off the volume, then compare full hashes. Load the checkpoint with the restricted reader to confirm configuration, finite tensors, optimizer moments and recorded step. Confirm that the weight/optimizer pair belongs to the same completed pretraining run. Preserve the copied metadata and record actual backup paths in a new inventory; do not mark unknown digests verified.

Run the read-only backup checker **on the host holding those paths**:

```sh
python qos_ai_stage4_recovery.py --inventory /path/to/completed_inventory.json \
  --checkpoint-directory /path/to/proposed/output --output /path/to/new_recovery_check.json
```

It exits 2 when any required backup is absent or mismatched. This is a byte/hash and space check, not a replacement for structural loading or proving that backups are independent of the volume.

The original dataset manifest is archived at `qos_ai_benchmark/evidence/runpod_557m_20261019/launch_snapshot/dataset.json`. Its file SHA-256 is `7451c6fabb883a9c87a893b2b84a80c536132c376698c0ba7cf352db894886b5`; the English training stream's SHA-256 is `520068a12af869ca405191b951fec23f6d70bb39ef50e8502f4f30baf229d2e9`.

After copying the verified manifest and original stream into one directory:

```sh
python qos_ai_stage4_data.py mix --pools /path/to/pools.pt \
  --pretraining-data /path/to/original_pretraining_data \
  --pretraining-manifest-sha256 7451c6fabb883a9c87a893b2b84a80c536132c376698c0ba7cf352db894886b5 \
  --output /path/to/new_mixture
```

Keep `mixture.json` with `sft.pt`. The manifest records actual token shares, repeated pool cycles, data hash and file hash. All variants consume this identical mixture with no extra replay sampling.

## Baseline and candidate evaluation

Run generation on the M1 for each chosen checkpoint, using the frozen new suite. Generation does not execute model code:

```sh
python qos_ai_stage4_eval.py generate \
  --suite qos_ai_benchmark/stage4_20261021/multi_input.json \
  --checkpoint /path/to/checkpoint.pt --device mps --output /path/to/new_answers.json
```

Score on a host with the exact pinned compiler (the current local compiler used to build this suite):

```sh
python qos_ai_stage4_eval.py score --fpr /path/to/pinned/fpr \
  --suite qos_ai_benchmark/stage4_20261021/multi_input.json \
  --answers /path/to/new_answers.json --output /path/to/new_scores.json
```

Keep the generation receipt, including checkpoint hash, with the answer file and score. Run the original `qos_ai_noguard.py` protocol separately with the old frozen compiler and benchmark bundle. Do not replace its compiler or revise old scores with the new suite. Retain the fixed English, documentation and Q&A loss probes; use arithmetic validation separately for selection and audit final numbers plus all intermediate values. The original reserved final test stays closed.

## Future training, once prerequisites are satisfied

Use one of `half_2e5`, `one_2e5` or `one_1e5`, each in its own output directory. Set `--data-sha256` to the exact `sft_file_sha256` from the independently retained mixture receipt. The common parent hash is enforced by the runner.

```sh
python qos_ai_stage4_train.py --load /path/to/m557_1b_domain.pt \
  --data /path/to/new_mixture/sft.pt --data-sha256 ACTUAL_SFT_FILE_SHA256 \
  --variant half_2e5 --output /path/to/new_run_half_2e5 \
  --deadline-unix AGREED_ABSOLUTE_UNIX_DEADLINE --device cuda
```

The default batch is 8 × 4 accumulation. It saves full progress every 100 updates, at completion, or five minutes before the deadline. Resume by repeating the same data, parent, variant, batch settings and output with a new deadline. Changed source or training settings are rejected. A failed/partial write leaves the prior complete checkpoint in place; inspect temporary files and available space before retrying.

The runner publishes weights and training history, not a promotion decision. Run the frozen probes and generation comparisons afterward. A separate controller must stop compute on success, failure and deadline and verify resource state; no credential handling or new cloud watchdog is embedded in this preparation. Full-size GPU/MPS recovery and save timing still require validation in the eventual bounded run.
