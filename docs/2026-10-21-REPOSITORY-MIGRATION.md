# Repository migration and source layout

Created: 2026-10-21T13:35:18+11:00 (Australia/Sydney). Kind: implementation and migration record.
Source checkout: `/Users/jasonqin/Documents/GitHub/Playground/python-playground-UV`.
Destination: `/Users/jasonqin/Documents/GitHub/ml-playground-py`.
Original source commit: `51672c75485be7ee6b67842f2503e399c0fe7648`; uncommitted Stage 4 results were included.

## What moved

The migration inventories **1,099 files / 20,726,878,588 bytes**, plus two opaque
credential files. It moves the active QOS LLM code, toy neural-network examples,
Qwen inference, test suite, model/data artifacts, cloud scripts, sample gallery,
benchmark evidence and narrative reports. ToyAV and the general playground's
Python environment remain in the original repository. No remote checkout or
RunPod resource was modified.

The root [README](../README.md) describes the new categories. Source remains in
its original modules under `src/ml_playground/model`, `data`, `training`,
`evaluation`, `workflows`, `cloud` and `examples`. Shell scripts are under
`scripts/runpod`; architecture and launcher configuration are under `configs`.
Twenty-three existing narrative documents were moved into date-prefixed pages
under `docs/`, with 63 relative links rebased. The chronological index is
[docs/README.md](README.md). Dates come from an existing filename, a dated sample
directory, or the first Git addition where no explicit filename date existed.
Exact original creation times are not invented; this record timestamps the move.

Frozen compiler/source snapshots, answers, split manifests and result JSON retain
their recorded bytes and historical paths. Large model/tensor files remain local
and ignored. New Stage 4 run receipts and candidate scores moved along with the
older records, even though they had not yet been committed in the source checkout.

## Compatibility and environment

`uv run --locked qos SCRIPT ...` supplies the original flat module and data paths
through links in ignored `.runtime/workspace`. The categorized files are the only
source copies. This preserves existing imports, subprocess script names and most
source hashes without mixing frozen job bundles with live source.

Only two cloud helper modules change their root lookup to retain the launcher's
path rather than resolving a link back into `src/cloud`. The host upload script
now archives the selected files' contents instead of transferring local links.
Historical source is preserved separately in
[original-source.tar.gz](data/migration/2026-10-21/original-source.tar.gz).
The archive includes the old project/lock configuration and excludes credentials.

Python is pinned to 3.13. The new project has its own ML environment and lockfile,
including the exact prior Transformers revision
`80a79d1bef2bacfa10b1b543669eebffa3f0ec36`. Unrelated hardware/web application
requirements stay with the original general playground. See the
[running guide](2026-10-21-RUNNING-EXPERIMENTS.md).

The user authorized moving `.env` and the RunPod key unchanged without reading
values. Their filesystem identity and size were preserved, permissions are 0600,
and `.secrets` is 0700. They are ignored and absent from the compatibility
workspace, source archive and source catalog. No environment values were printed
or rewritten.

## Verification

Before the move, the original suite ran 122 tests with one sandbox-related MPS
skip. Against the moved layout, the same environment ran **127 tests successfully**,
including five new tests of path resolution, source aliases, credential exclusion,
non-overwriting workspace creation and frozen benchmark identity. This run had
host access, so the MPS test also ran.

Full-file hashing verified every moved file: **1,081 files / 20,726,708,939 bytes
remain byte-identical**. The 18 adapted files are the declared path/transfer edits
and Markdown link changes; the detailed receipt lists old/new digests. Every
model checkpoint, tensor, frozen bundle and raw evidence file is unchanged.
Final validation completed at 2026-10-21T13:39:07+11:00. The new standalone environment also
passed **127 tests with no skips**. The launcher and model/data/example help
commands started successfully; the lockfile check passed. All 69 shared locked
packages retain their prior versions. The documentation audit checked 104 local
links with no broken links or missing catalog targets. Both credential files are
ignored, untracked and mode 0600. No unignored file exceeds 50 MB.
See [final validation](data/migration/2026-10-21/validation.json) and the
[standalone test log](data/migration/2026-10-21/standalone-environment-tests.log).
The original folder now contains a README pointing here; 83 obsolete bytecode
cache files for moved modules were removed. Unrelated caches and ToyAV remain.

- [Original paths, destinations and hashes](data/migration/2026-10-21/files.json)
- [File verification and declared adaptations](data/migration/2026-10-21/file-verification.json)
- [Relative-link changes](data/migration/2026-10-21/link_changes.json)
- [Original test run](data/migration/2026-10-21/before-tests.log)
- [Moved-layout test run](data/migration/2026-10-21/moved-layout-tests.log)

Run `uv run --locked python scripts/check_layout.py` to check catalog targets and
narrative links. Add `--full` to rehash the original migrated artifacts. The latter
reads about 21 GB and compares with migration-time evidence; it is not a new model
quality evaluation. No paid training or cloud deployment was tested by this move.

## Publication follow-up

On 2026-10-21, the user requested committing and pushing the migration to `main`
in both repositories. The original repository's `python-playground-UV/README.md`
now links to this repository and its documentation on GitHub. Six previously
versioned pod logs/data files are explicitly retained by the new ignore rules;
model weights and credentials remain ignored. The migration validation receipt
records the pre-publication snapshot. The old repository's Git history is retained.
An unrelated edit to `high-level-stuff/screw-cinematic/screw_cinematic.py` is
excluded from the migration commit.
