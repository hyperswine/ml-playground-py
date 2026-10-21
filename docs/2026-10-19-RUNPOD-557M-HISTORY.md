# 557M RunPod training preparation

Date: 2026-10-19 (Australia/Sydney). Status: **training complete; both compute pods stopped; backup and M1 grading pending**.

The user authorized trying a RunPod training run with a stated US$10.45 balance. The planned GPU is a Secure Cloud RTX 4090, US$0.89/hour from the live catalog. The eight-hour GPU ceiling, storage and a small CPU backup worker leave a reserve within that budget. No pretrained model weights or teacher embeddings are part of the new training lineage.

## Work ready locally

- [Public data preparation](../src/ml_playground/data/qos_ai_runpod_data.py): immutable FineWeb-Edu and CodeParrot dataset revisions, an 80/20 English/Python training mix, per-file permissive Python-license filtering, whole-host/repository validation groups, exact-text deduplication, literal benchmark-question screening and fixed-vocabulary coverage checks. The public corpus is not semantically decontaminated.
- [CUDA training](../src/ml_playground/training/qos_ai_runpod_train.py): disposable batch-size profiling, a token target frozen from measured speed and remaining time, bf16 full-parameter AdamW pretraining, finite-gradient checks, full optimizer checkpoints on a durable volume, verified answer-only SFT, and raw generations on the existing development benchmarks. Reserved test prompts are excluded from model inference.
- [Bounded job wrapper](../src/ml_playground/cloud/qos_ai_runpod_job.py) and [managed-MCP supervisor](../artifacts/runpod/557m/controller.js): setup timeout, GPU deadline, stop/read-back, CPU-only artifact transfer after GPU release, hash checks before cleanup. No API credential is injected into the pod. The supervisor must be rearmed with newly created resource IDs when the upload is approved.
- 2,784 M1-revalidated examples, 97,693 answer tokens, no unknown answer tokens, maximum encoded example length 192.
- Exact tokenizer parity and known token IDs on all 372 prompt encodings (132 original benchmark rows, 96 coding rows, 144 generalization rows). Only the original 90 development rows would receive model inference.
- Twelve focused tests pass. The [test log](data/qos_ai_benchmark/evidence/runpod_557m_20261019/tests.log) and [upload manifest](data/qos_ai_benchmark/evidence/runpod_557m_20261019/upload_review.json) are saved locally.

The allowlisted archive is `/private/tmp/qos_ai_557m_bundle_20261019.tar.gz`, 4,096,839 bytes. Its SHA-256 is `ac229705e7c518884c5f7ea972f8fa2c7e97ffb94d8a3c38ef78e9a8ca974ff6`. It contains 20 payload files plus its manifest: model/training code, tokenizer/vocabulary mapping, encoded verified training examples, fixed benchmarks and tests. Credential files and pretrained weights are excluded. Individual payload identities are recorded in the upload manifest.

## Provisioning proof and cleanup

One pod, `lqkftov98jyrha`, and one empty 40 GB network volume, `n30zs2a4mg`, were created in `EU-RO-1`. An SSH request verified an actual RTX 4090 with 24,564 MiB, driver 570.211.01, PyTorch 2.8.0+cu128, a successful CUDA tensor operation, and the network mount. This is hardware/environment proof only.

The attempted bundle upload was rejected **before execution** by automatic approval review. It stated that sending potentially sensitive custom code, vocabulary, verified examples and benchmarks to a third-party RunPod pod needs explicit approval. This attempt transferred no payload and launched no training. The user subsequently explicitly approved the reviewed payload, resolving this block. This approval requirement came from automatic approval review, not the RunPod skill.

The GPU was stopped immediately and its `EXITED` state verified. Both temporary resources were then removed because the volume contained no uploaded data. Fresh pod and volume lists are empty. There is no ongoing compute or storage charge from this preparation. The billing read had not yet posted records, so it must not be interpreted as proof of zero startup cost; a short provisioning/CUDA-check charge may appear later.

## Resume after approval

Re-read price, stock and storage capacity. Create fresh resources, arm the supervisor using their new IDs and creation time, and upload the exact reviewed archive. Install the pinned tokenizer dependency and a recorded PyArrow/Hugging Face environment into the official PyTorch image without replacing its CUDA PyTorch. Verify all bundle hashes and run the focused tests on the pod before launching the bounded job. Confirm a saved training checkpoint and measured loss before claiming training success. Final program grading belongs on the M1 with the frozen Sol compiler and source snapshot; generated code is not trusted merely because a lexical rubric matches.

No default checkpoint was changed, and no commit or push was performed.

## Approved GPU run

The user explicitly approved the payload. The exact reviewed archive was uploaded to fresh pod `xt9r2a1n5xij1r` and 40 GB network volume `xewpn3z8za`, both in EU-RO-1. The current catalog and pod read-back show US$0.89/hour. GPU creation was 2026-10-18 21:09:21 UTC; the hard GPU stop is 2026-10-19 05:09:21 UTC (16:09 Sydney). Managed MCP supervisor cell `49` is armed with these identities.

Fresh SSH proof: RTX 4090, 24,564 MiB, driver 595.91.07, PyTorch 2.8.0+cu128, successful CUDA tensor operation, network mount present. Dependencies installed: tokenizers 0.23.2, huggingface_hub 1.31.0, pyarrow 26.0.0. All twelve focused tests passed on the pod. The detached bounded job has been launched with the original GPU creation timestamp. Training success still requires measured updates and a saved checkpoint.

The initial preparation process exposed synchronous per-document writes to the network volume. A 512-document probe measured 6.239 seconds with NumPy `tofile` and 0.0195 seconds with buffered byte writes. The writer now buffers 8 MiB, with byte equivalence verified and all twelve tests re-passed on the pod. The original approved archive is unchanged; this single-file follow-up patch and its SHA-256 are recorded in `evidence/runpod_557m_20261019/io_fix.json`. Partial preparation was archived before restarting, with no model updates and no deadline reset. At 21:16 UTC the replacement process had prepared over 67 million English training tokens.

The supervisor also backs up the accepted corpus ledger, pinned manifest, runtime environment and tests. After verified weight backup and cloud cleanup, it runs `qos_ai_runpod_finalize.py` to grade saved code generations on the M1 using freshly frozen compiler, parser and benchmark identities.

Python preparation exposed a vocabulary-coverage issue: in 512 sampled source rows, most permitted documents exceeded the 1 percent unknown-token threshold. Sample median unknown coverage was 4.68 percent. The Python encoder now splits uncovered Qwen BPE tokens into existing student vocabulary pieces with exactly matching byte-level spelling; it does not add embedding rows or relax the unknown-token filter. Regression coverage includes non-greedy segmentation and rejection of unavailable bytes. Both changed payload files are hashed in the updated bundle manifest. Preparation was restarted before any model updates, preserving the original GPU creation time and hard deadline. The original English corpus is archived for identity checks.

A disposable full-architecture CUDA profile ran while the corrected corpus prepared. Batch 4 measured 15,771 tokens/second at 11,872,204,800 peak allocated bytes; batch 8 measured 25,126 tokens/second at 15,678,175,232 bytes. Both used bf16, fused AdamW and 512-token sequences. These are discarded profiling updates on archived token streams, not accepted training or quality evidence. The main job profiles again on its finalized corpus before freezing the schedule.

## Verified main training launch

The final corpus contains 400,000,397 English training tokens and 100,002,642 Python training tokens, plus 1,000,623 and 1,000,658 validation tokens respectively. All corpus identities are frozen in dataset hash `84e61e87b22ec26d80c81bcaf8891570a39f921ce35a0e49d58ea821541887cc`. Python expansion used 4,406,612 vocabulary-piece fallbacks in training; 2,090 residual unknown tokens remain in the entire 100-million-token Python training stream. Both replayed English stream hashes match the originals.

The main profile selected batch 8, accumulation 2, effective 8,192 tokens per update. Its 25,400 tokens/second measurement froze a target of **459,243,520 tokens / 56,060 updates**, with the original pretraining deadline unchanged. The first native optimizer checkpoint is 6,678,579,187 bytes and was successfully loaded on CPU: step 20, all 556,526,080 unique parameters, tied embedding/head storage, optimizer moments, matching schedule hash and twenty finite loss/gradient records. The main job subsequently passed update 100. This verifies working training, not answer quality.

All thirteen focused tests pass. The active managed-MCP supervisor remains cell `49`, with GPU cutoff **2026-10-19 05:09:21 UTC / 16:09 Sydney**. The detached pipeline will follow pretraining with verified SFT and the frozen development/coding/generalization prompts. The supervisor will release the GPU, copy the final weights and optimizer state using a budgeted CPU pod, verify hashes, remove its owned resources, and grade generated answers on the M1. Those later stages are not yet completed. The original/default and 168M checkpoints are preserved.

## Completion and supervisor recovery

The on-pod job finished pretraining, verified SFT and all six generation tracks at 2026-10-19 02:36:12 UTC (13:36 Sydney). It processed all 459,243,520 planned pretraining tokens, with no budget stop. The final native checkpoint hashes are `f7c9d03107c7532608fc6dca81120a62753c2e06e0f18f22193e9a5b1579fdc7` before SFT and `430915b08ee981d6efa32bca2aafdeeb4d33d4b18c093b82c16373ea0b7b0a63` after SFT.

The status check at approximately 03:57 UTC found that the previous background supervisor cell no longer existed. The GPU remained allocated despite job completion. Its stopped state was verified at approximately 03:58 UTC (14:58 Sydney). This created approximately 82 minutes of idle allocation. The controller now supports resuming artifact backup from a directly verified terminal status and checks readable job status before enforcing its setup timeout on a resumed session. A new controller is completing CPU-only backup, verification, cleanup and M1 grading within this active turn; a background exec cell is not a durable cross-session scheduler.

The recovery CPU pod is `kgeowatenfhc4t`, US$0.06/hour. Backup shell calls and the M1 frozen-input check were rejected before execution because the automatic approval-review service reported its selected model at capacity. This was a review-service failure, not a determination that the authorized transfer was unsafe. The recovery controller stopped further work and retained network volume `xewpn3z8za`. No verified local final checkpoint backup or M1 grading is claimed at this point.

The approval-service capacity failure also rejected the CPU stop and terminate requests before execution. A final native pod list still showed CPU pod `kgeowatenfhc4t` RUNNING at US$0.06/hour and GPU pod `xt9r2a1n5xij1r` EXITED. Pending review attempts have ended. Manual removal of the temporary CPU pod, or retry after service recovery, is required to stop its compute charge. Network volume `xewpn3z8za` must be retained because it holds the only verified final checkpoints. Posted main-pod billing at the status check totaled US$5.206635 through 03:00 UTC; later usage and separate resources were not yet included.

On the user-requested retry, the CPU stop action succeeded after an approval-review timeout and one retry. Fresh reads confirmed both GPU `xt9r2a1n5xij1r` and CPU `kgeowatenfhc4t` EXITED. No manual user action is needed to stop compute now. Checkpoint volume `xewpn3z8za` is retained. The controller now checks native stop errors immediately and can reuse an existing backup pod rather than creating another one. Its JavaScript compiles; it is explicitly documented as requiring an active client connection.
