# Qwen3.5-4B experiment plan: US$20 maximum

Updated: 2026-10-29T16:41:50+11:00 (Australia/Sydney).
Status: planned; training has not started. This page authorizes no deployment by
itself. The current request is to document and publish the plan.

## Goal and decision

Determine whether a full text-model fine-tune of `Qwen/Qwen3.5-4B` gives a useful
improvement over the current **Qwen3.5-2B round-4 MLX q8** reference, especially on
complete debugging/code answers, while remaining practical on the 16 GB M1.

The primary experiment is one pass through the audited round-4 curriculum, seed
7, using the existing full-SFT recipe. Keep curriculum changes separate from this
first size comparison. A second seed is conditional on measured remaining time
and money. The experiment can finish below US$20; spending the balance is not a
goal. Training from random weights, 9B models and vision training are outside scope.

Sources: [2B training report](2026-10-27-QWEN-2B.md),
[round-4 curriculum](2026-10-26-QWEN-ROUND4.md),
[controlled curriculum results](2026-10-28-QWEN-2B-CONTROLLED.md),
[internals audit](2026-10-28-QWEN-INTERNALS.md), and
[M1 runtime measurements](2026-10-27-MAC-INFERENCE.md).

## Budget and hardware

All figures are USD. The 2026-10-29 RunPod MCP reads returned:

| Secure Cloud GPU | VRAM | Hourly compute price | Observed capacity |
| --- | ---: | ---: | --- |
| RTX PRO 6000 Blackwell Server Edition | 96 GB | $2.49 | High globally; medium in EU-RO-1 |
| L40S | 48 GB | $1.09 | Low globally |
| A40 | 48 GB | $0.59 | High globally |

The retained volume `xewpn3z8za` is **40 GB, STANDARD, EU-RO-1**.
Choose one on-demand RTX PRO 6000 in that data center so the existing volume can
mount. Re-read price, stock, volume state and free space before execution.
The snapshot is evidence of availability when planning, not a reservation:
[raw planning reads](data/qos_ai_benchmark/qwen4b_plan_20261029/runpod_snapshot.json).

| Allocation | Maximum time at $2.49/hour | Budget ceiling |
| --- | ---: | ---: |
| Provisioning, setup and training feasibility check | 30 minutes | $1.25 |
| Primary one-pass training | 3 hours 30 minutes | $8.75 |
| Artifact export, backup and shutdown | 30 minutes | $1.25 |
| Optional independent repeat or bounded recovery | 1 hour 30 minutes | $3.75 |
| Disk/storage charges and applicable tax allowance | Measured separately | $2.50 |
| Unspent safety reserve for delayed shutdown/billing | Never assigned to extra training | $2.50 |
| **Total maximum** | **6 aggregate GPU hours** | **$20.00** |

Six hours of compute at the observed rate is $14.94; allocations round upward
to $15.00. Time includes provisioning/downloads, failed attempts, transfers and
idle waits while a GPU is allocated. The main run's first training steps occur
during the feasibility check and count toward its one pass.

Before each phase, require conservative accrued cost plus forecast completion,
artifact transfer, taxes/storage and the safety reserve to fit under $20.
Reduce the runtime ceiling if the contracted price or ancillary costs increase.
Do not start a retry or a second pod with a fresh independent $20 allowance.
A failure to fit the full planned pass produces a bounded partial-run report,
not permission to exceed the cap.

[RunPod's pricing documentation](https://docs.runpod.io/pods/pricing) lists
standard network storage below 1 TB at $0.07/GB/month: the existing 40 GB volume
is about $2.80/month before tax and keeps billing after GPU shutdown.
Count its prorated use during this experiment in the ledger; its indefinite
pre-existing retention is separate from this one-off experiment. Preserve it.
No new volume, volume expansion, serverless endpoint or standing GPU is planned.

The 2B reference used 47 GB of approximately 98 GB reported GPU memory and trained
in 46 minutes at 4,720 input tokens/s. A 4B pass taking roughly 1.5-3 hours is a
planning estimate, not a throughput measurement or completion guarantee. Full SFT
memory includes fp32 weights/gradients/Adam states plus activations; fitting 4B
in 96 GB must be established by the pilot. The cheaper 48 GB cards are possible
adapter-training alternatives, not drop-in replacements for this full-SFT recipe.

## Phase 1: freeze inputs and establish the local baseline

Do this on the M1 before renting compute.

1. Verify host identity and use `jq` over Tailscale, experiment root
   `/Users/jq/qos_ai_eval/experiments/qwen4b_round4_20261029`.
   Use a new directory if that name already contains results. Record repository
   revision, frozen compiler identity, dependency versions and available disk/RAM.
2. Download the public instruction checkpoint `Qwen/Qwen3.5-4B`, pin its immutable
   revision and file hashes, and save its license, config, tokenizer and template
   identities. Do not substitute the distinct Base checkpoint. Check the actual
   32-block hybrid implementation and parameter count before training.
3. Reconstruct the round-4 mixture (historically 92,314 examples), preserving row
   content, seed/order, answer masks and steps-first formatting. Hash the resulting
   rows and record source/target token totals, category shares and sequence lengths.
   Re-execute Sol targets and check supplied bindings, boundary inputs, repairs,
   no-change decisions and stated outputs with the pinned compiler/oracle.
4. Check the whole training mixture against development and final identifiers,
   normalized questions/programs and known composition-family boundaries.
   Preserve the existing split fixtures; use final signatures for exclusion only,
   without inspecting final answers for training. Reject overlap or incorrectly
   verified targets before any paid work. Record any exclusions. If the mixture
   changes, describe the result as a new recipe comparison rather than an isolated
   parameter-count effect; do not rewrite historical data or hashes.
5. Freeze a manifest of all training, validation, development and retention inputs,
   scorers, decoding settings and selection gates. The 528-row controlled curriculum
   is not appended to this run: that would change both size and data simultaneously.
6. Convert the untouched 4B checkpoint to text-only MLX q8 and generate its development
   baseline. Reuse the official template with **`enable_thinking=False` explicitly**,
   greedy decoding, the existing per-question token caps and stop rules.
   The public 4B card defaults to thinking, unlike the inspected small-model default;
   do not let that silently change this comparison. Record template parity.
7. Score the 48 coding items plus eight documentation items separately, the original
   740-prompt development/regression suite, and the strict calculation diagnostics.
   Reuse the current 2B's saved raw answers only after verifying prompt/scorer/compiler
   parity; otherwise regenerate its answers. Keep raw model results separate from
   execution-assisted answers and lexical/documentation proxies.

The final 24-item holdout remains sealed in this experiment. The older suites
have repeatedly informed development and must not be presented as untouched tests.
The upstream 4B baseline may struggle with Sol syntax; measure both its behavior
and the tuned model instead of assuming generic upstream benchmark gains transfer.

## Phase 2: prepare bounded execution before provisioning

The current [GPU job script](../scripts/runpod/qwen_sft/job.sh) is a historical
runner, not a ready-to-run budget guarantee. It saves weights only after training
and has an older self-stop path. The [trainer](../src/ml_playground/training/qos_ai_qwen_sft.py)
currently lacks the checkpoint/resume and deadline controls needed below.
Implement and validate these prerequisites as a separate execution change:

- Pin the image, framework, optimized DeltaNet kernels and model revision. Check
  actual imports and CUDA execution; a CPU/reference-kernel fallback is a failed
  feasibility check. Set a compatible host CUDA floor of at least 12.8; the
  planning read found capacity on 13.2. Do not blindly upgrade the old recipe.
- Add bounded telemetry, finite-loss/gradient checks, a monotonic deadline, periodic
  recoverable checkpoints, and receipts for actual input/answer tokens and updates.
  Verify prompt and padding contribute no loss, EOS does contribute, and no answer
  is silently truncated. Do not reuse the old MLX padding-mask bug.
- Continue pilot steps in the same optimizer/scheduler run toward the frozen total
  update count; do not train the pilot prefix twice. Resume must restore optimizer,
  scheduler, data-order position and RNG state. A weights-only restart is a new
  condition, not an exact resume.
- Reserve container disk for model cache and optimizer checkpoints, plus enough
  durable space for final weights and receipts. The 40 GB volume's configured size
  is not its free space. Require a tested persistence route to that volume and the
  M1; never prune old checkpoints or expand the volume just to make this run fit.
- Build a per-experiment ledger from contracted rates and aggregate billable time,
  and reconcile it with billing reads. Unrelated account charges must be identified
  separately, not mistaken for training cost or changed by the controller.
- Arm an independent supervisor on the M1 and a pod-local stop path. Persist the
  experiment pod ID and deadline before setup; read back both controls. Test the
  supervisor's deadline/exit handling without paid compute, then make the allocated
  pod's self-stop capability the first live check. Failure means immediate cleanup.
  Secret handling must follow the repository's credential policy and must not put
  credential values in prompts, logs, Git or training bundles.
- Terminate only this experiment's pod on completion/failure/deadline after its
  artifacts are durable. Use API read-back to confirm it is gone and the retained
  volume still exists. A stopped pod may retain billable disks. Killing Python or
  relying on a printed “training finished” message does not stop rental charges.
  Unsupported CLI timeout flags and an assumed account-balance cutoff are not guards.

The external supervisor checks at least once per minute, calculates the remaining
export/shutdown allowance, and initiates cleanup before the aggregate six-hour
limit or cost limit, whichever is earlier. A control-plane outage can defeat
software shutdown; the reserve reduces exposure but is not a provider-enforced
spending cap. Do not launch unless both stop routes are operational. All setup
failures and replacement pods remain inside the same ledger and wall-time budget.

## Phase 3: feasibility check and primary training

Use the previous full text-model SFT settings unless the frozen preflight manifest
records a necessary compatibility change:

| Setting | Planned value |
| --- | --- |
| Parent | Pinned public Qwen3.5-4B instruction model |
| Trainable weights | Full text backbone/output; vision frozen and unused |
| Epochs/order | One pass, seed 7, fixed example order |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.1 |
| Learning rate | 1e-5, cosine schedule, 20 warmup updates |
| Effective batch | 32 examples per update (4 x 8 before length grouping) |
| Precision | fp32 weights/optimizer with bf16 autocast |
| Memory controls | Gradient checkpointing; dynamically smaller micro-batches |
| Starting micro-batch ceilings | 4,096 padded tokens / 2,048 answer tokens |
| Loss | Completion tokens only, EOS included; no prompt or padding |
| Inference mode | Explicit non-thinking, frozen decoding budgets |

Use at least 20 representative optimizer updates after warmup to measure stable
throughput, loss/gradients and peak allocated/reserved VRAM, including the longest
examples. End the feasibility phase within 30 minutes including setup. Reducing
micro-batch ceilings may preserve the 32-example update; changing effective batch,
precision or optimizer requires a new recorded condition.

Proceed only if the full pass plus exports fits both the remaining budget and the
3.5-hour primary-training allowance, kernels are working, and enough VRAM headroom
remains for the longest batch. Freeze the estimate and limits before continuing.
Stop promptly on nonfinite values, repeated OOM or a forecast that no longer fits.

No automatic switch to LoRA/QLoRA is included. If full SFT is infeasible, save the
evidence and unused budget, then document an adapter fallback as a separate
experiment. That prevents an adaptation-method change being reported as a size-only
result. Keep recoverable checkpoints; label any deadline-limited output as partial.

## Phase 4: export first, then evaluate on the M1

Save the trained checkpoint/tokenizer/config and completion receipt before spending
GPU time on generation. Copy them to durable storage, transfer to the M1 and verify
all hashes. Do not leave a GPU waiting for a lengthy local benchmark. Complete the
cleanup read-back and ledger reconciliation, then use MLX on the M1.

1. Convert the tuned text model to q8 and q4. Keep the bf16 source and both converted
   variants, with configuration/weight hashes and conversion/runtime versions.
2. Compare **2B round 4 q8**, **untuned 4B q8**, and **tuned 4B q8** on the same
   frozen development and regression inputs. Compare tuned q4 against tuned q8 as
   a separate quantization question. Preserve raw answers and per-case differences.
3. Report complete answers /48 and each 12-item coding category, runnable programs
   /36, stated outputs, repair decisions, strict traces, Sol /360, programs /111 and
   multi-input /18. Keep documentation judgments, lexical proxies and supplementary
   ordered trace values in separate columns.
4. Measure cold load/first-token latency, warmed prompt-processing speed, generated
   tokens/s and peak MLX plus system memory. Use one request at a time, at least
   ten fixed prompts, and both short and approximately 900-token contexts. Keep
   model loading separate from warmed generation; record swap/memory pressure.
5. Run execution and retrieval demonstrations separately after raw scoring.
   Publish a fixed set of successful and failed outputs, not only best samples.

The existing 2B q8 measured 26.1 generation tokens/s and 2.6 GB peak MLX memory.
For 4B, q8 at 10-16 tokens/s and q4 at 15-25 tokens/s are **unverified estimates**.
Use actual measurements to judge usability. Practical target: median warmed
generation at least 10 tokens/s at short context, no sustained swap, and peak
process memory at most 8 GB so macOS retains headroom. Long-context performance
is not established by this short-context experiment.

## Predeclared development decision

Keep the current 2B reference unless the 4B q8 candidate meets every screening gate:

| Gate | Required result |
| --- | --- |
| Complete answers | At least 8/48, versus the current 2B's 4/48, and at least two more than untuned 4B |
| Actual debugging improvement | At least one complete repair and one complete no-change answer |
| Runnable code | At least 21/36; no coding category loses more than one complete answer versus 2B |
| Sol retention | At least 339/360 (2B reference 341) |
| Program and multi-input retention | At least 110/111 programs and 18/18 multi-input tasks |
| Calculation retention | At least 107/120 canonical strict traces (2B reference 108) |
| Local inference | Meets the M1 speed/memory target above |

These are exploratory development screening thresholds, not statistical proof.
A single passing seed is a promising candidate, not an automatic default change.
If q4 loses more than one complete answer /48, two Sol cases /360, one strict trace
/120, or any multi-input case relative to tuned q8, retain q8 for the comparison.

Only consider a second seed after the first passes, its artifacts are safe, and
the *same full exposure* plus export fits the optional allowance and total ledger.
Otherwise omit it. Do not compare an abbreviated second run as a seed confirmation,
search multiple seeds until one wins, or spend the safety reserve. The final
holdout remains unused; a later promotion/holdout plan requires confirmed results.

## Deliverables and execution checklist

- [ ] Freeze local source/compiler/model/data manifests and baseline scores.
- [ ] Implement and validate bounded runner, mask checks, checkpoints and dual stop controls.
- [ ] Re-read prices, account headroom, volume free space and EU-RO-1 capacity.
- [ ] Record contracted rate, exact pod ID, deadlines and aggregate budget ledger.
- [ ] Complete feasibility check, or stop with a documented reason and unused funds.
- [ ] Train one frozen pass only if its forecast fits.
- [ ] Hash-verify the trained artifacts on the M1 and durable storage.
- [ ] Confirm experiment pod removal, retained volume and final charges.
- [ ] Score q8/q4 on the same development benchmarks and measure M1 performance.
- [ ] Publish gates, per-case gains/regressions, limitations and the next decision.

Proposed outputs: dated narrative report under `docs/`, raw manifests/answers/
scores/pricing/billing receipts under `docs/data/qos_ai_benchmark/`, and ignored
weights/prepared rows under `artifacts/qos_ai/qwen4b_round4_20261029/`. Use the
`qos` launcher for categorized scripts and register implementation additions in
`configs/layout.json`. Store no credentials or large model weights in Git.

Official references:
[Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json),
[RunPod pricing](https://docs.runpod.io/pods/pricing).
