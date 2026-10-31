# Documentation

This is the dated record of the custom QOS language-model work and related ML experiments.
The layout follows the chronological pages used by FP-RISC and the separation of reports and raw measurements used by NeoProlog.

| Start here | Contents |
| --- | --- |
| [Project overview: what worked and what did not](2026-10-28-PROJECT-OVERVIEW.md) | Whole-project results, limitations and current priorities |
| [Repository layout and migration](2026-10-21-REPOSITORY-MIGRATION.md) | Source categories, preserved artifacts, timestamps and validation |
| [Running experiments](2026-10-21-RUNNING-EXPERIMENTS.md) | Launcher, dependencies, environments and unchanged M1 workflows |
| [Raw data and evidence](data/README.md) | Frozen benchmarks, source snapshots, receipts and migration provenance |

## Dated reports

Dates in filenames identify when a page began. Keep that date when editing a page;
record later updates with an ISO-8601 timestamp and timezone in a new section.
Preserve older reports as historical observations. A later report may supersede an
older plan without rewriting its original experiment results. The migration report
records its exact timestamp; original pages retain their own dates and recorded run times.

| Date | Document |
| --- | --- |
| 2026-08-24 | [2026-08-24-NEURAL-NOTES](2026-08-24-NEURAL-NOTES.txt) |
| 2026-10-16 | [Expanded verified curriculum and M1 comparison](2026-10-16-EXPANDED-FINETUNE.md) |
| 2026-10-16 | [M1 verified-example fine-tune and development comparison](2026-10-16-FINETUNE.md) |
| 2026-10-16 | [M1 benchmark and split implementation - 2026-10-16](2026-10-16-IMPLEMENTATION.md) |
| 2026-10-16 | [Input-preserving debugging and code-generation experiment](2026-10-16-INPUTS-FINETUNE.md) |
| 2026-10-16 | [qos_ai improvement plan](2026-10-16-QOS-AI-IMPROVEMENT-PLAN.md) |
| 2026-10-16 | [qos_ai: distilling a small FP-RISC / Sol / QOS model](2026-10-16-QOS-AI-REPORT.md) |
| 2026-10-17 | [qos_ai clean benchmark and verified teaching set](2026-10-17-QOS-AI-BENCHMARK.md) |
| 2026-10-18 | [Balanced repair curriculum and two-parent M1 comparison](2026-10-18-BALANCED-REPAIR.md) |
| 2026-10-18 | [Verified calculation and complete-repair follow-up](2026-10-18-OUTPUT-COMPLETE.md) |
| 2026-10-18 | [qos_ai: 32 real output samples](2026-10-18-QOS-AI-SAMPLE-GALLERY.md) |
| 2026-10-19 | [Broader verified curriculum and generalization experiment](2026-10-19-GENERALIZATION.md) |
| 2026-10-19 | [Next training steps for the custom 557M model](2026-10-19-NEXT-STEPS-557M.md) |
| 2026-10-19 | [557M RunPod training preparation](2026-10-19-RUNPOD-557M-HISTORY.md) |
| 2026-10-19 | [Custom 557M RunPod training pilot](2026-10-19-RUNPOD-557M.md) |
| 2026-10-19 | [Custom 557M model: M1 pilot](2026-10-19-SCALE-557M.md) |
| 2026-10-20 | [Fresh Sol question probe](2026-10-20-SOL-PROBE.md) |
| 2026-10-20 | [Stage 1: domain training for the 557M, with a matched 168M control](2026-10-20-STAGE1-557M.md) |
| 2026-10-20 | [Stage 2: continued 557M pretraining and a gentler mixed SFT](2026-10-20-STAGE2-557M.md) |
| 2026-10-20 | [Stage 3: training on 60,000 executed Sol programs](2026-10-20-STAGE3-SOL-PROGRAMS.md) |
| 2026-10-21 | [Stage 1: give the 557M the domain training the 168M had](2026-10-21-RUNPOD-STAGE1.md) |
| 2026-10-21 | [Stage 4 runbook](2026-10-21-RUNPOD-STAGE4.md) |
| 2026-10-21 | [Stage 4 preparation: retain language while strengthening Sol behavior](2026-10-21-STAGE4-PREPARATION.md) |
| 2026-10-21 | [Stage 5: varied wording, compiler-verified self-samples and retrieval context](2026-10-21-STAGE5.md) |
| 2026-10-21 | [Next steps 1-6: execution, compiler-verified self-training, wording, Stage 4 evaluation, retrieval](2026-10-21-NEXT-STEPS-PROGRESS.md) |
| 2026-10-21 | [Repository migration](2026-10-21-REPOSITORY-MIGRATION.md) |
| 2026-10-21 | [Running experiments](2026-10-21-RUNNING-EXPERIMENTS.md) |
| 2026-10-22 | [Reference: Qwen3.5-0.8B fine-tuned on the Stage 5 data](2026-10-22-QWEN-SFT-REFERENCE.md) |
| 2026-10-22 | [Fine-tuned Qwen: second seed, local M1 use, execution and retrieval](2026-10-22-QWEN-LOCAL-USE.md) |
| 2026-10-22 | [Qwen round 2: targeted Sol data, long-reference retrieval and hybrid search](2026-10-22-QWEN-ROUND2.md) |
| 2026-10-26 | [Qwen round 3: steps before the answer, and closed-book pairs](2026-10-26-QWEN-ROUND3.md) |
| 2026-10-26 | [Qwen round 4: steps first, retrieval, and different closed-book questions](2026-10-26-QWEN-ROUND4.md) |
| 2026-10-27 | [Qwen3.5-2B on the round-4 data](2026-10-27-QWEN-2B.md) |
| 2026-10-27 | [Mac inference: why Qwen3.5 was slow on the M1, and MLX as the fix](2026-10-27-MAC-INFERENCE.md) |
| 2026-10-24 | [Small-model pretraining precedents: 4-12B tokens](2026-10-24-SMALL-MODEL-PRETRAINING-PRECEDENTS.md) |
| 2026-10-27 | [Qwen 2B fresh diagnostic and M1 repair pilot](2026-10-27-QWEN-2B-REPAIR.md) |
| 2026-10-28 | [Qwen 2B controlled coding curriculum](2026-10-28-QWEN-2B-CONTROLLED.md) |
| 2026-10-28 | [Whole-project overview: what worked and what did not](2026-10-28-PROJECT-OVERVIEW.md) |
| 2026-10-28 | [Qwen internals, adaptation coverage and data/recipe gaps](2026-10-28-QWEN-INTERNALS.md) |
| 2026-10-29 | [Qwen3.5-4B experiment plan: US$20 maximum](2026-10-29-QWEN-4B-20USD-PLAN.md) |
| 2026-10-31 | [Windows PC: remote wake, the pc helper, and llama.cpp inference on the RX 7600](2026-10-31-WINDOWS-PC-INFERENCE.md) |
