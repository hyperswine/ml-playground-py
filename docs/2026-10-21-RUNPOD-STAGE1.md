# Stage 1: give the 557M the domain training the 168M had

Status: **run on 2026-10-19; results in [STAGE1_557M_2026-10-20.md](2026-10-20-STAGE1-557M.md).**

## Question

The 557M pilot never saw fprisc/qos text before its 2,784-example SFT, so its weak factual scores say
nothing about model size. Stage 1 gives it, and a matched 168M control, the same domain stage and the same
SFT, then compares them:

| | 557M | 168M control |
|---|---|---|
| Parent | pilot `pretrained.pt` (459M tokens, 80% English / 20% Python, random init) | `qos_ai/pod_pt/pretrained.pt` (500M FineWeb tokens, Qwen 0.8B embedding init) |
| Domain stage | identical | identical |
| SFT | pilot treatment (2 epochs, lr 5e-5, batch 8 x 4) | identical |

The parents still differ in initialization and data mix, so this narrows but does not isolate the size effect.

## Domain stage

`qos_ai_stage1_data.py` (run locally, output in `qos_ai/stage1_20261019/data/`, hash `7bb324d3...`):

- Training partition of the frozen split manifest (`623a7b7c...`) only: 2,302 source-chunk pieces from 444
  training files and 107,364 4B Q&A pairs. Benchmark questions and near duplicates are already excluded by
  the manifest. Zero unknown tokens.
- Q&A in the same chat format as SFT, loss on answer tokens only (as `qos_ai.py` lessons). Docs: every token.
- Two shuffled passes packed into 91,094 rows of 513 tokens (38.2M real tokens, 30.0M trained targets).
- Probes from the dev partition only: 94 doc windows and 512 Q&A pairs. Test partition unused.

`qos_ai_stage1_train.py`: fresh AdamW, lr 1e-4, 100 warmup, cosine to 10%, batch 8 x 2 (8,192 tokens/update).
**25% of each batch replays the pilot's English/Python training streams.** Without this, the 168M's
distillation wiped its English: FineWeb probe perplexity went from 27 to over 1,300 (see below).
Resumable, deadline-aware, publishes `domain.pt` then `student.pt`.

Probes before/after each phase: the pilot's English/Python validation windows, a 64-window FineWeb file-013
English probe nobody trained on, dev docs and dev Q&A answers. Generation benchmarks run afterwards on the
M1 (`qos_ai_noguard.py`, with and without the no-repeat guard).

## Cost

Secure RTX 4090, EU-RO-1 (the volume's data center), $0.89/hour on 2026-10-19, stock LOW.

| Step | Estimate |
|---|---|
| Pod start, upload 868 MB bundle, hash checks | ~15 min |
| 557M domain stage (62M tokens at ~25k tok/s) + SFT + probes | ~50 min |
| 168M control (about 3x faster) | ~20 min |
| **GPU total** | **~85 min, about $1.30** |
| Hard cap (pod stops itself) | 150 min, $2.23 |
| Backup of 4 checkpoints (6 GB) through the $0.06/hour CPU pod | ~$0.10 |

## Self-stop

`job.sh` arms a `sleep; runpodctl pod stop $RUNPOD_POD_ID` watchdog before anything else, and stops the pod
on completion and on any failure. It uses the pod-scoped runpodctl that RunPod provides inside pods; no
account key goes to the pod. Before launching, check over SSH that `runpodctl pod get $RUNPOD_POD_ID`
works on the pod. If it does not, keep the session open and stop the pod from here.

## Launch steps

1. Terminate the two stopped pilot pods (`xt9r2a1n5xij1r`, `kgeowatenfhc4t`); the volume is unaffected.
   Or reuse `kgeowatenfhc4t` for the backup step.
2. Create a Secure RTX 4090 pod in EU-RO-1 with volume `xewpn3z8za` at `/workspace`, image
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `minCudaVersion 13.0`, SSH on 22/tcp.
3. Verify the volume: `/workspace/qos_ai_557m/experiment/pretrained.pt` and
   `/workspace/qos_ai_557m/pretraining_data/dataset.json` exist.
4. Upload `qos_ai/stage1_20261019/bundle/` to `/workspace/qos_ai_stage1/bundle/`.
5. Check `runpodctl pod get $RUNPOD_POD_ID`, then start detached:
   `cd /workspace/qos_ai_stage1/bundle && setsid nohup bash job.sh > ../job.log 2>&1 < /dev/null &`
6. Watch `/workspace/qos_ai_stage1/m557/status.json`, and the pod status, until the pod stops itself.
7. Back up `m557/{domain,student}.pt`, `m168/{domain,student}.pt` and `result.json` files, verify hashes,
   then run `qos_ai_noguard.py` on the M1 for the four checkpoints, with no-repeat 0 and 3.
