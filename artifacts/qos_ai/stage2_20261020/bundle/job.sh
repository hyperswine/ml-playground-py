#!/bin/bash
# Stage 2 on a RunPod GPU pod with network volume xewpn3z8za at /workspace.
#   A. mixed SFT on the stage-1 557M domain checkpoint          -> /workspace/qos_ai_stage2/m557_459m_mix
#   B. mixed SFT on the stage-1 168M domain checkpoint          -> /workspace/qos_ai_stage2/m168_mix
#   C. continue 557M pretraining (pilot state, +541M tokens)    -> /workspace/qos_ai_stage2/m557_1b_pretrain
#   D. stage-1 domain stage + pilot SFT on the 1B model         -> /workspace/qos_ai_stage2/m557_1b
#   E. mixed SFT on the 1B domain checkpoint                    -> /workspace/qos_ai_stage2/m557_1b_mix
# The pod stops itself (pod-scoped runpodctl) on completion, on failure and at HARD_CAP_MINUTES.
# Usage on the pod: HARD_CAP_MINUTES=470 setsid nohup bash job.sh > ../job.log 2>&1 < /dev/null &
set -u
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
cd "$(dirname "$0")"
S1=/workspace/qos_ai_stage1/bundle
PILOT=/workspace/qos_ai_557m
OUT=/workspace/qos_ai_stage2
HARD_CAP_MINUTES="${HARD_CAP_MINUTES:-470}"
TAIL_MINUTES="${TAIL_MINUTES:-80}"   # reserved after pretraining for D, E, probes and margin
START=$(date +%s)
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT"

stamp() { echo "== $(date -u +%FT%TZ) (+$(( ($(date +%s) - START) / 60 )) min) $*"; }
stop_pod() { stamp "stopping pod $RUNPOD_POD_ID: $1"; sync; runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" 2>&1; }
( sleep $(( HARD_CAP_MINUTES * 60 )); echo "HARD CAP $HARD_CAP_MINUTES min reached" >> "$OUT/job_events.log"; sync
  runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" ) > /dev/null 2>&1 &
stamp "hard cap watchdog armed: $HARD_CAP_MINUTES min"
fail() { stamp "FAILED: $*"; echo "{\"phase\":\"failed\",\"error\":\"$*\"}" > "$OUT/job_status.json"; stop_pod failure; exit 1; }
step_done() { echo "{\"phase\":\"running\",\"done\":\"$1\",\"unix_time\":$(date +%s)}" > "$OUT/job_status.json"; }

stamp "verifying inputs"
sha256sum -c inputs.sha256 || fail "input hash mismatch"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=(--pretraining-data "$PILOT/pretraining_data" --stage1-data "$S1/data" --english-probe "$S1/english_probe.npy")

if [ "${SKIP_AB:-0}" != "1" ]; then
stamp "A: mixed SFT, 557M stage-1 domain"
python3 qos_ai_stage2_train.py sft "${P[@]}" --load /workspace/qos_ai_stage1/m557/domain.pt --data sft_mix.pt \
  --output "$OUT/m557_459m_mix" > "$OUT/a.log" 2>&1 || fail "A (a.log)"
step_done A
stamp "B: mixed SFT, 168M stage-1 domain"
python3 qos_ai_stage2_train.py sft "${P[@]}" --load /workspace/qos_ai_stage1/m168/domain.pt --data sft_mix.pt \
  --output "$OUT/m168_mix" > "$OUT/b.log" 2>&1 || fail "B (b.log)"
step_done B
fi

PRE_DEADLINE=$(( START + (HARD_CAP_MINUTES - TAIL_MINUTES) * 60 ))
stamp "C: continued pretraining until at most $(date -u -d @$PRE_DEADLINE +%H:%M) UTC"
python3 qos_ai_stage2_train.py pretrain "${P[@]}" --resume-from "$PILOT/experiment/progress_pretrain.pt" \
  --output "$OUT/m557_1b_pretrain" --deadline-unix "$PRE_DEADLINE" --save-minutes 30 ${TOTAL_STEPS:+--total-steps $TOTAL_STEPS} >> "$OUT/c.log" 2>&1 || fail "C (c.log)"
step_done C

stamp "D: domain stage + pilot SFT on the 1B model"
python3 qos_ai_stage1_train.py --load "$OUT/m557_1b_pretrain/pretrained.pt" --output "$OUT/m557_1b" \
  --data "$S1/data" --pretraining-data "$PILOT/pretraining_data" --english-probe "$S1/english_probe.npy" \
  --sft "$S1/verified_sft.pt" --save-minutes 999 --deadline-unix $(( START + HARD_CAP_MINUTES * 60 - 25 * 60 )) > "$OUT/d.log" 2>&1 || fail "D (d.log)"
step_done D

stamp "E: mixed SFT on the 1B domain checkpoint"
python3 qos_ai_stage2_train.py sft "${P[@]}" --load "$OUT/m557_1b/domain.pt" --data sft_mix.pt \
  --output "$OUT/m557_1b_mix" > "$OUT/e.log" 2>&1 || fail "E (e.log)"

stamp "done"
echo '{"phase":"complete"}' > "$OUT/job_status.json"
stop_pod complete
