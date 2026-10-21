#!/bin/bash
# Stage 5 on a RunPod GPU pod with network volume xewpn3z8za at /workspace (volume is read, never pruned).
#   1. mix the Stage 5 pools (Stage 4 verified + varied-wording tasks + retrieval-context examples +
#      compiler-verified self-samples) with English/docs/Q&A replay at Stage 4's 65/14/14/7 token shares
#   2. train one pass at 2e-5 from the 0.9B domain parent (Stage 4 `one_2e5` recipe) on the pod's local disk
#   3. fixed loss probes + arithmetic validation
#   4. wait (bounded) for the M1 to pull and acknowledge the weights, then stop the pod
# Self-stop via pod-scoped runpodctl on completion, failure and HARD_CAP_MINUTES; WAIT_MINUTES bounds step 4.
set -u
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
cd "$(dirname "$0")"
PILOT=/workspace/qos_ai_557m; S1=/workspace/qos_ai_stage1/bundle; S2=/workspace/qos_ai_stage2
VOL=/workspace/qos_ai_stage5; LOCAL=/root/s5; V=one_2e5
HARD_CAP_MINUTES="${HARD_CAP_MINUTES:-90}"; WAIT_MINUTES="${WAIT_MINUTES:-30}"
START=$(date +%s); export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p "$LOCAL" "$VOL"
stamp() { echo "== $(date -u +%FT%TZ) (+$(( ($(date +%s) - START) / 60 )) min) $*"; }
stop_pod() { stamp "stopping pod: $1"; sync; runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" 2>&1; }
( sleep $(( HARD_CAP_MINUTES * 60 )); echo "HARD CAP" >> "$VOL/job_events.log"; sync; runpodctl stop pod "$RUNPOD_POD_ID" ) > /dev/null 2>&1 &
fail() { stamp "FAILED: $*"; echo "{\"phase\":\"failed\",\"error\":\"$*\"}" > "$VOL/job_status.json"; cp "$LOCAL"/*.log "$VOL"/ 2>/dev/null; stop_pod failure; exit 1; }
phase() { echo "{\"phase\":\"$1\",\"unix_time\":$(date +%s)}" > "$VOL/job_status.json"; stamp "$1"; }

phase verify
sha256sum -c inputs.sha256 || fail "input hash mismatch"
PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q tokenizers==0.23.2 > "$LOCAL/pip.log" 2>&1 || fail "pip tokenizers"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h "$LOCAL" | tail -1

phase mix
python3 qos_ai_stage4_data.py mix --pools pools.pt --pretraining-data "$PILOT/pretraining_data" \
  --pretraining-manifest-sha256 7451c6fabb883a9c87a893b2b84a80c536132c376698c0ba7cf352db894886b5 \
  --output "$LOCAL/mixture" > "$LOCAL/mix.log" 2>&1 || fail "mix (mix.log)"
cp "$LOCAL/mixture/mixture.json" "$VOL/"
SFT_SHA=$(sha256sum "$LOCAL/mixture/sft.pt" | cut -d' ' -f1)
grep -q "$SFT_SHA" "$LOCAL/mixture/mixture.json" || fail "mixture receipt does not match sft.pt"

DEADLINE=$(( START + (HARD_CAP_MINUTES - WAIT_MINUTES / 2 - 10) * 60 ))
phase "train_$V"
python3 qos_ai_stage4_train.py --load "$S2/m557_1b/domain.pt" --data "$LOCAL/mixture/sft.pt" --data-sha256 "$SFT_SHA" \
  --variant "$V" --output "$LOCAL/$V" --deadline-unix "$DEADLINE" --device cuda --save-every 500 \
  > "$LOCAL/train_$V.log" 2>&1 || fail "train $V"
[ -f "$LOCAL/$V/student.pt" ] || fail "$V did not complete before its deadline"
phase probe
python3 qos_ai_stage4_probe.py --checkpoint "$LOCAL/$V/student.pt" --pretraining-data "$PILOT/pretraining_data" \
  --stage1-data "$S1/data" --english-probe "$S1/english_probe.npy" --arithmetic-validation arithmetic_validation.pt \
  --output "$LOCAL/$V/probes.json" > "$LOCAL/probe_$V.log" 2>&1 || fail "probe $V"
rm -f "$LOCAL/$V/progress.pt"
mkdir -p "$VOL/$V" && cp "$LOCAL/$V"/{settings.json,result.json,probes.json,status.json} "$VOL/$V/"
sha256sum "$LOCAL/$V/student.pt" | cut -d' ' -f1 > "$LOCAL/$V/student.sha256"
touch "$LOCAL/READY_$V"
cp "$LOCAL"/*.log "$VOL"/ 2>/dev/null

phase wait_for_pulls
date +%s > "$LOCAL/WAIT_STARTED"
until [ -f "$LOCAL/PULLED_$V" ]; do
  [ $(( $(date +%s) - START )) -gt $(( (HARD_CAP_MINUTES - 5) * 60 )) ] && { stamp "pull wait hit the hard cap"; break; }
  [ $(( $(date +%s) - $(cat "$LOCAL/WAIT_STARTED") )) -gt $(( WAIT_MINUTES * 60 )) ] && { stamp "pull wait timed out"; break; }
  sleep 20
done
echo "{\"phase\":\"complete\",\"pulled\":\"$(ls "$LOCAL" | grep PULLED_ | tr '\n' ' ')\"}" > "$VOL/job_status.json"
stop_pod complete
