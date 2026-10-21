#!/bin/bash
# Stage 4 on a RunPod GPU pod with network volume xewpn3z8za at /workspace (volume is read, never pruned).
#   1. hash the recovery artifacts on the volume (background)
#   2. assemble the token-accounted replay mixture next to the original English stream
#   3. train half_2e5, one_2e5, one_1e5 from the 0.9B domain parent; outputs on the pod's local disk
#   4. fixed loss probes + arithmetic validation for each variant
#   5. wait (bounded) for the M1 to pull and acknowledge each artifact, then stop the pod
# Self-stop via pod-scoped runpodctl on completion, failure and HARD_CAP_MINUTES; WAIT_MINUTES bounds step 5.
set -u
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
cd "$(dirname "$0")"
PILOT=/workspace/qos_ai_557m; S1=/workspace/qos_ai_stage1/bundle; S2=/workspace/qos_ai_stage2
VOL=/workspace/qos_ai_stage4; LOCAL=/root/s4
HARD_CAP_MINUTES="${HARD_CAP_MINUTES:-150}"; WAIT_MINUTES="${WAIT_MINUTES:-40}"
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

( cd /workspace && sha256sum qos_ai_stage2/m557_1b_pretrain/pretrained.pt qos_ai_stage2/m557_1b_pretrain/progress_stage2.pt \
    qos_ai_557m/pretraining_data/documents.jsonl.gz qos_ai_557m/pretraining_data/dataset.json \
    qos_ai_557m/pretraining_data/english_train.bin > "$LOCAL/recovery.sha256.partial" \
  && mv "$LOCAL/recovery.sha256.partial" "$LOCAL/recovery.sha256" && cp "$LOCAL/recovery.sha256" "$VOL/" \
  && touch "$LOCAL/READY_recovery" ) &

phase mix
python3 qos_ai_stage4_data.py mix --pools pools.pt --pretraining-data "$PILOT/pretraining_data" \
  --pretraining-manifest-sha256 7451c6fabb883a9c87a893b2b84a80c536132c376698c0ba7cf352db894886b5 \
  --output "$LOCAL/mixture" > "$LOCAL/mix.log" 2>&1 || fail "mix (mix.log)"
cp "$LOCAL/mixture/mixture.json" "$VOL/"
SFT_SHA=$(sha256sum "$LOCAL/mixture/sft.pt" | cut -d' ' -f1)
grep -q "$SFT_SHA" "$LOCAL/mixture/mixture.json" || fail "mixture receipt does not match sft.pt"

DEADLINE=$(( START + (HARD_CAP_MINUTES - WAIT_MINUTES / 2 - 10) * 60 ))
PROBE=(--pretraining-data "$PILOT/pretraining_data" --stage1-data "$S1/data" --english-probe "$S1/english_probe.npy"
       --arithmetic-validation arithmetic_validation.pt)
python3 qos_ai_stage4_probe.py --checkpoint "$S2/m557_1b/domain.pt" "${PROBE[@]}" --output "$LOCAL/parent_probes.json" \
  > "$LOCAL/parent_probes.log" 2>&1 || fail "parent probes"
cp "$LOCAL/parent_probes.json" "$VOL/"
for v in half_2e5 one_2e5 one_1e5; do
  phase "train_$v"
  python3 qos_ai_stage4_train.py --load "$S2/m557_1b/domain.pt" --data "$LOCAL/mixture/sft.pt" --data-sha256 "$SFT_SHA" \
    --variant "$v" --output "$LOCAL/$v" --deadline-unix "$DEADLINE" --device cuda --save-every 500 \
    > "$LOCAL/train_$v.log" 2>&1 || fail "train $v"
  [ -f "$LOCAL/$v/student.pt" ] || fail "$v did not complete before its deadline"
  python3 qos_ai_stage4_probe.py --checkpoint "$LOCAL/$v/student.pt" "${PROBE[@]}" --output "$LOCAL/$v/probes.json" \
    > "$LOCAL/probe_$v.log" 2>&1 || fail "probe $v"
  rm -f "$LOCAL/$v/progress.pt"           # completed; the published weights are what gets pulled
  mkdir -p "$VOL/$v" && cp "$LOCAL/$v"/{settings.json,result.json,probes.json,status.json} "$VOL/$v/"
  sha256sum "$LOCAL/$v/student.pt" | cut -d' ' -f1 > "$LOCAL/$v/student.sha256"
  touch "$LOCAL/READY_$v"
done
cp "$LOCAL"/*.log "$VOL"/ 2>/dev/null

phase wait_for_pulls
until [ -f "$LOCAL/PULLED_half_2e5" ] && [ -f "$LOCAL/PULLED_one_2e5" ] && [ -f "$LOCAL/PULLED_one_1e5" ] \
      && [ -f "$LOCAL/PULLED_recovery" ]; do
  [ $(( $(date +%s) - START )) -gt $(( (HARD_CAP_MINUTES - 5) * 60 )) ] && { stamp "pull wait hit the hard cap"; break; }
  [ -f "$LOCAL/WAIT_STARTED" ] || date +%s > "$LOCAL/WAIT_STARTED"
  [ $(( $(date +%s) - $(cat "$LOCAL/WAIT_STARTED") )) -gt $(( WAIT_MINUTES * 60 )) ] && { stamp "pull wait timed out"; break; }
  sleep 20
done
echo "{\"phase\":\"complete\",\"pulled\":\"$(ls "$LOCAL" | grep PULLED_ | tr '\n' ' ')\"}" > "$VOL/job_status.json"
stop_pod complete
