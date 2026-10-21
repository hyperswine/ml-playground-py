#!/bin/bash
# Stage 3: SFT the 0.9B 557M domain checkpoint on 60k executed Sol programs + 4,709 verified examples,
# with 10% 4B Q&A replay. Uses the stage-2 trainer already on the volume. Self-stops; hard cap HARD_CAP_MINUTES.
set -u
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
cd "$(dirname "$0")"
S2=/workspace/qos_ai_stage2/bundle; S1=/workspace/qos_ai_stage1/bundle; OUT=/workspace/qos_ai_stage3
HARD_CAP_MINUTES="${HARD_CAP_MINUTES:-90}"; START=$(date +%s); export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
stamp() { echo "== $(date -u +%FT%TZ) (+$(( ($(date +%s) - START) / 60 )) min) $*"; }
stop_pod() { stamp "stopping pod: $1"; sync; runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" 2>&1; }
( sleep $(( HARD_CAP_MINUTES * 60 )); echo "HARD CAP" >> "$OUT/job_events.log"; sync; runpodctl stop pod "$RUNPOD_POD_ID" ) > /dev/null 2>&1 &
fail() { stamp "FAILED: $*"; echo "{\"phase\":\"failed\",\"error\":\"$*\"}" > "$OUT/job_status.json"; stop_pod failure; exit 1; }
stamp "verify"
sha256sum -c inputs.sha256 || fail "input hash mismatch"
cp $S2/qos_ai.py $S2/qos_ai_data.py $S2/qos_ai_finetune.py $S2/qos_ai_resume.py $S2/qos_ai_stage1_train.py $S2/qos_ai_stage2_train.py . || fail "copy code"
stamp "SFT on executed Sol programs"
python3 qos_ai_stage2_train.py sft --pretraining-data /workspace/qos_ai_557m/pretraining_data --stage1-data $S1/data \
  --english-probe $S1/english_probe.npy --load /workspace/qos_ai_stage2/m557_1b/domain.pt --data sft_sol.pt \
  --epochs 2 --learning-rate 3e-5 --warmup-steps 100 --replay-ratio 0.1 --output $OUT/m557_sol > $OUT/sft.log 2>&1 || fail "sft (sft.log)"
stamp "done"; echo '{"phase":"complete"}' > "$OUT/job_status.json"; stop_pod complete
