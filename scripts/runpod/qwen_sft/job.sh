#!/bin/bash
# Reference run: fine-tune Qwen3.5-0.8B on the identical Stage 5 mixture and generate answers for every frozen
# evaluation prompt, before and after. Network volume xewpn3z8za at /workspace is read, never pruned; outputs go to
# the pod's local disk, small receipts and answers to /workspace/qos_ai_qwen_sft.
# Self-stop via pod-scoped runpodctl on completion, failure and HARD_CAP_MINUTES; WAIT_MINUTES bounds the pull wait.
set -u
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
cd "$(dirname "$0")"
PILOT=/workspace/qos_ai_557m; VOL=/workspace/qos_ai_qwen_sft; LOCAL=/root/q
STAGE5_SFT_SHA=937452d144b39e99cbc0870e103d9948eb08444e101678319ea6e5f1b690a9ee
HARD_CAP_MINUTES="${HARD_CAP_MINUTES:-120}"; WAIT_MINUTES="${WAIT_MINUTES:-30}"
START=$(date +%s); export PYTHONUNBUFFERED=1 HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p "$LOCAL" "$VOL"
stamp() { echo "== $(date -u +%FT%TZ) (+$(( ($(date +%s) - START) / 60 )) min) $*"; }
stop_pod() { stamp "stopping pod: $1"; sync; runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" 2>&1; }
( sleep $(( HARD_CAP_MINUTES * 60 )); echo "HARD CAP" >> "$VOL/job_events.log"; sync; runpodctl stop pod "$RUNPOD_POD_ID" ) > /dev/null 2>&1 &
fail() { stamp "FAILED: $*"; echo "{\"phase\":\"failed\",\"error\":\"$*\"}" > "$VOL/job_status.json"; cp "$LOCAL"/*.log "$VOL"/ 2>/dev/null; stop_pod failure; exit 1; }
phase() { echo "{\"phase\":\"$1\",\"unix_time\":$(date +%s)}" > "$VOL/job_status.json"; stamp "$1"; }

phase verify
sha256sum -c inputs.sha256 || fail "input hash mismatch"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
phase install
pip install -q "transformers @ git+https://github.com/huggingface/transformers.git@80a79d1bef2bacfa10b1b543669eebffa3f0ec36" \
  safetensors huggingface_hub tokenizers flash-linear-attention > "$LOCAL/pip.log" 2>&1 || fail "pip (pip.log)"
python3 -c "import transformers, fla; print('transformers', transformers.__version__)" >> "$LOCAL/pip.log" 2>&1 || fail "imports (pip.log)"
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-0.8B'))" > "$LOCAL/download.log" 2>&1 \
  || fail "model download (download.log)"

phase mix
python3 qos_ai_stage4_data.py mix --pools pools.pt --pretraining-data "$PILOT/pretraining_data" \
  --pretraining-manifest-sha256 7451c6fabb883a9c87a893b2b84a80c536132c376698c0ba7cf352db894886b5 \
  --output "$LOCAL/mixture" > "$LOCAL/mix.log" 2>&1 || fail "mix (mix.log)"
[ "$(sha256sum "$LOCAL/mixture/sft.pt" | cut -d' ' -f1)" = "$STAGE5_SFT_SHA" ] || fail "mixture differs from Stage 5"
python3 qos_ai_qwen_sft.py convert --sft "$LOCAL/mixture/sft.pt" --vocabulary vocabulary.pt --output "$LOCAL/qwen_sft.pt" \
  > "$LOCAL/convert.log" 2>&1 || fail "convert (convert.log)"

phase generate_base
python3 qos_ai_qwen_sft.py generate --model Qwen/Qwen3.5-0.8B --prompts prompts.jsonl --output "$LOCAL/base_answers.jsonl" \
  > "$LOCAL/generate_base.log" 2>&1 || fail "generate base (generate_base.log)"
cp "$LOCAL/base_answers.jsonl" "$VOL/"
phase train
python3 qos_ai_qwen_sft.py train --data "$LOCAL/qwen_sft.pt" --output "$LOCAL/run" > "$LOCAL/train.log" 2>&1 || fail "train (train.log)"
cp "$LOCAL/run/result.json" "$VOL/train_result.json"
phase generate_tuned
python3 qos_ai_qwen_sft.py generate --model "$LOCAL/run/model" --prompts prompts.jsonl --output "$LOCAL/tuned_answers.jsonl" \
  > "$LOCAL/generate_tuned.log" 2>&1 || fail "generate tuned (generate_tuned.log)"
cp "$LOCAL/tuned_answers.jsonl" "$LOCAL"/*.log "$VOL/"
(cd "$LOCAL/run/model" && sha256sum * > ../model.sha256) && cp "$LOCAL/run/model.sha256" "$VOL/"
touch "$LOCAL/READY"

phase wait_for_pulls
date +%s > "$LOCAL/WAIT_STARTED"
until [ -f "$LOCAL/PULLED" ]; do
  [ $(( $(date +%s) - START )) -gt $(( (HARD_CAP_MINUTES - 5) * 60 )) ] && { stamp "pull wait hit the hard cap"; break; }
  [ $(( $(date +%s) - $(cat "$LOCAL/WAIT_STARTED") )) -gt $(( WAIT_MINUTES * 60 )) ] && { stamp "pull wait timed out"; break; }
  sleep 20
done
echo "{\"phase\":\"complete\",\"pulled\":$( [ -f "$LOCAL/PULLED" ] && echo true || echo false )}" > "$VOL/job_status.json"
stop_pod complete
