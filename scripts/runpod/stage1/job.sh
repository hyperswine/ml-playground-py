#!/bin/bash
# Stage 1 on a RunPod GPU pod with network volume xewpn3z8za at /workspace.
#   557M: pilot pretraining parent -> domain stage -> pilot SFT
#   168M control: qos_ai/pod_pt/pretrained.pt (500M FineWeb tokens) -> same domain stage -> same SFT
# The pod stops itself when done, on failure, and at a hard wall-clock cap, using the
# pod-scoped runpodctl. No account API key is placed on the pod.
#
# Usage on the pod (detached):  setsid nohup bash job.sh > job.log 2>&1 < /dev/null &
# Settings: HARD_CAP_MINUTES (default 150), SKIP_168M=1 to run the 557M only.
set -u
# SSH sessions do not inherit the container's RUNPOD_* variables; runpodctl needs the pod-scoped key.
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
cd "$(dirname "$0")"
HERE=$(pwd)
PILOT=/workspace/qos_ai_557m
OUT=/workspace/qos_ai_stage1
HARD_CAP_MINUTES="${HARD_CAP_MINUTES:-150}"
START=$(date +%s)
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

stamp() { echo "== $(date -u +%FT%TZ) (+$(( ($(date +%s) - START) / 60 )) min) $*"; }
stop_pod() {
  stamp "stopping pod $RUNPOD_POD_ID: $1"
  sync
  runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" 2>&1
}

# Hard cap first, so nothing below can leave the GPU billing.
( sleep $(( HARD_CAP_MINUTES * 60 )); echo "HARD CAP $HARD_CAP_MINUTES min reached" >> "$OUT/job_events.log"; sync
  runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" ) > /dev/null 2>&1 &
echo "$!" > "$OUT/watchdog.pid"
stamp "hard cap watchdog armed: $HARD_CAP_MINUTES min (pid $(cat "$OUT/watchdog.pid"))"

fail() { stamp "FAILED: $*"; echo "{\"phase\":\"failed\",\"error\":\"$*\"}" > "$OUT/job_status.json"; stop_pod failure; exit 1; }

stamp "checking runpodctl can see this pod"
runpodctl pod get "$RUNPOD_POD_ID" > /dev/null 2>&1 || runpodctl get pod "$RUNPOD_POD_ID" > /dev/null 2>&1 \
  || echo "WARNING: runpodctl could not read this pod; self-stop may not work" | tee -a "$OUT/job_events.log"

stamp "verifying inputs"
sha256sum -c "$HERE/inputs.sha256" || fail "input hash mismatch"
python3 - <<'EOF' || fail "stage-1 data hash mismatch"
import hashlib, json, numpy as np
meta = json.load(open('data/stage1_data.json'))
for name, sha in meta['sha256'].items():
  a = np.load(f'data/{name}.npy')
  assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == sha, name
print('stage-1 data verified', meta['hash'])
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DEADLINE=$(( START + HARD_CAP_MINUTES * 60 - 15 * 60 ))
COMMON=(--data data --pretraining-data "$PILOT/pretraining_data" --english-probe english_probe.npy
        --sft verified_sft.pt --deadline-unix "$DEADLINE")

stamp "557M domain stage + SFT"
python3 qos_ai_stage1_train.py --load "$PILOT/experiment/pretrained.pt" --output "$OUT/m557" "${COMMON[@]}" \
  > "$OUT/m557.log" 2>&1 || fail "557M run (see m557.log)"

if [ "${SKIP_168M:-0}" != "1" ]; then
  stamp "168M control domain stage + SFT"
  python3 qos_ai_stage1_train.py --load pretrained_168m_pod_pt.pt --output "$OUT/m168" "${COMMON[@]}" \
    > "$OUT/m168.log" 2>&1 || fail "168M run (see m168.log)"
fi

stamp "done"
echo '{"phase":"complete"}' > "$OUT/job_status.json"
stop_pod complete
