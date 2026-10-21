#!/usr/bin/env bash
# Runs ON the pod: ask (all rounds) -> teach -> train -> sample answers.  Everything resumes, so
# after an interruption just run it again.  Settings come from the environment:
#
#   ROUNDS=9         ask rounds; each round asks QUESTIONS new questions about every chunk
#   FIRST_ROUND=1    number of the first round, e.g. FIRST_ROUND=10 ROUNDS=6 adds rounds 10-15
#   STYLE_SET=original  question styles: original, or extended for new angles in later rounds
#   QUESTIONS=5      question styles per chunk per round (5 = all of them)
#   ASK_LIMIT=0      only use N random chunks (0 = all); ROUNDS=1 ASK_LIMIT=64 is a quick benchmark
#   TEACH_LIMIT=0    only score the first N windows per source (smoke runs)
#   STUDENT="--embedding-size 512 --layers 8 --heads 8"      ~60M parameters
#   TRAIN_ARGS="--epochs 3 --batch-size 16 --grad-accumulation 2 --learning-rate 6e-4"
#   STOP_WHEN_DONE=0 set to 1 to stop this pod when finished (billing stops; the volume is kept).
#                    Needs runpodctl configured with an API key on the pod, which RunPod pods do not
#                    have by default; otherwise stop the pod from the console or the RunPod MCP tools.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf}" PYTHONUNBUFFERED=1
ROUNDS="${ROUNDS:-9}" QUESTIONS="${QUESTIONS:-5}" ASK_LIMIT="${ASK_LIMIT:-0}" TEACH_LIMIT="${TEACH_LIMIT:-0}"
FIRST_ROUND="${FIRST_ROUND:-1}" STYLE_SET="${STYLE_SET:-original}"
STUDENT="${STUDENT:---embedding-size 512 --layers 8 --heads 8}"
TRAIN_ARGS="${TRAIN_ARGS:---epochs 3 --batch-size 16 --grad-accumulation 2 --learning-rate 6e-4}"
PY="${PY:-python3}"
ASK=(qos_ai.py ask --round "$FIRST_ROUND" --rounds "$ROUNDS" --questions-per-chunk "$QUESTIONS" --style-set "$STYLE_SET" --limit "$ASK_LIMIT")
started=$(date +%s)
stamp() { echo "== $(date '+%F %T') (+$(( ($(date +%s) - started) / 60 )) min) $*"; }

stamp "ask: rounds $FIRST_ROUND-$((FIRST_ROUND + ROUNDS - 1)), $QUESTIONS $STYLE_SET questions per chunk"
# vLLM does the asking; transformers (10x slower) only takes over if vLLM is missing or fails.
# Rejected pairs are not recorded, so a second pass would re-ask them -- not worth it at that speed.
if [ -x .venv-vllm/bin/python ] && .venv-vllm/bin/python "${ASK[@]}" --engine vllm --batch-size 1024; then
  :
else
  stamp "vLLM unavailable or failed; asking with transformers"
  $PY "${ASK[@]}" --engine hf --batch-size 48 || exit 1
fi

stamp "teach"
$PY qos_ai.py teach --batch-size 8 --limit "$TEACH_LIMIT" || exit 1

stamp "train"
# shellcheck disable=SC2086  # STUDENT and TRAIN_ARGS are deliberately word-split into flags
$PY qos_ai.py train $STUDENT $TRAIN_ARGS --report-every 100 --sample-every 500 --save-every 500 || exit 1

stamp "sample answers"
for question in "What is FP-RISC?" "What is Sol?" "How do I run a .sol script?" "What is the std tier?" \
    "Write a Sol function that adds two numbers." "How does QOS schedule actors?"; do
  echo "Q: $question"
  $PY qos_ai.py chat --max-new-tokens 150 "$question" | tail -n +2
  echo
done
stamp "done"

if [ "${STOP_WHEN_DONE:-0}" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
  stamp "stopping pod $RUNPOD_POD_ID"
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
