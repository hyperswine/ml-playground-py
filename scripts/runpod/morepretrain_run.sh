#!/usr/bin/env bash
# Runs ON the pod: more English pretraining for an existing pretrained student, then re-distill.
#
#   pretokenize --append  APPEND_TOKENS more FineWeb-Edu tokens from file FIRST_FILE on (vocabulary unchanged)
#   pretrain --load       continue qos_ai/pretrained.pt up to TOTAL_TOKENS in all (re-warms the learning rate)
#   train                 distill the result on the 4B lessons already scored on this pod -> qos_ai/student.pt
#   evalloops             loops / finished on held-out questions (400-token answers)
# Settings: APPEND_TOKENS=1000000000 FIRST_FILE=1 TOTAL_TOKENS=1500000000 PRETRAIN_MINUTES=240
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf}" PYTHONUNBUFFERED=1
export QOS_AI_TEACHER="${QOS_AI_TEACHER:-Qwen/Qwen3.5-4B}"
APPEND_TOKENS="${APPEND_TOKENS:-1000000000}" FIRST_FILE="${FIRST_FILE:-1}"
TOTAL_TOKENS="${TOTAL_TOKENS:-1500000000}" PRETRAIN_MINUTES="${PRETRAIN_MINUTES:-240}"
started=$(date +%s)
stamp() { echo "== $(date '+%F %T') (+$(( ($(date +%s) - started) / 60 )) min) $*"; }

# keep the checkpoint this run starts from, and append each batch of new data only once
[ -s "qos_ai/pretrained_before_$TOTAL_TOKENS.pt" ] || cp qos_ai/pretrained.pt "qos_ai/pretrained_before_$TOTAL_TOKENS.pt"
marker="qos_ai/appended_for_$TOTAL_TOKENS.done"
if [ ! -e "$marker" ]; then
  stamp "pretokenize --append $APPEND_TOKENS tokens from file $FIRST_FILE"
  python3 qos_ai.py pretokenize --append --first-file "$FIRST_FILE" --tokens "$APPEND_TOKENS" || exit 1
  touch "$marker"
fi

stamp "pretrain to $TOTAL_TOKENS tokens"
python3 qos_ai.py pretrain --load qos_ai/pretrained.pt --tokens "$TOTAL_TOKENS" --batch-size 16 --grad-accumulation 2 \
  --compile --max-minutes "$PRETRAIN_MINUTES" --sample-every 4000 --save-every 2000 || exit 1

stamp "teach (only new or changed lessons)"
python3 qos_ai.py teach --batch-size 8 || exit 1

stamp "distill the longer-pretrained student"
python3 qos_ai.py train --load qos_ai/pretrained.pt --output qos_ai/student.pt --epochs 2 --batch-size 8 --grad-accumulation 4 \
  --learning-rate 3e-4 --report-every 100 --sample-every 500 --save-every 500 || exit 1

stamp "evalloops"
python3 qos_ai.py evalloops --load qos_ai/student.pt --batch-size 200 --max-new-tokens 400 || exit 1

stamp "sample answers"
for question in "What is FP-RISC?" "What is Sol?" "What is a hart?" "How does the Sol runtime handle a failed transaction?" \
    "Write a Sol function that adds two numbers."; do
  echo "Q: $question"
  python3 qos_ai.py chat --max-new-tokens 150 --temperature 0.4 "$question" | tail -n +2
  echo
done
stamp "done"
