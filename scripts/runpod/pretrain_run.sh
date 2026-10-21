#!/usr/bin/env bash
# Runs ON the pod: general English first, then the teacher's lessons.  Everything resumes.
#
#   pretokenize  PRETRAIN_TOKENS of FineWeb-Edu -> qos_ai/pretrain_tokens.bin + qos_ai/vocab.pt
#   pretrain     the student on that text (next-token loss)        -> qos_ai/pretrained.pt
#   teach        the 4B teacher scores the existing Q&A + docs       -> qos_ai/lessons/
#   train        continue the pretrained student on the lessons      -> qos_ai/student.pt
#
# Settings (environment):
#   PRETRAIN_TOKENS=500000000   PRETRAIN_MINUTES=0 (0 = no time limit; otherwise stop and move on)
#   STUDENT="--embedding-size 1024 --layers 8 --heads 16"
#   PRETRAIN_ARGS="--batch-size 16 --grad-accumulation 2"
#   TRAIN_ARGS="--epochs 2 --batch-size 8 --grad-accumulation 4 --learning-rate 3e-4"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf}" PYTHONUNBUFFERED=1
export QOS_AI_TEACHER="${QOS_AI_TEACHER:-Qwen/Qwen3.5-4B}"
PRETRAIN_TOKENS="${PRETRAIN_TOKENS:-500000000}" PRETRAIN_MINUTES="${PRETRAIN_MINUTES:-0}"
STUDENT="${STUDENT:---embedding-size 1024 --layers 8 --heads 16}"
PRETRAIN_ARGS="${PRETRAIN_ARGS:---batch-size 16 --grad-accumulation 2}"
TRAIN_ARGS="${TRAIN_ARGS:---epochs 2 --batch-size 8 --grad-accumulation 4 --learning-rate 3e-4}"
started=$(date +%s)
stamp() { echo "== $(date '+%F %T') (+$(( ($(date +%s) - started) / 60 )) min) $*"; }

if [ ! -s qos_ai/vocab.pt ]; then
  stamp "pretokenize $PRETRAIN_TOKENS tokens"
  python3 qos_ai.py pretokenize --tokens "$PRETRAIN_TOKENS" || exit 1
fi

stamp "pretrain"
resume=()
[ -s qos_ai/pretrained.pt ] && resume=(--load qos_ai/pretrained.pt)
# shellcheck disable=SC2086  # STUDENT and PRETRAIN_ARGS are word-split into flags on purpose
python3 qos_ai.py pretrain $STUDENT $PRETRAIN_ARGS --teacher-embeddings --compile --max-minutes "$PRETRAIN_MINUTES" "${resume[@]}" || exit 1

stamp "teach"
python3 qos_ai.py teach --batch-size 8 || exit 1

stamp "train from the pretrained student"
# shellcheck disable=SC2086
python3 qos_ai.py train --load qos_ai/pretrained.pt $TRAIN_ARGS --report-every 100 --sample-every 500 --save-every 500 || exit 1

stamp "sample answers"
for question in "What is FP-RISC?" "What is Sol?" "What is a hart?" "How does the Sol runtime handle a failed transaction?" \
    "Write a Sol function that adds two numbers."; do
  echo "Q: $question"
  python3 qos_ai.py chat --max-new-tokens 120 --temperature 0.5 "$question" | tail -n +2
  echo
done
stamp "done"
