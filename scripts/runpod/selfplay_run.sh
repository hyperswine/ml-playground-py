#!/usr/bin/env bash
# Runs ON the pod: self-correction rounds on top of an existing student (qos_ai/student.pt).
#
#   each round:  selfplay  the student answers SELF_QUESTIONS questions itself   -> qos_ai/self.jsonl
#                teach     the teacher scores every token of those answers        -> qos_ai/lessons/self_r*.pt
#                train     continue the student; SELF_FRACTION of each batch is its own scored answers
#   evalloops before the first round and after every round: how often answers loop on held-out questions
#
# Settings: ROUNDS=2 SELF_QUESTIONS=16000 SELF_STEPS=800 SELF_FRACTION=0.5 QA_FRACTION=0.25
#           TRAIN_ARGS="--batch-size 8 --grad-accumulation 4 --learning-rate 1e-4"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf}" PYTHONUNBUFFERED=1
export QOS_AI_TEACHER="${QOS_AI_TEACHER:-Qwen/Qwen3.5-4B}"
ROUNDS="${ROUNDS:-2}" SELF_QUESTIONS="${SELF_QUESTIONS:-16000}" SELF_STEPS="${SELF_STEPS:-800}"
SELF_FRACTION="${SELF_FRACTION:-0.5}" QA_FRACTION="${QA_FRACTION:-0.25}"
TRAIN_ARGS="${TRAIN_ARGS:---batch-size 8 --grad-accumulation 4 --learning-rate 1e-4}"
started=$(date +%s)
stamp() { echo "== $(date '+%F %T') (+$(( ($(date +%s) - started) / 60 )) min) $*"; }

[ -s qos_ai/student_before_self.pt ] || cp qos_ai/student.pt qos_ai/student_before_self.pt
stamp "baseline"
python3 qos_ai.py evalloops --load qos_ai/student_before_self.pt --batch-size 200 || exit 1

for round in $(seq 1 "$ROUNDS"); do
  stamp "round $round: selfplay"
  python3 qos_ai.py selfplay --load qos_ai/student.pt --round "$round" --questions "$SELF_QUESTIONS" || exit 1
  stamp "round $round: teach"
  python3 qos_ai.py teach --batch-size 8 || exit 1
  stamp "round $round: train"
  # shellcheck disable=SC2086  # TRAIN_ARGS is word-split into flags on purpose
  python3 qos_ai.py train --load qos_ai/student.pt --steps "$SELF_STEPS" --self-fraction "$SELF_FRACTION" --qa-fraction "$QA_FRACTION" \
    $TRAIN_ARGS --warmup-steps 50 --report-every 100 --sample-every 400 --save-every 400 || exit 1
  cp qos_ai/student.pt "qos_ai/student_self_r$round.pt"
  stamp "round $round: evalloops"
  python3 qos_ai.py evalloops --load qos_ai/student.pt --batch-size 200 || exit 1
done

stamp "sample answers"
for question in "What is FP-RISC?" "What is Sol?" "What is a hart?" "How does the Sol runtime handle a failed transaction?" \
    "Write a Sol function that adds two numbers."; do
  echo "Q: $question"
  python3 qos_ai.py chat --max-new-tokens 120 --temperature 0.5 "$question" | tail -n +2
  echo
done
stamp "done"
