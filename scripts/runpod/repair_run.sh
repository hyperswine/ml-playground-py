#!/usr/bin/env bash
# Runs ON the pod: teacher-repaired self-play rounds, starting from qos_ai/student_before_self.pt
# (or qos_ai/student.pt if that does not exist yet).
#
#   each round:  selfplay  the student answers questions itself (round 1 reuses existing answers)
#                repair    the teacher finishes each answer from just before its first repeat (vLLM)
#                teach     the teacher scores the repaired answers
#                train     continue the student; REPAIR_FRACTION of each batch is repaired answers
#                evalloops how often the student loops / finishes on held-out questions
# Settings: ROUNDS=2 SELF_QUESTIONS=16000 SELF_STEPS=800 REPAIR_FRACTION=0.5 QA_FRACTION=0.25
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf}" PYTHONUNBUFFERED=1
export QOS_AI_TEACHER="${QOS_AI_TEACHER:-Qwen/Qwen3.5-4B}"
ROUNDS="${ROUNDS:-2}" SELF_QUESTIONS="${SELF_QUESTIONS:-16000}" SELF_STEPS="${SELF_STEPS:-800}"
REPAIR_FRACTION="${REPAIR_FRACTION:-0.5}" QA_FRACTION="${QA_FRACTION:-0.25}"
started=$(date +%s)
stamp() { echo "== $(date '+%F %T') (+$(( ($(date +%s) - started) / 60 )) min) $*"; }

[ -s qos_ai/student_before_self.pt ] || cp qos_ai/student.pt qos_ai/student_before_self.pt
cp qos_ai/student_before_self.pt qos_ai/student.pt
# keep only self-play answers from the original student (round 1); later rounds are regenerated
if [ -s qos_ai/self.jsonl ]; then
  python3 -c "import json; rows=[l for l in open('qos_ai/self.jsonl') if json.loads(l)['round']==1]; open('qos_ai/self.jsonl','w').writelines(rows); print(len(rows), 'round-1 self-play answers kept')"
fi
rm -f qos_ai/lessons/self_r00[2-9]_*.pt

stamp "baseline"
python3 qos_ai.py evalloops --load qos_ai/student.pt --batch-size 200 || exit 1

for round in $(seq 1 "$ROUNDS"); do
  stamp "round $round: selfplay"
  python3 qos_ai.py selfplay --load qos_ai/student.pt --round "$round" --questions "$SELF_QUESTIONS" || exit 1
  stamp "round $round: repair"
  .venv-vllm/bin/python qos_ai.py repair --round "$round" --source-round "$round" --questions "$SELF_QUESTIONS" || exit 1
  stamp "round $round: teach"
  python3 qos_ai.py teach --batch-size 8 || exit 1
  stamp "round $round: train"
  python3 qos_ai.py train --load qos_ai/student.pt --steps "$SELF_STEPS" --repair-fraction "$REPAIR_FRACTION" --qa-fraction "$QA_FRACTION" \
    --batch-size 8 --grad-accumulation 4 --learning-rate 1e-4 --warmup-steps 50 --report-every 100 --sample-every 400 --save-every 400 || exit 1
  cp qos_ai/student.pt "qos_ai/student_repair_r$round.pt"
  stamp "round $round: evalloops"
  python3 qos_ai.py evalloops --load qos_ai/student.pt --batch-size 200 || exit 1
done

stamp "sample answers"
for question in "What is FP-RISC?" "What is Sol?" "What is a hart?" "How does the Sol runtime handle a failed transaction?" \
    "Write a Sol function that adds two numbers."; do
  echo "Q: $question"
  python3 qos_ai.py chat --max-new-tokens 150 --temperature 0.4 "$question" | tail -n +2
  echo
done
stamp "done"
