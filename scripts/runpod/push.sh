#!/usr/bin/env bash
# From this Mac: copy the pipeline to a RunPod pod, install it, and start run.sh in the background.
#
#   runpod/push.sh <ip> <port> [ssh key]
#
# <ip> and <port> are on the pod's Connect tab, under "SSH over exposed TCP".  Settings for
# run.sh (ROUNDS, QUESTIONS, ASK_LIMIT, ...) are passed through from this shell, for example:
#   ROUNDS=1 ASK_LIMIT=64 runpod/push.sh 203.0.113.7 22022     # 10-minute benchmark
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { sed -n '2,9p' "$0"; exit 1; }
HOST="$1" PORT="$2" KEY="${3:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "root@$HOST")
RUN_SCRIPT="${RUN_SCRIPT:-runpod/run.sh}"           # runpod/pretrain_run.sh = general English, then the lessons
REMOTE="${REMOTE_DIR:-/workspace/qos_ai_run}"  # a separate REMOTE_DIR keeps an experiment apart from earlier runs
QA_SEED="${QA_SEED:-qos_ai/qa.jsonl}"          # QA_SEED=/dev/null starts the pod with no Q&A pairs

[ -s qos_ai/corpus.jsonl ] || uv run python qos_ai.py corpus
touch qos_ai/qa.jsonl
# Never ship secrets (runpod/API_KEY) or macOS metadata; extract without trying to keep Mac file owners.
COPYFILE_DISABLE=1 tar chzf - --no-mac-metadata --no-xattrs --exclude 'runpod/API_KEY' --exclude '*.key' --exclude '.env' \
    qos_ai.py runpod/*.sh qos_ai/corpus.jsonl \
  | "${SSH[@]}" "mkdir -p $REMOTE && tar xzf - --no-same-owner --warning=no-unknown-keyword -C $REMOTE"
# The pod's qa.jsonl grows with every run and is the expensive part: only seed it when the pod has none.
"${SSH[@]}" "test -s $REMOTE/qos_ai/qa.jsonl || cat > $REMOTE/qos_ai/qa.jsonl" < "$QA_SEED"
"${SSH[@]}" "cd $REMOTE && SKIP_VLLM=${SKIP_VLLM:-0} bash runpod/setup.sh 2>&1 | tee qos_ai/setup.log | tail -n 12"

settings=""
for name in QOS_AI_TEACHER ROUNDS FIRST_ROUND STYLE_SET QUESTIONS ASK_LIMIT TEACH_LIMIT STUDENT TRAIN_ARGS STOP_WHEN_DONE \
    PRETRAIN_TOKENS PRETRAIN_MINUTES PRETRAIN_ARGS SELF_QUESTIONS SELF_STEPS SELF_FRACTION QA_FRACTION REPAIR_FRACTION \
    APPEND_TOKENS FIRST_FILE TOTAL_TOKENS; do
  [ -n "${!name:-}" ] && settings+="$name=$(printf '%q' "${!name}") "
done
# -n and a fully detached child, so this ssh returns immediately instead of waiting on the run
ssh -n -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "root@$HOST" "cd $REMOTE && ($settings setsid nohup bash ${RUN_SCRIPT:-runpod/run.sh} > qos_ai/pod.log 2>&1 < /dev/null &) ; echo started run.sh"
echo "follow it:  ssh -p $PORT -i $KEY root@$HOST tail -f $REMOTE/qos_ai/pod.log"
echo "fetch it:   REMOTE_DIR=$REMOTE runpod/pull.sh $HOST $PORT"
