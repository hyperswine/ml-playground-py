#!/usr/bin/env bash
# From this Mac: fetch the pod's results into qos_ai/pod/ (Q&A pairs, student checkpoint, logs).
# Safe to run while the pod is still working; qos_ai/ itself is never overwritten.
#
#   runpod/pull.sh <ip> <port> [ssh key]
#   uv run python qos_ai.py chat --load qos_ai/pod/student.pt "What is Sol?"
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { sed -n '2,7p' "$0"; exit 1; }
HOST="$1" PORT="$2" KEY="${3:-$HOME/.ssh/id_ed25519}"
REMOTE="${REMOTE_DIR:-/workspace/qos_ai_run}" DEST="${PULL_DIR:-qos_ai/pod}"
mkdir -p "$DEST"
ssh -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "root@$HOST" \
  "cd $REMOTE/qos_ai && tar czf - \$(ls qa.jsonl student.pt *.log 2>/dev/null)" | tar xzf - -C "$DEST"
ls -la "$DEST"
