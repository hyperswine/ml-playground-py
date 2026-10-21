#!/bin/bash
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.:$HOME/qos_ai_eval HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=$HOME/qos_ai_eval/.venv/bin/python; M=$HOME/qos_ai_eval/experiments/qwen_sft_20261022/model
D="--corpus corpus.jsonl --qa qa.jsonl --splits splits.json"
echo "== $(date) mps fp32"; $PY qos_ai_qwen_chat.py mps-check --model $M --prompts prompts.jsonl --reference tuned_answers.jsonl --n 120 --output mps_fp32.json
echo "== $(date) mps bf16"; $PY qos_ai_qwen_chat.py mps-check --model $M --dtype bfloat16 --prompts prompts.jsonl --reference tuned_answers.jsonl --n 120 --output mps_bf16.json
echo "== $(date) rag tuned"; $PY qos_ai_qwen_chat.py rag-loss --model $M $D --output rag_tuned.json
echo "== $(date) rag base"; $PY qos_ai_qwen_chat.py rag-loss --model Qwen/Qwen3.5-0.8B $D --output rag_base.json
echo "== $(date) done"
