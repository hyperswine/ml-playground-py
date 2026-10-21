#!/bin/sh
set -eu
cd /Users/jq/qos_ai_eval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv/bin/python -m unittest test_qos_ai_data test_qos_ai_finetune test_qos_ai_compare test_qos_ai_curriculum > curricula/verified_expanded/tests.log 2>&1
.venv/bin/python qos_ai_finetune.py --load experiments/verified_sft5/student.pt --curriculum curricula/verified_expanded --data curricula/verified_expanded/verified_train_ready.jsonl --splits curricula/verified_expanded/splits.json --root fprisc=source_snapshot/fprisc --root qos=source_snapshot/qos --output experiments/verified_expanded_sft2 --epochs 2 --learning-rate 0.00003 --max-minutes 10 --device mps > curricula/verified_expanded/training.log 2>&1
.venv/bin/python qos_ai_bench.py run --load experiments/verified_expanded_sft2/student.pt --splits curricula/verified_expanded/splits.json --output reports/expanded_tuned_raw --split dev --seed 7 --temperature 0 --top-k 20 --no-repeat 0 --max-new-tokens 200 --device mps > curricula/verified_expanded/evaluation_raw.log 2>&1
.venv/bin/python qos_ai_bench.py run --load experiments/verified_expanded_sft2/student.pt --splits curricula/verified_expanded/splits.json --output reports/expanded_tuned_guard --split dev --seed 7 --temperature 0 --top-k 20 --no-repeat 3 --max-new-tokens 200 --device mps > curricula/verified_expanded/evaluation_guard.log 2>&1
.venv/bin/python qos_ai_compare_expanded.py --reports reports --benchmark qos_ai_benchmark/benchmark.jsonl --base-splits clean_benchmark_v2/splits.json --splits curricula/verified_expanded/splits.json --training experiments/verified_expanded_sft2/result.json --output reports/expanded_comparison.json > curricula/verified_expanded/comparison.log
