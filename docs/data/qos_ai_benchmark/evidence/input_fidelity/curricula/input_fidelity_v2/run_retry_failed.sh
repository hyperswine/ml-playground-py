#!/bin/sh
set -eu
cd /Users/jq/qos_ai_eval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
train() {
.venv/bin/python qos_ai_finetune.py --load experiments/verified_expanded_sft2_retry/student.pt --curriculum curricula/input_fidelity_v2 --selection curricula/input_fidelity_v2/training_selection.json --data curricula/input_fidelity_v2/focused_train_ready.jsonl --splits curricula/input_fidelity_v2/splits.json --root fprisc=source_snapshot/fprisc --root qos=source_snapshot/qos --output experiments/input_fidelity_sft2_retry --fixed-padding --epochs 2 --learning-rate 0.00003 --max-minutes 10 --pad-multiple 32 --mps-cache-interval 20 --device mps "$@"
}
.venv/bin/python -m unittest test_qos_ai_data test_qos_ai_finetune test_qos_ai_compare test_qos_ai_curriculum test_qos_ai_inputs > curricula/input_fidelity_v2/tests.log 2>&1
train --dry-run > curricula/input_fidelity_v2/preflight.json
.venv/bin/python -c 'import json; from pathlib import Path; p=json.loads(Path("curricula/input_fidelity_v2/preflight.json").read_text()); assert p["unknown_answer_tokens_skipped"]==0; assert p["examples"]==598'
train > curricula/input_fidelity_v2/training.log 2>&1
.venv/bin/python qos_ai_bench.py run --load experiments/input_fidelity_sft2_retry/student.pt --splits curricula/input_fidelity_v2/splits.json --output reports/inputs_tuned_raw --split dev --seed 7 --temperature 0 --top-k 20 --no-repeat 0 --max-new-tokens 200 --device mps > curricula/input_fidelity_v2/evaluation_raw.log 2>&1
.venv/bin/python qos_ai_bench.py run --load experiments/input_fidelity_sft2_retry/student.pt --splits curricula/input_fidelity_v2/splits.json --output reports/inputs_tuned_guard --split dev --seed 7 --temperature 0 --top-k 20 --no-repeat 3 --max-new-tokens 200 --device mps > curricula/input_fidelity_v2/evaluation_guard.log 2>&1
.venv/bin/python qos_ai_compare_followup.py --reports reports --base-splits curricula/verified_expanded/splits.json --splits curricula/input_fidelity_v2/splits.json --training experiments/input_fidelity_sft2_retry/result.json --parent-stem expanded_tuned --tuned-stem inputs_tuned --prefix inputs --output reports/inputs_comparison.json > curricula/input_fidelity_v2/comparison.log
