#!/usr/bin/env bash
set -euo pipefail

# Run the end-to-end tutorial notebooks with real training/evaluation outputs.
#
# Required environment variables:
#   PROJECT_ROOT   Path to 04mujoco ACT/Pi0/SmolVLA runnable project
#   DATA_ROOT      Dataset root used by the notebooks
#   MODEL_ROOT     Checkpoint/model root used by the notebooks
#   OUTPUT_ROOT    Output root for configs, logs, checkpoints, eval JSONL
#
# Common switches:
#   RUN_SMOKE=1
#   RUN_LONG_TRAIN=1
#   RUN_EVAL=1
#   RENDER_EVAL=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_ROOT}/eval_policy_success.py}"

mkdir -p "${OUTPUT_ROOT}"

echo "TOPIC_ROOT=${TOPIC_ROOT}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "RUN_SMOKE=${RUN_SMOKE:-0}"
echo "RUN_LONG_TRAIN=${RUN_LONG_TRAIN:-0}"
echo "RUN_EVAL=${RUN_EVAL:-0}"

cd "${TOPIC_ROOT}"

if [ "$#" -gt 0 ]; then
  notebooks=("$@")
else
  notebooks=(
    "notebooks/14_smolvla_end_to_end.ipynb"
    "notebooks/15_pi0_end_to_end.ipynb"
    "notebooks/16_act_end_to_end.ipynb"
  )
fi

"${PYTHON_BIN}" code/execute_tutorial_notebooks.py "${notebooks[@]}"
