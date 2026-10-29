#!/usr/bin/env bash
# Source this file before running any AMD LeRobot Notebook.
# The legacy checkout is intentional: this tutorial uses lerobot.common.

export LEROBOT_LEGACY_ROOT="${LEROBOT_LEGACY_ROOT:-/home/aup/jiahang/lerobot_10b7_legacy}"
export LEROBOT_SRC="$LEROBOT_LEGACY_ROOT"
export PYTHONPATH="$LEROBOT_LEGACY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "LEROBOT_SRC=$LEROBOT_SRC"
echo "PYTHONPATH=$PYTHONPATH"
