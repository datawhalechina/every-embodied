#!/usr/bin/env bash
set -euo pipefail

# Run native HIP conditional nodes and the eager fallback under one workload.
# The runtime must be built from the HIP/CLR patches before this script runs.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV="${ENV:-/home/aup/envs/mujoco-warp-amd-py312}"
MODEL="${MODEL:-$ROOT/upstream/mujoco_warp/mujoco_warp/test_data/humanoid/humanoid.xml}"
WORLDS="${WORLDS:-1024}"
STEPS="${STEPS:-1000}"
OUT="${OUT:-$ROOT/results/native_mjwarp_amd395}"
HIP_CLR_LIB="${HIP_CLR_LIB:-}"

# Accept either the runtime directory or the full libamdhip64.so path.
if [[ -n "$HIP_CLR_LIB" && -f "$HIP_CLR_LIB" ]]; then
  HIP_CLR_DIR="$(dirname "$HIP_CLR_LIB")"
else
  HIP_CLR_DIR="$HIP_CLR_LIB"
fi

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/upstream/warp:$ROOT/upstream/mujoco_warp"
export LD_LIBRARY_PATH="${HIP_CLR_DIR:+$HIP_CLR_DIR:}$ROOT/upstream/warp/warp/bin:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export MJW_HIP_SINGLE_STREAM=1
export WP_HIP_GRAPH_ENABLE=0
export WP_HIP_CONDITIONAL_EMULATION=0
export WP_HIP_CONDITIONAL_MAX_ITERS=100

run_variant() {
  local variant="$1"
  local native_flag=0
  [[ "$variant" == native ]] && native_flag=1
  WP_HIP_CONDITIONAL_NATIVE="$native_flag" \
    "$ENV/bin/python" -m mujoco_warp.testspeed "$MODEL" \
    --function=step --nworld="$WORLDS" --nstep="$STEPS" \
    --nconmax=128 --njmax=128 --device=cuda:0 --format=json \
    > "$OUT/${variant}.log" 2>&1
}

run_variant native
run_variant eager

WP_HIP_CONDITIONAL_NATIVE=1 "$ENV/bin/python" "$ROOT/scripts/run_mjwarp_variant.py" \
  "$MODEL" --variant=native --worlds=256 --steps=100 --output "$OUT/native_state.json"
WP_HIP_CONDITIONAL_NATIVE=0 "$ENV/bin/python" "$ROOT/scripts/run_mjwarp_variant.py" \
  "$MODEL" --variant=eager --worlds=256 --steps=100 --output "$OUT/eager_state.json"

"$ENV/bin/python" "$ROOT/scripts/compare_native_mjwarp.py" \
  --native-state "$OUT/native_state.json" --eager-state "$OUT/eager_state.json" \
  --native-benchmark "$OUT/native.log" --eager-benchmark "$OUT/eager.log" \
  --output "$OUT/summary.json"

sha256sum "$OUT"/*.json "$OUT"/*.log > "$OUT/SHA256SUMS"
printf 'Wrote %s\n' "$OUT/summary.json"
