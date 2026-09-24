#!/usr/bin/env bash
# The smallest RED control for card t_97f1bc93: the operator's command, on the parent tree, with
# no subshells in the loop (the box's pid cap is tight when sibling cards are running).
#
#   red_min.sh <label> <count>
set -uo pipefail
LABEL=${1:?label}
COUNT=${2:?count}
CARD_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$CARD_DIR/../.." && pwd)
PARENT_TREE=${PARENT_TREE:-/work/t97-red}
PY="$REPO/.venv/bin/python"
MODEL=${MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
export GGUFONE_RUNTIME_DIR=${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}
export HOME=${HOME:-/var/home/rybens}
export PYTHONPATH="$PARENT_TREE/src"
export PYTHONFAULTHANDLER=1
LOGS="$CARD_DIR/logs"; mkdir -p "$LOGS"
SUMMARY="$LOGS/$LABEL.summary"; : > "$SUMMARY"

cd "$PARENT_TREE"
for ((i = 1; i <= COUNT; i++)); do
  "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
      --runs 1 --threads 4 --sizes 64 --json > "$LOGS/$LABEL-$i.raw" 2>&1
  printf 'run=%s exit=%s\n' "$i" "$?" >> "$SUMMARY"
done
cat "$SUMMARY"
