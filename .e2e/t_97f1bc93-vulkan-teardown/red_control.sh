#!/usr/bin/env bash
# RED control for card t_97f1bc93: the operator's command against the *parent* tree (f738315),
# under the same pressure the live gate uses.
#
#   red_control.sh <label> <count> [hog-layers]
#
# `PARENT_TREE` is a `git worktree` of f738315 (no `.venv` of its own: the checkout's `src/` is put
# on `PYTHONPATH` and the card's venv interpreter runs it — the same shape the sibling cards used).
set -uo pipefail
LABEL=${1:?label}
COUNT=${2:?count}
HOG_LAYERS=${3:-12}
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

HOG_PID=""
if [ "$HOG_LAYERS" != "off" ]; then
  GGUFONE_RUNTIME_DIR="$GGUFONE_RUNTIME_DIR" nohup "$PY" "$CARD_DIR/hog.py" --model "$MODEL" \
      --layers "$HOG_LAYERS" --hold 7200 > "$LOGS/$LABEL.hog" 2>&1 &
  HOG_PID=$!
  for _ in $(seq 1 300); do
    grep -q HOG_HOLDING "$LOGS/$LABEL.hog" && break
    sleep 1
  done
fi

cd "$PARENT_TREE"
for i in $(seq 1 "$COUNT"); do
  free_before=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
  "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
      --runs 1 --threads 4 --sizes 64 --json > "$LOGS/$LABEL-$i.raw" 2>&1
  code=$?
  report=$(grep -c '"schema": "ggufone.bench/v1"' "$LOGS/$LABEL-$i.raw")
  ok_true=$(grep -c '"ok": true' "$LOGS/$LABEL-$i.raw")
  printf 'run=%s exit=%s report_lines=%s ok_true=%s vram_free_before=%s\n' \
      "$i" "$code" "$report" "$ok_true" "$free_before" | tee -a "$SUMMARY"
done

if [ -n "$HOG_PID" ]; then
  kill "$HOG_PID" 2>/dev/null
  wait "$HOG_PID" 2>/dev/null
fi
echo "--- $LABEL summary ---"; cat "$SUMMARY"
