#!/usr/bin/env bash
# Pressure driver for card t_97f1bc93: one Vulkan bundle, one real model, a starved device.
#
#   pressure.sh <label> <hog-layers> <target-model> [hog-n-ctx]
#
# Starts the hog (`hog.py`) on the pinned Vulkan bundle, waits for it to hold device memory,
# snapshots `nvidia-smi` before/after, then runs the *single-bundle* target command with
# faulthandler on. Everything lands in logs/<label>.* .
set -uo pipefail

LABEL=${1:?label}
HOG_LAYERS=${2:?hog layers (-1 = all, 0 = none)}
TARGET_MODEL=${3:?target model path}
HOG_CTX=${4:-0}

CARD_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$CARD_DIR/../.." && pwd)
PY="$REPO/.venv/bin/python"
HOG_MODEL=${HOG_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
export GGUFONE_RUNTIME_DIR=${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}
export PYTHONFAULTHANDLER=1
LOGS="$CARD_DIR/logs"
mkdir -p "$LOGS"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,utilization.gpu --format=csv,noheader; }

echo "=== $LABEL ===" | tee "$LOGS/$LABEL.meta"
echo "hog layers=$HOG_LAYERS model=$HOG_MODEL ctx=$HOG_CTX" | tee -a "$LOGS/$LABEL.meta"
echo "runtime=$GGUFONE_RUNTIME_DIR" | tee -a "$LOGS/$LABEL.meta"
echo "target: bench --suite throughput --model $TARGET_MODEL --backend vulkan --runs 1 --threads 4 --sizes 64 --json" | tee -a "$LOGS/$LABEL.meta"

smi > "$LOGS/$LABEL.vram_before_hog"
HOG_PID=""
if [ "$HOG_LAYERS" != "off" ]; then
  "$PY" "$CARD_DIR/hog.py" --model "$HOG_MODEL" --layers "$HOG_LAYERS" --n-ctx "$HOG_CTX" \
      --hold 1800 > "$LOGS/$LABEL.hog" 2>&1 &
  HOG_PID=$!
  for _ in $(seq 1 120); do
    grep -q HOG_HOLDING "$LOGS/$LABEL.hog" && break
    grep -qiE "Traceback|Error" "$LOGS/$LABEL.hog" && break
    sleep 1
  done
  echo "hog ready? $(grep -c HOG_HOLDING "$LOGS/$LABEL.hog") (pid $HOG_PID)" | tee -a "$LOGS/$LABEL.meta"
  sleep 2
fi
smi | tee "$LOGS/$LABEL.vram_before_target"

cd "$REPO"
"$PY" -m ggufone bench --suite throughput --model "$TARGET_MODEL" --backend vulkan \
    --runs 1 --threads 4 --sizes 64 --json > "$LOGS/$LABEL.raw" 2>&1
TARGET_EXIT=$?
echo "EXIT=$TARGET_EXIT" | tee "$LOGS/$LABEL.exit"
smi | tee "$LOGS/$LABEL.vram_after"

if [ -n "$HOG_PID" ]; then
  kill "$HOG_PID" 2>/dev/null
  wait "$HOG_PID" 2>/dev/null
fi
echo "done: $LABEL (target exit $TARGET_EXIT)"
exit 0
