#!/usr/bin/env bash
# Independent control for card t_97f1bc93: the *bundle's own* tool, no ggufone in the picture.
#
# The pinned Vulkan bundle ships `llama-bench`. If the teardown SIGSEGV is ggufone's, this tool
# should not show it; if the crash is the ICD's exit path under device pressure, the vendor's own
# binary dies the same way on the same model and the same device.
#
#   vendor_control.sh <label> <count> [hog-layers|off] [model]
#
# The SIGSEGV handler (`segv_bt.so`) is preloaded so a crash leaves its backtrace in the raw file
# even when the tool dies before it can print anything.
set -uo pipefail
LABEL=${1:?label}
COUNT=${2:?count}
HOG_LAYERS=${3:-off}
MODEL=${4:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
CARD_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$CARD_DIR/../.." && pwd)
PY="$REPO/.venv/bin/python"
BUNDLE=${BUNDLE:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
export LD_PRELOAD=${LD_PRELOAD:-$CARD_DIR/segv_bt.so}
LOGS="$CARD_DIR/logs"; mkdir -p "$LOGS"
SUMMARY="$LOGS/$LABEL.summary"; : > "$SUMMARY"

HOG_PID=""
if [ "$HOG_LAYERS" != "off" ]; then
  GGUFONE_RUNTIME_DIR="$BUNDLE" "$PY" "$CARD_DIR/hog.py" --model "$MODEL" --layers "$HOG_LAYERS" \
      --hold 3600 > "$LOGS/$LABEL.hog" 2>&1 &
  HOG_PID=$!
  for _ in $(seq 1 300); do
    grep -q HOG_HOLDING "$LOGS/$LABEL.hog" && break
    sleep 1
  done
fi

for i in $(seq 1 "$COUNT"); do
  free_before=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
  "$BUNDLE/llama-bench" -m "$MODEL" -ngl 99 -p 64 -n 0 -r 1 -o json \
      > "$LOGS/$LABEL-$i.raw" 2>&1
  code=$?
  bytes=$(wc -c < "$LOGS/$LABEL-$i.raw")
  segv=$(grep -c "SEGV_BT" "$LOGS/$LABEL-$i.raw")
  printf 'run=%s exit=%s bytes=%s segv_bt=%s vram_free_before=%s\n' \
      "$i" "$code" "$bytes" "$segv" "$free_before" | tee -a "$SUMMARY"
done

if [ -n "$HOG_PID" ]; then
  kill "$HOG_PID" 2>/dev/null
  wait "$HOG_PID" 2>/dev/null
fi
echo "--- $LABEL summary ---"; cat "$SUMMARY"
