#!/usr/bin/env bash
# Independent control for card t_97f1bc93: the *bundle's own* tool, no ggufone in the picture.
#
# The pinned Vulkan bundle ships `llama-bench`. If the teardown SIGSEGV is ggufone's, this tool
# should not show it; if the crash is the ICD's exit path, the vendor's own binary dies the same
# way on the same model and the same device.
#
#   vendor_control.sh <label> <count> [model]
set -uo pipefail
LABEL=${1:?label}
COUNT=${2:?count}
MODEL=${3:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
CARD_DIR=$(cd "$(dirname "$0")" && pwd)
BUNDLE=${BUNDLE:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
export LD_PRELOAD=${LD_PRELOAD:-$CARD_DIR/segv_bt.so}
LOGS="$CARD_DIR/logs"; mkdir -p "$LOGS"
SUMMARY="$LOGS/$LABEL.summary"; : > "$SUMMARY"

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
echo "--- $LABEL summary ---"; cat "$SUMMARY"
