#!/usr/bin/env bash
# Run the single-bundle target under gdb and capture the teardown backtrace (card t_97f1bc93).
#
#   gdb_target.sh <label> [model]
#
set -uo pipefail
LABEL=${1:?label}
MODEL=${2:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
CARD_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$CARD_DIR/../.." && pwd)
LOGS="$CARD_DIR/logs"; mkdir -p "$LOGS"
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
export GGUFONE_RUNTIME_DIR=${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}
export PYTHONFAULTHANDLER=1

exec gdb -batch -q \
  -ex "set pagination off" \
  -ex "set print frame-arguments none" \
  -ex "handle SIGSEGV stop print nopass" \
  -ex "run" \
  -ex "echo \n===== SIGNAL BACKTRACE =====\n" \
  -ex "bt 40" \
  -ex "echo \n===== SHARED LIBRARIES =====\n" \
  -ex "info sharedlibrary" \
  -ex "echo \n===== THREADS =====\n" \
  -ex "thread apply all bt 15" \
  --args "$REPO/.venv/bin/python" -m ggufone bench --suite throughput --model "$MODEL" \
      --backend vulkan --runs 1 --threads 4 --sizes 64 --json \
  > "$LOGS/$LABEL.gdb" 2>&1
