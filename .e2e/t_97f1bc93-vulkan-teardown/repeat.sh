#!/usr/bin/env bash
# Repeat the single-bundle target K times and record the exit codes (crash-rate evidence).
#
#   repeat.sh <label> <count> [model] [hog-layers|-] [hog-model] [hog-ctx]
#
# The hog (optional) is started once, holds device memory for the whole batch and is killed at
# the end; each repetition writes logs/<label>-<i>.{raw,exit} plus one summary line.
set -uo pipefail

LABEL=${1:?label}
COUNT=${2:?count}
MODEL=${3:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
HOG_LAYERS=${4:-off}
HOG_MODEL=${5:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
HOG_CTX=${6:-0}

CARD_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$CARD_DIR/../.." && pwd)
PY="$REPO/.venv/bin/python"
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
export GGUFONE_RUNTIME_DIR=${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}
export PYTHONFAULTHANDLER=1
LOGS="$CARD_DIR/logs"; mkdir -p "$LOGS"
SUMMARY="$LOGS/$LABEL.summary"
: > "$SUMMARY"

HOG_PID=""
if [ "$HOG_LAYERS" != "off" ]; then
  "$PY" "$CARD_DIR/hog.py" --model "$HOG_MODEL" --layers "$HOG_LAYERS" --n-ctx "$HOG_CTX" \
      --hold 7200 > "$LOGS/$LABEL.hog" 2>&1 &
  HOG_PID=$!
  for _ in $(seq 1 180); do
    grep -q HOG_HOLDING "$LOGS/$LABEL.hog" && break
    sleep 1
  done
  echo "hog: $(grep -c HOG_HOLDING "$LOGS/$LABEL.hog") (pid $HOG_PID) $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
fi

cd "$REPO"
for i in $(seq 1 "$COUNT"); do
  FREE_BEFORE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
  "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
      --runs 1 --threads 4 --sizes 64 --json > "$LOGS/$LABEL-$i.raw" 2>&1
  code=$?
  report=$(grep -c '"schema": "ggufone.bench/v1"' "$LOGS/$LABEL-$i.raw")
  ok=$(grep -c '"ok": true' "$LOGS/$LABEL-$i.raw")
  faulthandler=$(grep -c "Fatal Python error" "$LOGS/$LABEL-$i.raw")
  printf 'run=%s exit=%s report_lines=%s ok_true=%s faulthandler=%s vram_free_before=%s\n' \
      "$i" "$code" "$report" "$ok" "$faulthandler" "$FREE_BEFORE" | tee -a "$SUMMARY"
done

if [ -n "$HOG_PID" ]; then
  kill "$HOG_PID" 2>/dev/null
  wait "$HOG_PID" 2>/dev/null
fi
echo "--- $LABEL summary ---"
cat "$SUMMARY"
