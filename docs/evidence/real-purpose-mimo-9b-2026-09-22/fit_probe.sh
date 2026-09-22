#!/usr/bin/env bash
# Fit probes for the MiMo arm (card t_d199e09c AC2).
#
# Usage: fit_probe.sh <model.gguf> <label> [extra fit args...]
# Records `typed-gguf fit --print --json --no-cache` output into
# $BASE/fit/<label>.json and prints the placement fields that decide residency.
set -u
MODEL="$1"; LABEL="$2"; shift 2
BASE="${BASE:-/work/t_d199e09c/mimo}"
export TYPED_GGUF_HOME="${TYPED_GGUF_HOME:-$BASE/home}"
export TYPED_GGUF_RUNTIME_DIR="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
cd /workspace/ggufone || exit 1
mkdir -p "$BASE/fit"
out="$BASE/fit/$LABEL.json"
{
  echo "# model=$MODEL"
  echo "# cmd: uv run typed-gguf fit $MODEL --print --json --no-cache $*"
  echo "# nvidia-smi: $(nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader)"
  uv run typed-gguf fit "$MODEL" --print --json --no-cache "$@" 2>&1
  echo "# exit=$?"
} > "$out"
echo "=== $LABEL -> $out"
cat "$out"
