#!/usr/bin/env bash
# Sample device memory until killed: sample_vram.sh <out-file> [interval-seconds]
OUT=${1:?out file}
INTERVAL=${2:-2}
: > "$OUT"
while true; do
  printf '%s %s\n' "$(date +%H:%M:%S)" \
    "$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader)" >> "$OUT"
  sleep "$INTERVAL"
done
