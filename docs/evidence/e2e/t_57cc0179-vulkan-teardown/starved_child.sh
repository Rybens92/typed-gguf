#!/bin/sh
# Card t_57cc0179 — the operator-box control: one Vulkan child, one bundle, a memory-starved
# device (the E3 campaign holds the rest). This is the *pre-fix* tree (`/work/t57cc-red`, HEAD
# f738315), so it is the raw defect the card is about: a complete report, then a fatal signal.
#
#   sh .e2e/t_57cc0179-vulkan-teardown/starved_child.sh <tree> <tag>
#
# `tree` is the checkout to run (its `src` wins through PYTHONPATH); `tag` names the artefacts.
set -u
TREE=${1:-/work/t57cc-red}
TAG=${2:-pre}
HERE=$(cd "$(dirname "$0")" && pwd)
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
BIG=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf
PY=/work/t57cc-ggufone/.venv/bin/python

export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
export PYTHONPATH=$TREE/src
export PYTHONFAULTHANDLER=1

(nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
    || echo "nvidia-smi n/a") > "$HERE/logs/${TAG}_vram_before.txt" 2>&1

GGUFONE_RUNTIME_DIR=$VULK "$PY" -m ggufone bench --suite throughput --model "$BIG" \
    --backend vulkan --runs 1 --threads 4 --sizes 64 --json \
    > "$HERE/logs/${TAG}.raw" 2> "$HERE/logs/${TAG}.stderr"
echo "$?" > "$HERE/logs/${TAG}.exit"

(nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
    || echo "nvidia-smi n/a") > "$HERE/logs/${TAG}_vram_after.txt" 2>&1

printf '%s exit=%s\n' "$TAG" "$(cat "$HERE/logs/${TAG}.exit")"
printf 'vram before: %s\n' "$(cat "$HERE/logs/${TAG}_vram_before.txt")"
printf 'vram after:  %s\n' "$(cat "$HERE/logs/${TAG}_vram_after.txt")"
printf 'stderr tail:\n'
tail -c 400 "$HERE/logs/${TAG}.stderr"
