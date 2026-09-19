#!/bin/sh
# Card t_57cc0179 — the integration check after the sibling card's fix (t_97f1bc93):
# a Vulkan child now ends *itself* (`runtime/teardown.end_process` → `os._exit` after flush), so the
# exit code and the report can no longer contradict each other, and this card's crash shape is not
# produced by the shipped command any more. What must still hold is the parent's side of the seam:
# the isolated child's answer is verified and **published** (no withheld row, no warning).
#
#   sh integration_check.sh <tree> <tag>
#
# `--gpu-layers 0` keeps the placement on the host so the run is deterministic on a box whose device
# is held by the E3 campaign; the bundle is still dlopened, which is the point of the check.
set -u
TREE=${1:?tree}
TAG=${2:?tag}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
CPU=/work/t603-runtime/b11026-linux-x64-cpu
HOST_RT=/var/home/rybens/.local/share/ggufone/runtime
SMALL=${REPRO_MODEL:-/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"

(
    echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "tree: $TREE ($(cd "$TREE" && git rev-parse --short HEAD 2>/dev/null || echo no-git))"
    echo "cmd: GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT" \
         "$PY -m ggufone bench --suite throughput --model $SMALL --backend all" \
         "--gpu-layers 0 --runs 1 --threads 4 --sizes 64 --json"
) > "$LOGS/${TAG}.meta"

cd "$TREE" || exit 1
env "PYTHONPATH=$TREE/src" "GGUFONE_RUNTIME_DIR=$CPU" "GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT" \
    "VK_DRIVER_FILES=$VK_DRIVER_FILES" \
    "$PY" -m ggufone bench --suite throughput --model "$SMALL" --backend all \
    --gpu-layers 0 --runs 1 --threads 4 --sizes 64 --json \
    > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr"
code=$?
echo "$code" > "$LOGS/${TAG}.exit"
"$PY" "$HERE/../render_from_raw.py" "$LOGS/${TAG}.raw" "$LOGS/${TAG}_rendered.md" \
    > "$LOGS/${TAG}_render.log" 2>&1

echo "tag=$TAG exit=$code"
echo "withheld-rows=$(grep -c ISOLATED_CHILD_FAILED "$LOGS/${TAG}.raw" || true)"
echo "teardown-warnings=$(grep -c W_BACKEND_CRASHED_AT_TEARDOWN "$LOGS/${TAG}.raw" || true)"
echo "recovered-notes=$(grep -c RECOVERED_AFTER_TEARDOWN_CRASH "$LOGS/${TAG}.raw" || true)"
echo "--- rendered table ---"
cat "$LOGS/${TAG}_rendered.md" 2>/dev/null
