#!/bin/sh
# Card t_57cc0179 — the retry chain, end to end, on the documented two-bundle command.
#
#   sh fault_demo.sh <tree> <mode>
#     mode = retry-ok  -> attempt 1 dies of SIGSEGV after its report, the degraded retry answers
#     mode = always    -> both attempts die; the withheld row carries W_BACKEND_CRASHED_AT_TEARDOWN
#
# The child's fatal signal is **injected** (`repro/fault_inject/sitecustomize.py` arms an atexit
# SIGSEGV for the isolated `--backend vulkan` child only) because the natural crash is flaky; every
# other part is the real thing — the real CLI, real child processes, the real report the child
# writes, the real verification, the real degrade ladder and the real rendered table. The *natural*
# occurrence of the same shape is recorded in `.e2e/t_dd62ec29-mixed-bundle-teardown/logs/`
# (exit -11 after a complete report) and re-measured live by the sibling card `t_97f1bc93`.
set -u
TREE=${1:?tree}
MODE=${2:?mode}
: "${3:=}"                      # optional tag suffix, so the same mode can be run per tree
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
CPU=/work/t603-runtime/b11026-linux-x64-cpu
HOST_RT=/var/home/rybens/.local/share/ggufone/runtime
SMALL=${REPRO_MODEL:-/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
TAG="fault_${MODE}${3:-}"
#: The demo pins the placement to the host (`--gpu-layers 0`) so it is deterministic on a box whose
#: device is held by the E3 campaign: the child then *always* writes a complete report, which is the
#: shape's first half, and the injected signal supplies the second. Set DEMO_GPU_LAYERS=-1 to let the
#: loader pick the device placement instead.
GPU_LAYERS=${DEMO_GPU_LAYERS:-0}
#: The smallest KV rung the loader has (`--kv-type q4_0`), so the demo's child can *finish* its
#: measurement on a box whose device is held by the E3 campaign: the shape under test needs the
#: report on disk, and a child that dies of a typed OOM never writes one.
KV_TYPE=${DEMO_KV_TYPE:-q4_0}
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
        || echo "nvidia-smi n/a"; }
smi > "$LOGS/${TAG}_vram_before.txt"

cd "$TREE" || exit 1
(
    echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "tree: $TREE ($(git rev-parse --short HEAD 2>/dev/null || echo no-git))"
    echo "mode: $MODE (injected SIGSEGV at the isolated vulkan child's exit)"
    echo "cmd: PYTHONPATH=$HERE/fault_inject:$TREE/src GGUFONE_T57_FAULT=$MODE" \
         "GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT" \
         "$PY -m ggufone bench --suite throughput --model $SMALL --backend all" \
         "--gpu-layers $GPU_LAYERS --kv-type $KV_TYPE" \
         "--runs 1 --threads 4 --sizes 64 --json"
) > "$LOGS/${TAG}.meta"

env "PYTHONPATH=$HERE/fault_inject:$TREE/src" "GGUFONE_T57_FAULT=$MODE" \
    "GGUFONE_RUNTIME_DIR=$CPU" "GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT" \
    "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
    "$PY" -m ggufone bench --suite throughput --model "$SMALL" --backend all \
    "--gpu-layers" "$GPU_LAYERS" "--kv-type" "$KV_TYPE" \
    --runs 1 --threads 4 --sizes 64 --json > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr"
code=$?
echo "$code" > "$LOGS/${TAG}.exit"
smi > "$LOGS/${TAG}_vram_after.txt"

"$PY" "$HERE/../render_from_raw.py" "$LOGS/${TAG}.raw" "$LOGS/${TAG}_rendered.md" \
    > "$LOGS/${TAG}_render.log" 2>&1

echo "mode=$MODE exit=$code"
echo "recovered-notes=$(grep -c RECOVERED_AFTER_TEARDOWN_CRASH "$LOGS/${TAG}.raw" || true)"
echo "named-warning=$(grep -c W_BACKEND_CRASHED_AT_TEARDOWN "$LOGS/${TAG}.raw" || true)"
echo "withheld-rows=$(grep -c ISOLATED_CHILD_FAILED "$LOGS/${TAG}.raw" || true)"
echo "--- rendered table ---"
cat "$LOGS/${TAG}_rendered.md" 2>/dev/null
