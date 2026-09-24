#!/bin/sh
# Card t_57cc0179 — the *documented band* starve: hold the device at the operator's failing reading
# (~3272 MiB free of 8192) **before** the child starts, and keep holding until the child is gone.
#
# Why a new driver next to `starve_and_run.sh`:
#
#   * `starve_and_run.sh` targets the same band, but its hold is a best-effort adaptation (`auto`):
#     the three passes it needs to converge take ~10 s, and on a device held by the E3 campaign it
#     never reaches the target — the run that landed (185-298 MiB free) tested the *refusal* side
#     (`E_BACKEND_OOM`), not the documented 3272 MiB band;
#   * `starve_at_teardown.sh` starts the hold *after* the child's context marker, which starves the
#     child's own remaining allocations (`E_BACKEND_OOM` mid-run) or lands after the child is gone.
#
# This driver instead:
#
#   1. waits until the ambient free memory leaves room for the hold (the campaign gives GiBs);
#   2. starts the dummy allocator and **confirms** the band with nvidia-smi before the child starts
#      (`<tag>_vram_held.txt` shows the confirmed reading, not a hopeful one);
#   3. starts the one Vulkan child and samples the device every 2 s while it runs, so the reading
#      *at teardown* is a trace (`<tag>_trace.txt`), not a single guess;
#   4. records the child's exit code, streams and the hog's own control line.
#
#   sh hold_and_run.sh <tree> <tag> [keep_free_mib] [wait_minutes] [gdb]
set -u
TREE=${1:?tree}
TAG=${2:?tag}
KEEP_FREE=${3:-3272}
WAIT_MIN=${4:-2}
GDB_MODE=${5:-off}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
MODEL=${REPRO_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
        || echo "nvidia-smi n/a"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9'; }

smi > "$LOGS/${TAG}_vram_before.txt"

# 1. the ambient window: the hold only ever removes memory, so wait for the campaign to give some
WANTED=$((KEEP_FREE + 256))
WAITED=0
while [ "$WAITED" -lt $((WAIT_MIN * 60)) ]; do
    NOW=$(free_mib)
    echo "$(date -u '+%H:%M:%S') free=${NOW}MiB wanted=${WANTED}MiB" >> "$LOGS/${TAG}_window.txt"
    [ "${NOW:-0}" -ge "$WANTED" ] && break
    sleep 10
    WAITED=$((WAITED + 10))
done
smi > "$LOGS/${TAG}_vram_window.txt"

# 2. the hold, confirmed before the child starts
HOLD=$(( $(free_mib) - KEEP_FREE ))
[ "$HOLD" -lt 0 ] && HOLD=0
"$HERE/vram_hog" "$HOLD" > "$LOGS/${TAG}_hog.log" 2>&1 &
HOG=$!
HELD=0
for _ in $(seq 1 30); do
    NOW=$(free_mib)
    echo "$(date -u '+%H:%M:%S') hog=$HOLD free=${NOW}MiB target=${KEEP_FREE}MiB" \
        >> "$LOGS/${TAG}_hold_control.txt"
    if [ "${NOW:-9999}" -le "$((KEEP_FREE + 256))" ]; then HELD=1; break; fi
    kill -0 "$HOG" 2>/dev/null || break
    sleep 2
done
smi > "$LOGS/${TAG}_vram_held.txt"

# 3. the child, with the device sampled while it runs
cd "$TREE" || exit 1
export PYTHONPATH=$TREE/src
export PYTHONFAULTHANDLER=1
case "$GDB_MODE" in
    gdb) set -- env "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
             "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
             timeout 900 gdb -batch -nx -ex 'set pagination off' -ex run -ex 'bt 40' --args \
             "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
             --runs 1 --threads 4 --sizes 64 --json ;;
    *)   set -- env "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
             "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
             "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
             --runs 1 --threads 4 --sizes 64 --json ;;
esac

"$@" > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr" &
CHILD=$!
: > "$LOGS/${TAG}_trace.txt"
while kill -0 "$CHILD" 2>/dev/null; do
    echo "$(date -u '+%H:%M:%S') $(smi)" >> "$LOGS/${TAG}_trace.txt"
    sleep 2
done
wait "$CHILD"
echo "$?" > "$LOGS/${TAG}.exit"
if [ "$GDB_MODE" = "gdb" ]; then
    grep -c "Program received signal" "$LOGS/${TAG}.raw" > "$LOGS/${TAG}_signal_count.txt" 2>/dev/null
fi
smi > "$LOGS/${TAG}_vram_atexit.txt"

kill -TERM "$HOG" 2>/dev/null
sleep 1
kill -KILL "$HOG" 2>/dev/null
smi > "$LOGS/${TAG}_vram_after.txt"

echo "tag=$TAG tree=$TREE keep_free=${KEEP_FREE}MiB hold=${HOLD}MiB confirmed=$HELD"
echo "vram before:  $(cat "$LOGS/${TAG}_vram_before.txt")"
echo "vram window:  $(cat "$LOGS/${TAG}_vram_window.txt")"
echo "vram held:    $(cat "$LOGS/${TAG}_vram_held.txt")"
echo "vram at exit: $(cat "$LOGS/${TAG}_vram_atexit.txt")"
echo "vram after:   $(cat "$LOGS/${TAG}_vram_after.txt")"
echo "exit=$(cat "$LOGS/${TAG}.exit")"
echo "hog: $(head -c 200 "$LOGS/${TAG}_hog.log" | tr '\n' ' ')"
echo "--- stderr tail ---"
tail -c 600 "$LOGS/${TAG}.stderr"
