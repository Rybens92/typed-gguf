#!/bin/sh
# Card t_57cc0179 — the minimal starved-device repro recipe.
#
#   1. a dummy allocator (`vram_hog`) holds device-local memory until the device is as full as the
#      operator's failing run (~3272 MiB free of 8192 MiB), *outside* the process under test;
#   2. one Vulkan bench child (4B model, all layers off the host) runs against that device;
#   3. the script records nvidia-smi before/after, the child's stdout/stderr/exit code, and (with
#      GDB=1) a C backtrace of the crash.
#
#   sh starve_and_run.sh <tree> <tag> [hold_mib|auto] [keep_free_mib] [GDB] [wait_minutes]
#
# `tree` is the checkout whose `src` wins through PYTHONPATH (the pre-fix tree reproduces the raw
# defect; a tree with the retry shows the recovery). `keep_free_mib` defaults to 3272 (the reading
# of `.e2e/t_dd62ec29-mixed-bundle-teardown/logs/vram_before_controls.txt`). `wait_minutes` polls
# until the *ambient* free memory leaves room for the hold (the E3 campaign takes and gives GiBs).
set -u
TREE=${1:?tree}
TAG=${2:?tag}
HOLD=${3:-auto}
KEEP_FREE=${4:-3272}
GDB_MODE=${5:-off}
WAIT_MIN=${6:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
BIG=${REPRO_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
        || echo "nvidia-smi n/a"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9'; }

smi > "$LOGS/${TAG}_vram_before.txt"
# Wait for a *window*: the campaign has to leave room for the hold, and my own hold only ever
# removes memory — it can never give the campaign's memory back.
if [ "$WAIT_MIN" -gt 0 ]; then
    WANTED=$((KEEP_FREE + 128))
    WAITED=0
    while [ "$WAITED" -lt $((WAIT_MIN * 60)) ]; do
        NOW=$(free_mib)
        echo "$(date -u '+%H:%M:%S') free=${NOW}MiB wanted=${WANTED}MiB" \
            >> "$LOGS/${TAG}_window.txt"
        [ "$NOW" -ge "$WANTED" ] && break
        sleep 20
        WAITED=$((WAITED + 20))
    done
    smi > "$LOGS/${TAG}_vram_window.txt"
fi
if [ "$HOLD" = "auto" ]; then
    HOLD=$(( $(free_mib) - KEEP_FREE ))
    [ "$HOLD" -lt 0 ] && HOLD=0
fi

# The box is shared: the E3 campaign takes and gives GiBs while this recipe runs, so the hold is
# adjusted against a *fresh* reading of the free memory until the device is at the target, or the
# loop gives up (each pass is 3 s, and the child starts immediately after the last reading).
start_hog() {
    "$HERE/vram_hog" "$1" > "$LOGS/${TAG}_hog.log" 2>&1 &
    HOG=$!
    sleep 3
}
start_hog "$HOLD"
PASS=0
while [ "$PASS" -lt 4 ]; do
    if ! kill -0 "$HOG" 2>/dev/null; then            # the hold did not fit: ask for less
        HOLD=$((HOLD - 512))
        [ "$HOLD" -lt 0 ] && HOLD=0
        start_hog "$HOLD"
        PASS=$((PASS + 1))
        continue
    fi
    FREE_NOW=$(free_mib)
    DELTA=$((FREE_NOW - KEEP_FREE))
    echo "pass=$PASS hold=$HOLD free=$FREE_NOW delta=$DELTA" >> "$LOGS/${TAG}_hog_control.txt"
    if [ "$DELTA" -le 256 ] && [ "$DELTA" -ge -256 ]; then break; fi
    kill -TERM "$HOG" 2>/dev/null
    sleep 1
    HOLD=$((HOLD + DELTA))
    [ "$HOLD" -lt 0 ] && HOLD=0
    start_hog "$HOLD"
    PASS=$((PASS + 1))
done
sleep 1
smi > "$LOGS/${TAG}_vram_starved.txt"
HOG_GONE=1
if kill -0 "$HOG" 2>/dev/null; then HOG_GONE=0; fi

cd "$TREE" || exit 1
export PYTHONPATH=$TREE/src
export PYTHONFAULTHANDLER=1
COMMAND="GGUFONE_RUNTIME_DIR=$VULK $PY -m ggufone bench --suite throughput --model $BIG \
--backend vulkan --runs 1 --threads 4 --sizes 64 --json"
case "$GDB_MODE" in
    gdb) timeout 900 gdb -batch -nx -ex "set pagination off" -ex "run" -ex "bt 40" \
             --args env "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
             "PYTHONFAULTHANDLER=1" "VK_DRIVER_FILES=$VK_DRIVER_FILES" \
             "$PY" -m ggufone bench --suite throughput --model "$BIG" --backend vulkan \
             --runs 1 --threads 4 --sizes 64 --json \
             > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr" ;;
    *)   GGUFONE_RUNTIME_DIR=$VULK "$PY" -m ggufone bench --suite throughput --model "$BIG" \
             --backend vulkan --runs 1 --threads 4 --sizes 64 --json \
             > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr" ;;
esac
echo "$?" > "$LOGS/${TAG}.exit"

kill -TERM "$HOG" 2>/dev/null
sleep 1
kill -KILL "$HOG" 2>/dev/null
smi > "$LOGS/${TAG}_vram_after.txt"

echo "tag=$TAG tree=$TREE hold=${HOLD}MiB keep_free=${KEEP_FREE}MiB hog_survived_start=$((1 - HOG_GONE))"
echo "command: $COMMAND"
echo "exit=$(cat "$LOGS/${TAG}.exit")"
echo "vram before: $(cat "$LOGS/${TAG}_vram_before.txt")"
echo "vram with hog: $(cat "$LOGS/${TAG}_vram_starved.txt")"
echo "vram after: $(cat "$LOGS/${TAG}_vram_after.txt")"
echo "hog: $(head -c 200 "$LOGS/${TAG}_hog.log" | tr '\n' ' ')"
echo "--- stderr tail ---"
tail -c 600 "$LOGS/${TAG}.stderr"
