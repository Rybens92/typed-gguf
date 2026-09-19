#!/bin/sh
# Card t_57cc0179 — the *rate* of the teardown crash, with the device held in the documented band.
#
# The teardown SIGSEGV is flaky (the sibling card t_97f1bc93 measures ~1 in 4 identical runs on this
# box). A single run proves nothing, so this driver repeats the documented single-bundle Vulkan
# child and stops at the first fatal signal, recording for every attempt:
#
#   * the free device memory *before* the child (the band this attempt ran in),
#   * the child's own stdout/stderr/exit code,
#   * whether its report was complete (`"ok": true` present) when the signal arrived.
#
# With a `keep_free_mib` the device is first held **down to that reading** with the dummy allocator
# (`vram_hog`, pressure from outside the process under test) and the hold is confirmed with
# nvidia-smi before the child starts; `none` runs against the ambient device. The hog is released
# between attempts.
#
#   sh crashrate.sh <tree> <prefix> [runs] [keep_free_mib|none] [wait_min] [gdb]
set -u
TREE=${1:?tree}
PREFIX=${2:?prefix}
RUNS=${3:-4}
KEEP_FREE=${4:-none}
WAIT_MIN=${5:-0}
GDB_MODE=${6:-off}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
MODEL=${REPRO_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"
SUMMARY=$LOGS/${PREFIX}_rate.txt
: > "$SUMMARY"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
        || echo "nvidia-smi n/a"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9'; }
count() { grep -c "$1" "$2" 2>/dev/null || true; }

# one window wait for the whole batch: the hold only ever removes memory, and the E3 campaign gives
# it back in steps, so the batch starts at the first reading that leaves room for the hold.
if [ "$KEEP_FREE" != "none" ] && [ "$WAIT_MIN" -gt 0 ]; then
    WANTED=$((KEEP_FREE + 256))
    WAITED=0
    while [ "$WAITED" -lt $((WAIT_MIN * 60)) ]; do
        NOW=$(free_mib)
        echo "$(date -u '+%H:%M:%S') free=${NOW}MiB wanted=${WANTED}MiB" \
            >> "$LOGS/${PREFIX}_window.txt"
        [ "${NOW:-0}" -ge "$WANTED" ] && break
        sleep 15
        WAITED=$((WAITED + 15))
    done
    smi > "$LOGS/${PREFIX}_vram_window.txt"
fi

cd "$TREE" || exit 1
for index in $(seq 1 "$RUNS"); do
    tag="${PREFIX}_run${index}"
    HOG=""
    if [ "$KEEP_FREE" != "none" ]; then
        HOLD=$(( $(free_mib) - KEEP_FREE ))
        [ "$HOLD" -lt 0 ] && HOLD=0
        "$HERE/vram_hog" "$HOLD" > "$LOGS/${tag}_hog.log" 2>&1 &
        HOG=$!
        CONFIRMED=0
        for _ in $(seq 1 30); do
            NOW=$(free_mib)
            [ "${NOW:-9999}" -le "$((KEEP_FREE + 256))" ] && CONFIRMED=1 && break
            kill -0 "$HOG" 2>/dev/null || break
            sleep 2
        done
        echo "attempt=$index hold=${HOLD}MiB confirmed=${CONFIRMED}" \
            >> "$LOGS/${tag}_hold_control.txt"
    fi
    smi > "$LOGS/${tag}_vram_before.txt"

    case "$GDB_MODE" in
        gdb) timeout 900 gdb -batch -nx -ex 'set pagination off' -ex run -ex 'bt 40' --args env \
                 "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
                 "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
                 "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
                 --runs 1 --threads 4 --sizes 64 --json \
                 > "$LOGS/${tag}.raw" 2> "$LOGS/${tag}.stderr" ;;
        *)   env "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
                 "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
                 "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
                 --runs 1 --threads 4 --sizes 64 --json \
                 > "$LOGS/${tag}.raw" 2> "$LOGS/${tag}.stderr" ;;
    esac
    code=$?
    smi > "$LOGS/${tag}_vram_after.txt"
    [ -n "$HOG" ] && kill -TERM "$HOG" 2>/dev/null
    [ -n "$HOG" ] && sleep 1 && kill -KILL "$HOG" 2>/dev/null
    ok=$(count '"ok": true' "$LOGS/${tag}.raw")
    printf 'run=%s exit=%s report_ok_true=%s vram_before=%s\n' \
        "$index" "$code" "$ok" "$(cat "$LOGS/${tag}_vram_before.txt" | tr '\n' ' ')" \
        | tee -a "$SUMMARY"
    if [ "$code" -gt 128 ] 2>/dev/null; then
        printf 'CRASH at run %s: exit=%s (a signal) — raw kept: %s\n' "$index" "$code" "$tag" \
            | tee -a "$SUMMARY"
        break
    fi
done
echo "--- $SUMMARY ---"
cat "$SUMMARY"
