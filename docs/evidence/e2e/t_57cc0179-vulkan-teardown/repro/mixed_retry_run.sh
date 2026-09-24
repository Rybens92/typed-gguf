#!/bin/sh
# Card t_57cc0179 — the live half: `bench --backend all` on the two-bundle host, repeated, until the
# isolation seam sees a Vulkan child die at teardown and this card's retry answers it.
#
# The crash is *flaky* on this box (the sibling card t_97f1bc93 measures ~2 of 6 identical runs,
# exit 139 after a complete `ok: true` report), so one run proves nothing: this driver repeats the
# documented mixed command and stops at the first run whose report shows either
# `RECOVERED_AFTER_TEARDOWN_CRASH` (the retry worked) or `W_BACKEND_CRASHED_AT_TEARDOWN` (it did
# not, and the row is withheld with the name on it).
#
#   sh mixed_retry_run.sh <tree> <prefix> [runs] [keep_free_mib|none]
#
# With a `keep_free_mib`, a dummy allocator holds the device down to that reading before each run
# (the operator's failing reading was 3272 MiB free of 8192) and is released between runs;
# `none` runs against the ambient device (the E3 campaign's own pressure).
set -u
TREE=${1:?tree}
PREFIX=${2:?prefix}
RUNS=${3:-3}
KEEP_FREE=${4:-none}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
CPU=/work/t603-runtime/b11026-linux-x64-cpu
HOST_RT=/var/home/rybens/.local/share/ggufone/runtime
VULK=$HOST_RT/b11026-linux-x64-vulkan
MODEL=${REPRO_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"
SUMMARY=$LOGS/${PREFIX}_batch.txt
: > "$SUMMARY"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
        || echo "nvidia-smi n/a"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9'; }

for index in $(seq 1 "$RUNS"); do
    tag="${PREFIX}_run${index}"
    HOG=""
    if [ "$KEEP_FREE" != "none" ]; then
        NOW=$(free_mib)
        HOLD=$((NOW - KEEP_FREE))
        [ "$HOLD" -lt 0 ] && HOLD=0
        "$HERE/vram_hog" "$HOLD" > "$LOGS/${tag}_hog.log" 2>&1 &
        HOG=$!
        for _ in $(seq 1 30); do
            NOW=$(free_mib)
            [ "${NOW:-9999}" -le "$((KEEP_FREE + 256))" ] && break
            sleep 2
        done
    fi
    smi > "$LOGS/${tag}_vram_before.txt"

    cd "$TREE" || exit 1
    (
        echo "cmd: GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT $PY -m ggufone bench" \
             "--suite throughput --model $MODEL --backend all --runs 1 --threads 4 --sizes 64 --json"
        echo "tree: $TREE ($(git rev-parse --short HEAD 2>/dev/null || echo no-git))"
        echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    ) > "$LOGS/${tag}.meta"

    : > "$LOGS/${tag}_trace.txt"
    GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT \
        PYTHONPATH=$TREE/src PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
        "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend all \
        --runs 1 --threads 4 --sizes 64 --json > "$LOGS/${tag}.raw" 2> "$LOGS/${tag}.stderr" &
    CHILD=$!
    while kill -0 "$CHILD" 2>/dev/null; do
        echo "$(date -u '+%H:%M:%S') $(smi)" >> "$LOGS/${tag}_trace.txt"
        sleep 2
    done
    wait "$CHILD"
    code=$?
    echo "$code" > "$LOGS/${tag}.exit"
    smi > "$LOGS/${tag}_vram_after.txt"

    if [ -n "$HOG" ]; then
        kill -TERM "$HOG" 2>/dev/null
        sleep 1
        kill -KILL "$HOG" 2>/dev/null
    fi

    recovered=$(grep -c "RECOVERED_AFTER_TEARDOWN_CRASH" "$LOGS/${tag}.raw" 2>/dev/null || true)
    warning=$(grep -c "W_BACKEND_CRASHED_AT_TEARDOWN" "$LOGS/${tag}.raw" 2>/dev/null || true)
    withheld=$(grep -c "ISOLATED_CHILD_FAILED" "$LOGS/${tag}.raw" 2>/dev/null || true)
    recovered=${recovered:-0}
    warning=${warning:-0}
    withheld=${withheld:-0}
    printf 'RUN %s exit=%s recovered=%s warning=%s withheld=%s vram_before=%s\n' \
        "$index" "$code" "$recovered" "$warning" "$withheld" \
        "$(cat "$LOGS/${tag}_vram_before.txt" | tr '\n' ' ')" | tee -a "$SUMMARY"
    if [ "$recovered" -gt 0 ] || [ "$warning" -gt 0 ]; then
        printf 'STOP at run %s: the crash shape appeared (recovered=%s warning=%s)\n' \
            "$index" "$recovered" "$warning" | tee -a "$SUMMARY"
        break
    fi
done
echo "--- $SUMMARY ---"
cat "$SUMMARY"
