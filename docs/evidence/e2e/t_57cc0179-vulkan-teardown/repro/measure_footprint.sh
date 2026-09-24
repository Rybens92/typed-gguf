#!/bin/sh
# Card t_57cc0179 — what does one Vulkan child actually take on the device?
#
# The starve recipe needs three numbers to be honourable — how much free memory the load needs, how
# much the child holds at its peak, and what is left at teardown. This measures them with the
# *device's* own accounting (nvidia-smi `used`, sampled while the child runs: the child's footprint
# is the difference against the process-external baseline, which is reported next to it).
#
#   sh measure_footprint.sh <tree> <tag> [model]
set -u
TREE=${1:?tree}
TAG=${2:?tag}
MODEL=${3:-${REPRO_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"

used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9'; }

# The window matters even for a *measurement*: with less free memory than the fit plan needs the
# child degrades or fails and the footprint is not a footprint. Wait for `WANT_FREE` MiB, up to
# `WAIT_MIN` minutes.
WANT_FREE=${WANT_FREE:-4600}
WAIT_MIN=${WAIT_MIN:-20}
WAITED=0
while [ "$WAITED" -lt $((WAIT_MIN * 60)) ]; do
    NOW=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9')
    echo "$(date -u '+%H:%M:%S') free=${NOW}MiB want=${WANT_FREE}MiB" >> "$LOGS/${TAG}_window.txt"
    [ "${NOW:-0}" -ge "$WANT_FREE" ] && break
    sleep 15
    WAITED=$((WAITED + 15))
done

BEFORE=$(used_mib)
echo "baseline_used=${BEFORE}MiB model=$MODEL" > "$LOGS/${TAG}_footprint.txt"

cd "$TREE" || exit 1
GGUFONE_RUNTIME_DIR=$VULK PYTHONPATH=$TREE/src "$PY" -m ggufone bench --suite throughput \
    --model "$MODEL" --backend vulkan --runs 1 --threads 4 --sizes 64 --json \
    > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr" &
CHILD=$!
PEAK=0
SAMPLES=0
while kill -0 "$CHILD" 2>/dev/null; do
    NOW=$(used_mib)
    [ "${NOW:-0}" -gt "$PEAK" ] && PEAK=$NOW
    SAMPLES=$((SAMPLES + 1))
    # the moment the child's own allocations are released (teardown) is the last sample it exists
    echo "$(date -u '+%H:%M:%S') used=${NOW}MiB" >> "$LOGS/${TAG}_footprint.txt"
    sleep 2
done
wait "$CHILD"
echo "$?" > "$LOGS/${TAG}.exit"
AFTER=$(used_mib)
{
    echo "samples=$SAMPLES peak_used=${PEAK}MiB after_used=${AFTER}MiB"
    echo "child_footprint_at_peak=$((PEAK - BEFORE))MiB exit=$(cat "$LOGS/${TAG}.exit")"
} >> "$LOGS/${TAG}_footprint.txt"
tail -4 "$LOGS/${TAG}_footprint.txt"
