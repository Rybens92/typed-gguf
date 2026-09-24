#!/bin/sh
# Card t_57cc0179 — repeat the teardown-time starve until the crash appears.
#
#   sh batch_teardown.sh <tree> <prefix> [runs] [load_free_mib] [leave_free_mib]
set -u
TREE=${1:?tree}
PREFIX=${2:?prefix}
RUNS=${3:-3}
LOAD_FREE=${4:-4200}
LEAVE_FREE=${5:-500}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
SUMMARY=$LOGS/${PREFIX}_batch.txt

: > "$SUMMARY"
for index in $(seq 1 "$RUNS"); do
    tag="${PREFIX}_run${index}"
    sh "$HERE/starve_at_teardown.sh" "$TREE" "$tag" "$LOAD_FREE" "$LEAVE_FREE" off \
        >> "$SUMMARY" 2>&1
    code=$(cat "$LOGS/${tag}.exit" 2>/dev/null || echo missing)
    starved_free=$(awk -F, '{gsub(/[^0-9]/, "", $2); print $2}' "$LOGS/${tag}_vram_starved.txt" \
        2>/dev/null || echo "?")
    reason=$(grep -o 'E_BACKEND_OOM' "$LOGS/${tag}.raw" 2>/dev/null | head -1)
    printf 'RUN %s exit=%s free_when_starved=%sMiB typed_failure=%s\n' \
        "$index" "$code" "$starved_free" "${reason:-none}" >> "$SUMMARY"
    if [ "$code" -gt 128 ] 2>/dev/null; then
        printf 'CRASH-SHAPE at run %s (exit %s)\n' "$index" "$code" >> "$SUMMARY"
        break
    fi
done
tail -20 "$SUMMARY"
