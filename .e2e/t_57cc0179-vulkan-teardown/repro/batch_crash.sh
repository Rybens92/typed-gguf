#!/bin/sh
# Card t_57cc0179 — run the starved-device recipe until the crash appears (or the budget runs out).
#
# The band that produces the defect is narrow: too little free memory and the child's own fit
# ladder refuses the load with a typed `E_BACKEND_OOM` (no crash — see `pre_starved.raw`); too much
# and the teardown has nothing to fail on. This driver waits for the band, runs one child, records
# the exit code, and stops at the first **signal** (a shell exit code over 128).
#
#   sh batch_crash.sh <tree> <prefix> [runs] [keep_free_mib]
set -u
TREE=${1:?tree}
PREFIX=${2:?prefix}
RUNS=${3:-4}
KEEP_FREE=${4:-3272}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
SUMMARY=$LOGS/${PREFIX}_batch.txt

: > "$SUMMARY"
for index in $(seq 1 "$RUNS"); do
    tag="${PREFIX}_run${index}"
    sh "$HERE/starve_and_run.sh" "$TREE" "$tag" auto "$((KEEP_FREE - 128))" off 12 \
        >> "$SUMMARY" 2>&1
    code=$(cat "$LOGS/${tag}.exit" 2>/dev/null || echo missing)
    starved_free=$(awk -F, '{gsub(/[^0-9]/, "", $2); print $2}' "$LOGS/${tag}_vram_starved.txt" \
        2>/dev/null || echo "?")
    printf 'RUN %s exit=%s free_at_start=%sMiB\n' "$index" "$code" "$starved_free" >> "$SUMMARY"
    if [ "$code" -gt 128 ] 2>/dev/null; then
        printf 'CRASH-SHAPE at run %s (exit %s)\n' "$index" "$code" >> "$SUMMARY"
        break
    fi
done
tail -30 "$SUMMARY"
