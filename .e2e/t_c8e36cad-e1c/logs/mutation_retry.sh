#!/usr/bin/env bash
# Resumable mutmut sweep for the E1c tree (Tier M).
#
# This container shares a cgroup with sibling workloads: `pids.max` is 256 and `os.fork()` in
# mutmut's isolation worker dies with BlockingIOError when a sibling eats the budget. mutmut
# saves every verdict to `mutants/<file>.meta` as it goes, and a run started with NO mutant names
# resumes from that cache, so the fix is a retry loop that only retries on that error.
#
# Usage: bash .e2e/t_c8e36cad-e1c/logs/mutation_retry.sh [attempts]
set -u
attempts="${1:-8}"
cd /workspace/ggufone || exit 1
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache
LOG=.e2e/t_c8e36cad-e1c/logs/mutation_r2.log
TMP=.e2e/t_c8e36cad-e1c/logs/.mutation_attempt.log
for attempt in $(seq 1 "$attempts"); do
  stamp=$(date -u +%H:%M:%S)
  printf '\n=== attempt %s/%s start %s ===\n' "$attempt" "$attempts" "$stamp" >> "$LOG"
  uv run --extra dev --with 'mutmut==3.8' python tools/mutmut_driver.py run --max-children 2 \
     > "$TMP" 2>&1
  code=$?
  cat "$TMP" >> "$LOG"
  printf '=== attempt %s/%s exit=%s at %s ===\n' "$attempt" "$attempts" "$code" \
     "$(date -u +%H:%M:%S)" >> "$LOG"
  if [ "$code" -eq 0 ]; then
    echo "sweep finished (exit 0) on attempt $attempt" >> "$LOG"
    break
  fi
  if ! grep -q 'BlockingIOError' "$TMP"; then
    echo "sweep stopped with a non-fork error on attempt $attempt" >> "$LOG"
    break
  fi
  sleep 30
done
