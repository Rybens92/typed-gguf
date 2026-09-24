#!/bin/sh
# Card t_7e24cea4 — the Tier-M sweep of `src/typed_gguf/keep`, retried around the shared box.
#
# Two facts about this box and mutmut 3.8 (learned on card t_57cc0179; still true here):
#
# * `max_children` in `[tool.mutmut]` is NOT read (a CLI-only option in 3.8), so `mutmut run`
#   forks `cpu_count` runners at once and the container's shared pid cgroup (256, with the sibling
#   cards' campaigns inside it) answers `os.fork()` with `EAGAIN` — observed as
#   `BlockingIOError: [Errno 11] Resource temporarily unavailable`, which kills the whole run.
#   Hence the explicit `--max-children 2`.
# * verdicts already written to `mutants/<pkg>/<file>.py.meta` persist and a fresh `mutmut run`
#   resumes from them (`exit_code_by_key` `None` = pending), so this driver re-runs until a run
#   comes back without a `BlockingIOError` instead of losing a partial sweep.
#
#   sh mutmut_sweep.sh [attempts]      (logs to logs/mutmut.out)
set -u
cd /var/home/rybens/workspace/ggufone || exit 1
ATTEMPTS=${1:-4}
LOG=.e2e/t_7e24cea4-warm-host/logs
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/work/mutmut-tmp
mkdir -p "$LOG" "$TMPDIR"
: > "$LOG/mutmut.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }

for attempt in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while [ "$(pids_now)" -gt 200 ] && [ "$waited" -lt 900 ]; do
        echo "attempt $attempt: pid cgroup at $(pids_now)/256 - waiting" >> "$LOG/mutmut.out"
        sleep 30
        waited=$((waited + 30))
    done
    echo "=== attempt $attempt - $(date -u '+%H:%M:%S') pids=$(pids_now)/256 ===" >> "$LOG/mutmut.out"
    timeout 5400 uv run --extra dev --with mutmut mutmut run --max-children 2 \
        >> "$LOG/mutmut.out" 2>&1
    rc=$?
    echo "attempt $attempt exit=$rc" >> "$LOG/mutmut.out"
    if [ "$rc" -eq 0 ] && ! tail -40 "$LOG/mutmut.out" | grep -q "BlockingIOError"; then
        break
    fi
    sleep 10
done
timeout 900 uv run --extra dev --with mutmut mutmut results >> "$LOG/mutmut.out" 2>&1
echo "=== sweep done $(date -u '+%H:%M:%S') ===" >> "$LOG/mutmut.out"
