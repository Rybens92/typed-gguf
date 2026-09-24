#!/bin/sh
# Card t_57cc0179 — the Tier-M sweep of `bench/isolation.py`, retried around the shared box.
#
# Two facts about this box and this mutmut version (3.8):
#
# * `max_children` in `[tool.mutmut]` is NOT read (it is a CLI-only option there — see
#   `mutmut/configuration.py`), so `mutmut run` forks `cpu_count` (= 24) runners at once and the
#   container's shared pid cgroup (256, with 24 uv-build pythons, the voice-companion stryker tree
#   and the other kanban workers inside it) answers `os.fork()` with `EAGAIN` — observed as
#   `BlockingIOError: [Errno 11] Resource temporarily unavailable`, which kills the whole run.
#   Hence the explicit `--max-children 2` below.
# * verdicts already written to `mutants/<pkg>/<file>.py.meta` persist, and a fresh `mutmut run`
#   resumes from them (`exit_code_by_key` `None` = pending), so this driver re-runs until nothing
#   is pending instead of losing a partial sweep.
#
#   sh mutmut_sweep.sh [attempts]      (logs to logs/mutmut.out)
set -u
cd /work/t57cc-ggufone || exit 1
ATTEMPTS=${1:-8}
META=mutants/src/ggufone/bench/isolation.py.meta
LOG=.e2e/t_57cc0179-vulkan-teardown/logs
PY=/work/t57cc-ggufone/.venv/bin/python
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/dev/shm
mkdir -p "$LOG"
: > "$LOG/mutmut.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }
pending() { "$PY" /work/t57cc-scratch/census.py "$META" 2>/dev/null | sed -n 's/^PENDING=//p'; }

for attempt in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while [ "$(pids_now)" -gt 200 ] && [ "$waited" -lt 600 ]; do
        echo "attempt $attempt: pid cgroup at $(pids_now)/256 — waiting" >> "$LOG/mutmut.out"
        sleep 30
        waited=$((waited + 30))
    done
    echo "=== attempt $attempt — $(date -u '+%H:%M:%S') pids=$(pids_now)/256 ===" >> "$LOG/mutmut.out"
    uv run --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG/mutmut.out" 2>&1
    echo "attempt $attempt exit=$?" >> "$LOG/mutmut.out"
    "$PY" /work/t57cc-scratch/census.py "$META" >> "$LOG/mutmut.out" 2>&1
    [ "$(pending)" = "0" ] && break
    sleep 5
done
"$PY" /work/t57cc-scratch/census.py "$META" survivors >> "$LOG/mutmut.out" 2>&1
echo "=== sweep done $(date -u '+%H:%M:%S') ===" >> "$LOG/mutmut.out"
