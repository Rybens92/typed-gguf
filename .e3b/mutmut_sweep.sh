#!/bin/bash
# E3b Tier-M mutation sweep (card t_6952f0dd), pid-cgroup aware.
#
# This box's pid cgroup is shared (256 total, with sibling workers inside it), and mutmut 3.8
# ignores `max_children` from pyproject (CLI-only) — a failed os.fork() takes the whole run down
# with `BlockingIOError: [Errno 11]`. So: wait for headroom, run with --max-children 2, and retry
# until every mutant has a verdict (mutmut resumes from mutants/<file>.py.meta).
#
#   bash .e3b/mutmut_sweep.sh [attempts]
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
ATTEMPTS=${1:-6}
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/dev/shm
LOG=.e3b/logs/mutmut_sweep.out
: > "$LOG"
pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }
for attempt in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while [ "$(pids_now)" -gt 190 ] && [ "$waited" -lt 1800 ]; do
        echo "attempt $attempt: pid cgroup at $(pids_now)/256 — waiting" >> "$LOG"
        sleep 30; waited=$((waited + 30))
    done
    echo "=== attempt $attempt $(date -u '+%H:%M:%S') pids=$(pids_now)/256 ===" >> "$LOG"
    uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG" 2>&1
    echo "attempt $attempt exit=$?" >> "$LOG"
    sleep 10
done
echo "=== sweep finished $(date -u '+%H:%M:%S') ===" >> "$LOG"
