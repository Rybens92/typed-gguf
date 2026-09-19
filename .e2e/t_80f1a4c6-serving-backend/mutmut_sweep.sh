#!/bin/sh
# Card t_80f1a4c6 — the Tier-M sweep of the two engine modules the fix moves, retried around the
# shared, pid-capped box (the recipe card t_57cc0179's driver established; see pyproject.toml's
# `[tool.mutmut]` comment block for the config this reads).
#
# Two facts about this box and mutmut 3.8:
#
# * `max_children` in `[tool.mutmut]` is NOT read (CLI-only there), so `mutmut run` forks
#   `cpu_count` runners at once and the container's shared pid cgroup (256, with the sibling
#   cards' campaigns inside it) answers `os.fork()` with EAGAIN — observed as
#   `BlockingIOError: [Errno 11]`, which kills the whole run. Hence `--max-children 2`.
# * verdicts already written to `mutants/<pkg>/<file>.py.meta` persist — but this driver starts
#   from a FRESH `mutants/` (the caller removes it) so the score belongs to the frozen tree, and
#   re-runs only until nothing is pending.
#
#   sh .e2e/t_80f1a4c6-serving-backend/mutmut_sweep.sh [attempts]
set -u
cd /work/t80serve || exit 1
ATTEMPTS=${1:-8}
LOG=.e2e/t_80f1a4c6-serving-backend
PY=/work/t80serve/.venv/bin/python
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
mkdir -p "$LOG"
: > "$LOG/mutmut.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }

for attempt in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while [ "$(pids_now)" -gt 200 ] && [ "$waited" -lt 600 ]; do
        echo "attempt $attempt: pid cgroup at $(pids_now)/256 — waiting" >> "$LOG/mutmut.out"
        sleep 30
        waited=$((waited + 30))
    done
    echo "=== attempt $attempt — $(date -u '+%H:%M:%S') pids=$(pids_now)/256 ===" >> "$LOG/mutmut.out"
    uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG/mutmut.out" 2>&1
    echo "attempt $attempt exit=$?" >> "$LOG/mutmut.out"
    "$PY" tools/t80_census.py >> "$LOG/mutmut.out" 2>&1
    pending=$("$PY" tools/t80_census.py 2>/dev/null | sed -n 's/^PENDING=//p')
    [ "$pending" = "0" ] && break
    echo "attempt $attempt: $pending pending — retrying" >> "$LOG/mutmut.out"
    sleep 5
done
"$PY" tools/mutation_score.py >> "$LOG/mutation_score.txt" 2>&1
"$PY" tools/mutation_score.py | sed -n '/survivors by symbol/,$p' > "$LOG/mutation_survivors.txt" 2>&1
echo "=== sweep done $(date -u '+%H:%M:%S') ===" >> "$LOG/mutmut.out"
