#!/bin/sh
# Card t_635124bf — the Tier-M sweep of `engine/cue.py`, retried around the shared box.
#
# The box facts this driver exists for (inherited from `.e2e/t_57cc0179-vulkan-teardown/mutmut_sweep.sh`):
#
# * `max_children` in `[tool.mutmut]` is NOT read by mutmut 3.8 (CLI-only option), so `mutmut run`
#   forks `cpu_count` runners and the container's shared pid cgroup answers `os.fork()` with
#   `EAGAIN` (`BlockingIOError: [Errno 11]`) — hence `--max-children 2`.
# * verdicts persist in `mutants/<pkg>/<file>.py.meta` and a fresh run resumes from them (None =
#   pending), so this driver retries until nothing is pending instead of losing a partial sweep.
#
# The previous card's verdicts for THIS file were deleted first: mutmut keys are per *function*
# (`ggufone.engine.cue.x_cue_verdict__mutmut_3`), and both mutated functions changed body, so a
# resumed run would have applied stale verdicts to renumbered mutants. A score has to be a sweep
# of this source.
#
#   sh mutmut_sweep.sh [attempts]        (logs to /work/t635/mutmut.out)
set -u
cd /workspace/ggufone || exit 1
ATTEMPTS=${1:-8}
META=mutants/src/ggufone/engine/cue.py.meta
LOG=/work/t635
PY=/workspace/ggufone/.venv/bin/python
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
rm -f "$META"
: > "$LOG/mutmut.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }
pending() { "$PY" "$LOG/census.py" "$META" 2>/dev/null | sed -n 's/.*PENDING=\([0-9]*\).*/\1/p'; }

for attempt in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while [ "$(pids_now)" -gt 190 ] && [ "$waited" -lt 900 ]; do
        echo "attempt $attempt: pid cgroup at $(pids_now) — waiting" >> "$LOG/mutmut.out"
        sleep 30
        waited=$((waited + 30))
    done
    echo "=== attempt $attempt — $(date -u '+%H:%M:%S') pids=$(pids_now) ===" >> "$LOG/mutmut.out"
    uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG/mutmut.out" 2>&1
    echo "attempt $attempt exit=$?" >> "$LOG/mutmut.out"
    [ -f "$META" ] || { echo "no meta yet" >> "$LOG/mutmut.out"; sleep 5; continue; }
    "$PY" "$LOG/census.py" "$META" >> "$LOG/mutmut.out" 2>&1
    [ "$(pending)" = "0" ] && break
    sleep 5
done
echo "=== sweep done $(date -u '+%H:%M:%S') ===" >> "$LOG/mutmut.out"
"$PY" "$LOG/census.py" "$META" survivors >> "$LOG/mutmut.out" 2>&1
"$PY" "$LOG/census.py" "$META" killed >> "$LOG/mutmut.out" 2>&1
