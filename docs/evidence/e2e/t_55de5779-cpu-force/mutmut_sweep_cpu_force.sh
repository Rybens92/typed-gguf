#!/bin/bash
# Card t_55de5779 — the Tier-M sweep of `src/ggufone/engine/session.py` (the module the fix's rule
# lives in: `open_model(cpu_only=…)` — the CPU-device pin, the typed refusal, the zero-layer
# executed plan and the skipped ladder walk), with the card's own pin gate plus the three fast
# bench/loader gates that build real placement dicts.
#
# Same two facts as the earlier drivers: mutmut 3.8 reads `max_children` only from the CLI (so
# `--max-children 2`, or a failed `os.fork()` takes the run down with `BlockingIOError`), and the
# verdicts persist in `mutants/<path>.meta`, so a fresh attempt resumes instead of restarting. The
# log appends per attempt (a truncating driver once threw away the verdict log of the run the
# published numbers came from).
#
#   bash .e2e/t_55de5779-cpu-force/mutmut_sweep_cpu_force.sh [attempts] [seconds-per-attempt]
set -u
cd "$(dirname "$0")/../.." || exit 126
ATTEMPTS=${1:-6}
BUDGET=${2:-900}
META=mutants/src/ggufone/engine/session.py.meta
LOG=.e2e/t_55de5779-cpu-force/logs
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$LOG"
echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ) — mutmut_sweep_cpu_force.sh $* pids=$(cat /sys/fs/cgroup/pids.current)" \
    >> "$LOG/mutmut_cpu_force.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }

pending() {
    [ -f "$META" ] || { echo unknown; return; }
    python3 - "$META" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(sum(1 for value in data["exit_code_by_key"].values() if value is None))
PY
}

for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "=== attempt $attempt (pids $(pids_now), pending $(pending)) ===" >> "$LOG/mutmut_cpu_force.out"
    while [ "$(pids_now)" -gt 175 ]; do sleep 20; done
    timeout "$BUDGET" uv run --frozen --extra dev --with mutmut \
        python tools/mutmut_driver.py run --max-children 2 >> "$LOG/mutmut_cpu_force.out" 2>&1
    status=$?
    echo "--- attempt $attempt exit $status, pending $(pending)" >> "$LOG/mutmut_cpu_force.out"
    [ "$(pending)" = "0" ] && break
    sleep 20
done
uv run --frozen --extra dev --with mutmut mutmut results >> "$LOG/mutmut_cpu_force.out" 2>&1
echo "=== sweep done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG/mutmut_cpu_force.out"
