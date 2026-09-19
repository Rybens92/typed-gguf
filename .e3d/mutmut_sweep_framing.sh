#!/bin/bash
# Card t_6de5fc53 — the Tier-M sweep of `src/ggufone/bench/harness.py` (the module the parity fix
# moves), retried around the shared box. Same two facts as the earlier drivers: mutmut 3.8 reads
# `max_children` only from the CLI (so `--max-children 2`, or a fork failure takes the run down with
# `BlockingIOError`), and verdicts persist in `mutants/<path>.meta`, so a fresh attempt resumes.
#
#   bash .e3d/mutmut_sweep_framing.sh [attempts]     (logs to .e3d/logs/mutmut_framing.out)
set -u
cd "$(dirname "$0")/.." || exit 126
ATTEMPTS=${1:-4}
META=mutants/src/ggufone/bench/harness.py.meta
LOG=.e3d/logs
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$LOG"
echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ) — mutmut_sweep_framing.sh $*" >> "$LOG/mutmut_framing.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== attempt $attempt (pids $(pids_now)) ===" >> "$LOG/mutmut_framing.out"
  while [ "$(pids_now)" -gt 175 ]; do sleep 20; done
  timeout 900 uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 \
    >> "$LOG/mutmut_framing.out" 2>&1
  status=$?
  echo "--- attempt $attempt exit $status" >> "$LOG/mutmut_framing.out"
  if [ -f "$META" ]; then
    pending=$(python3 - "$META" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(sum(1 for value in data["exit_code_by_key"].values() if value is None))
PY
)
    echo "--- pending after attempt $attempt: $pending" >> "$LOG/mutmut_framing.out"
    [ "$pending" = "0" ] && break
  fi
  sleep 20
done
uv run --frozen --extra dev --with mutmut mutmut results >> "$LOG/mutmut_framing.out" 2>&1
echo "=== sweep done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG/mutmut_framing.out"
