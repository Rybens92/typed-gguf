#!/bin/sh
# Card t_d90404ac (E3d) — the Tier-M sweep of `engine/prompt.py`, retried around the shared box.
#
# Same two facts as the E2 FIX sweep driver (`.e2e/t_57cc0179-vulkan-teardown/mutmut_sweep.sh`):
#
# * mutmut 3.8 does not read `max_children` from `[tool.mutmut]` (CLI-only), so `mutmut run` forks
#   `cpu_count` runners and the container's shared pid cgroup answers `os.fork()` with `EAGAIN`
#   (`BlockingIOError`), which kills the run — hence the explicit `--max-children 2`;
# * verdicts persist in `mutants/<pkg>/<file>.meta`, and a fresh `mutmut run` resumes from them, so
#   this driver re-runs until nothing is pending instead of losing a partial sweep.
#
#   sh .e3d/mutmut_sweep.sh [attempts]      (logs to .e3d/logs/mutmut.out)
set -u
cd "$(dirname "$0")/.." || exit 126
ATTEMPTS=${1:-6}
META=mutants/src/ggufone/engine/prompt.py.meta
LOG=.e3d/logs
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$LOG"
# Append per invocation (and say which invocation): the first version truncated, which cost the
# verdict log of the run the numbers in pyproject.toml come from when a follow-up attempt followed.
echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ) — mutmut_sweep.sh $*" >> "$LOG/mutmut.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== attempt $attempt (pids $(pids_now)) ===" >> "$LOG/mutmut.out"
  while [ "$(pids_now)" -gt 175 ]; do sleep 20; done
  uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG/mutmut.out" 2>&1
  status=$?
  echo "--- attempt $attempt exit $status" >> "$LOG/mutmut.out"
  if [ -f "$META" ]; then
    pending=$("${PYTHON:-python3}" - "$META" <<'PY'
import json, sys
print(sum(1 for v in json.load(open(sys.argv[1]))["exit_code_by_key"].values() if v is None))
PY
)
    echo "--- pending after attempt $attempt: $pending" >> "$LOG/mutmut.out"
    [ "$pending" = "0" ] && break
  fi
  sleep 30
done
uv run --frozen --extra dev --with mutmut mutmut results >> "$LOG/mutmut.out" 2>&1
echo "=== sweep done ===" >> "$LOG/mutmut.out"
