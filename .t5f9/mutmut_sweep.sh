#!/bin/sh
# Card t_5f9c15fe — the Tier-M sweep of the rename, run against the *same* pair the tree was
# already configured for (`tools/e3e_roles_decision.py` + tests/test_e3e_roles_decision.py), so the
# score can be compared with the pre-rename numbers the E3e audit card (t_c8c76cc8) recorded:
# 1941 mutants, round 1 = 1404 killed / 537 survived (72.3 %), round 2 = 1412 killed / 529 (72.7 %).
#
# Box facts (mutmut 3.8, shared pid cgroup at 256): `max_children` in `[tool.mutmut]` is CLI-only,
# so a bare `mutmut run` forks `cpu_count` runners and the cgroup answers EAGAIN — hence
# `--max-children 2`, the pid wait, and the retry loop around `BlockingIOError`.
set -u
cd /workspace/ggufone || exit 1
ATTEMPTS=${1:-6}
LOG=/work/t5f9/logs
PY=/workspace/ggufone/.venv/bin/python
CENSUS=/work/t57cc-scratch/census.py
META=mutants/tools/e3e_roles_decision.py.meta
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/dev/shm
mkdir -p "$LOG"
: > "$LOG/mutmut.out"

pids_now() { cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0; }
pending() { "$PY" "$CENSUS" "$META" 2>/dev/null | sed -n 's/^PENDING=//p'; }

# stale tree from the pre-rename sweep: move it aside so the verdicts are this tree's
[ -d mutants ] && mv mutants "mutants.stale-t5f9" 2>/dev/null

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
    "$PY" "$CENSUS" "$META" >> "$LOG/mutmut.out" 2>&1
    left=$(pending)
    echo "attempt $attempt pending=$left" >> "$LOG/mutmut.out"
    [ "${left:-1}" = "0" ] && break
done
"$PY" "$CENSUS" "$META" /work/t5f9/logs/mutmut_survivors.txt >> "$LOG/mutmut.out" 2>&1
echo "=== done — $(date -u '+%H:%M:%S') ===" >> "$LOG/mutmut.out"
