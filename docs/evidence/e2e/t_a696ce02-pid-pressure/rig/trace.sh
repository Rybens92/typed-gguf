#!/bin/sh
# Card t_a696ce02 — pids.current per test, deterministic order (measurement only).
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp PYTHONPATH=/work/t_a696ce02
cd /work/t_a696ce02/repo || exit 1
OUT=/work/t_a696ce02/out
mkdir -p "$OUT"
: > "$OUT/pidtrace.log"
: > "$OUT/pidtrace_run.txt"
timeout 900 .venv/bin/python -m pytest -q --tb=line -p no:randomly -p pidtrace "$@" \
    >> "$OUT/pidtrace_run.txt" 2>&1
echo "exit=$? pids=$(cat /sys/fs/cgroup/pids.current)" >> "$OUT/pidtrace_run.txt"
tail -3 "$OUT/pidtrace_run.txt"
