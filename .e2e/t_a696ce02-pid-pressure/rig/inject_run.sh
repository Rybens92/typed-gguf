#!/bin/sh
# Card t_a696ce02 — deterministic fork-denial runs (the "starved box" column).
# usage: sh /work/t_a696ce02/inject_run.sh <label> <mode: N|always> [pytest args...]
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp PYTHONPATH=/work/t_a696ce02
cd /work/t_a696ce02/repo || exit 1
OUT=/work/t_a696ce02/out
mkdir -p "$OUT"
LABEL=${1:-inject}; MODE=${2:-always}; shift 2
echo "pids before=$(cat /sys/fs/cgroup/pids.current)" > "$OUT/$LABEL.txt"
GGUFONE_TEST_DENY_SPAWN="$MODE" timeout 900 .venv/bin/python -m pytest -q -p no:randomly -p inject \
    --tb=line "$@" >> "$OUT/$LABEL.txt" 2>&1
echo "exit=$?" >> "$OUT/$LABEL.txt"
echo "pids after=$(cat /sys/fs/cgroup/pids.current)" >> "$OUT/$LABEL.txt"
cat "$OUT/$LABEL.txt"
