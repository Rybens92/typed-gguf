#!/bin/sh
# Card t_a696ce02 — seed matrix on the FLAKY runtime-probe files (pytest-randomly ordering).
#
# usage: sh /work/t_a696ce02/seed_matrix.sh <label> <seed-list>
# Writes one line per run: label seed pids_before pids_after result
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
mkdir -p "$OUT"
LABEL=${1:-matrix}
SEEDS=${2:-"1 2 3 4 5"}
FILES="tests/test_runtime_fallback.py tests/test_runtime_install.py tests/test_runtime_contract.py"
LOG="$OUT/${LABEL}.log"
: > "$LOG"
cd "$REPO" || exit 1
for seed in $SEEDS; do
    before=$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo '?')
    summary=$(timeout 600 .venv/bin/python -m pytest -q --tb=line --randomly-seed="$seed" $FILES 2>&1 | tail -25)
    code=$?
    after=$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo '?')
    line=$(printf '%s' "$summary" | tail -1)
    echo "=== $LABEL seed=$seed pids=$before->$after exit=$code" >> "$LOG"
    printf '%s\n' "$summary" >> "$LOG"
    echo "-----"
    echo "$LABEL seed=$seed pids=$before->$after exit=$code :: $line"
done
