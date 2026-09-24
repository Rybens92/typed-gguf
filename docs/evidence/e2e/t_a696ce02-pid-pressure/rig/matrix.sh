#!/bin/sh
# Card t_a696ce02 — the run matrix: quiet box vs starved box (real cgroup + injected spawn denial).
# usage: sh /work/t_a696ce02/matrix.sh
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp PYTHONPATH=/work/t_a696ce02
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
mkdir -p "$OUT" || exit 1
cd "$REPO" || exit 1
PY=.venv/bin/python
FILES="tests/test_runtime_fallback.py tests/test_runtime_install.py tests/test_runtime_contract.py"
FIVE="$FILES tests/test_probe_isolation.py tests/test_capability.py"

run() {   # run <label> <env VAR=val...> -- <cmd...>
    label=$1; shift
    echo "### $label pids=$(cat /sys/fs/cgroup/pids.current)" >> "$OUT/matrix.txt"
    env "$@" >> "$OUT/matrix.txt" 2>&1
    echo "    exit=$? pids=$(cat /sys/fs/cgroup/pids.current)" >> "$OUT/matrix.txt"
}

: > "$OUT/matrix.txt"
# A. every spawn denied -> every fork gate must skip by name, nothing may fail
run "A inject=always (5 files)" GGUFONE_TEST_DENY_SPAWN=always \
    $PY -m pytest -q -p no:randomly -p inject --tb=line $FIVE
# B. a transient denial below the retry budget -> the retry must absorb it (green)
run "B inject=3 (transient, 5 files)" GGUFONE_TEST_DENY_SPAWN=3 \
    $PY -m pytest -q -p no:randomly -p inject --tb=line $FIVE
# C. a denial that outlives the budget at the first sites -> named pressure, never a loader lie
run "C inject=9 (5 files)" GGUFONE_TEST_DENY_SPAWN=9 \
    $PY -m pytest -q -p no:randomly -p inject --tb=line $FIVE
# D. the repo's deterministic gate on the real box
run "D full suite -p no:randomly" \
    $PY -m pytest -q -p no:randomly --tb=line
# E. the repo's default gate: pytest-randomly, fresh order each time
run "E full suite seed=11" $PY -m pytest -q --randomly-seed=11 --tb=line
run "E full suite seed=12" $PY -m pytest -q --randomly-seed=12 --tb=line
run "E full suite seed=13" $PY -m pytest -q --randomly-seed=13 --tb=line
echo "MATRIX DONE"
