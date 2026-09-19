#!/bin/sh
# Card t_a696ce02 — matrix round 2: the cgroup reading the gate acts on is forced, so the rows are
# comparable on a shared box (the real cgroup is only ever "whatever the siblings are doing").
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp PYTHONPATH=/work/t_a696ce02
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
cd "$REPO" || exit 1
PY=.venv/bin/python
FIVE="tests/test_runtime_fallback.py tests/test_runtime_install.py tests/test_runtime_contract.py
      tests/test_probe_isolation.py tests/test_capability.py"

run() {   # run <label> <env VAR=val...> -- <cmd...>
    label=$1; shift
    echo "### $label pids=$(cat /sys/fs/cgroup/pids.current)" >> "$OUT/matrix2.txt"
    env "$@" >> "$OUT/matrix2.txt" 2>&1
    echo "    exit=$? pids=$(cat /sys/fs/cgroup/pids.current)" >> "$OUT/matrix2.txt"
}

: > "$OUT/matrix2.txt"
# A2. cgroup starved + every spawn denied: the marked fork gates must skip by name, 0 failures
run "A2 starved cgroup + inject=always (5 files)" \
    GGUFONE_TEST_PID_HEADROOM=250/256 GGUFONE_TEST_DENY_SPAWN=always \
    $PY -m pytest -q -p no:randomly -p inject -rs --tb=line $FIVE
# B2. healthy cgroup + a transient denial below the retry budget: green
run "B2 healthy cgroup + inject=3 (5 files)" \
    GGUFONE_TEST_PID_HEADROOM=0/256 GGUFONE_TEST_DENY_SPAWN=3 \
    $PY -m pytest -q -p no:randomly -p inject --tb=line $FIVE
# C2. healthy cgroup + a denial that outlives the budget: named pressure, never a loader lie
run "C2 healthy cgroup + inject=9 (5 files)" \
    GGUFONE_TEST_PID_HEADROOM=0/256 GGUFONE_TEST_DENY_SPAWN=9 \
    $PY -m pytest -q -p no:randomly -p inject --tb=line $FIVE
# D2. the repo's deterministic gate on this box, cgroup forced healthy
run "D2 healthy cgroup, full suite -p no:randomly" \
    GGUFONE_TEST_PID_HEADROOM=0/256 $PY -m pytest -q -p no:randomly --tb=line
echo "MATRIX2 DONE"
