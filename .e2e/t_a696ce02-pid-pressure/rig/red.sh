#!/bin/sh
# Card t_a696ce02 — RED on the landed parent tree (8474802 = the shared tree's main).
#
# RED tree = the parent product code + the new test files only. What it must show:
#   * the probe path has no retry and no E_PID_PRESSURE name (3 gates fail on the parent code);
#   * the same mislabelled failures the card was filed for, under injected spawn denial.
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp PYTHONPATH=/work/t_a696ce02
REPO=/work/t_a696ce02/repo
RED=/work/t_a696ce02/red
OUT=/work/t_a696ce02/out
mkdir -p "$OUT" || exit 1
cd "$REPO" || exit 1
if [ ! -d "$RED" ]; then
    git worktree add --detach "$RED" 8474802 || exit 2
fi
cp tests/test_probe_pressure.py tests/conftest.py "$RED/tests/" || exit 3
cp tests/test_runtime_contract.py "$RED/tests/" || exit 3
cp src/ggufone/runtime/pressure.py "$RED/src/ggufone/runtime/" || exit 3
cd "$RED" || exit 1
PY="$REPO/.venv/bin/python"
LOG="$OUT/red_parent.txt"
: > "$LOG"
echo "### tree: parent 8474802 + the new gate file (+ pressure.py, the new module)" >> "$LOG"
$PY -m pytest -q -p no:randomly --tb=line "$RED/tests/test_probe_pressure.py" >> "$LOG" 2>&1
echo "    exit=$?" >> "$LOG"
echo "### the 3 flaky files on the parent tree, every spawn denied (the filed flake)" >> "$LOG"
GGUFONE_TEST_DENY_SPAWN=always $PY -m pytest -q -p no:randomly -p inject --tb=line \
    "$RED/tests/test_runtime_fallback.py" "$RED/tests/test_runtime_install.py" \
    "$RED/tests/test_runtime_contract.py" >> "$LOG" 2>&1
echo "    exit=$?" >> "$LOG"
grep -E "^FAILED|passed|failed|Error" "$LOG" | tail -40
