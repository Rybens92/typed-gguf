#!/bin/sh
# Card t_a696ce02 — assemble the raw material bundle in the private clone.
set -u
REPO=/work/t_a696ce02/repo
SRC=/work/t_a696ce02
B="$REPO/.e2e/t_a696ce02-pid-pressure"
mkdir -p "$B/rig" "$B/logs" || exit 1
cp "$SRC/setup.sh" "$SRC/seed_matrix.sh" "$SRC/trace.sh" "$SRC/pidtrace.py" "$SRC/inject.py" \
   "$SRC/inject_run.sh" "$SRC/matrix.sh" "$SRC/matrix2.sh" "$SRC/red.sh" "$SRC/mark.py" \
   "$SRC/retarget.py" "$SRC/sweep.sh" "$SRC/gates.sh" "$B/rig/" || exit 2
cp "$SRC/out/baseline.log" "$B/logs/baseline_seed_matrix.log" 2>/dev/null
mv "$SRC/out/baseline.log" "$SRC/out/baseline_seed_matrix.log" 2>/dev/null
cp "$SRC/out/pidtrace.log" "$B/logs/pidtrace.log" 2>/dev/null
cp "$SRC/out/pidtrace_run.txt" "$B/logs/pidtrace_run.txt" 2>/dev/null
cp "$SRC/out/matrix.txt" "$B/logs/matrix.txt" 2>/dev/null
cp "$SRC/out/matrix2.txt" "$B/logs/matrix2.txt" 2>/dev/null
cp "$SRC/out/red_parent.txt" "$B/logs/red_parent.txt" 2>/dev/null
cp "$SRC/out/ruff.txt" "$B/logs/ruff.txt" 2>/dev/null
cp "$SRC/out/coverage.txt" "$B/logs/coverage.txt" 2>/dev/null
cp "$SRC/out/inject-always.txt" "$B/logs/inject_always_3files.txt" 2>/dev/null
cp "$SRC/out/inject-always2.txt" "$B/logs/inject_always_5files.txt" 2>/dev/null
cp "$SRC/out/inject-isolation.txt" "$B/logs/inject_always_isolation_capability.txt" 2>/dev/null
cp "$SRC/out/mutation.txt" "$B/logs/mutation.txt" 2>/dev/null
cp "$SRC/out/sweep.log" "$B/logs/sweep_driver.log" 2>/dev/null
ls -la "$B/logs" | head -20
echo "--- rig ---"
ls "$B/rig"
