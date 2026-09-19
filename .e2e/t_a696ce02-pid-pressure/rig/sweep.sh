#!/bin/sh
# Card t_a696ce02 — Tier-M sweep of the two runtime modules the fix moves.
# Retargets [tool.mutmut] in the PRIVATE CLONE only (the shared tree's copy is a live sibling's
# WIP): source_paths -> pressure.py + isolated.py, selection -> the pressure/probe gate files.
# Waits for pid headroom before forking mutmut (the shared 256-pid cgroup kills it otherwise).
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
cd "$REPO" || exit 1
LOG="$OUT/mutation.txt"
: > "$LOG"

python3 /work/t_a696ce02/retarget.py "$REPO/pyproject.toml" >> "$LOG" 2>&1 || exit 2
grep -n "source_paths\|pytest_add_cli_args_test_selection" pyproject.toml | head -4 >> "$LOG"

wait_for_room() {
    waited=0
    while [ "$(cat /sys/fs/cgroup/pids.current)" -gt 150 ] && [ "$waited" -lt 900 ]; do
        sleep 30
        waited=$((waited + 30))
    done
}

for attempt in 1 2 3; do
    rm -rf mutants
    wait_for_room
    echo "=== attempt $attempt pids=$(cat /sys/fs/cgroup/pids.current)" >> "$LOG"
    uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2 >> "$LOG" 2>&1
    echo "attempt $attempt exit=$?" >> "$LOG"
    python3 tools/mutation_score.py mutants >> "$LOG" 2>&1
    if grep -qE "not-checked *: *0\b" "$LOG"; then break; fi
done
tail -30 "$LOG"
