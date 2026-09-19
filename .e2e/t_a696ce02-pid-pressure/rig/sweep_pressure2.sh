#!/bin/sh
# Card t_a696ce02 — the post-survivor-fix sweep, but only on a box that can fork.
#
# The gate this card adds forces a non-zero exit when a starved cgroup skips fork gates — which
# would make *every* mutant look killed to mutmut. So this waits for real headroom first.
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
LOG="$OUT/mutation_pressure2.txt"
cd "$REPO" || exit 1

waited=0
while [ "$(cat /sys/fs/cgroup/pids.current)" -gt 130 ]; do
    if [ "$waited" -ge 1800 ]; then
        echo "no headroom after ${waited}s — sweep skipped (box starved: pids=$(cat /sys/fs/cgroup/pids.current))" | tee "$LOG"
        exit 3
    fi
    sleep 30
    waited=$((waited + 30))
done
echo "=== sweep 2 (post-killers) pids=$(cat /sys/fs/cgroup/pids.current)" > "$LOG"

python3 /work/t_a696ce02/retarget_pressure.py "$REPO/pyproject.toml" >> "$LOG" 2>&1 || exit 2
rm -rf mutants
uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
echo "pids after=$(cat /sys/fs/cgroup/pids.current)" >> "$LOG"
python3 tools/mutation_score.py mutants >> "$LOG" 2>&1
python3 /work/t_a696ce02/survivors.py mutants survived >> "$LOG" 2>&1
python3 /work/t_a696ce02/survivors_diff.py mutants/src/ggufone/runtime/pressure.py >> "$LOG" 2>&1
grep -v "⠋\|⠙\|⠹\|⠸\|⠼\|⠴\|⠦\|⠧\|⠇\|⠏" "$LOG" | tail -20
