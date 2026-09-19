#!/bin/sh
# Card t_a696ce02 — targeted re-run of the pressure.py mutants, to classify the 4 survivors
# (the full sweep's artifacts were deleted by the driver's next attempt).
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
cd "$REPO" || exit 1
LOG="$OUT/mutation_pressure.txt"
: > "$LOG"
python3 /work/t_a696ce02/retarget_pressure.py "$REPO/pyproject.toml" >> "$LOG" 2>&1 || exit 2
rm -rf mutants
uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
python3 tools/mutation_score.py mutants >> "$LOG" 2>&1
python3 /work/t_a696ce02/survivors.py mutants survived >> "$LOG" 2>&1
grep -v "⠋\|⠙\|⠹\|⠸\|⠼\|⠴\|⠦\|⠧\|⠇\|⠏" "$LOG" | tail -22
