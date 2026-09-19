#!/bin/bash
# Tier-M mutation sweep for card t_6952f0dd, round 2.
#
# Round 1 (`.e3b/logs/mutmut_sweep.out`) scored 225 killed / 59 survived = 79.2 % against the
# *first* version of the gate file. The mutation verdicts are cached in `mutants/`, and mutmut
# only reruns a mutant when its own source changed — so after strengthening the gates (the exact
# cue line, the exact renderings, first-token-only coverage, the empty-token error) the honest
# move is a clean tree: every mutant runs against the current gates, not against stale verdicts.
# `mutants.stale/` keeps the round-1 verdicts for comparison (rename, never delete).
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/dev/shm
LOG=.e3b/logs/mutmut_final.out
: > "$LOG"
stamp=$(date -u +%m%d-%H%M)
if [ -d mutants ]; then
    mv mutants "mutants.stale-$stamp"
    echo "round-1 tree kept as mutants.stale-$stamp" >> "$LOG"
fi
echo "=== clean sweep $(date -u '+%H:%M:%S') pids=$(cat /sys/fs/cgroup/pids.current)/256 ===" >> "$LOG"
uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
uv run --frozen python tools/mutation_score.py mutants >> "$LOG" 2>&1
echo "=== done $(date -u '+%H:%M:%S') ===" >> "$LOG"
