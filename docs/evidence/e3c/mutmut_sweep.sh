#!/bin/bash
# Tier-M mutation sweep for card t_6c119626 (E3c): the cue verdict module.
#
# The pair in pyproject.toml targets `src/ggufone/engine/cue.py` with the card's two gate files.
# mutmut 3.8 does not read `max_children` from the config (CLI-only), and on this box the flag
# needs a retry driver: a failed `os.fork()` takes the whole run down with `BlockingIOError`
# (the cgroup pids cap is shared with sibling sessions). A previous `mutants/` tree is renamed,
# never deleted, so the verdicts stay comparable.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp OMP_NUM_THREADS=2
LOG=.e3c/logs/mutmut_cue.out
: > "$LOG"
stamp=$(date -u +%m%d-%H%M)
if [ -d mutants ]; then
    mv mutants "mutants.stale-$stamp"
    echo "previous tree kept as mutants.stale-$stamp" >> "$LOG"
fi
code=1
for attempt in 1 2 3; do
    while [ "$(cat /sys/fs/cgroup/pids.current)" -gt 200 ]; do sleep 20; done
    echo "=== attempt $attempt $(date -u '+%H:%M:%S') pids=$(cat /sys/fs/cgroup/pids.current)/256 ===" >> "$LOG"
    uv run --frozen --extra dev --with mutmut mutmut run --max-children 2 >> "$LOG" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then break; fi
    if grep -q "BlockingIOError" "$LOG"; then
        echo "retrying after BlockingIOError" >> "$LOG"
        sleep 30
        continue
    fi
    break
done
echo "exit=$code" >> "$LOG"
uv run --frozen python tools/mutation_score.py mutants >> "$LOG" 2>&1
echo "=== done $(date -u '+%H:%M:%S') ===" >> "$LOG"
