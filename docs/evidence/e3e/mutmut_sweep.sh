#!/bin/bash
# E3e (card t_4c48f40a) — the Tier-M mutation sweep.
#
# Scope comes from pyproject.toml (`source_paths = engine/prompt.py + engine/cue.py`,
# `pytest_add_cli_args_test_selection = the E3e/E3d/E3c gate files`) — mutmut 3.8 does not scope on
# a name pattern, so the source_paths entry *is* the scope and this script only drives it.
#
# The pid-capped box: a failed os.fork() takes the whole run down (EAGAIN/BlockingIOError), so a
# crash gets a bounded retry, and a run that finished is reported by status from the cache.
set -u
cd "$(dirname "$0")/.." || exit 126
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp

mkdir -p .e3e/logs
log=.e3e/logs/mutmut.log

for attempt in 1 2 3; do
    echo "=== sweep attempt $attempt ($(date -Is)) ===" | tee -a "$log"
    if uv run --extra dev --with mutmut mutmut run --max-children 2 >>"$log" 2>&1; then
        echo "sweep finished on attempt $attempt" | tee -a "$log"
        break
    fi
    if grep -qE "BlockingIOError|EAGAIN|OSError: \[Errno 11\]" "$log"; then
        echo "attempt $attempt hit the pid cap; retrying" | tee -a "$log"
        rm -rf .mutmut-cache
        sleep 5
        continue
    fi
    echo "sweep finished on attempt $attempt (exit non-zero: survivors or a kill)" | tee -a "$log"
    break
done

uv run --extra dev --with mutmut mutmut results > .e3e/mutmut_results.txt 2>&1 || true
echo "--- results ---" | tee -a "$log"
cat .e3e/mutmut_results.txt | head -40 | tee -a "$log"

python3 - <<'PY' | tee -a "$log"
import sqlite3
try:
    con = sqlite3.connect(".mutmut-cache")
    rows = con.execute("select status, count(*) from mutant group by status order by 2 desc").fetchall()
    total = sum(n for _, n in rows)
    print(f"mutants: {total}")
    for status, n in rows:
        print(f"  {status}: {n}  ({n / total:.1%})" if total else f"  {status}: {n}")
    killers = con.execute(
        "select count(distinct mutant_id) from mutant_killed_by").fetchone()
    print("killed-by rows:", killers)
except Exception as error:                      # noqa: BLE001
    print("cache summary failed:", error)
PY
echo "=== sweep done ($(date -Is)) ===" | tee -a "$log"
