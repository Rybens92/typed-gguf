#!/usr/bin/env bash
# Quick-preset evidence (card t_f46cec41): the five suites through the CLI, one model, one log dir.
#
#   tools/rehearse_quick.sh <log-dir> <model.gguf> [threads]
#
# Writes <log-dir>/quick_<suite>.json (the report), .md (the rendered table) and .exit (the exit
# code), plus <log-dir>/quick_walls.txt (the wall time the report itself measured).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
LOG="${1:?usage: rehearse_quick.sh <log-dir> <model.gguf> [threads]}"
MODEL="${2:?usage: rehearse_quick.sh <log-dir> <model.gguf> [threads]}"
THREADS="${3:-2}"
RT="${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu}"
mkdir -p "$LOG"
echo "log dir: $LOG"
echo "model:   $MODEL"
echo "threads: $THREADS"
echo "runtime: $RT"

: >"$LOG/quick_walls.txt"
for suite in latency throughput quality calibration determinism; do
    echo
    echo "== quick $suite"
    start=$(date +%s)
    GGUFONE_RUNTIME_DIR="$RT" uv run ggufone bench --suite "$suite" --quick --model "$MODEL" \
        --backend auto --threads "$THREADS" --out "$LOG/quick_$suite.json" \
        >"$LOG/quick_$suite.md" 2>"$LOG/quick_$suite.err"
    code=$?
    end=$(date +%s)
    echo "$code" >"$LOG/quick_$suite.exit"
    wall=$(python3 -c "import json,sys;print(json.load(open('$LOG/quick_$suite.json'))['wall_ms']/1000.0)" 2>/dev/null || echo "?")
    echo "   -> exit=$code shell=$((end - start))s report_wall=${wall}s"
    printf '%s\t%s\t%s\n' "$suite" "$code" "$wall" >>"$LOG/quick_walls.txt"
done
echo
echo "== walls (suite, exit, report wall_ms/1000)"
cat "$LOG/quick_walls.txt"
python3 - "$LOG/quick_walls.txt" <<'PY'
import sys
rows = [line.split() for line in open(sys.argv[1]) if line.strip()]
total = sum(float(row[2]) for row in rows if row[2] != "?")
print(f"quick campaign total: {total:.1f} s over {len(rows)} suites")
PY
