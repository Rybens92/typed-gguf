#!/bin/bash
# Collect `mutmut show` diffs for a verdict sample, per function, into one file.
#
#   bash /work/t55/show_sample.sh <meta> <out> <fn-substr>:<verdict>:<limit> ...
set -u
META=$1; OUT=$2; shift 2
: > "$OUT"
for spec in "$@"; do
    fn=${spec%%:*}; rest=${spec#*:}; verdict=${rest%%:*}; limit=${rest#*:}
    keys=$(python3 /work/t55/pick_mutants.py "$META" "$fn" "$verdict" "$limit")
    for key in $keys; do
        echo "##### $key" >> "$OUT"
        uv run --frozen --extra dev --with mutmut mutmut show "$key" >> "$OUT" 2>&1
    done
done
echo "wrote $OUT: $(grep -c '^#####' "$OUT") mutants"
