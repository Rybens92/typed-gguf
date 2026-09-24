#!/bin/sh
# M1 acceptance (card t_ba767a2b): the exact committed keep sub-gate on py3.11 —
# 25 consecutive runs (green at the wave head) and then >=4 rounds of 4-way concurrent copies of
# that same command (the load under which the pre-fix leak fires: 2/32 runs on the review box).
# Usage: sh gate-m1.sh <label> <python>
set -u
cd /workspace/ggufone || exit 1
LABEL=${1:-m1}
PY=${2:-.venv/bin/python}
GATE="tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py"
OUT=/workspace/t_ba767a2b/gate-311-subgate
mkdir -p "$OUT"
: > "$OUT/exits.txt"

run() {   # run <tag> <logfile>
  env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 \
    timeout 300 "$PY" -m pytest -q $GATE > "$2" 2>&1
  code=$?
  printf '%s exit=%s %s\n' "$1" "$code" "$(tail -1 "$2")" | tee -a "$OUT/exits.txt"
}

i=1
while [ "$i" -le 25 ]; do run "$LABEL-seq-$i" "$OUT/$LABEL-seq-$i.log"; i=$((i + 1)); done
r=1
while [ "$r" -le 4 ]; do
  p=1
  while [ "$p" -le 4 ]; do run "$LABEL-par-$r-$p" "$OUT/$LABEL-par-$r-$p.log" & p=$((p + 1)); done
  wait
  echo "round $r done"
  r=$((r + 1))
done
echo "--- totals ($LABEL)"
echo "runs:   $(wc -l < "$OUT/exits.txt")"
echo "green:  $(grep -c 'exit=0' "$OUT/exits.txt")"
echo "red:    $(grep -c 'exit=[^0]' "$OUT/exits.txt")"
