#!/bin/sh
# Gates for card t_ba767a2b (M1 acceptance on py3.12 + the CI-shape suite at the new head).
# Usage: sh gates-rest.sh <python> <label>
set -u
cd /workspace/ggufone || exit 1
PY=${1:-.venv/bin/python}
LABEL=${2:-311}
OUT=/workspace/t_ba767a2b
GATE="tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py"
STUB=$OUT/offline-bundle

env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 \
  TYPED_GGUF_BENCH_RUNTIME_DIR="$STUB" timeout 900 "$PY" -m pytest -q -rs --timeout=120 \
  > "$OUT/gate-$LABEL-full.txt" 2>&1
echo "full-$LABEL exit=$? $(tail -1 "$OUT/gate-$LABEL-full.txt")"

: > "$OUT/gate-$LABEL-subgate.txt"
i=1
while [ "$i" -le 10 ]; do
  env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 timeout 300 "$PY" -m pytest -q \
    $GATE > "$OUT/subgate-$LABEL-$i.log" 2>&1
  printf '%s-seq-%s exit=%s %s\n' "$LABEL" "$i" "$?" "$(tail -1 "$OUT/subgate-$LABEL-$i.log")" \
    >> "$OUT/gate-$LABEL-subgate.txt"
  i=$((i + 1))
done
echo "--- subgate $LABEL"
echo "runs:  $(wc -l < "$OUT/gate-$LABEL-subgate.txt")"
echo "green: $(grep -c 'exit=0' "$OUT/gate-$LABEL-subgate.txt")"
echo "red:   $(grep -c 'exit=[^0]' "$OUT/gate-$LABEL-subgate.txt")"

env -u PYTHONPATH uv build > "$OUT/gate-uvbuild.txt" 2>&1
echo "uv-build exit=$? $(tail -1 "$OUT/gate-uvbuild.txt")"
