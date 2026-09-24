#!/bin/sh
# Final gates at the fixed head (card t_ba767a2b). Usage: sh gates-final.sh
set -u
cd /workspace/ggufone || exit 1
OUT=/workspace/t_ba767a2b
STUB=$OUT/offline-bundle
GATE="tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py"

env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 \
  TYPED_GGUF_BENCH_RUNTIME_DIR="$STUB" timeout 900 .venv/bin/python -m pytest -q -rs --timeout=120 \
  > "$OUT/final-311-full.txt" 2>&1
echo "final-311-full exit=$? $(tail -1 "$OUT/final-311-full.txt")"

env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 \
  TYPED_GGUF_BENCH_RUNTIME_DIR="$STUB" timeout 900 "$OUT/venv312/bin/python" -m pytest -q -rs --timeout=120 \
  > "$OUT/final-312-full.txt" 2>&1
echo "final-312-full exit=$? $(tail -1 "$OUT/final-312-full.txt")"

env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 timeout 300 .venv/bin/python -m pytest -q \
  $GATE > "$OUT/final-subgate-311.log" 2>&1
echo "final-subgate-311 exit=$? $(tail -1 "$OUT/final-subgate-311.log")"

env -u PYTHONPATH .venv/bin/python -m ruff check src tests tools docs .github > "$OUT/final-ruff.txt" 2>&1
echo "final-ruff exit=$? $(tail -1 "$OUT/final-ruff.txt")"

env -u PYTHONPATH uv build > "$OUT/final-uvbuild.txt" 2>&1
echo "final-uvbuild exit=$? $(tail -1 "$OUT/final-uvbuild.txt")"
