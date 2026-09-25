#!/bin/bash
# Card t_16067777 — the CI-shape gates, re-run on the fixed tree (resumed run).
# Same shape as the card's GATES line; `.venv/bin/python` is used instead of a bare
# `uv run` so the sweep/venv is never re-synced out from under the run (see the
# README's reproduce note). Usage: bash gates.sh > ../gates.log 2>&1
set -u
cd /workspace/ggufone || exit 1
STUB=/tmp/offline-bundle-023
mkdir -p "$STUB"
for lib in libllama.so libggml.so libggml-base.so libggml-cpu.so; do : > "$STUB/$lib"; done
PY=.venv/bin/python

echo "== host =="
date -u '+%Y-%m-%dT%H:%M:%SZ'
"$PY" -V
uname -srm

echo
echo "== full suite (offline: TYPED_GGUF_TEST_BLOCK_NET=1, stub bundle, no PYTHONPATH/TYPED_GGUF_HOME) =="
LOG=$(mktemp)
env -u PYTHONPATH -u TYPED_GGUF_HOME TYPED_GGUF_TEST_BLOCK_NET=1 \
    TYPED_GGUF_BENCH_RUNTIME_DIR="$STUB" \
    "$PY" -m pytest -q -rs --timeout=120 > "$LOG" 2>&1
echo "pytest exit=$?"
tail -n 6 "$LOG"
echo "  skipped, by reason:"
grep -E '^SKIPPED' "$LOG" | sed 's/^SKIPPED \[[0-9]*\] //' | sort | uniq -c | sort -rn

echo
echo "== ruff check src tests tools docs .github =="
env -u PYTHONPATH ruff check src tests tools docs .github
echo "ruff exit=$?"

echo
echo "== uv build =="
env -u PYTHONPATH uv build 2>&1 | tail -n 4
echo "uv build exit=$?"

echo
echo "== tree under test =="
git status --porcelain | sed 's/^/  /'
git diff --stat | tail -n 3 | sed 's/^/  /'
