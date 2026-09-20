#!/usr/bin/env bash
# The CI gate, executed verbatim in a fresh clone of the rename commit (card t_5f9c15fe).
# Mirrors .github/workflows/ci.yml: ruff -> offline suite with an empty bundle stub ->
# live gates must skip (red path) -> the oracle's bundle-free shape.
set -uo pipefail
cd /work/t5f9/clone-t5f9
export HOME=/work/t5f9/empty-home
export UV_CACHE_DIR=/work/.uv-cache
export UV_PYTHON=3.11
LOG=/work/t5f9/logs
step() { echo; echo "== $*"; }

step "uv sync --extra dev"
uv sync --extra dev --python 3.11 2>&1 | tail -2

step "ruff check src tests tools docs .github"
uv run --extra dev ruff check src tests tools docs .github; echo "ruff rc=$?"

bundle="$(mktemp -d)/typed-gguf-offline-bundle"
mkdir -p "$bundle"
for lib in libllama.so libggml.so libggml-base.so libggml-cpu.so; do
  : > "$bundle/$lib"
done

step "offline suite (TYPED_GGUF_TEST_BLOCK_NET=1, empty bundle stub)"
env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR="$bundle" \
  uv run --extra dev pytest -q -rs --timeout=120 > "$LOG/clone_suite.txt" 2>&1
echo "suite rc=$?"; tail -2 "$LOG/clone_suite.txt"

step "red path: live gates must skip, never fail (runtime vars unset)"
out=$(env -u PYTHONPATH -u TYPED_GGUF_RUNTIME_DIR -u TYPED_GGUF_BENCH_RUNTIME_DIR \
      uv run --extra dev pytest -q -m 'model or network' 2>&1)
echo "$out" | tail -2

step "oracle, bundle-free shape"
uv run --extra dev python3 docs/verify_runtime_contract.py > "$LOG/clone_oracle.txt" 2>&1
echo "oracle rc=$?"; tail -2 "$LOG/clone_oracle.txt"
