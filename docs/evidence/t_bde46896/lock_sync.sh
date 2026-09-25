#!/usr/bin/env bash
# The lock after the bump: `uv lock` refreshed the project pin, and the frozen sync still resolves.
#   bash docs/evidence/t_bde46896/lock_sync.sh > docs/evidence/t_bde46896/green_lock_sync.log 2>&1
set -u
cd "$(dirname "$0")/../../.." || exit 1
export UV_PYTHON=/home/rybens/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13

echo "\$ uv lock"
uv lock
echo "EXIT=$?"
echo
echo "\$ uv sync --frozen --extra dev"
uv sync --frozen --extra dev
echo "EXIT=$?"
echo
echo "\$ grep -n -A2 '^name = \"typed-gguf\"' uv.lock"
grep -n -A2 '^name = "typed-gguf"' uv.lock
