#!/bin/sh
# Card t_a696ce02 — static + coverage gates in the private clone.
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
REPO=/work/t_a696ce02/repo
OUT=/work/t_a696ce02/out
cd "$REPO" || exit 1
PY=.venv/bin/python
: > "$OUT/ruff.txt"
.venv/bin/ruff check src tests tools >> "$OUT/ruff.txt" 2>&1
echo "ruff exit=$?" >> "$OUT/ruff.txt"
tail -3 "$OUT/ruff.txt"
uv pip install -q -p .venv coverage || exit 2
: > "$OUT/coverage.txt"
$PY -m coverage run --source=src/ggufone -m pytest -q -p no:randomly \
    tests/test_probe_pressure.py tests/test_runtime_fallback.py tests/test_runtime_install.py \
    tests/test_runtime_contract.py tests/test_probe_isolation.py tests/test_capability.py \
    >> "$OUT/coverage.txt" 2>&1
echo "pytest exit=$?" >> "$OUT/coverage.txt"
$PY -m coverage report --include="*/runtime/pressure.py,*/runtime/isolated.py,*/runtime/capability.py" \
    >> "$OUT/coverage.txt" 2>&1
tail -12 "$OUT/coverage.txt"
