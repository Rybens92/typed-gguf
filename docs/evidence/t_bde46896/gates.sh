#!/usr/bin/env bash
# t_bde46896 — the release-prep gates, exactly as the card spells them.
#
#   bash docs/evidence/t_bde46896/gates.sh <label>
#
# `UV_PYTHON=<host 3.13>` is the one prefix this container needs (transparency note in the
# receipt): the sandbox's `uv` defaults to CPython 3.12.13, and a bare `uv run --extra dev` here
# would re-create the host's `.venv` on that interpreter. With UV_PYTHON set, `uv run` syncs to
# the host's own 3.13.14 instead, and the gate is byte-for-byte the card's command otherwise.
# A long TMPDIR breaks one unix-socket race test when the suite is run from the *host* shell (the
# card's note); TMPDIR is /tmp in this container, so that note does not apply here.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=/home/rybens/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13
export UV_PYTHON="$PY"

run() {
  echo "\$ $*"
  "$@"
  echo "EXIT=$?"
  echo
}

echo "### tree under test"
git log --oneline -1
git status --short
echo

run env -u PYTHONPATH uv run --extra dev pytest -q

echo "\$ uv run python -c \"...version assertion...\""
env -u PYTHONPATH uv run python -c "import tomllib,pathlib,typed_gguf; pyv=tomllib.load(open('pyproject.toml','rb'))['project']['version']; print(pyv, typed_gguf.__version__); assert pyv=='0.3.0'==typed_gguf.__version__"
echo "EXIT=$?"
echo

run env -u PYTHONPATH uv run --extra dev pytest -q tests/test_public_docs.py tests/test_release_publish.py

# an isolated data home: this container's *default* home (`/root/.local/share/typed-gguf`) carries
# leftover `runtime.json` cruft from earlier cards' pytest runs (2026-09-24), so an unfixed
# `version` here would report a foreign runtime and read as if this card had installed one.
run env -u PYTHONPATH TYPED_GGUF_HOME="$(mktemp -d)" uv run typed-gguf version

run env -u PYTHONPATH uv run --extra dev ruff check src tests tools docs .github

run env -u PYTHONPATH uv build

echo "\$ ls dist/"
ls dist/
