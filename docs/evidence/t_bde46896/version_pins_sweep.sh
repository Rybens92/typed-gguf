#!/usr/bin/env bash
# What pins the version (or the notes file) in this repository — and what merely *mentions* it.
# Run from the repository root: bash docs/evidence/t_bde46896/version_pins_sweep.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1
CUTOFF="$(git rev-parse --short=12 HEAD)"

echo "### tree: $CUTOFF"
echo
echo "### 1. every tracked reference to a release-notes file (the file-name pin surface)"
echo "###    (\$ git grep -n RELEASE_NOTES -- . | grep -v '^docs/evidence/' | grep -v '^docs/qa/')"
git grep -n 'RELEASE_NOTES' -- . | grep -v '^docs/evidence/' | grep -v '^docs/qa/'

echo
echo "### 2. every tracked reference to 0.2.3 outside the receipt trees"
echo "###    (the live surface: src · tests · tools · workflows · packaging · docs/*.md)"
git grep -n '0\.2\.3' -- src tests tools .github pyproject.toml uv.lock | grep -v '^docs/evidence/'
git grep -n '0\.2\.3' -- docs | grep -v '^docs/evidence/' | grep -v '^docs/qa/'

echo
echo "### 3. the version spellings that must agree, read from the tree"
grep -n '^version' pyproject.toml
grep -n '^__version__' src/typed_gguf/__init__.py

echo
echo "### 4. the lock's project pin"
grep -n -A2 '^name = "typed-gguf"' uv.lock

echo
echo "### 5. the two test gates that pin the version / the notes file"
git grep -n 'assert version == ' -- tests
git grep -n 'RELEASE_NOTES_v' -- tests
