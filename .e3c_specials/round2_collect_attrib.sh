#!/bin/bash
# Card t_635124bf round 2: attribute the live-tree count vs the committed tree.
# collect-only in both trees; the delta must be the sibling's untracked E3d gate files.
set -u
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/work/t635/tmp
echo "== committed tree /work/t635/committed-head (8938950, no sibling WIP)"
cd /work/t635/committed-head && uv run --frozen --extra dev python -m pytest -q --collect-only 2>&1 | tail -3
echo
echo "== live shared tree /var/home/rybens/workspace/ggufone (sibling WIP untracked files present)"
cd /var/home/rybens/workspace/ggufone && uv run --frozen --extra dev python -m pytest -q --collect-only 2>&1 | tail -3
echo
echo "== untracked test files in the live tree"
git -C /var/home/rybens/workspace/ggufone status --porcelain -- tests
