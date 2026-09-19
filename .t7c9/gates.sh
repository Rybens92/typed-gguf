#!/bin/bash
# t_7c926398 — the card's gates, run in the worktree at the committed corrected instrument.
#
#   * oracle: docs/verify_runtime_contract.py (offline pins; `--run-network`-free)
#   * suite:  the repository gate is `uv run pytest -q`; here it is `--frozen --offline` plus the
#     `dev` extra (pytest/ruff), because this worktree has no site-wide pytest and the run must
#     not touch the network
#   * ruff:   the paths this card touches (`.t7c9`), since the worktree at HEAD already carries
#     lint findings in other cards' scratch dirs (`.e2e`, `.e3c_tiel`) — reported, not hidden
set -u
cd /var/home/rybens/workspace/ggufone-wt-t7c9 || exit 126
echo "== card t_7c926398 gates $(date -Is)"
echo "== tree: $(git rev-parse --short HEAD) ($(git branch --show-current))"
echo "== scope: cgroup=$(cat /proc/self/cgroup | cut -d: -f3) memory.max=$(cat /sys/fs/cgroup$(cat /proc/self/cgroup | cut -d: -f3)/memory.max 2>/dev/null)"
echo
echo "== gate 1: oracle docs/verify_runtime_contract.py"
python3 docs/verify_runtime_contract.py > .t7c9/oracle.txt 2>&1
echo "   exit=$? (full output: .t7c9/oracle.txt)"
tail -2 .t7c9/oracle.txt
echo
echo "== gate 2: ruff (the paths this card touches)"
uv run --frozen --offline --extra dev ruff check .t7c9 2>&1 | tail -3
echo
echo "== gate 3: the test suite"
uv run --frozen --offline --extra dev pytest -q -rs -p no:cacheprovider 2>&1 | tail -25
echo "   suite exit=$?"
echo
echo "== gates done $(date -Is)"
