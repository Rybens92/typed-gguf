#!/bin/bash
# t_9bcbecff — the card's gates, run in the worktree (the committed tree + this card's `.t9bcb/`).
#
#   * oracle: docs/verify_runtime_contract.py (offline pins)
#   * suite:  the repository gate is `uv run pytest -q`; here it is `--frozen --offline` plus the
#     `dev` extra (pytest/ruff), because this worktree has no site-wide pytest and the run must
#     not touch the network
#   * ruff:   the paths this card touches (`.t9bcb`), since the worktree at HEAD already carries
#     lint findings in other cards' scratch dirs — reported, not hidden
set -u
cd /var/home/rybens/workspace/ggufone-wt-t9bcb || exit 126
echo "== card t_9bcbecff gates $(date -Is)"
echo "== tree: $(git rev-parse --short HEAD) ($(git branch --show-current))"
echo "== scope: cgroup=$(cat /proc/self/cgroup | cut -d: -f3) memory.max=$(cat /sys/fs/cgroup$(cat /proc/self/cgroup | cut -d: -f3)/memory.max 2>/dev/null)"
echo
echo "== gate 1: oracle docs/verify_runtime_contract.py"
python3 docs/verify_runtime_contract.py > .t9bcb/oracle.txt 2>&1
echo "   exit=$? (full output: .t9bcb/oracle.txt)"
tail -2 .t9bcb/oracle.txt
echo
echo "== gate 2: ruff (the paths this card touches)"
uv run --frozen --offline --extra dev ruff check .t9bcb 2>&1 | tail -3
echo
echo "== gate 3: the test suite"
uv run --frozen --offline --extra dev pytest -q -rs -p no:cacheprovider > .t9bcb/gates.txt 2>&1
rc=$?
tail -25 .t9bcb/gates.txt
echo "   suite exit=$rc (full output: .t9bcb/gates.txt)"
echo
echo "== gates done $(date -Is)"
