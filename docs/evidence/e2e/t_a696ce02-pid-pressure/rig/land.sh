#!/bin/sh
# Card t_a696ce02 — land the private clone's commits onto the shared tree (main), ff-only.
#
# Siblings landed 8474802..37fc224 (E3 review + E3b) while this card worked; the clone is rebased
# onto the shared main first. Nothing in this card touches pyproject.toml (the recurring hotspot)
# or the siblings' dirty files, and the WIP payload is hashed before/after the merge.
export GIT_CONFIG_GLOBAL=/root/.gitconfig
CLONE=/work/t_a696ce02/repo
SHARED=/var/home/rybens/workspace/ggufone
OUT=/work/t_a696ce02/out
mkdir -p "$OUT"

cd "$CLONE" || exit 1
if ! git diff --quiet -- pyproject.toml; then
    git stash push -q -- pyproject.toml || exit 2
    echo "parked the sweep retarget in the stash"
fi
git fetch -q "$SHARED" main || exit 3
echo "shared main: $(git log --oneline -1 FETCH_HEAD)"
git rebase FETCH_HEAD || { echo "REBASE CONFLICT — stop"; exit 4; }
echo "rebased: $(git log --oneline -3 | tr '\n' ' ')"
git diff --quiet 8474802..HEAD -- pyproject.toml && echo "no pyproject hunk in this card" || exit 5

cd "$SHARED" || exit 6
git status --short > "$OUT/shared_status_before.txt"
sha256sum pyproject.toml state/groupchat/ggufone-e1.md > "$OUT/wip_before.sha" 2>/dev/null
echo "before: $(cat "$OUT/wip_before.sha" | tr '\n' ' ')"

git fetch -q "$CLONE" main || exit 7
git merge --ff-only FETCH_HEAD || { echo "FF REFUSED — inspect"; exit 8; }
git log --oneline -1

sha256sum pyproject.toml state/groupchat/ggufone-e1.md > "$OUT/wip_after.sha" 2>/dev/null
if diff -q "$OUT/wip_before.sha" "$OUT/wip_after.sha" > /dev/null; then
    echo "sibling WIP byte-identical after the landing"
else
    echo "SIBLING WIP CHANGED — investigate"
    diff "$OUT/wip_before.sha" "$OUT/wip_after.sha"
    exit 9
fi
git status --short | head -10
echo "LANDED: $(git log --oneline -1)"
