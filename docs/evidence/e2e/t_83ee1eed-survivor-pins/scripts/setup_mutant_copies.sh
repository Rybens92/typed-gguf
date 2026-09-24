#!/usr/bin/env bash
# Fresh-copy mutant check for card t_83ee1eed (defender side, mechanical).
# For each mutant: HEAD copy + patches/mNN.patch, then run the updated task-0 file.
set -u
REPO=/workspace/ggufone
FIGHT=/workspace/state/fights/e1a-t0fc576df
ROOT=/workspace/tmp/e1a-fix-t83ee1eed
rm -rf "$ROOT"
mkdir -p "$ROOT"

for name in base m08 m09 m10 m11; do
  mkdir -p "$ROOT/$name"
  git -C "$REPO" archive HEAD | tar -x -C "$ROOT/$name"
done

for name in m08 m09 m10 m11; do
  ( cd "$ROOT/$name" && git apply "$FIGHT/patches/$name.patch" && echo "patched: $name" )
done

# Sanity: which ggufone does each copy import?
for name in base m08 m09 m10 m11; do
  cat > "$ROOT/$name/tests/_whoami_test.py" <<'PY'
import pathlib
import ggufone


def test_which_ggufone():
    here = pathlib.Path(ggufone.__file__).resolve()
    assert str(here).startswith(str(pathlib.Path(__file__).resolve().parents[1])), here
PY
done

echo "copies ready in $ROOT"
