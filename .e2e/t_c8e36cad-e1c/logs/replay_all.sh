#!/usr/bin/env bash
# Replay mutant keys from the E1c sweep against the FINAL test tree (the fix for the flaky RSS
# assertion included), with mutmut's own selection plus the pin file.
#
# Why the whole selection and not just the mapped tests: this is the *stricter* check. A key that
# fails here is genuinely killable by the committed tests; a key that passes here means the
# sweep's "killed" verdict (if it had one) came from a test version that no longer exists.
#
# Usage: bash .e2e/t_c8e36cad-e1c/logs/replay_all.sh [keysfile]
set -u
keys="${1:-/workspace/ggufone/.e2e/t_c8e36cad-e1c/logs/replay_keys_all.txt}"
cd /workspace/ggufone/mutants || exit 1
cp /workspace/ggufone/tests/test_e1c_mutation_pins.py tests/
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache
SELECTION=(tests/test_templates.py tests/test_fit.py tests/test_cli_e1c.py
           tests/test_engine_fork.py tests/test_cli.py tests/test_e1c_mutation_pins.py)
while IFS= read -r key; do
  [ -z "$key" ] && continue
  if MUTANT_UNDER_TEST="$key" uv run --no-project --with mutmut==3.8 --with pytest \
       --with pytest-timeout python -m pytest -q -x -p no:randomly "${SELECTION[@]}" \
       >/dev/null 2>&1; then
    printf 'SURVIVED %s\n' "$key"
  else
    printf 'killed   %s\n' "$key"
  fi
done < "$keys"
