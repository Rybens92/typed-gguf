#!/usr/bin/env bash
# Replay specific mutant keys from the E1c sweep against the new pin file alone.
# "killed" = the new tests fail with that mutant active (i.e. a fresh sweep would record a kill).
set -u
cd /workspace/ggufone/mutants || exit 1
cp /workspace/ggufone/tests/test_e1c_mutation_pins.py tests/
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache
RUN=(uv run --no-project --with mutmut==3.8 --with pytest --with pytest-timeout
     python -m pytest -q -x -p no:randomly tests/test_e1c_mutation_pins.py)
while IFS= read -r key; do
  [ -z "$key" ] && continue
  if MUTANT_UNDER_TEST="$key" "${RUN[@]}" >/dev/null 2>&1; then
    printf 'SURVIVED %s\n' "$key"
  else
    printf 'killed   %s\n' "$key"
  fi
done < /workspace/ggufone/.e2e/t_c8e36cad-e1c/logs/replay_keys.txt
