#!/usr/bin/env bash
# (b) BEFORE-receipt, take 2: find the "ready host that cannot serve" state without a CUDA hog
# (CUDA allocations are refused in this container: rc=201 with 6.6 GiB free).
#
# The state's cause (field case): the plan fits the *weights* plus its KV estimate, but each
# request needs its own context on top — and llama_init_from_model then fails instantly, on every
# request, for as long as the host lives. Variants differ only in how much the plan spends.
set -u
cd /work/t176614c6-live
source env.sh
mkdir -p out

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }

one() {  # one LABEL -- <ask flags...>
  local label="$1"; shift; shift
  echo "=== $label $(date -u +%H:%M:%S)"
  python3 tg.py keep stop >/dev/null 2>&1 || true
  bash runbg.sh "b2_${label}_first" -- python3 tg.py ask --state @states/state_short.txt \
    --choice "$Q" --model "$MODEL" --keep-alive 10m --out "out/b2_${label}_first.json" "$@"
  echo "first: exit=$(cat out/b2_${label}_first.exit) wall_ms=$(cat out/b2_${label}_first.wall_ms) smi=$(smi)"
  head -c 200 "out/b2_${label}_first.stderr"; echo
  bash runbg.sh "b2_${label}_second" -- python3 tg.py ask --state @states/state_short.txt \
    --choice "$Q" --model "$MODEL" --keep-alive 10m --out "out/b2_${label}_second.json" "$@"
  echo "second: exit=$(cat out/b2_${label}_second.exit) wall_ms=$(cat out/b2_${label}_second.wall_ms)"
  head -c 200 "out/b2_${label}_second.stderr"; echo
  python3 tg.py keep status --json > "out/b2_${label}_status.json" 2>&1
  python3 -c "import json,sys; d=json.load(open('out/b2_${label}_status.json')); print('status:', d.get('state'), 'pid', d.get('pid'), 'requests', d.get('requests'), 'placement', json.dumps(d.get('placement')))" 2>/dev/null
  python3 tg.py keep stop >/dev/null 2>&1 || true
}

python3 tg.py keep stop >/dev/null 2>&1 || true
rm -rf home/fit
echo "--- smi quiet: $(smi)"
one v1_nctx51200_t0 -- --n-ctx 51200 --fit-target 0
one v2_nctx65536_t0 -- --n-ctx 65536 --fit-target 0
one v3_default_t0 -- --fit-target 0
echo "=== done $(date -u +%H:%M:%S)"
