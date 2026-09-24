#!/usr/bin/env bash
# (c) BEFORE-receipt: does a swap plan the *replacement* against the outgoing host's VRAM?
#
# Phase 1: quiet box -> ask the small model (Ling-3.0-tiny, policy call) -> resident host A.
# Phase 2: with A resident, ask the default 4B (policy call) -> the swap -> capture the plan
#          the *new* host used (engine.fit / engine.placement) and the wall time.
# Phase 3: `keep stop` (A gone, box quiet) -> what the box plans for the 4B now -> compare.
set -u
cd /work/t176614c6-live
source env.sh
mkdir -p out

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
LING=/var/home/rybens/.hermes/models/Ling-3.0-tiny-Q5_K_M.gguf

smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }
snaplans() { mkdir -p "out/$1"; cp -f home/fit/*.json "out/$1/" 2>/dev/null; ls -1 "out/$1"; }

echo "=== t0 $(date -u +%H:%M:%S) keep stop"; python3 tg.py keep stop || true
echo "--- smi quiet: $(smi)"
echo "--- fit dir: $(ls -1 home/fit 2>/dev/null)"

echo "=== t1 $(date -u +%H:%M:%S) phase 1: Ling resident (policy call, cold)"
bash runbg.sh c_ling -- python3 tg.py ask --state @states/state_medium.txt --choice "$Q" \
  --model "$LING" --keep-alive 10m --out out/c_ling.json
echo "exit=$(cat out/c_ling.exit) wall_ms=$(cat out/c_ling.wall_ms)"
echo "--- smi with A resident: $(smi)"
python3 tg.py keep status --json > out/c_keepA.json 2>&1
echo "--- plans now: $(snaplans c_plans_afterA)"

echo "=== t2 $(date -u +%H:%M:%S) phase 2: swap -> 4B (policy call) while A resident"
bash runbg.sh c_swap4b -- python3 tg.py ask --state @states/state_medium.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/c_swap4b.json
echo "exit=$(cat out/c_swap4b.exit) wall_ms=$(cat out/c_swap4b.wall_ms)"
echo "--- smi after swap: $(smi)"
python3 tg.py keep status --json > out/c_keepB.json 2>&1
echo "--- plans after swap: $(snaplans c_plans_after_swap)"

echo "=== t3 $(date -u +%H:%M:%S) phase 3: keep stop -> what the box plans for the 4B when quiet"
python3 tg.py keep stop || true
echo "--- smi quiet: $(smi)"
python3 tg.py fit "$MODEL" --json > out/c_fit_quiet.json 2>&1 || true
echo "--- plans after quiet fit: $(snaplans c_plans_after_quietfit)"

echo "=== t4 $(date -u +%H:%M:%S) phase 4: the 4B again, quiet box (the compare run)"
bash runbg.sh c_quiet4b -- python3 tg.py ask --state @states/state_medium.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/c_quiet4b.json
echo "exit=$(cat out/c_quiet4b.exit) wall_ms=$(cat out/c_quiet4b.wall_ms)"
echo "--- smi after quiet ask: $(smi)"
python3 tg.py keep stop || true
echo "=== t5 $(date -u +%H:%M:%S) done"
