#!/usr/bin/env bash
# (c) BEFORE-receipt, take 2: the load-time re-plan writes a *depleted* plan over a healthy
# cached one, and the entry then sticks (shrink-only never climbs back).
#
# 1. quiet box: `fit <4B>` -> the policy's own plan (36 layers) -> the cache entry.
# 2. start the 4B warm host (the device is now occupied by our own resident model).
# 3. `fit <4B>` again, host resident: the same load-time re-plan the swap path runs
#    (fit.replan_for_host, shrink-only) -> does it shrink, and does it WRITE that back?
# 4. `keep stop` (box quiet again) -> `fit <4B>` -> does the box climb back to 36 layers?
# 5. `ask <4B>` on that entry -> the user-visible cost (CPU run) if the entry stuck.
set -u
cd /work/t176614c6-live
source env.sh
mkdir -p out

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }
snap() { mkdir -p "out/$1"; cp -f home/fit/*.json "out/$1/" 2>/dev/null; }
grab() { python3 show.py "plan=$1" 2>/dev/null | head -1; }

echo "=== t0 $(date -u +%H:%M:%S) quiet start"
python3 tg.py keep stop || true
rm -rf home/fit
echo "--- smi: $(smi)"

echo "=== t1 $(date -u +%H:%M:%S) step 1: quiet fit (the policy's own answer)"
python3 tg.py fit "$MODEL" --json > out/c2_fit_quiet1.json 2>&1
snap c2_plans_step1
grab out/c2_fit_quiet1.json

echo "=== t2 $(date -u +%H:%M:%S) step 2: the 4B host becomes resident"
bash runbg.sh c2_host -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/c2_host.json
echo "exit=$(cat out/c2_host.exit) wall_ms=$(cat out/c2_host.wall_ms)"
echo "--- smi: $(smi)"

echo "=== t3 $(date -u +%H:%M:%S) step 3: fit again, host resident (depleted device)"
python3 tg.py fit "$MODEL" --json > out/c2_fit_busy.json 2>&1
snap c2_plans_step3
grab out/c2_fit_busy.json

echo "=== t4 $(date -u +%H:%M:%S) step 4: keep stop -> fit on the quiet box again"
python3 tg.py keep stop || true
echo "--- smi: $(smi)"
python3 tg.py fit "$MODEL" --json > out/c2_fit_quiet2.json 2>&1
snap c2_plans_step4
grab out/c2_fit_quiet2.json

echo "=== t5 $(date -u +%H:%M:%S) step 5: ask the 4B on the entry that survived"
bash runbg.sh c2_ask -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 0 --out out/c2_ask.json
echo "exit=$(cat out/c2_ask.exit) wall_ms=$(cat out/c2_ask.wall_ms)"
python3 tg.py keep stop || true
echo "=== t6 $(date -u +%H:%M:%S) done"
