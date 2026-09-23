#!/usr/bin/env bash
# (c) AFTER-receipt: the same four steps as the before-run (repro_c2.sh, artifacts out/c2_*),
# with the fix in place. What must change:
#   step 3 (`fit` while our own host is resident): still re-plans to 0 layers for *that* reading
#          (the answer for the moment is honest) — but
#   step 4 (`fit` after `keep stop`): the box must climb back to the 36-layer policy plan, because
#          the depleted reading was never written over the cache entry, and
#   step 5 (`ask`): must be served by the device again (~16-17 s host-served, not the ~26 s CPU run).
set -u
cd /work/t176614c6-live
source env.sh
mkdir -p out

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }
snap() { mkdir -p "out/$1"; cp -f home/fit/*.json "out/$1/" 2>/dev/null; }

echo "=== t0 $(date -u +%H:%M:%S) quiet start"
python3 tg.py keep stop || true
rm -rf home/fit
echo "--- smi: $(smi)"

echo "=== t1 $(date -u +%H:%M:%S) step 1: quiet fit (the policy's own answer)"
python3 tg.py fit "$MODEL" --json > out/c3_fit_quiet1.json 2>&1
snap c3_plans_step1

echo "=== t2 $(date -u +%H:%M:%S) step 2: the 4B host becomes resident"
bash runbg.sh c3_host -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/c3_host.json
echo "exit=$(cat out/c3_host.exit) wall_ms=$(cat out/c3_host.wall_ms)"
echo "--- smi: $(smi)"

echo "=== t3 $(date -u +%H:%M:%S) step 3: fit again, host resident (depleted device)"
python3 tg.py fit "$MODEL" --json > out/c3_fit_busy.json 2>&1
snap c3_plans_step3

echo "=== t4 $(date -u +%H:%M:%S) step 4: keep stop -> fit on the quiet box again"
python3 tg.py keep stop || true
echo "--- smi: $(smi)"
python3 tg.py fit "$MODEL" --json > out/c3_fit_quiet2.json 2>&1
snap c3_plans_step4

echo "=== t5 $(date -u +%H:%M:%S) step 5: ask the 4B on the entry that survived"
bash runbg.sh c3_ask -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 0 --out out/c3_ask.json
echo "exit=$(cat out/c3_ask.exit) wall_ms=$(cat out/c3_ask.wall_ms)"
python3 tg.py keep stop || true
echo "=== t6 $(date -u +%H:%M:%S) done"

echo "--- the four plan readings that decide this receipt:"
python3 plans.py "step1_quiet_fit=out/c3_fit_quiet1.json" "step3_fit_resident=out/c3_fit_busy.json" \
  "step4_after_keepstop=out/c3_fit_quiet2.json" "step5_ask=out/c3_ask.json" 2>&1 | grep -v "^    note"
echo "--- the cache entry itself, at each step (the file in home/fit that ask/run reads):"
for step in step1 step3 step4; do
  echo "-- $step"
  for f in out/c3_plans_$step/*.json; do
    [ -f "$f" ] || continue
    python3 plans.py "entry.$(basename "$f" | cut -c1-12)=$f" 2>&1 | grep -v "^    note"
  done
done
