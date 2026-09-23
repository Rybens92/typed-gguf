#!/usr/bin/env bash
# (a) receipt, runnable against either tree:
#   $1 = label ("before" | "after"), $2 = launcher script (tg_before.py | tg.py)
#
# ask A: the resident key, 12.8k-token state — a decision longer than STOP_GRACE (5 s).
# ask B: +0.5 s later, the same model with --n-ctx 32768 -> a different host key -> a SWAP.
# Before the fix the swap SIGKILLs A's host at the 5 s grace and A's answer dies (A falls back
# inline; both calls end in E_BACKEND_OOM). After: A's answer is delivered, B waits for the drain.
set -u
cd /work/t176614c6-live
source env.sh
label="$1"; launcher="$2"
mkdir -p out

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }
p() { echo "=== [$label] $*"; }

p "t0 $(date -u +%H:%M:%S) quiet start"
python3 "$launcher" keep stop >/dev/null 2>&1 || true
rm -rf home/fit
rm -f out/a_${label}_*.json out/a_${label}_*.exit out/a_${label}_*.stderr out/a_${label}_*.wall_ms
echo "--- smi: $(smi)"

p "t1 prime: the 4B host on the default key (12.8k state, cold)"
bash runbg.sh "a_${label}_prime" -- python3 "$launcher" ask --state @states/state_xlong.txt \
  --choice "$Q" --model "$MODEL" --keep-alive 10m --out "out/a_${label}_prime.json"
echo "prime: exit=$(cat out/a_${label}_prime.exit) wall_ms=$(cat out/a_${label}_prime.wall_ms) smi=$(smi)"

(
  for i in $(seq 1 45); do
    echo "--- t=${i} $(date -u +%H:%M:%S.%3N) smi=$(smi)"
    ps -eo pid,etime,args | grep -E "keep _host|tg.*ask" | grep -v grep | cut -c1-100
    sleep 1
  done
) >> "out/a_${label}_timeline.txt" 2>&1 &

p "t2 ask A (resident key, 12.8k) ..."
bash runbg.sh "a_${label}_first" -- python3 "$launcher" ask --state @states/state_xlong.txt \
  --choice "$Q" --model "$MODEL" --keep-alive 10m --out "out/a_${label}_first.json" &
apid=$!
sleep 0.5
p "t3 ask B (--n-ctx 32768 -> the swap) at +0.5 s ..."
bash runbg.sh "a_${label}_second" -- python3 "$launcher" ask --state @states/state_xlong.txt \
  --choice "$Q" --model "$MODEL" --keep-alive 10m --n-ctx 32768 --out "out/a_${label}_second.json"
echo "B(swap): exit=$(cat out/a_${label}_second.exit) wall_ms=$(cat out/a_${label}_second.wall_ms)"
wait $apid
echo "A(first): exit=$(cat out/a_${label}_first.exit) wall_ms=$(cat out/a_${label}_first.wall_ms)"
echo "--- A stderr:"; head -c 420 out/a_${label}_first.stderr; echo
echo "--- B stderr:"; head -c 420 out/a_${label}_second.stderr; echo
echo "--- A answer delivered: $([ -s out/a_${label}_first.json ] && echo YES || echo NO)"
python3 plans.py "A=out/a_${label}_first.json" "B=out/a_${label}_second.json" 2>/dev/null

p "t4 request 3 on whatever is resident now (is it serving?)"
bash runbg.sh "a_${label}_third" -- python3 "$launcher" ask --state @states/state_short.txt \
  --choice "$Q" --model "$MODEL" --keep-alive 10m --out "out/a_${label}_third.json"
echo "third: exit=$(cat out/a_${label}_third.exit) wall_ms=$(cat out/a_${label}_third.wall_ms)"
head -c 300 out/a_${label}_third.stderr; echo
python3 "$launcher" keep status --json > "out/a_${label}_status.json" 2>&1
python3 -c "import json; d=json.load(open('out/a_${label}_status.json')); print('status:', d.get('state'), 'pid', d.get('pid'), 'requests', d.get('requests'))" 2>/dev/null
python3 "$launcher" keep stop >/dev/null 2>&1 || true
echo "--- smi after: $(smi)"
p "done $(date -u +%H:%M:%S)"
