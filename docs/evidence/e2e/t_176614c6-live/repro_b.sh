#!/usr/bin/env bash
# (b) BEFORE-receipt: a host whose weights are resident but whose per-request context can no longer
# be allocated serves an *instant* E_BACKEND_OOM on every request and stays "ready" until a human
# types `keep stop`.
#
# The card's own shape (a swap under pressure) is not repeatable on demand, so the state is built
# directly with its cause: the device is filled *after* the host loaded (hog.py = "the desktop
# took the card"), which is exactly what the field case left behind.
set -u
cd /work/t176614c6-live
source env.sh
mkdir -p out

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }

echo "=== t0 $(date -u +%H:%M:%S) quiet start"
python3 tg.py keep stop || true
rm -rf home/fit out/b_*.json
echo "--- smi: $(smi)"

echo "=== t1 $(date -u +%H:%M:%S) the 4B host becomes resident (36 layers)"
bash runbg.sh b_host -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/b_host.json
echo "exit=$(cat out/b_host.exit) wall_ms=$(cat out/b_host.wall_ms) smi=$(smi)"

echo "=== t2 $(date -u +%H:%M:%S) the card is taken (hog holds 2200 MiB for 120 s)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
bash runbg.sh b_hog -- python3 hog.py 2200 120
sleep 2
echo "hog: exit=$(cat out/b_hog.exit 2>/dev/null || echo running) smi=$(smi)"
cat out/b_hog.stdout

echo "=== t3 $(date -u +%H:%M:%S) request 1 on the 'ready' host"
bash runbg.sh b_ask1 -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/b_ask1.json
echo "exit=$(cat out/b_ask1.exit) wall_ms=$(cat out/b_ask1.wall_ms)"
head -c 300 out/b_ask1.stderr; echo

echo "=== t4 $(date -u +%H:%M:%S) request 2 on the same 'ready' host"
bash runbg.sh b_ask2 -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/b_ask2.json
echo "exit=$(cat out/b_ask2.exit) wall_ms=$(cat out/b_ask2.wall_ms)"
head -c 300 out/b_ask2.stderr; echo
python3 tg.py keep status --json > out/b_status.json 2>&1
python3 plans.py "status=out/b_status.json" >/dev/null 2>&1 || true

echo "=== t5 $(date -u +%H:%M:%S) cleanup"
python3 tg.py keep stop || true
pkill -f "hog.py 2200" || true
sleep 1
echo "--- smi after: $(smi)"
echo "=== t6 $(date -u +%H:%M:%S) done"
