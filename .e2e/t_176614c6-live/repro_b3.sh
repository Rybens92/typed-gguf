#!/usr/bin/env bash
# (b) BEFORE-receipt, take 3: build the "ready host that cannot serve" state with a *second VRAM
# tenant* (a second data home, which SPEC 2.12 allows: one host per data home). The field case's
# cause was the same — the card was taken after the host had loaded, so its weights are resident but
# every request's fresh context no longer fits.
set -u
cd /work/t176614c6-live
source env.sh
mkdir -p out homeB

Q="cause=What is the root cause of the empty invoice list, according to the state?:deploy|infra|provider"
smi() { nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }
BENV="TYPED_GGUF_HOME=/work/t176614c6-live/homeB HOME=/work/t176614c6-live/homeB"
bhub() { env $BENV "$@"; }

echo "=== t0 $(date -u +%H:%M:%S) quiet start"
python3 tg.py keep stop >/dev/null 2>&1 || true
bhub python3 tg.py keep stop >/dev/null 2>&1 || true
rm -rf home/fit homeB/fit out/b3_*.json
echo "--- smi: $(smi)"

echo "=== t1 $(date -u +%H:%M:%S) home A: the product host (default policy, 36 layers)"
bash runbg.sh b3_hostA -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/b3_hostA.json
echo "A: exit=$(cat out/b3_hostA.exit) wall_ms=$(cat out/b3_hostA.wall_ms) smi=$(smi)"

echo "=== t2 $(date -u +%H:%M:%S) home B: the second tenant takes what is left of the card"
bash runbg.sh b3_hostB -- env $BENV python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --fit-target 0 --n-ctx 8192 --out out/b3_hostB.json
echo "B: exit=$(cat out/b3_hostB.exit) wall_ms=$(cat out/b3_hostB.wall_ms) smi=$(smi)"
head -c 200 out/b3_hostB.stderr; echo

echo "=== t3 $(date -u +%H:%M:%S) request 1 on host A (the card is gone)"
bash runbg.sh b3_ask1 -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/b3_ask1.json
echo "ask1: exit=$(cat out/b3_ask1.exit) wall_ms=$(cat out/b3_ask1.wall_ms)"
head -c 400 out/b3_ask1.stderr; echo

echo "=== t4 $(date -u +%H:%M:%S) request 2 on the same host (is it still 'ready'?)"
bash runbg.sh b3_ask2 -- python3 tg.py ask --state @states/state_short.txt --choice "$Q" \
  --model "$MODEL" --keep-alive 10m --out out/b3_ask2.json
echo "ask2: exit=$(cat out/b3_ask2.exit) wall_ms=$(cat out/b3_ask2.wall_ms)"
head -c 400 out/b3_ask2.stderr; echo
python3 tg.py keep status --json > out/b3_status.json 2>&1
python3 -c "import json; d=json.load(open('out/b3_status.json')); print('status: state=%s pid=%s requests=%s placement=%s' % (d.get('state'), d.get('pid'), d.get('requests'), json.dumps(d.get('placement'))))" 2>/dev/null

echo "=== t5 $(date -u +%H:%M:%S) cleanup"
python3 tg.py keep stop >/dev/null 2>&1 || true
bhub python3 tg.py keep stop >/dev/null 2>&1 || true
echo "--- smi after: $(smi)"
echo "=== done $(date -u +%H:%M:%S)"
