#!/usr/bin/env bash
# Story 1: an installed wheel's `serve` + a real typesafe-sdk 0.7.1 client, on a real model.
# Cold then warm, `served_by=host`, and the teardown checked for leaks.
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
unset PYTHONPATH
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
unset HTTPS_PROXY https_proxy
cd /workspace/e2e-t_559ed8c8
GG=venv/bin/typed-gguf
BASE=http://127.0.0.1:8088

echo "### keep ledger before (must be empty)"
ls -la "$TYPED_GGUF_HOME/keep" 2>&1
echo
echo "### command: typed-gguf serve --host 127.0.0.1 --port 8088 --keep-alive 600"
$GG serve --host 127.0.0.1 --port 8088 --keep-alive 600 > logs/serve.log 2>&1 &
SERVE=$!
echo "serve pid=$SERVE"
for _ in $(seq 1 40); do
  if curl -s --max-time 2 "$BASE/health" > /dev/null 2>&1; then break; fi
  sleep 0.25
done

echo
echo "### command: curl -s \$BASE/health   (before any decision: nothing is resident)"
curl -sS "$BASE/health"; echo
echo
echo "### command: curl -s \$BASE/v1/models"
curl -sS "$BASE/v1/models"; echo
echo
echo "### command: <sdk-venv>/bin/python tools/host_gate_serve_client.py --base-url $BASE"
echo "###          (typesafe-sdk $(sdkvenv/bin/pip show typesafe-sdk 2>/dev/null | awk '/^Version:/{print $2}'))"
TYPESAFE_API_KEY=local TYPESAFE_BASE_URL=$BASE \
  sdkvenv/bin/python /workspace/ggufone/tools/host_gate_serve_client.py \
  --base-url "$BASE" --out logs/sdk_body.json
echo "client EXIT=$?"
echo
echo "### the served body the SDK itself parsed (logs/sdk_body.json)"
cat logs/sdk_body.json
echo
echo "### command: curl -s \$BASE/health   (after: the host is resident)"
curl -sS "$BASE/health"; echo
echo
echo "### command: curl -s -X POST \$BASE/v1/decide (native format, engine.keep block)"
curl -sS -X POST "$BASE/v1/decide" -H 'Content-Type: application/json' \
  -d '{"state": "Customer Brightline Utilities: seats down 41% -> 26%, champion left, invoice disputed then paid late.", "model": "qwen3.5-0.8b", "questions": {"escalate": {"type": "noul", "instructions": "Escalate to an exec-level conversation this week?"}}, "options": {"keep_alive": 600}}' \
  > logs/decide_native.json
echo "curl EXIT=$?"
env -u PYTHONPATH venv/bin/python src/native_summary.py logs/decide_native.json
echo
echo "### command: typed-gguf keep status --json"
$GG keep status --json; echo "EXIT=$?"
echo
echo "### the server's own log (served_by per request)"
cat logs/serve.log
echo
echo "### command: kill -TERM $SERVE   (teardown)"
kill -TERM "$SERVE" 2>/dev/null; wait "$SERVE" 2>/dev/null; echo "serve exit=$?"
sleep 2
echo
echo "### after serve exits: keep ledger, keep status, processes, port"
ls -la "$TYPED_GGUF_HOME/keep" 2>&1
$GG keep status; echo "keep status EXIT=$?"
ps -eo pid,stat,args | grep -E "keep_host|typed-gguf serve|fake_keep" | grep -v grep || echo "(no keep/serve process)"
curl -s --max-time 2 "$BASE/health" || echo "(port 8088 is closed: curl exit $?)"
echo
echo "### command: typed-gguf keep stop   (after the fact: nothing to stop)"
$GG keep stop; echo "EXIT=$?"
