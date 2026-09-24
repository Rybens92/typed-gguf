#!/usr/bin/env bash
# A-E5-4 on the real box: the served answer *is* the CLI's answer, on the same warm host.
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
unset PYTHONPATH
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
unset HTTPS_PROXY https_proxy
cd /workspace/e2e-t_559ed8c8
GG=venv/bin/typed-gguf
BASE=http://127.0.0.1:8088
STATE="Customer Brightline Utilities: seats down 41% -> 26%, champion left in July, invoice disputed then paid late."
QUESTION="Escalate to an exec-level conversation this week?"

$GG keep stop > /dev/null 2>&1
$GG serve --host 127.0.0.1 --port 8088 --keep-alive 600 > logs/serve_same_engine.log 2>&1 &
SERVE=$!
for _ in $(seq 1 40); do curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 0.25; done

echo "### command: curl POST \$BASE/v1/decide (native) — cold, spawns the host"
curl -sS -X POST "$BASE/v1/decide" -H 'Content-Type: application/json' \
  -d "{\"state\": \"$STATE\", \"model\": \"qwen3.5-0.8b\", \"questions\": {\"escalate\": {\"type\": \"noul\", \"instructions\": \"$QUESTION\"}}, \"options\": {\"keep_alive\": 600}}" \
  > logs/same_engine_served.json
echo "EXIT=$?"
env -u PYTHONPATH venv/bin/python src/native_summary.py logs/same_engine_served.json

echo
echo "### command: typed-gguf ask --state ... --noul \"escalate=...\" --keep-alive 600   (the CLI, same host)"
$GG ask --state "$STATE" --noul "escalate=$QUESTION" --keep-alive 600 > logs/same_engine_ask.json 2> logs/same_engine_ask.err
echo "EXIT=$?"
env -u PYTHONPATH venv/bin/python src/native_summary.py logs/same_engine_ask.json

echo
echo "### compare (served vs ask): model, noul value, keep pid"
env -u PYTHONPATH venv/bin/python src/compare_answers.py logs/same_engine_served.json logs/same_engine_ask.json
echo "COMPARE_EXIT=$?"

echo
echo "### server log + the host ledger (one host, reused by both callers)"
cat logs/serve_same_engine.log
$GG keep status --json | head -20
kill -TERM "$SERVE" 2>/dev/null; wait "$SERVE" 2>/dev/null
$GG keep stop
