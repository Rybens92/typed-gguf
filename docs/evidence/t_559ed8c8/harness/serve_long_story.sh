#!/usr/bin/env bash
# Story 1, take 2: the same SDK client, with an HTTP timeout that fits this box's engine speed.
# Cold + warm typed answers over loopback, `served_by=host`, then the teardown/leak checks.
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
unset PYTHONPATH
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
unset HTTPS_PROXY https_proxy
cd /workspace/e2e-t_559ed8c8
GG=venv/bin/typed-gguf
BASE=http://127.0.0.1:8088

echo "### command: typed-gguf keep stop   (start from a clean ledger)"
$GG keep stop; echo "EXIT=$?"
echo
echo "### command: typed-gguf serve --host 127.0.0.1 --port 8088 --keep-alive 600"
$GG serve --host 127.0.0.1 --port 8088 --keep-alive 600 > logs/serve_long.log 2>&1 &
SERVE=$!
echo "serve pid=$SERVE"
for _ in $(seq 1 40); do
  if curl -s --max-time 2 "$BASE/health" > /dev/null 2>&1; then break; fi
  sleep 0.25
done
echo
echo "### curl \$BASE/health   (cold: nothing resident)"
curl -sS "$BASE/health"; echo
curl -sS "$BASE/health" > logs/health_cold.json
echo
echo "### command: sdkvenv/bin/python src/sdk_client_long.py --base-url $BASE"
TYPESAFE_API_KEY=local TYPESAFE_BASE_URL="$BASE" \
  sdkvenv/bin/python src/sdk_client_long.py --base-url "$BASE" --out logs/sdk_body
SDK_EXIT=$?
echo "SDK client EXIT=$SDK_EXIT"
echo
echo "### curl \$BASE/health   (warm: the host is resident)"
curl -sS "$BASE/health"; echo
curl -sS "$BASE/health" > logs/health_warm.json
echo
echo "### the server's own request log (the served_by column is the server's, not a claim)"
cat logs/serve_long.log
echo
echo "### command: kill -TERM $SERVE"
kill -TERM "$SERVE" 2>/dev/null; wait "$SERVE" 2>/dev/null; echo "serve exit=$?"
sleep 2
echo "### port after the server stops"
curl -s --max-time 2 "$BASE/health" || echo "(port 8088 closed: curl exit $?)"
echo
echo "### command: typed-gguf keep stop   (the host the gate's own shell stops too)"
$GG keep stop; echo "EXIT=$?"
sleep 1
echo
echo "### leaks after teardown"
echo "--- keep ledger:"; ls -la "$TYPED_GGUF_HOME/keep"
echo "--- keep status:"; $GG keep status; echo "EXIT=$?"
echo "--- processes:"; ps -eo pid,args | grep -E "keep_host|keep.host|serve --host|llama" | grep -v grep || echo "(none)"
echo "--- host.json / socket still there?"; ls "$TYPED_GGUF_HOME/keep/host.json" "$TYPED_GGUF_HOME/keep"/*.sock 2>&1
echo
echo "### SDK_EXIT was $SDK_EXIT (0 = every check in the gate's verify() passed)"
