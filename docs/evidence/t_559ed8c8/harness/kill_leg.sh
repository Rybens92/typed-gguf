#!/usr/bin/env bash
# Story 2, leg 4: `kill -9` in the middle of a real (throttled) download.
# SPEC 2.8 step 4: "a kill -9 anywhere earlier keeps the old runtime active".
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
export SSL_CERT_FILE=/workspace/e2e-t_559ed8c8/ca/ca.pem
export HTTPS_PROXY=http://127.0.0.1:8443 https_proxy=http://127.0.0.1:8443
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
cd /workspace/e2e-t_559ed8c8
GG="env -u PYTHONPATH venv/bin/typed-gguf"

before=$(sha256sum "$TYPED_GGUF_HOME/runtime.json" | awk '{print $1}')
echo "### record before: $before"
printf '{"releases": "ok", "asset": "ok", "throttle_ms": 100}\n' > fixtures/scenario.json
echo "### scenario: $(cat fixtures/scenario.json)"
rm -f home/downloads/llama-b99998*.part

echo
echo "### command: typed-gguf runtime update --tag b99998 --json   (throttled, then SIGKILL)"
$GG runtime update --tag b99998 --json > logs/kill_update.out 2> logs/kill_update.err &
CLI=$!
echo "pid=$CLI"
part=""
for _ in $(seq 1 200); do
  part=$(ls home/downloads/llama-b99998*.part 2>/dev/null | head -1 || true)
  if [ -n "$part" ] && [ "$(stat -c%s "$part")" -gt 2097152 ]; then break; fi
  sleep 0.3
done
echo "partial download: ${part:-<none>} $( [ -n "$part" ] && stat -c%s "$part" || echo 0 ) bytes"
kill -9 "$CLI"
wait "$CLI"; echo "cli exit after kill -9: $? (137 = SIGKILL)"
sleep 1
echo
echo "### record after the kill (must equal the 'before' hash)"
sha256sum "$TYPED_GGUF_HOME/runtime.json"
after=$(sha256sum "$TYPED_GGUF_HOME/runtime.json" | awk '{print $1}')
[ "$before" = "$after" ] && echo "RECORD BYTE-IDENTICAL: yes" || echo "RECORD BYTE-IDENTICAL: NO"
echo
echo "### runtime root (no b99998 bundle, no .pending-* staging debris)"
ls -la "$TYPED_GGUF_HOME/runtime"
echo
echo "### downloads after the kill (the resumable .part stays, by design)"
ls -la "$TYPED_GGUF_HOME/downloads"
echo
echo "### the still-active runtime, as the app itself reports it"
$GG version; echo "EXIT=$?"
echo
echo "### no process of the killed update survives"
ps -eo pid,stat,args | grep -E "typed-gguf|fixture" | grep -v grep
echo
echo "### command (retry, unthrottled): typed-gguf runtime update --tag b99998 --json"
printf '{"releases": "ok", "asset": "ok", "throttle_ms": 0}\n' > fixtures/scenario.json
$GG runtime update --tag b99998 --json > logs/kill_update_retry.json 2> logs/kill_update_retry.err
echo "EXIT=$?"
env -u PYTHONPATH venv/bin/python src/record_summary.py "$TYPED_GGUF_HOME/runtime.json"
echo
echo "### command: typed-gguf runtime rollback  (back to the b11026 bundle for the serve story)"
$GG runtime rollback; echo "EXIT=$?"
$GG version; echo "EXIT=$?"
