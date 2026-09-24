#!/usr/bin/env bash
# Story 3, part 2b: the three legs that need a modified *environment* (not a modified fixture).
# (part 2's helper could not express them; these are the same legs, run plainly.)
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
export SSL_CERT_FILE=/workspace/e2e-t_559ed8c8/ca/ca.pem
export HTTPS_PROXY=http://127.0.0.1:8443 https_proxy=http://127.0.0.1:8443
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
cd /workspace/e2e-t_559ed8c8
REC="$TYPED_GGUF_HOME/runtime.json"

show() {
  echo "--- record before: $1"
  echo "--- record after:  $2"
  [ "$1" = "$2" ] && echo "--- runtime.json byte-identical: YES" \
                  || echo "--- runtime.json byte-identical: NO"
  echo "--- fixture saw:"; sed 's/^/      /' logs/fixture.log
  echo "--- runtime root:"; ls "$TYPED_GGUF_HOME/runtime" | sed 's/^/      /'
}

before=$(sha256sum "$REC" | awk '{print $1}')

echo
echo "======================================================================================="
echo "### U1. TYPED_GGUF_RUNTIME_DIR is set -> the rung update may not touch"
: > logs/fixture.log
echo "### command: TYPED_GGUF_RUNTIME_DIR=/opt/managed-runtime env -u PYTHONPATH venv/bin/typed-gguf runtime update --check"
env TYPED_GGUF_RUNTIME_DIR=/opt/managed-runtime env -u PYTHONPATH venv/bin/typed-gguf runtime update --check; echo "EXIT=$?"
show "$before" "$(sha256sum "$REC" | awk '{print $1}')"

echo
echo "======================================================================================="
echo "### U2. nothing installed at all (empty data home) -> E_RUNTIME_MISSING"
: > logs/fixture.log
echo "### command: TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home-clean env -u PYTHONPATH venv/bin/typed-gguf runtime update"
env TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home-clean env -u PYTHONPATH venv/bin/typed-gguf runtime update; echo "EXIT=$?"
show "$before" "$(sha256sum "$REC" | awk '{print $1}')"
ls -la /workspace/e2e-t_559ed8c8/home-clean

echo
echo "======================================================================================="
echo "### U3. no reachable network -> E_DOWNLOAD_FAILED naming the URL (offline --check)"
: > logs/fixture.log
echo "### command: HTTPS_PROXY=http://127.0.0.1:1 https_proxy=http://127.0.0.1:1 env -u PYTHONPATH venv/bin/typed-gguf runtime update --check"
env HTTPS_PROXY=http://127.0.0.1:1 https_proxy=http://127.0.0.1:1 \
  env -u PYTHONPATH venv/bin/typed-gguf runtime update --check; echo "EXIT=$?"
show "$before" "$(sha256sum "$REC" | awk '{print $1}')"
