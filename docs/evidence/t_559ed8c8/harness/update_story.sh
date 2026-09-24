#!/usr/bin/env bash
# Story 2, legs 2-3: a real update (download -> extract -> probe -> atomic switch -> record) and
# the rollback that returns.  Every command is echoed with its exit code; the record's hash and
# the record's own `previous` block are quoted, not asserted.
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
export SSL_CERT_FILE=/workspace/e2e-t_559ed8c8/ca/ca.pem
export HTTPS_PROXY=http://127.0.0.1:8443 https_proxy=http://127.0.0.1:8443
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
cd /workspace/e2e-t_559ed8c8
GG="env -u PYTHONPATH venv/bin/typed-gguf"

echo "### 0. before: record + doctor"
sha256sum "$TYPED_GGUF_HOME/runtime.json"
$GG doctor --json > logs/doctor_before.json 2>logs/doctor_before.err; echo "doctor before EXIT=$?"
echo
echo "### 1. command: typed-gguf runtime update --json"
$GG runtime update --json > logs/update.json 2> logs/update.err
echo "EXIT=$?"
echo "--- stderr:"; cat logs/update.err
echo "--- stdout:"; cat logs/update.json
echo
echo "### 2. record after the update"
sha256sum "$TYPED_GGUF_HOME/runtime.json"
echo
echo "### 3. what the record now says (dir / tag / build / variant / previous / update_from)"
env -u PYTHONPATH venv/bin/python src/record_summary.py "$TYPED_GGUF_HOME/runtime.json"
echo
echo "### 4. runtime root after the update (both bundles on disk: nothing was deleted)"
ls -la "$TYPED_GGUF_HOME/runtime"
echo
echo "### 5. command: typed-gguf doctor (build now comes from the new bundle's own probe)"
$GG doctor; echo "EXIT=$?"
echo
echo "### 6. command: typed-gguf version"
$GG version; echo "EXIT=$?"
echo
echo "### 7. command: typed-gguf runtime update --check (already at the target)"
$GG runtime update --check; echo "EXIT=$?"
echo
echo "### 8. command: typed-gguf runtime rollback --json"
$GG runtime rollback --json > logs/rollback.json 2> logs/rollback.err
echo "EXIT=$?"
cat logs/rollback.json
echo
echo "### 9. record after the rollback"
sha256sum "$TYPED_GGUF_HOME/runtime.json"
env -u PYTHONPATH venv/bin/python src/record_summary.py "$TYPED_GGUF_HOME/runtime.json"
echo
echo "### 10. command: typed-gguf doctor (back on the old bundle)"
$GG doctor; echo "EXIT=$?"
echo
echo "### 11. command: typed-gguf runtime rollback (nothing left to roll back)"
$GG runtime rollback; echo "EXIT=$?"
