#!/usr/bin/env bash
# Story 2, leg 1: the read-only plan report + the record's byte-identity across it.
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
export SSL_CERT_FILE=/workspace/e2e-t_559ed8c8/ca/ca.pem
export HTTPS_PROXY=http://127.0.0.1:8443 https_proxy=http://127.0.0.1:8443
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
cd /workspace/e2e-t_559ed8c8

echo "### baseline: runtime.json"
sha256sum "$TYPED_GGUF_HOME/runtime.json"
echo
echo "### command: typed-gguf runtime update --check"
env -u PYTHONPATH venv/bin/typed-gguf runtime update --check
echo "EXIT=$?"
echo
echo "### runtime.json after --check (must be identical)"
sha256sum "$TYPED_GGUF_HOME/runtime.json"
echo
echo "### downloads dir (must be empty: nothing was fetched)"
ls -la "$TYPED_GGUF_HOME/downloads"
echo
echo "### runtime root (must hold only the installed bundle — no staging debris)"
ls -la "$TYPED_GGUF_HOME/runtime"
