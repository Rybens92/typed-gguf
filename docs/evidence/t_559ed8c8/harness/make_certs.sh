#!/usr/bin/env bash
# Local CA + a leaf certificate for api.github.com / github.com, so the fixture server in
# src/fixture_server.py can stand in for GitHub over real HTTPS (the update path does a real
# TLS GET; nothing in the app is patched).  Nothing here leaves this scratch dir.
set -euo pipefail

CA_DIR="${1:-/workspace/e2e-t_559ed8c8/ca}"
mkdir -p "$CA_DIR"
cd "$CA_DIR"

if [ ! -f ca.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 3 \
    -subj "/CN=e2e-typed-gguf-local-ca" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" >/dev/null 2>&1
fi

cat > server.ext <<'EOF'
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:api.github.com,DNS:github.com,DNS:localhost,IP:127.0.0.1
EOF

if [ ! -f server.pem ]; then
  openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr \
    -subj "/CN=api.github.com" >/dev/null 2>&1
  openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -out server.pem -days 3 -extfile server.ext >/dev/null 2>&1
fi

openssl x509 -in server.pem -noout -subject -ext subjectAltName
echo "ca:     $CA_DIR/ca.pem"
echo "server: $CA_DIR/server.pem"
