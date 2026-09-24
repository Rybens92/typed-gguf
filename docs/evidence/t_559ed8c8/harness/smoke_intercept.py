"""Smoke test the interception: does the *unmodified* urllib in the fresh venv reach our local
fixture as if it were GitHub, trusting the local CA via SSL_CERT_FILE?
"""
from __future__ import annotations

import json
import os
import urllib.request

URL = ("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20")
print("SSL_CERT_FILE:", os.environ.get("SSL_CERT_FILE"))
print("proxy env:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()} or "none")

request = urllib.request.Request(URL, headers={
    "Accept": "application/vnd.github+json",
    "User-Agent": "typed-gguf/0.2.3",
})
with urllib.request.urlopen(request, timeout=15) as response:
    body = response.read()
print("status:", response.status, "bytes:", len(body))
payload = json.loads(body)
print("tags:", [entry["tag_name"] for entry in payload])
print("peer:", response.fp.raw._sock.getpeername() if hasattr(response.fp, "raw") else "?")
