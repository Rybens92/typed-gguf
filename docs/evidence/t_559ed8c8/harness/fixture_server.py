"""The offline GitHub stand-in for the `runtime update` story.

Binds 0.0.0.0:443 with a certificate for api.github.com / github.com (src/make_certs.sh) and
answers exactly the two URLs the update path asks for:

  GET /repos/<repo>/releases?per_page=N        -> fixtures/releases.json (or a variant)
  GET /repos/<repo>/releases/tags/<tag>        -> the one release from that list
  GET /<repo>/releases/download/<tag>/<asset>  -> fixtures/assets/<asset>

Everything is driven by fixtures/scenario.json, re-read per request:

    {"releases": "ok" | "no_target_asset" | "http_500",
     "asset":    "ok" | "missing" | "short",
     "throttle_ms": 0}

The throttle makes the asset download slow enough to `kill -9` mid-transfer.

No TLS session is cached and every request is appended to $E2E_FIXTURE_LOG (one line, plus the
client's User-Agent), so the receipt can quote what the app actually asked for.
"""
from __future__ import annotations

import http.server
import json
import os
import pathlib
import ssl
import sys
import time
import urllib.parse

ROOT = pathlib.Path(os.environ.get("E2E_FIXTURE_ROOT", "/workspace/e2e-t_559ed8c8/fixtures"))
CA_DIR = pathlib.Path(os.environ.get("E2E_CA_DIR", "/workspace/e2e-t_559ed8c8/ca"))
LOG = pathlib.Path(os.environ.get("E2E_FIXTURE_LOG", "/workspace/e2e-t_559ed8c8/logs/fixture.log"))
PORT = int(os.environ.get("E2E_FIXTURE_PORT", "443"))
CHUNK = 64 * 1024


def scenario() -> dict:
    path = ROOT / "scenario.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"releases": "ok", "asset": "ok", "throttle_ms": 0}


def releases_payload() -> list:
    """The release list, in the API's order (newest first)."""
    declared = json.loads((ROOT / "releases.json").read_text())
    mode = scenario().get("releases", "ok")
    if mode == "no_target_asset":
        # every release loses the asset for this host: the naming rule finds nothing
        for release in declared:
            release["assets"] = [asset for asset in release.get("assets", [])
                                 if "nonmatching" in asset.get("name", "")]
    return declared


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "github-fixture/1.0"

    def log_message(self, fmt: str, *args) -> None:  # keep the stdlib chatter out of the receipt
        pass

    def _record(self, note: str = "") -> None:
        line = (f"{time.strftime('%H:%M:%S')} {self.command} {self.path} "
                f"ua={self.headers.get('User-Agent')!r} {note}\n")
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(line)
        sys.stderr.write(line)
        sys.stderr.flush()

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = urllib.parse.urlparse(self.path).path
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[1]
            for release in releases_payload():
                if release.get("tag_name") == tag:
                    self._record(f"tag={tag}")
                    return self._send_json(release)
            self._record(f"tag={tag} NOT FOUND")
            return self._send_json({"message": "Not Found"}, 404)
        if path.endswith("/releases"):
            mode = scenario().get("releases", "ok")
            if mode == "http_500":
                self._record("releases -> 500")
                return self._send_json({"message": "boom"}, 500)
            payload = releases_payload()
            self._record(f"releases per_page={query.get('per_page')} -> {len(payload)} entries")
            return self._send_json(payload)
        if "/releases/download/" in path:
            asset = path.rsplit("/", 1)[1]
            return self._send_asset(asset)
        self._record("unhandled")
        self._send_json({"message": "Not Found"}, 404)

    def _send_asset(self, asset: str) -> None:
        mode = scenario().get("asset", "ok")
        source = ROOT / "assets" / asset
        if mode == "missing" or not source.exists():
            self._record(f"asset={asset} -> 404 ({mode})")
            return self._send_json({"message": "Not Found"}, 404)
        size = source.stat().st_size
        if mode == "short":
            size = size // 3
        self._record(f"asset={asset} -> {size} bytes (mode={mode})")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        throttle = float(scenario().get("throttle_ms", 0)) / 1000.0
        sent = 0
        with open(source, "rb") as handle:
            while sent < size:
                block = handle.read(min(CHUNK, size - sent))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    self._record(f"asset={asset} client vanished after {sent} bytes")
                    return
                sent += len(block)
                if throttle:
                    time.sleep(throttle)


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CA_DIR / "server.pem", CA_DIR / "server.key")
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"github fixture on https://0.0.0.0:{PORT} root={ROOT} log={LOG}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
