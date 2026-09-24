"""The offline GitHub stand-in, as an HTTPS **proxy** (no system config touched).

Why a proxy: `runtime update` asks for `https://api.github.com/...` and downloads from
`https://github.com/...` (both from `runtime.lock` itself), and nothing in the app exposes a
base-URL override — that is the point of the story, so nothing is patched either. The one seam
left is the transport: point the process at `HTTPS_PROXY` and this proxy

  1. answers `CONNECT api.github.com:443` with the stdlib's own "200 Connection Established",
  2. wraps the tunnel with TLS using a leaf certificate for api.github.com / github.com
     (src/make_certs.sh) — the client trusts it because `SSL_CERT_FILE` names the local CA, so
     *the client's* TLS verification still runs, against our root,
  3. speaks HTTP/1.1 inside the tunnel: the release list, or the asset bytes.

Everything the app asks for is logged (path, `User-Agent`, bytes sent) so the receipt can quote
the real requests. Content is driven by fixtures/scenario.json, re-read per request:

    {"releases": "ok" | "no_target_asset" | "http_500",
     "asset":    "ok" | "missing" | "short",
     "throttle_ms": 0}      # a slow asset download is what the `kill -9` leg needs
"""
from __future__ import annotations

import json
import os
import pathlib
import socketserver
import ssl
import sys
import time
import urllib.parse

ROOT = pathlib.Path(os.environ.get("E2E_FIXTURE_ROOT", "/workspace/e2e-t_559ed8c8/fixtures"))
CA_DIR = pathlib.Path(os.environ.get("E2E_CA_DIR", "/workspace/e2e-t_559ed8c8/ca"))
LOG = pathlib.Path(os.environ.get("E2E_FIXTURE_LOG", "/workspace/e2e-t_559ed8c8/logs/fixture.log"))
PORT = int(os.environ.get("E2E_FIXTURE_PORT", "8443"))
CHUNK = 64 * 1024


def log(line: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")
    sys.stderr.write(f"{stamp} {line}\n")
    sys.stderr.flush()


def scenario() -> dict:
    try:
        return json.loads((ROOT / "scenario.json").read_text())
    except Exception:
        return {"releases": "ok", "asset": "ok", "throttle_ms": 0}


def releases_payload() -> list:
    declared = json.loads((ROOT / "releases.json").read_text())
    mode = scenario().get("releases")
    if mode == "no_target_asset":
        for release in declared:
            release["assets"] = [asset for asset in release.get("assets", [])
                                 if "nonmatching" in asset.get("name", "")]
    if mode == "wrong_digest":
        # the listing advertises a digest the bytes will not match: the app must refuse the
        # archive by SHA-256 and leave runtime.json alone
        for release in declared:
            for asset in release.get("assets", []):
                asset["digest"] = "sha256:" + "f" * 64
    return declared


def http_response(status: int, body: bytes, content_type: str = "application/json",
                  extra: dict[str, str] | None = None) -> bytes:
    head = [f"HTTP/1.1 {status} {'OK' if status == 200 else 'Err'}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body)}",
            "Connection: keep-alive"]
    for key, value in (extra or {}).items():
        head.append(f"{key}: {value}")
    return ("\r\n".join(head) + "\r\n\r\n").encode("utf-8") + body


class Proxy(socketserver.StreamRequestHandler):
    """One client connection: CONNECT, TLS, then plain HTTP/1.1 inside the tunnel."""

    def _read_headers(self, stream) -> bytes:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            block = stream.recv(4096)
            if not block:
                return b""
            buffer += block
        return buffer

    def handle(self) -> None:  # noqa: C901 - a two-stage protocol in one place
        first = self._read_headers(self.connection)
        if not first:
            return
        request_line = first.split(b"\r\n", 1)[0].decode("latin-1")
        if not request_line.upper().startswith("CONNECT"):
            log(f"non-CONNECT request: {request_line!r}")
            self.connection.sendall(http_response(405, b"{}"))
            return
        authority = request_line.split()[1]
        log(f"CONNECT {authority}")
        self.connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CA_DIR / "server.pem", CA_DIR / "server.key")
        try:
            tls = context.wrap_socket(self.connection, server_side=True)
        except ssl.SSLError as exc:
            log(f"TLS handshake failed: {exc.__class__.__name__}: {exc}")
            return
        with tls:
            while True:
                raw = self._read_headers(tls)
                if not raw:
                    return
                head, _, rest = raw.partition(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                method, target, *_ = lines[0].split(" ")
                headers = dict(
                    line.split(": ", 1) for line in lines[1:] if ": " in line)
                try:
                    if not self._serve(tls, method, target, headers):
                        return
                except (BrokenPipeError, ConnectionResetError) as exc:
                    log(f"client vanished: {exc.__class__.__name__}")
                    return

    def _serve(self, stream, method: str, target: str, headers: dict[str, str]) -> bool:
        parsed = urllib.parse.urlparse(target)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        agent = headers.get("User-Agent")
        mode = scenario()
        if "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[1]
            for release in releases_payload():
                if release.get("tag_name") == tag:
                    log(f"GET {path} ua={agent!r} -> tag {tag}")
                    stream.sendall(http_response(200, json.dumps(release).encode()))
                    return True
            log(f"GET {path} -> 404")
            stream.sendall(http_response(404, b'{"message":"Not Found"}'))
            return True
        if path.endswith("/releases"):
            if mode.get("releases") == "http_500":
                log(f"GET {path} ua={agent!r} -> 500 (scenario)")
                stream.sendall(http_response(500, b'{"message":"boom"}'))
                return True
            payload = releases_payload()
            log(f"GET {path} ua={agent!r} per_page={query.get('per_page')} "
                f"-> {len(payload)} entries")
            stream.sendall(http_response(200, json.dumps(payload).encode()))
            return True
        if "/releases/download/" in path:
            return self._serve_asset(stream, path.rsplit("/", 1)[1], agent)
        log(f"GET {path} ua={agent!r} -> 404 (unhandled)")
        stream.sendall(http_response(404, b'{"message":"Not Found"}'))
        return True

    def _serve_asset(self, stream, asset: str, agent: str | None) -> bool:
        mode = scenario()
        source = ROOT / "assets" / asset
        if mode.get("asset") == "missing" or not source.exists():
            log(f"GET asset={asset} ua={agent!r} -> 404 ({mode.get('asset')})")
            stream.sendall(http_response(404, b'{"message":"Not Found"}'))
            return True
        size = source.stat().st_size
        if mode.get("asset") == "short":
            size = size // 3
        log(f"GET asset={asset} ua={agent!r} -> {size} bytes "
            f"(mode={mode.get('asset')}, throttle={mode.get('throttle_ms')} ms)")
        stream.sendall(("HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                        f"Content-Length: {size}\r\nConnection: keep-alive\r\n\r\n").encode())
        throttle = float(mode.get("throttle_ms", 0)) / 1000.0
        sent = 0
        with open(source, "rb") as handle:
            while sent < size:
                block = handle.read(min(CHUNK, size - sent))
                if not block:
                    break
                stream.sendall(block)
                sent += len(block)
                if throttle:
                    time.sleep(throttle)
        return True


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with Server(("127.0.0.1", PORT), Proxy) as server:
        print(f"github fixture proxy on 127.0.0.1:{PORT} root={ROOT} log={LOG}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
