"""Raw-socket probes for the serve wave's two HTTP bounds (t_ba767a2b):

  too_large  — send only the headers of a >1 MiB body and wait: SPEC 2.9 says the server refuses
               it from `Content-Length` alone and never reads the body.
  idle       — open a connection, send half a request, and time how long the server keeps it.
               SPEC 2.9: an idle connection is dropped after 30 s.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time


def too_large(host: str, port: int, path: str, declared: int) -> int:
    with socket.create_connection((host, port), timeout=15) as sock:
        head = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                "Content-Type: application/json\r\nAuthorization: Bearer E2E-SENTINEL-AUTH\r\n"
                f"Content-Length: {declared}\r\n\r\n")
        sock.sendall(head.encode())
        started = time.monotonic()
        chunks = []
        while True:
            try:
                block = sock.recv(4096)
            except socket.timeout:
                break
            if not block:
                break
            chunks.append(block)
            if b"\r\n\r\n" in b"".join(chunks) and len(b"".join(chunks)) > 100:
                break
        elapsed = time.monotonic() - started
    body = b"".join(chunks).decode("utf-8", "replace")
    print(f"--- POST {path} with Content-Length: {declared} and NO body sent")
    print(f"    answered in {elapsed:.3f}s: {body!r}")
    return 0


def idle(host: str, port: int, wait: float) -> int:
    with socket.create_connection((host, port), timeout=wait + 15) as sock:
        sock.sendall(b"POST /v1/decide HTTP/1.1\r\nHost: 127.0.0.1\r\n")
        started = time.monotonic()
        sock.settimeout(wait + 15)
        try:
            block = sock.recv(4096)
        except socket.timeout:
            print(f"--- connection still open after {wait + 15:.0f}s (NOT dropped)")
            return 1
        elapsed = time.monotonic() - started
    print(f"--- half a request, no more bytes: the server closed the connection after "
          f"{elapsed:.1f}s (recv -> {block!r})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("too_large", "idle"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--path", default="/v1/systemone")
    parser.add_argument("--declared", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--wait", type=float, default=45.0)
    opts = parser.parse_args()
    if opts.action == "too_large":
        return too_large(opts.host, opts.port, opts.path, opts.declared)
    return idle(opts.host, opts.port, opts.wait)


if __name__ == "__main__":
    sys.exit(main())
