"""The warm engine host (SPEC 2.12, card t_7e24cea4): one resident model, at most one host.

The surface:

* `identity` — how long a host stays (`--keep-alive`), and which host a request belongs to (the
  key: model + sha + the placement-affecting options);
* `state` — the ledger in `<data-home>/keep/`: the record, the socket path, liveness, cleanup;
* `host` — the server side: load once, answer decisions over a unix socket, exit on idle;
* `client` — the caller side: reuse/swap/spawn transparently, fall back inline once on a broken
  host, and the `keep status` / `keep stop` verbs.

Window into the whole thing:
`tests/test_keep.py` (identity + ledger), `tests/test_keep_host.py` (the server),
`tests/test_keep_client.py` (reuse/swap/stale/fallback), `tests/test_keep_cli.py` (the CLI wiring)
and `tests/test_keep_live.py` (a real 4B staying resident).
"""
from __future__ import annotations

from typed_gguf.keep import client, host, identity, state
from typed_gguf.keep.client import Client, KeepUnavailable
from typed_gguf.keep.identity import (
                                      DEFAULT_KEEP_ALIVE,
                                      KEEP_ALIVE_ENV,
                                      KEEP_ALIVE_FLAG,
                                      KeepKey,
                                      parse_duration,
                                      resolve_keep_alive,
                                      supported,
)
from typed_gguf.keep.state import HostRecord

__all__ = ["Client", "DEFAULT_KEEP_ALIVE", "HostRecord", "KEEP_ALIVE_ENV", "KEEP_ALIVE_FLAG",
           "KeepKey", "KeepUnavailable", "client", "host", "identity", "parse_duration",
           "resolve_keep_alive", "state", "supported"]
