"""The keep-alive identity: how long a host stays, and which host a request belongs to.

Milestone: E4 (SPEC 2.12, card t_7e24cea4). Two questions live here, both pure bookkeeping:

* **how long** — `--keep-alive <dur|0>` on `ask`/`run`, then `$TYPED_GGUF_KEEP_ALIVE`, then
  `DEFAULT_KEEP_ALIVE` (10 minutes). `0` is the explicit "no host" switch (today's behaviour:
  answer and unload), and it survives the chain in every spelling;
* **which host** — `KeepKey`: the resolved model (path + the registry's sha256) plus every option
  that changes what the loader does (backend, n_ctx, n_seq_max, kv_type, threads, fit). A request
  whose key differs from the resident host's is a *swap* (unload, load, restart the countdown),
  never a second host: one model resident at a time is the whole point on an 8 GiB device.

Everything that cannot move the model — the questions, the state text, `--format`, the readout,
the cue, the temperature, a `--state-id` — is deliberately *not* in the key: two `ask` calls about
different tickets must land on the same warm host, or the feature would never fire.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typed_gguf.errors import UserError

#: SPEC 2.12: "a few minutes" for the request became ten of them — long enough for a human
#: between two `ask` calls, short enough that a forgotten host frees the device.
DEFAULT_KEEP_ALIVE = 600.0
#: The env half of the precedence chain (`--keep-alive` > this > `DEFAULT_KEEP_ALIVE`).
KEEP_ALIVE_ENV = "TYPED_GGUF_KEEP_ALIVE"
#: The flag half, named in the error a bad value produces.
KEEP_ALIVE_FLAG = "--keep-alive"
_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}
_DURATION = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smh]?)$")


def parse_duration(value: str | int | float, *, source: str = KEEP_ALIVE_FLAG) -> float:
    """`600` / `10m` / `5s` / `1h` / `0` -> seconds. Anything else is the user's typo, named.

    A bare number is seconds (what the request asked for), and `0` is legal because it is the
    documented "do not keep a host" switch. Negative durations do not exist: a timeout that
    cannot expire is not a feature.
    """
    if isinstance(value, bool):                       # bool is an int; `True` is not "1 second"
        raise _bad_duration(value, source)
    text = repr(float(value)) if isinstance(value, (int, float)) else str(value).strip()
    match = _DURATION.match(text)
    if match is None:
        raise _bad_duration(value, source)
    return float(match.group("value")) * _UNITS[match.group("unit")]


def _bad_duration(value: object, source: str) -> UserError:
    return UserError(
        f"{source} must be a duration in seconds (`600`), with a unit (`10m`, `5s`, `1h`) or `0` "
        f"to keep no host at all (got {value!r})", code="E_UNKNOWN_KEY")


def resolve_keep_alive(flag: str | int | float | None = None, *,
                       environ: Mapping[str, str] | None = None) -> float:
    """The precedence chain (SPEC 2.12): `--keep-alive` flag > `$TYPED_GGUF_KEEP_ALIVE` > 600 s."""
    if flag is not None and flag != "":
        return parse_duration(flag, source=KEEP_ALIVE_FLAG)
    raw = (os.environ if environ is None else environ).get(KEEP_ALIVE_ENV)
    if raw is None or str(raw).strip() == "":
        return float(DEFAULT_KEEP_ALIVE)
    return parse_duration(raw, source=f"${KEEP_ALIVE_ENV}")


def supported(*, platform: str | None = None) -> bool:
    """Is a warm host possible here? (A unix socket in the data home — Linux/macOS, not Windows.)

    `platform` is the seam the Windows fallback is tested through; the live answer is what the
    interpreter's own socket module offers. A host that cannot be reached has to *say so* rather
    than be silently absent: `cli` prints the named warning and runs inline.
    """
    name = platform if platform is not None else os.sys.platform
    if name.startswith("win"):
        return False
    return hasattr(socket, "AF_UNIX")


@dataclass(frozen=True, slots=True)
class KeepKey:
    """The identity of a warm host: one model placed one way (SPEC 2.12)."""

    model_path: str
    backend: str = "auto"
    n_ctx: int | None = None
    n_seq_max: int | None = None
    kv_type: str = "auto"
    threads: int | None = None
    fit: bool = True
    fit_target_mb: int | None = None
    fit_ctx: int | None = None
    fit_cache: bool = True
    #: the registry's recorded sha256 for this file ("" when the alias carries none). It is the
    #: free half of "path + sha": the path alone would let a *replaced* file keep a warm host
    #: whose weights no longer match the alias.
    model_sha: str = ""

    @classmethod
    def of(cls, request: Any, *, model_path: str, model_sha: str = "",
           fit: Mapping[str, Any] | None = None) -> KeepKey:
        """The key a request resolves to, from the request's own options + the fit arguments.

        `request` is a `schema.Request` (anything with `.options`): only the placement-affecting
        fields are read, so a caller that changed nothing about the load lands on the same host.
        """
        options = getattr(request, "options", None)
        args = dict(fit or {})
        return cls(
            model_path=str(model_path),
            backend=str(getattr(options, "backend", "auto") or "auto"),
            n_ctx=_optional_int(getattr(options, "n_ctx", None)),
            n_seq_max=_optional_int(getattr(options, "n_seq_max", None)),
            kv_type=str(getattr(options, "kv_type", "auto") or "auto"),
            threads=_optional_int(getattr(options, "threads", None)),
            fit=bool(args.get("fit_enabled", True)),
            fit_target_mb=_optional_int(args.get("fit_target_mb")),
            fit_ctx=_optional_int(args.get("fit_ctx")),
            fit_cache=bool(args.get("fit_cache", True)),
            model_sha=str(model_sha or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path, "model_sha": self.model_sha, "backend": self.backend,
            "n_ctx": self.n_ctx, "n_seq_max": self.n_seq_max, "kv_type": self.kv_type,
            "threads": self.threads, "fit": self.fit, "fit_target_mb": self.fit_target_mb,
            "fit_ctx": self.fit_ctx, "fit_cache": self.fit_cache,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KeepKey:
        return cls(
            model_path=str(payload["model_path"]),
            backend=str(payload.get("backend") or "auto"),
            n_ctx=_optional_int(payload.get("n_ctx")),
            n_seq_max=_optional_int(payload.get("n_seq_max")),
            kv_type=str(payload.get("kv_type") or "auto"),
            threads=_optional_int(payload.get("threads")),
            fit=bool(payload.get("fit", True)),
            fit_target_mb=_optional_int(payload.get("fit_target_mb")),
            fit_ctx=_optional_int(payload.get("fit_ctx")),
            fit_cache=bool(payload.get("fit_cache", True)),
            model_sha=str(payload.get("model_sha") or ""),
        )

    @property
    def digest(self) -> str:
        """16 hex characters — the socket name, the spec name and the `keep status` key."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """One line a human reads in `keep status`: what this host was loaded for."""
        parts = [f"backend={self.backend}",
                 f"n_ctx={self.n_ctx if self.n_ctx is not None else 'fit'}",
                 f"n_seq_max={self.n_seq_max if self.n_seq_max is not None else 'fit'}",
                 f"kv_type={self.kv_type}",
                 f"threads={self.threads if self.threads is not None else 'fit'}",
                 f"fit={'on' if self.fit else 'off'}"]
        if self.fit_target_mb is not None:
            parts.append(f"fit_target={self.fit_target_mb}MiB")
        if self.fit_ctx is not None:
            parts.append(f"fit_ctx={self.fit_ctx}")
        if not self.fit_cache:
            parts.append("fit_cache=off")
        return " ".join(parts)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def model_identity(path: str | os.PathLike[str] | None, sha: str = "") -> str:
    """The model half of a host key: the recorded sha256 when there is one, else the file's stat.

    Hashing a multi-GB model on every call is the cost this feature exists to avoid (SPEC 2.12),
    so a bare path is keyed by `stat:<size>:<mtime_ns>` — cheap, and enough to notice a replaced
    file. The content-addressed truth stays where it belongs: the fit plan's own `sha256`.
    """
    if sha:
        return str(sha)
    if not path:
        return ""
    try:
        info = os.stat(path)
    except OSError:
        return ""
    return f"stat:{info.st_size}:{info.st_mtime_ns}"
