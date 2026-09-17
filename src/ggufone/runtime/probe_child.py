"""Child-process half of `ggufone.runtime.isolated` — the disposable probe (E1a FIX t_eae35404)

One JSON request on stdin, one JSON object on stdout. Run as:

    python -m ggufone.runtime.probe_child <<< '{"mode": "probe", "runtime_dir": "/…"}'

Modes:
    probe   dlopen the bundle: missing required symbols + which accelerators load on this host
    warmup  load the model, decode once, report milliseconds
    libs    can this host dlopen the given system libraries at all (pre-flight)

Exit code 0 means "the answer is data" — including "this bundle is broken", which is the
normal outcome for a GPU bundle whose cudart is missing. A non-zero exit means the probe
itself could not run, and the caller records that instead of guessing.

Nothing here is allowed to outlive the question: this process dlopens third-party GPU
libraries and drivers, and its own teardown is exactly what aborted `ggufone init` on the
operator's host. It dies alone.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

# Run from a checkout without an installed dist, like `tools/live_probe.py` does.
_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_SOURCE_ROOT / "ggufone" / "runtime").is_dir():
    sys.path.insert(0, str(_SOURCE_ROOT))

from ggufone.runtime import capability, install  # noqa: E402


def probe_mode(request: dict[str, Any]) -> dict[str, Any]:
    """The deep probe: symbols, then every accelerator backend's real `dlopen`."""
    scan = capability.scan_in_process(
        request["runtime_dir"], symbols_llama=tuple(request.get("symbols_llama") or ()),
        symbols_ggml=tuple(request.get("symbols_ggml") or ()), system=request.get("system"))
    return scan.to_dict()


def warmup_mode(request: dict[str, Any]) -> dict[str, Any]:
    """The warm-up: never raises for a bundle it cannot use, reports why instead."""
    try:
        ms = install.warmup(request["runtime_dir"], request["warmup_model"],
                            n_ctx=int(request.get("n_ctx") or 128),
                            n_threads=int(request.get("n_threads") or 1))
    except Exception as exc:  # noqa: BLE001 - the caller records the text, verbatim
        return {"warmup_ms": None, "error": f"{exc.__class__.__name__}: {exc}"}
    return {"warmup_ms": ms, "error": None}


def libs_mode(request: dict[str, Any]) -> dict[str, Any]:
    """Pre-flight: which of these system sonames this host can load (None = loadable)."""
    return {"libs": {str(name): capability.load_system_lib(str(name))
                     for name in request.get("libs") or ()}}


MODES = {"probe": probe_mode, "warmup": warmup_mode, "libs": libs_mode}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    mode = str(request.get("mode") or "")
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r} (known: {', '.join(sorted(MODES))})")
    return MODES[mode](request)


def main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    payload = handle(request)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
