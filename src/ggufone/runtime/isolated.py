"""Run bundle probes outside the command process (E1a FIX t_eae35404)

Why this module exists: on the operator's RTX host `ggufone init --json` printed its JSON and
then died with `double free or corruption (!prev)` (SIGABRT, exit 134) — reproducibly, also on
the idempotent re-run. The process had dlopened the CUDA bundle it rejected, deleted that
directory, dlopened the vulkan bundle and held a live Vulkan device; the third-party
destructors that run at interpreter exit are not ours to trust.

So no ggufone command dlopens a bundle. Every probe runs in a disposable child
(`ggufone.runtime.probe_child`), **one bundle per process** — which also keeps the rejected
directory's copies out of the next probe — and the command only reads JSON. A probe that dies
is data, not a crash: the caller records `the isolated probe … failed`, and a bundle that
cannot be verified here must never win over a tier that does load (requirement 4).

Same rule the live tests already live by (tests/test_runtime_live.py): a C library must never
be able to kill the caller.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field

from ggufone.runtime import pressure

CHILD_MODULE = "ggufone.runtime.probe_child"
CHILD_TIMEOUT = 300.0            # seconds for one probe child
WARMUP_TIMEOUT = 900.0
TAIL = 400                       # characters of child stderr kept in the failure message


class ChildFailure(RuntimeError):
    """The isolated probe could not run: it exited non-zero, hung, or answered garbage."""


@dataclass(frozen=True)
class ProbeScan:
    """What one deep probe (one child, one runtime directory) learned."""

    missing_llama: tuple[str, ...] = ()
    missing_ggml: tuple[str, ...] = ()
    error: str | None = None
    backend_errors: dict[str, str] = field(default_factory=dict)
    child_error: str | None = None      # the probe itself could not run

    def to_dict(self) -> dict[str, object]:
        return {"missing_llama": list(self.missing_llama), "missing_ggml": list(self.missing_ggml),
                "error": self.error, "backend_errors": dict(self.backend_errors),
                "child_error": self.child_error}


def child_command() -> list[str]:
    """How a probe child is started (one seam, so tests can stand in for a dying probe)."""
    return [sys.executable, "-m", CHILD_MODULE]


def child_env() -> dict[str, str]:
    """The environment for a probe child: ours, plus the checkout's `src` when there is one."""
    env = {**os.environ}
    source_root = pathlib.Path(__file__).resolve().parents[2]  # <repo>/src in a checkout
    if (source_root / "ggufone" / "runtime").is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing}" if existing else str(source_root)
    return env


def run_child(request: dict[str, object], *, timeout: float | None = None) -> dict[str, object]:
    """One request in, one JSON object out. Raises `ChildFailure` for anything else.

    The spawn itself goes through `pressure.spawn` (card t_a696ce02): this container's pid
    cgroup is shared and small, so the *kernel* refusing a fork must not be reported as "the
    backend does not load on this host" — a momentarily full pid table is retried, a sustained
    cap is named `E_PID_PRESSURE` with the live reading.
    """
    command = child_command()
    limit = CHILD_TIMEOUT if timeout is None else timeout
    try:
        result = pressure.spawn(                                      # noqa: S603
            command, input=json.dumps(request), capture_output=True, text=True,
            errors="replace", timeout=limit, check=False, env=child_env())
    except subprocess.TimeoutExpired as exc:
        raise ChildFailure(
            f"the isolated probe timed out after {limit:g}s ({' '.join(command)}); the bundle "
            f"may be hanging in its own dlopen/init") from exc
    except OSError as exc:
        raise ChildFailure(
            f"the isolated probe could not be started ({' '.join(command)}): {exc}") from exc
    if result.returncode != 0:
        raise ChildFailure(
            f"the isolated probe child exited {result.returncode} "
            f"({' '.join(command)}): {(result.stderr or '').strip()[-TAIL:] or '<no stderr>'}")
    for line in reversed((result.stdout or "").strip().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ChildFailure(
        f"the isolated probe printed no JSON object (exit 0): "
        f"{(result.stdout or '').strip()[-TAIL:] or '<no stdout>'}")


def scan_bundle(runtime_dir: str | os.PathLike[str], *, symbols_llama: tuple[str, ...] = (),
                symbols_ggml: tuple[str, ...] = (), system: str | None = None,
                timeout: float | None = None) -> ProbeScan:
    """Deep-probe `runtime_dir` in a child: missing symbols + which backends dlopen here."""
    request: dict[str, object] = {"mode": "probe", "runtime_dir": str(runtime_dir),
                                  "system": system, "symbols_llama": list(symbols_llama),
                                  "symbols_ggml": list(symbols_ggml)}
    try:
        payload = run_child(request, timeout=timeout)
    except ChildFailure as exc:
        return ProbeScan(child_error=str(exc))
    return ProbeScan(
        missing_llama=tuple(str(name) for name in payload.get("missing_llama", ())),
        missing_ggml=tuple(str(name) for name in payload.get("missing_ggml", ())),
        error=str(payload["error"]) if payload.get("error") else None,
        backend_errors={str(k): str(v) for k, v in (payload.get("backend_errors") or {}).items()})


def warmup_in_child(runtime_dir: str | os.PathLike[str], model_path: str | os.PathLike[str], *,
                    n_ctx: int = 128, n_threads: int = 1, timeout: float | None = None) -> float:
    """One tiny decode in a child process; returns milliseconds.

    Loading a model and freeing it leaves the bundle's own state behind, and that teardown is
    exactly what has aborted at exit before — `init` needs the *number*, not the process.
    """
    payload = run_child({"mode": "warmup", "runtime_dir": str(runtime_dir),
                         "warmup_model": str(model_path), "n_ctx": int(n_ctx),
                         "n_threads": int(n_threads)},
                        timeout=WARMUP_TIMEOUT if timeout is None else timeout)
    if payload.get("error"):
        from ggufone.errors import RuntimeMissingError
        raise RuntimeMissingError(str(payload["error"]))
    ms = payload.get("warmup_ms")
    if ms is None:
        raise ChildFailure("the isolated warm-up answered neither a number nor an error")
    return float(ms)


def system_libs(names: tuple[str, ...] | list[str], *, timeout: float | None = None
                ) -> dict[str, str | None]:
    """`{soname: None}` when *this host* can dlopen it, else `{soname: error}`.

    Used by the pre-flight: the pinned CUDA bundle links libcudart/libcublas/libcuda, so a
    host without them must not download 168.8 MB just to find out (finding 2). Loading a
    driver library is also third-party code in a process — hence: in a child, like everything
    else here.
    """
    wanted = [str(name) for name in names]
    try:
        payload = run_child({"mode": "libs", "libs": wanted}, timeout=timeout)
    except ChildFailure as exc:
        return dict.fromkeys(wanted, str(exc))
    answer = payload.get("libs") or {}
    return {name: (str(answer[name]) if answer.get(name) else None) for name in wanted}
