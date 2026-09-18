"""One bundle per process: the bench stack's isolation seam (card t_dd62ec29).

Two llama.cpp bundles in **one** process abort it at teardown. Measured on the operator box
(`.e2e/t_603a35a0-backend-attribution/logs/after_mixed.raw`): `bench --suite throughput --backend
all` prints the whole `ggufone.bench/v1` report and then dies with `double free or corruption
(!prev)`, exit 134. The second bundle's own libraries are shadowed by the first one's (identical
SONAMEs under `RTLD_GLOBAL` — the loader caches one `Runtime` per directory and the second
`libllama.so` binds the first `libggml.so`), so its engine emits no log line at all and its model
runs on the host CPU under its own label; the crash is the same interposition seen from glibc's
side. Single-bundle runs exit 0/1 normally, and the neighbour card t_603a35a0 (which fixed *what
the report says*) deliberately left this alone.

The remedy is the auditor's F4: **one bundle per process**. A run that would dlopen two distinct
bundle directories does not dlopen anything — it measures every backend through this module, one
child process per backend, by re-entering the public single-bundle path:

    python -m ggufone bench --suite throughput --backend vulkan --model … --out <row.json> --json

The parent then *verifies* the child instead of trusting it — the exit code must agree with the
child's own report (`0` when `ok`, `1` when flagged, SPEC 2.5), the row must name the backend and
the bundle the parent selected, and the child's echoed config must be the run it was asked for. A
child that fails any of those checks (the `double free` case included: a complete report *and* exit
134) is never published as a row: the row is withheld with the reason, the report is flagged
`ok: false`, and `cli._bench` exits 1 — never 134, and never a silent 0 for a crash.

Nothing here imports a bundle: the module is stdlib-only, like the rest of `bench/` (A-E2-7), and
the parent process of an isolated run dlopens no library at all (`tests/test_bench_isolation.py`).
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: How much of a child's stderr travels into the report (`reason`/`detail`): the first line of a
#: glibc abort is the evidence, the rest is the engine's log.
STDERR_TAIL_CHARS = 200
DEFAULT_OUT_PREFIX = "ggufone-bench-isolated-"
#: The config fields a child's report must echo back for the row to be publishable, per suite.
COMMON_ECHO = ("suite", "model_path", "backend", "runs", "threads", "kv_type", "gpu_layers")
SUITE_ECHO = {"throughput": ("prefill_sizes",), "determinism": ("determinism_repeats",)}
BUNDLE_ISOLATION_REASON = (
    "two distinct local bundles cannot be dlopened into one process: the second bundle's "
    "libllama/libggml are shadowed by the first one's SONAMEs (RTLD_GLOBAL, identical names) and "
    "the process aborts at teardown with `double free or corruption` — exit 134 (card t_dd62ec29)"
)


def _bundle_key(directory: str | os.PathLike[str]) -> str:
    """The key `ctypes_binding._LOADED` uses: the *resolved* directory."""
    return str(pathlib.Path(directory).resolve())


def isolation_needed(backends: Sequence[str], runtimes: Mapping[str, Any]) -> bool:
    """True when these backends would dlopen two *distinct* bundle directories in one process.

    One bundle answering two labels (a host with a single install: `Vulkan0` op-offload under a
    `cpu` row) is a single dlopen and stays in-process — that is the case the W_BACKEND_MISMATCH
    gate was built on. Two directories is the case that aborts, and a backend without a local
    bundle is never loaded at all.
    """
    return len({_bundle_key(runtimes[backend]) for backend in backends
                if backend in runtimes}) > 1


def child_command(config: Any, backend: str, *, python: str, out_path: pathlib.Path) -> list[str]:
    """The child's argv: this CLI, one backend, the report written to `out_path`.

    Only flags the run actually sets travel (`--threads`/`--kv-type`/`--gpu-layers` are optional);
    the scale flags are the run's own, and a quick run states `--quick` instead of re-stating what
    the preset fixes (re-stating one is an `E_BENCH_QUICK` error). `--max-seconds` is deliberately
    absent: a row is one measurement unit and a unit that started always finishes.
    """
    parts = [python, "-m", "ggufone", "bench", "--suite", str(config.suite)]
    if config.model_path:
        parts += ["--model", str(config.model_path)]
    parts += ["--backend", backend]
    if config.quick:
        parts.append("--quick")
    else:
        parts += ["--runs", str(int(config.runs))]
    if config.threads:
        parts += ["--threads", str(int(config.threads))]
    if config.suite == "throughput" and not config.quick:
        parts += ["--sizes", ",".join(str(int(size)) for size in config.prefill_sizes)]
    if config.kv_type and config.kv_type != "auto":
        parts += ["--kv-type", str(config.kv_type)]
    if config.gpu_layers is not None:
        parts += ["--gpu-layers", str(int(config.gpu_layers))]
    parts += ["--out", str(out_path), "--json"]
    return parts


@dataclass(frozen=True)
class ChildRun:
    """One backend measured by one child process, and whether its answer is publishable."""

    backend: str
    command: tuple[str, ...]
    exit_code: int | None
    report: Mapping[str, Any] | None
    row: Mapping[str, Any] | None
    ok: bool
    detail: str | None
    stderr_tail: str


def _exit_desc(exit_code: int | None) -> str:
    """`-6 (SIGABRT; 134 in a shell)` — both numberings, because a CI log carries the shell one."""
    if exit_code is None:
        return "never started"
    if exit_code < 0:
        try:
            name = signal.Signals(-exit_code).name
        except ValueError:  # pragma: no cover - a signal Python does not know about
            name = f"signal {-exit_code}"
        return f"{exit_code} ({name}; {128 - exit_code} in a shell)"
    return str(exit_code)


def _tail(text: str | None) -> str:
    tail = (text or "").strip()
    if len(tail) <= STDERR_TAIL_CHARS:
        return tail
    return "…" + tail[-STDERR_TAIL_CHARS:]


def _tail_hint(tail: str) -> str:
    return f" (stderr tail: {tail})" if tail else ""


def _unusable(backend: str, command: Sequence[str], exit_code: int | None, detail: str,
              stderr_tail: str, report: Mapping[str, Any] | None = None,
              row: Mapping[str, Any] | None = None) -> ChildRun:
    return ChildRun(backend=backend, command=tuple(command), exit_code=exit_code, report=report,
                    row=row, ok=False, detail=detail, stderr_tail=stderr_tail)


def run_backend_child(config: Any, backend: str, *, runner: Callable[..., Any] | None = None,
                      out_dir: str | os.PathLike[str] | None = None, python: str | None = None,
                      env: Mapping[str, str] | None = None, timeout: float | None = None,
                      runtime_dir: str | None = None) -> ChildRun:
    """Measure `backend` in its own process and verify the child's answer.

    The child is the documented single-bundle path (`--backend <one>`), run through the same
    interpreter; the parent reads the report file it was told to write and accepts it only when
    the exit code, the row's backend/bundle and the echoed config all corroborate the request.

    The child's report is scratch: it is read into the verdict and removed with the temporary
    directory it lived in, unless the caller named `out_dir` (an evidence run that wants to keep
    the raws passes one). A child whose answer could **not** be verified is the exception: its
    scratch directory is kept — `row-<backend>.json`, `.stdout`, `.stderr` — and the row's reason
    names it, because a crash is exactly when the child's own log is worth reading.
    """
    keeps = out_dir is not None
    directory = pathlib.Path(out_dir) if keeps else pathlib.Path(
        tempfile.mkdtemp(prefix=DEFAULT_OUT_PREFIX))
    directory.mkdir(parents=True, exist_ok=True)
    verdict, diagnostics = _child_verdict(config, backend, directory, runner=runner, python=python,
                                          env=env, timeout=timeout, runtime_dir=runtime_dir)
    if verdict.ok or keeps:
        if not keeps:
            shutil.rmtree(directory, ignore_errors=True)
        return verdict
    _keep_diagnostics(directory, backend, diagnostics)
    return dataclasses.replace(
        verdict, detail=f"{verdict.detail} (the child's own report and logs are kept at "
                        f"{directory})")


def _keep_diagnostics(directory: pathlib.Path, backend: str,
                      diagnostics: Mapping[str, str]) -> None:
    """Keep a broken child's streams next to its report (the crash's only full evidence)."""
    for stream, text in diagnostics.items():
        if text:
            (directory / f"row-{backend}.{stream}").write_text(text, encoding="utf-8")


def _child_verdict(config: Any, backend: str, directory: pathlib.Path, *,
                   runner: Callable[..., Any] | None, python: str | None,
                   env: Mapping[str, str] | None, timeout: float | None,
                   runtime_dir: str | None) -> tuple[ChildRun, dict[str, str]]:
    """Spawn the child, read its report, and decide whether the row may be published."""
    diagnostics: dict[str, str] = {}
    out_path = directory / f"row-{backend}.json"
    command = child_command(config, backend, python=python or sys.executable, out_path=out_path)
    if not config.model_path:
        return _unusable(backend, command, None,
                         "this run has no --model: a child is told the model path explicitly, and "
                         "one that resolved `GGUFONE_BENCH_MODEL` on its own could measure a model "
                         "this report never names", ""), diagnostics
    launcher = runner or subprocess.run
    try:
        completed = launcher(command, env=dict(os.environ) if env is None else dict(env),
                             capture_output=True, text=True, timeout=timeout)
    except OSError as exc:
        return _unusable(backend, command, None,
                         f"E_BENCH_CHILD: the child process never started ({exc})", ""), diagnostics
    diagnostics = {"stdout": getattr(completed, "stdout", "") or "",
                   "stderr": getattr(completed, "stderr", "") or ""}
    exit_code = int(completed.returncode)
    stderr_tail = _tail(diagnostics["stderr"])
    report = _read_report(out_path)
    if report is None:
        return _unusable(
            backend, command, exit_code,
            f"the isolated child exited {_exit_desc(exit_code)} without writing a report to "
            f"{out_path}{_tail_hint(stderr_tail)}", stderr_tail), diagnostics
    expected_exit = 0 if report.get("ok", True) else 1
    if exit_code != expected_exit:
        return _unusable(
            backend, command, exit_code,
            f"the isolated child exited {_exit_desc(exit_code)} while its own report says "
            f"`ok: {report.get('ok', True)}`: the exit code and the report contradict each other, "
            f"so neither can be trusted{_tail_hint(stderr_tail)}",
            stderr_tail, report=report), diagnostics
    row = _row_for(report, backend)
    if row is None:
        measured = ", ".join(str(entry.get("backend")) for entry in report.get("backends") or []
                             if isinstance(entry, Mapping)) or "nothing"
        return _unusable(
            backend, command, exit_code,
            f"the child's report carries no row for backend {backend!r} (it measured: {measured})",
            stderr_tail, report=report), diagnostics
    if runtime_dir is not None and str(row.get("runtime_dir")) != str(runtime_dir):
        return _unusable(
            backend, command, exit_code,
            f"the child measured {row.get('runtime_dir')}, this run selected {runtime_dir}: the "
            f"row would name a bundle it did not load", stderr_tail, report=report,
            row=row), diagnostics
    mismatch = _echo_mismatch(config, row.get("backend"), report)
    if mismatch is not None:
        return _unusable(
            backend, command, exit_code,
            f"the child's report echoes a different run than it was asked for: {mismatch}",
            stderr_tail, report=report, row=row), diagnostics
    return ChildRun(backend=backend, command=tuple(command), exit_code=exit_code, report=report,
                    row=row, ok=True, detail=None,
                    stderr_tail=stderr_tail), diagnostics


def _read_report(path: pathlib.Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _row_for(report: Mapping[str, Any], backend: str) -> Mapping[str, Any] | None:
    for row in report.get("backends") or []:
        if isinstance(row, Mapping) and row.get("backend") == backend:
            return row
    return None


def _echo_mismatch(config: Any, backend: str, report: Mapping[str, Any]) -> str | None:
    """The first field where the child's echoed config disagrees with the run it was asked for."""
    echo = report.get("config")
    if not isinstance(echo, Mapping):
        return "the report carries no `config` block (the child's run cannot be checked)"
    expected: dict[str, Any] = {
        "suite": config.suite, "model_path": config.model_path, "backend": backend,
        "runs": int(config.runs), "threads": config.threads, "kv_type": config.kv_type,
        "gpu_layers": config.gpu_layers,
    }
    if config.suite == "throughput":
        expected["prefill_sizes"] = list(config.prefill_sizes)
    if config.suite == "determinism":
        expected["determinism_repeats"] = int(config.determinism_repeats)
    for field in COMMON_ECHO + SUITE_ECHO.get(str(config.suite), ()):
        if echo.get(field) != expected[field]:
            return f"{field}={echo.get(field)!r}, this run asked {expected[field]!r}"
    return None


def process_block(child: ChildRun) -> dict[str, Any]:
    """The `process` block every isolated row carries: which child produced it, and whether."""
    return {"isolated": True, "exit_code": child.exit_code, "ok": child.ok, "detail": child.detail}


def isolated_row(child: ChildRun, *, gap: Mapping[str, Any]) -> dict[str, Any]:
    """The row the parent publishes: the child's own row, or the suite's `gap` row + the reason.

    A row the child could not prove is withheld entirely (no devices, no numbers): what the parent
    publishes is the fact that the measurement did not happen and why.
    """
    block = process_block(child)
    if child.ok and child.row is not None:
        return {**child.row, "process": block}
    return {**gap, "backend": child.backend, "reason": child.detail, "process": block}


def broken_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows whose isolated child produced nothing verifiable (each one fails the report)."""
    return [row for row in rows if (row.get("process") or {}).get("ok") is False]


def isolation_record(config: Any, runtimes: Mapping[str, Any],
                     backends: Sequence[str]) -> dict[str, Any]:
    """The machine-readable record of an isolated run: what was measured, and by which bundle."""
    return {"one_bundle_per_process": True,
            "bundles": {backend: str(runtimes[backend]) for backend in backends
                        if backend in runtimes},
            "reason": BUNDLE_ISOLATION_REASON,
            "suite": config.suite}


def isolation_used_note(config: Any, runtimes: Mapping[str, Any],
                        backends: Sequence[str]) -> str:
    """The note an isolated report carries: why its rows came from children."""
    distinct = len({_bundle_key(runtimes[backend]) for backend in backends if backend in runtimes})
    return (f"one bundle per process: `--backend {config.backend}` selected "
            f"{', '.join(backends)} over {distinct} distinct local bundle directories, and "
            f"{BUNDLE_ISOLATION_REASON}. Every row was measured by its own child process — the "
            f"documented `--backend <one>` path — and carries the `process` block that produced "
            f"it; a child whose answer the parent could not verify leaves no row at all.")


def isolation_note(row: Mapping[str, Any]) -> str:
    """The report note for a withheld row: the child's exit code is never the answer's."""
    process = row.get("process") or {}
    return (f"ISOLATED_CHILD_FAILED: the row for backend `{row.get('backend')}` was measured in "
            f"its own child process (one bundle per process — {BUNDLE_ISOLATION_REASON}) and that "
            f"child exited {_exit_desc(process.get('exit_code'))}: {row.get('reason')}. The row is "
            f"withheld (`measured: false`) and the report is not ok.")
