"""One bundle per process: the bench stack's isolation seam (card t_dd62ec29).

Two llama.cpp bundles in **one** process abort it at teardown. Measured on the operator box
(`docs/evidence/e2e/t_603a35a0-backend-attribution/logs/after_mixed.raw`): `bench --suite
throughput --backend all` prints the whole `typed_gguf.bench/v1` report and then dies with
`double free or corruption (!prev)`, exit 134. The second bundle's own libraries are shadowed by
the first one's (identical
SONAMEs under `RTLD_GLOBAL` — the loader caches one `Runtime` per directory and the second
`libllama.so` binds the first `libggml.so`), so its engine emits no log line at all and its model
runs on the host CPU under its own label; the crash is the same interposition seen from glibc's
side. Single-bundle runs exit 0/1 normally, and the neighbour card t_603a35a0 (which fixed *what
the report says*) deliberately left this alone.

The remedy is the auditor's F4: **one bundle per process**. A run that would dlopen two distinct
bundle directories does not dlopen anything — it measures every backend through this module, one
child process per backend, by re-entering the public single-bundle path:

    python -m typed_gguf bench --suite throughput --backend vulkan --model … --out <row.json> --json

The parent then *verifies* the child instead of trusting it — the exit code must agree with the
child's own report (`0` when `ok`, `1` when flagged, SPEC 2.5), the row must name the backend and
the bundle the parent selected, and the child's echoed config must be the run it was asked for. A
child that fails any of those checks (the `double free` case included: a complete report *and* exit
134) is never published as a row: the row is withheld with the reason, the report is flagged
`ok: false`, and `cli._bench` exits 1 — never 134, and never a silent 0 for a crash.

Nothing here imports a bundle: the module is stdlib-only, like the rest of `bench/` (A-E2-7), and
the parent process of an isolated run dlopens no library at all (`tests/test_bench_isolation.py`).

**The second half of the story (card t_57cc0179): the child can crash at teardown.** On the
operator's box, while the E3 campaign holds the device, a *single* Vulkan child writes its whole
report, prints `ok: true`, and then dies with exit **-11 (SIGSEGV)**; the same command on a free
device exits 0. Withholding the row (t_dd62ec29) contains the crash but throws the measurement away
on a host where this shape is routine. So an isolated child that dies on a fatal signal *after*
writing its report earns **one retry** with the loader's next degrade rung (`retry_placement`:
fewer layers, or the next KV rung when the weights are already on the host) — the placement the
starved device can still hold. If the retry crashes too, the containment stands and the withheld
row is named: `W_BACKEND_CRASHED_AT_TEARDOWN`, carrying both exit codes and the free-device-memory
reading at each attempt's start, in the row's `reason` (which the rendered table prints) and in the
report's `notes`. The starvation reading is the driver's own answer read through
`device_memory()`; on a box with no device it is *unknown*, never zero.
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
DEFAULT_OUT_PREFIX = "typed-gguf-bench-isolated-"
#: The config fields a child's report must echo back for the row to be publishable, per suite.
COMMON_ECHO = ("suite", "model_path", "backend", "runs", "threads", "kv_type", "gpu_layers")
SUITE_ECHO = {"throughput": ("prefill_sizes",), "determinism": ("determinism_repeats",)}
BUNDLE_ISOLATION_REASON = (
    "two distinct local bundles cannot be dlopened into one process: the second bundle's "
    "libllama/libggml are shadowed by the first one's SONAMEs (RTLD_GLOBAL, identical names) and "
    "the process aborts at teardown with `double free or corruption` — exit 134 (card t_dd62ec29)"
)

# ------------------------------------------------------------------ a crash at teardown
#: The *exit codes* that mean the engine died rather than answered: `-11` (SIGSEGV) and `-6`
#: (SIGABRT). A `-9` (SIGKILL) is the OOM killer or a human and says nothing about teardown.
FATAL_SIGNALS: tuple[int, ...] = (-int(signal.SIGSEGV), -int(signal.SIGABRT))
#: The warning a withheld row carries when the crash survived the one retry (card t_57cc0179).
W_BACKEND_CRASHED_AT_TEARDOWN = "W_BACKEND_CRASHED_AT_TEARDOWN"
#: The reserve an all-layers placement must leave free (`fit.DEFAULT_FIT_TARGET_MB`, SPEC 2.4):
#: a device whose free memory cannot hold the weights *plus* this margin is the starved case.
FIT_MARGIN_BYTES = 1024 * 1024 * 1024
MIB = 1024 * 1024
#: One retry, per the card: the first crash earns a second (degraded) child, the second is final.
DEFAULT_ATTEMPTS = 2

#: The device memory the free/starved accounting speaks in: bytes, or `known is False` when there
#: is no device to ask (a CI box) — an unknown reading is never silently read as "0 MiB free".
@dataclass(frozen=True)
class DeviceMemory:
    """Total + free device memory at one point in time."""
    total_bytes: int = 0
    free_bytes: int = 0
    source: str = "unknown"

    @property
    def known(self) -> bool:
        return self.total_bytes > 0

    def sentence(self) -> str:
        if not self.known:
            return "device memory unknown"
        return f"{self.free_bytes / MIB:.0f} MiB free of {self.total_bytes / MIB:.0f} MiB"

    def to_dict(self) -> dict[str, Any]:
        return {"total_bytes": self.total_bytes, "free_bytes": self.free_bytes,
                "source": self.source}


def device_memory(*, probe: Callable[[], Any] | None = None) -> DeviceMemory:
    """The device's free memory *now* — the driver's own answer, not a guess.

    The reading comes from the same query `fit.host_facts` uses (`registry.recommend`), imported
    lazily: this module stays stdlib-only at import time (A-E2-7). An injected `probe` owns the
    answer completely — `(total, free)` bytes, or `None` for "no device" — which is what keeps
    every gate on this path deterministic on a box that *has* a GPU.
    """
    if probe is not None:
        answer: Any = probe()
        source = "probe"
    else:
        try:
            from typed_gguf.registry import recommend

            reading = recommend.device_memory()
        except Exception:  # noqa: BLE001 - a driver that misbehaves reads as "unknown", never fatal
            return DeviceMemory()
        answer, source = (reading.total_bytes, reading.free_bytes), reading.source
    if not answer or len(tuple(answer)) < 2:
        return DeviceMemory()
    total, free = (int(value) for value in tuple(answer)[:2])
    return DeviceMemory(total_bytes=max(0, total),
                        free_bytes=max(0, min(free, max(0, total))), source=source)


def starved(*, memory: DeviceMemory, weights_bytes: int | None) -> bool | None:
    """Could this model's weights *plus* the fit margin have fit in the free device memory?

    `True` is the operator's shape: a 4B model (2.4 GiB of weights) on a device holding ~3 GiB
    free while the E3 campaign uses the rest — an all-layers placement has to fit the weights, the
    KV cache *and* the compute buffers, so "the weights alone do not fit with the documented
    margin" is the conservative side of "this run was starved". `None` when either number is
    unknown: nothing is invented for a box without a device.
    """
    if not memory.known or not weights_bytes:
        return None
    return memory.free_bytes - FIT_MARGIN_BYTES < int(weights_bytes)


@dataclass(frozen=True)
class TeardownCrash:
    """The defect's shape: the child answered *completely* and then died on a fatal signal."""

    exit_code: int
    memory: DeviceMemory
    weights_bytes: int | None
    starved: bool | None

    def sentence(self) -> str:
        where = f"{self.memory.sentence()} at the attempt's start" if self.memory.known \
            else "device memory unknown at the attempt's start"
        if self.starved is True:
            return (f"{where} (starved: the model's {self.weights_bytes / MIB:.0f} MiB of weights "
                    f"plus a {FIT_MARGIN_BYTES / MIB:.0f} MiB reserve do not fit)")
        if self.starved is False:
            return f"{where} (the weights would fit: the crash was not memory pressure)"
        return f"{where} (the model's weights are unknown, so starvation was not checked)"

    def to_dict(self) -> dict[str, Any]:
        """The machine-readable record a recovered row carries under `process.crash`."""
        return {"exit_code": self.exit_code, "signal": signal.Signals(-self.exit_code).name,
                "free_bytes": self.memory.free_bytes, "total_bytes": self.memory.total_bytes,
                "weights_bytes": self.weights_bytes, "starved": self.starved,
                "free_source": self.memory.source}


def teardown_crash(exit_code: int | None, *, report: Mapping[str, Any] | None,
                   memory: DeviceMemory, weights_bytes: int | None) -> TeardownCrash | None:
    """The crash-at-teardown shape, or `None` when this failure is something else.

    Two conditions: the child died on a fatal signal (-11/-6), *and* it had already written its
    report — a complete answer followed by a dead process is exactly the contradiction this card is
    about (card t_dd62ec29 withheld the row for it; this card retries it). A child with no report
    crashed before it measured anything, and a clean or flagged exit is the child's own answer.
    """
    if exit_code is None or int(exit_code) not in FATAL_SIGNALS:
        return None
    if report is None:
        return None
    return TeardownCrash(exit_code=int(exit_code), memory=memory, weights_bytes=weights_bytes,
                         starved=starved(memory=memory, weights_bytes=weights_bytes))


@dataclass(frozen=True)
class Attempt:
    """One child process of a row: what it was asked for, how it ended, and the device it met."""

    index: int
    exit_code: int | None
    ok: bool
    memory: DeviceMemory
    placement: Mapping[str, Any]
    detail: str | None
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "exit_code": self.exit_code, "ok": self.ok,
                "free_bytes": self.memory.free_bytes, "total_bytes": self.memory.total_bytes,
                "placement": dict(self.placement), "detail": self.detail,
                "stderr_tail": self.stderr_tail}


@dataclass(frozen=True)
class RetryPlacement:
    """The one degraded placement a retry runs with (the loader's own ladder, one rung down)."""

    n_gpu_layers: int
    kv_type: str
    degraded: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"n_gpu_layers": self.n_gpu_layers, "kv_type": self.kv_type,
                "degraded": self.degraded, "note": self.note}


def _asked_layers(config: Any, backend: str, first_row: Mapping[str, Any] | None) -> int:
    """The layers the first attempt really asked the device for (its own report wins).

    A `n_gpu_layers < 0` count is llama.cpp's "every layer" — the benchmark's GPU default — so it
    is the *largest* footprint, and the ladder has to be able to reduce from it.
    """
    row = first_row if isinstance(first_row, Mapping) else {}
    used = (row.get("placement_used") or {})
    used = used.get("used") if isinstance(used, Mapping) else None
    value = used.get("n_gpu_layers") if isinstance(used, Mapping) else None
    if value is None:
        value = config.gpu_layers
    if value is None:
        value = 0 if backend == "cpu" else -1
    return int(value)


def _next_kv(fit_module: Any, kv_type: str) -> str:
    """One rung down the KV ladder (`f16` -> `q8_0` -> `q4_0`), or the same value at the bottom."""
    order = tuple(fit_module.KV_DOWNGRADE_ORDER)
    start = fit_module.kv_start(kv_type)
    index = order.index(start)
    return order[index + 1] if index + 1 < len(order) else start


def model_bytes(path: str | None) -> int | None:
    """The model file's size — the honest lower bound for what an offload must place on device."""
    if not path:
        return None
    try:
        size = pathlib.Path(path).stat().st_size
    except OSError:
        return None
    return size or None


def model_facts(path: str | None) -> Any | None:
    """The GGUF header facts the retry needs (the layer count), or `None` — never a new failure.

    Reading the header of a multi-GB model costs a second or two; it happens only after a crash,
    and only when the placement really says "all layers" (a count the parent cannot know without
    asking the file).
    """
    if not path or not pathlib.Path(path).is_file():
        return None
    try:
        from typed_gguf.runtime import fit

        return fit.ModelFacts.read(path, want_sha256=False)
    except Exception:  # noqa: BLE001 - an unreadable header only means "take the safe rung"
        return None


def retry_placement(config: Any, backend: str, *, first_row: Mapping[str, Any] | None = None,
                    facts: Any | None = None) -> RetryPlacement:
    """The degraded placement of the one retry: the loader's own ladder, one rung down.

    `fit.degrade_ladder` is the documented ordering (fewer layers -> smaller kv_type -> CPU-only),
    so the retry does not invent a policy — it takes the first rung below what the first attempt
    really ran (the child's own report's `placement_used`, or the run's request when it crashed
    before saying), which is `max(1, layers // 2)` with the same KV type. Two cases do not need the
    layer count at all:

    * the placement already asks the device for no weights (`n_gpu_layers=0`): the KV rung is the
      only thing left to move, and `auto` is pinned to the rung the engine would have walked to;
    * the layers cannot be counted (no readable header): the retry takes the rung that asks the
      device for nothing, which is always available.
    """
    from typed_gguf.runtime import fit

    asked_kv = str(getattr(config, "kv_type", None) or "auto")
    counted = int(getattr(facts, "n_layer", 0) or 0) if facts is not None else 0
    start = _asked_layers(config, backend, first_row)
    if start < 0 and counted > 0:
        start = counted                    # `< 0` means "every layer": the largest footprint
    if start > 0 and counted > 0:
        start = min(start, counted)        # the ladder counts against the model, not the request
    if start > 0:
        half = start // 2                  # the ladder's first rung (`max(1, layers // 2)`, and for
        return RetryPlacement(             # a single layer the 0-layer rung just below it)
            n_gpu_layers=max(0, half), kv_type=asked_kv, degraded=max(0, half) < start,
            note=f"with n_gpu_layers={max(0, half)} (fewer than the {start} the first attempt ran)")
    if start == 0:
        lower = _next_kv(fit, asked_kv)
        if lower != asked_kv:
            return RetryPlacement(n_gpu_layers=0, kv_type=lower, degraded=True,
                                  note=f"with n_gpu_layers=0 and kv_type={lower} (the next KV rung "
                                       f"of the ladder)")
        return RetryPlacement(n_gpu_layers=0, kv_type=asked_kv, degraded=False,
                              note=f"with the placement it already had (n_gpu_layers=0, kv_type="
                                   f"{asked_kv} is already the bottom rung)")
    return RetryPlacement(
        n_gpu_layers=0, kv_type=_next_kv(fit, asked_kv), degraded=True,
        note="with n_gpu_layers=0 (the layers could not be counted: the rung that asks the device "
             "for nothing)")


def retry_config(config: Any, backend: str, placement: RetryPlacement) -> Any:
    """The config the retry child runs with: the same run, one rung down the placement ladder."""
    return dataclasses.replace(config, gpu_layers=int(placement.n_gpu_layers),
                               kv_type=placement.kv_type)


def _placement_of(config: Any, backend: str) -> dict[str, Any]:
    """What one attempt asked for, as the run states it (`None` layers = the bench default)."""
    return {"n_gpu_layers": config.gpu_layers, "kv_type": config.kv_type, "backend": backend}


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
    parts = [python, "-m", "typed_gguf", "bench", "--suite", str(config.suite)]
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
    #: Every child that was spawned for this row, in order (one entry unless a crash was retried).
    attempts: tuple[Attempt, ...] = ()
    #: Set when the row is the retry's *because* the first attempt crashed at teardown.
    crash: TeardownCrash | None = None

    @property
    def warning(self) -> str | None:
        """The named warning a withheld row carries: the crash survived the one retry."""
        return W_BACKEND_CRASHED_AT_TEARDOWN if self.crash is not None else None


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
              row: Mapping[str, Any] | None = None,
              attempts: tuple[Attempt, ...] = (),
              crash: TeardownCrash | None = None) -> ChildRun:
    return ChildRun(backend=backend, command=tuple(command), exit_code=exit_code, report=report,
                    row=row, ok=False, detail=detail, stderr_tail=stderr_tail, attempts=attempts,
                    crash=crash)


def crash_warning(crash: TeardownCrash, attempts: Sequence[Attempt],
                  placement: RetryPlacement | None) -> str:
    """The named sentence of a withheld row whose teardown crash survived the one retry.

    It carries what the card asks for — the child's exit code and the free-device-memory reading —
    for *each* attempt, plus the starvation verdict, so the row a reader sees in the table says
    what happened and why (the same sentence is the report's `notes` entry).
    """
    first = attempts[0]
    parts = [f"{W_BACKEND_CRASHED_AT_TEARDOWN}: the isolated child exited "
             f"{_exit_desc(first.exit_code)} after writing a complete report"
             f"{_tail_hint(first.stderr_tail)}; {crash.sentence()}"]
    if len(attempts) > 1 and placement is not None:
        last = attempts[-1]
        where = last.memory.sentence() if last.memory.known else "device memory unknown"
        retry = (f"; the retry {placement.note} exited {_exit_desc(last.exit_code)} "
                 f"{_tail_hint(last.stderr_tail)} ({where} at its start)")
        if last.exit_code not in FATAL_SIGNALS:
            retry += f", and its own verdict was: {last.detail}"
        parts.append(retry)
    elif len(attempts) == 1:
        parts.append("; no retry was attempted")
    parts.append("; the row is withheld (`measured: false`) and the report is not ok")
    return "".join(parts)


def run_backend_child(config: Any, backend: str, *, runner: Callable[..., Any] | None = None,
                      out_dir: str | os.PathLike[str] | None = None, python: str | None = None,
                      env: Mapping[str, str] | None = None, timeout: float | None = None,
                      runtime_dir: str | None = None, attempts: int = DEFAULT_ATTEMPTS,
                      probe: Callable[[], Any] | None = None,
                      facts: Any | None = None) -> ChildRun:
    """Measure `backend` in its own process and verify the child's answer.

    The child is the documented single-bundle path (`--backend <one>`), run through the same
    interpreter; the parent reads the report file it was told to write and accepts it only when
    the exit code, the row's backend/bundle and the echoed config all corroborate the request.

    The child's report is scratch: it is read into the verdict and removed with the temporary
    directory it lived in, unless the caller named `out_dir` (an evidence run that wants to keep
    the raws passes one). A child whose answer could **not** be verified is the exception: its
    scratch directory is kept — `row-<backend>.json`, `.stdout`, `.stderr` — and the row's reason
    names it, because a crash is exactly when the child's own log is worth reading.

    **A crash at teardown earns one retry** (card t_57cc0179): when a child dies on a fatal signal
    *after* writing a complete report (`teardown_crash` — the shape measured on the operator's
    memory-starved device), a second child measures the same run with the loader's next degraded
    rung (`retry_placement`), because that is the placement the device can still hold. The retry is
    verified exactly like the first attempt; if it also fails, the row stays withheld and carries
    `W_BACKEND_CRASHED_AT_TEARDOWN` with both exit codes and free-device readings. `probe` (the
    device-memory reader) and `facts` (the model header) are injectable seams: the gates on this
    path must not depend on the box's GPU.
    """
    keeps = out_dir is not None
    directory = pathlib.Path(out_dir) if keeps else pathlib.Path(
        tempfile.mkdtemp(prefix=DEFAULT_OUT_PREFIX))
    directory.mkdir(parents=True, exist_ok=True)
    resolved = facts if facts is not None else model_facts(config.model_path)
    header_bytes = int(getattr(resolved, "weights_bytes", 0) or 0)
    weights = model_bytes(config.model_path) or header_bytes or None
    verdict, diagnostics = _attempts(
        config, backend, directory, runner=runner, python=python, env=env, timeout=timeout,
        runtime_dir=runtime_dir, attempts=max(1, int(attempts)), probe=probe, facts=resolved,
        weights=weights)
    if verdict.ok or keeps:
        if not keeps:
            shutil.rmtree(directory, ignore_errors=True)
        return verdict
    for index, streams in diagnostics.items():
        _keep_diagnostics(directory, backend, streams,
                          suffix="" if index == 1 else f"-retry{index - 1}")
    return dataclasses.replace(
        verdict, detail=f"{verdict.detail} (the child's own report and logs are kept at "
                        f"{directory})")


def _attempts(config: Any, backend: str, directory: pathlib.Path, *,
              runner: Callable[..., Any] | None, python: str | None,
              env: Mapping[str, str] | None, timeout: float | None, runtime_dir: str | None,
              attempts: int, probe: Callable[[], Any] | None,
              facts: Any | None, weights: int | None,
              ) -> tuple[ChildRun, dict[int, dict[str, str]]]:
    """Run the child, and (once) the degraded retry a teardown crash earns.

    Returns the verdict of the last attempt and the streams of every attempt that produced any
    (keyed by attempt index) — the `Attempt` records travel on the verdict itself.
    """
    seen: list[Attempt] = []
    streams: dict[int, dict[str, str]] = {}
    crash: TeardownCrash | None = None
    placement: RetryPlacement | None = None
    verdict: ChildRun | None = None
    run_config = config
    for index in range(1, attempts + 1):
        memory = device_memory(probe=probe)
        verdict, diagnostics = _child_verdict(
            run_config, backend, directory, runner=runner, python=python, env=env, timeout=timeout,
            runtime_dir=runtime_dir, out_path=directory / _out_name(backend, index))
        if diagnostics:
            streams[index] = diagnostics
        seen.append(Attempt(index=index, exit_code=verdict.exit_code, ok=verdict.ok, memory=memory,
                            placement=_placement_of(run_config, backend), detail=verdict.detail,
                            stderr_tail=verdict.stderr_tail))
        if verdict.ok:
            break
        detected = teardown_crash(verdict.exit_code, report=verdict.report, memory=memory,
                                  weights_bytes=weights)
        if detected is None:
            # a different failure: no retry, and the first crash (if any) stands
            break
        crash = detected
        if index >= attempts:
            break
        first_row = _row_for(verdict.report, backend) if verdict.report is not None else None
        placement = retry_placement(config, backend, first_row=first_row, facts=facts)
        run_config = retry_config(config, backend, placement)
    assert verdict is not None                     # `attempts` is at least one
    if verdict.ok:
        return dataclasses.replace(verdict, attempts=tuple(seen), crash=crash), streams
    if crash is None:
        return dataclasses.replace(verdict, attempts=tuple(seen)), streams
    return _unusable(backend, verdict.command, verdict.exit_code,
                     crash_warning(crash, seen, placement), verdict.stderr_tail,
                     report=verdict.report, row=None, attempts=tuple(seen),
                     crash=crash), streams


def _out_name(backend: str, index: int) -> str:
    """The report file of one attempt: the first keeps the documented name, retries are labelled."""
    return f"row-{backend}.json" if index == 1 else f"row-{backend}-retry{index - 1}.json"


def _keep_diagnostics(directory: pathlib.Path, backend: str,
                      diagnostics: Mapping[str, str], *, suffix: str = "") -> None:
    """Keep a broken child's streams next to its report (the crash's only full evidence)."""
    for stream, text in diagnostics.items():
        if text:
            (directory / f"row-{backend}{suffix}.{stream}").write_text(text, encoding="utf-8")


def _child_verdict(config: Any, backend: str, directory: pathlib.Path, *,
                   runner: Callable[..., Any] | None, python: str | None,
                   env: Mapping[str, str] | None, timeout: float | None,
                   runtime_dir: str | None,
                   out_path: pathlib.Path | None = None) -> tuple[ChildRun, dict[str, str]]:
    """Spawn the child, read its report, and decide whether the row may be published."""
    diagnostics: dict[str, str] = {}
    out_path = out_path if out_path is not None else directory / f"row-{backend}.json"
    command = child_command(config, backend, python=python or sys.executable, out_path=out_path)
    if not config.model_path:
        return _unusable(backend, command, None,
                         "this run has no --model: a child is told the model path explicitly, and "
                         "one that resolved `TYPED_GGUF_BENCH_MODEL` alone could measure a model "
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
    """The `process` block every isolated row carries: which child produced it, and whether.

    A withheld row keeps the shape the containment pinned (card t_dd62ec29): the whole story is its
    `reason` sentence, which the table renders. A **recovered** row adds the structured record of
    what happened — every attempt with its exit code and the device memory it met, plus the crash
    dict — because that row *is* published, so the crash must not hide in prose.
    """
    block: dict[str, Any] = {"isolated": True, "exit_code": child.exit_code, "ok": child.ok,
                             "detail": child.detail}
    if child.ok and len(child.attempts) > 1:
        block["attempts"] = [attempt.to_dict() for attempt in child.attempts]
        if child.crash is not None:
            block["crash"] = child.crash.to_dict()
    return block


def isolated_row(child: ChildRun, *, gap: Mapping[str, Any]) -> dict[str, Any]:
    """The row the parent publishes: the child's own row, or the suite's `gap` row + the reason.

    A row the child could not prove is withheld entirely (no devices, no numbers): what the parent
    publishes is the fact that the measurement did not happen and why — plus, when a teardown crash
    survived the retry, the named warning (`W_BACKEND_CRASHED_AT_TEARDOWN`) the reader needs.
    """
    block = process_block(child)
    if child.ok and child.row is not None:
        return {**child.row, "process": block}
    row: dict[str, Any] = {**gap, "backend": child.backend, "reason": child.detail,
                           "process": block}
    if child.warning is not None:
        row["warnings"] = [child.warning]
    return row


def broken_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows whose isolated child produced nothing verifiable (each one fails the report)."""
    return [row for row in rows if (row.get("process") or {}).get("ok") is False]


def verified_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows no child of which the parent refuted: in-process rows, and verified children.

    The complement of `broken_rows`, and the set the suites aggregate their own gates over: a row
    withheld *because of* a crash must not be read as "the engine's log refutes this label" (its
    warnings are the crash's, not a mismatch's).
    """
    return [row for row in rows if (row.get("process") or {}).get("ok") is not False]


def recovered_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows a retry recovered: a child crashed at teardown and the row still measured."""
    return [row for row in rows if (row.get("process") or {}).get("crash")]


def _placement_text(attempt: Mapping[str, Any]) -> str:
    """`n_gpu_layers=16 kv_type=auto` — what one attempt asked for, as the run states it."""
    placement = attempt.get("placement") or {}
    return f"n_gpu_layers={placement.get('n_gpu_layers')} kv_type={placement.get('kv_type')}"


def recovered_note(row: Mapping[str, Any]) -> str:
    """The report note for a recovered row: the crash is named, the row is published."""
    process = row.get("process") or {}
    attempts = list(process.get("attempts") or [])
    first = attempts[0] if attempts else {}
    walked = " then ".join(_placement_text(attempt) for attempt in attempts) or "unknown"
    return (f"RECOVERED_AFTER_TEARDOWN_CRASH: the row for backend `{row.get('backend')}` was first "
            f"measured by a child that exited {_exit_desc(first.get('exit_code'))} after writing a "
            f"complete report; the one retry ({walked}) measured it, so the published row is the "
            f"retry's own (its placement, its bundle, its `process.attempts`) and the report keeps "
            f"`ok: true` — a recovered row is a measurement, not a failure.")


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
