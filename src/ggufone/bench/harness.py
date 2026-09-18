"""Benchmark harness: statistics, the model seam, reports and the isolation guarantees.

Milestone: E2 (SPEC 5, A-E2-1..8).

Two hard rules shape this module:

* **Offline, registry-free.** A benchmark never resolves a registry alias and never opens a
  socket: a model is an existing *file path* (`resolve_model_path`) and the runtime is a local
  bundle (`backend_runtimes`). `A-E2-7` is asserted in `tests/test_bench.py` with the network
  and the registry store poisoned — and the module deliberately does not import
  `ggufone.registry.*` at import time (the runtime finder is imported lazily, inside the live
  path only).
* **One seam.** Every suite talks to a `ModelLike` — `load()`, `tokenize()`, `session()` and
  `decide()`. `LiveModel` implements it over libllama; the tests implement it over the
  deterministic fake in `tests/fake_engine.py`. Nothing in `suites.py` knows which one it has.

Percentiles are the interpolated definition (`idx = q/100 * (n-1)`, linear between neighbours),
so `p50`/`p95` of a sample are reproducible by hand from the JSON report.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

from ggufone.errors import RuntimeMissingError, UserError

SCHEMA = "ggufone.bench/v1"
SUITES = ("latency", "throughput", "quality", "calibration", "determinism")
DEFAULT_BACKENDS = ("cpu", "vulkan", "cuda")
CPU_BACKEND = "cpu"
DEFAULT_RUNS = 5
PREFILL_SIZES = (256, 2048, 8192)
CANDIDATE_COUNTS = (2, 4, 10)
WAVE_SCALING = tuple(range(1, 17))
N_BINS = 10
#: the fields stripped before two responses are compared byte-for-byte (SPEC A5 / A-E2-5)
VOLATILE_KEYS = ("timings",)
#: the backend each bundle variant advertises, by the ggml library it carries
BACKEND_LIBRARIES = {"vulkan": "libggml-vulkan.so", "cuda": "libggml-cuda.so",
                     "metal": "libggml-metal.dylib"}
MODEL_SUFFIX = ".gguf"

# ----------------------------------------------------------- the quick preset (card t_f46cec41)
#: A short *scale* of every suite, sized for a GPU-less worker's iteration loop (`--quick`).
#: Nothing here changes *what* is measured — only how much of it: the same code paths, the same
#: report shape, the same rows-per-measurement conventions. Published tables (docs/BENCHMARKS.md)
#: are full-campaign only and are never produced with `--quick`.
QUICK_RUNS = 1
QUICK_PREFILL_SIZES = (256,)
QUICK_CANDIDATE_COUNTS = (2, 4)
QUICK_WAVE_SCALING = (1, 2)
QUICK_ITEMS = 6
QUICK_ITEMS_PER_TYPE = 2
QUICK_DETERMINISM_REPEATS = 2
QUICK_BACKENDS = 1
#: the card's target for `--quick` end to end on a CPU-only container; `tests/test_bench_live.py`
#: asserts the measured wall time against it (the number is printed in every report)
QUICK_TARGET_SECONDS = 180.0
#: the *scale* knobs the preset already fixes: passing one of them next to `--quick` is a typed
#: E_BENCH_QUICK error instead of a run that silently ignores half of what it was told
QUICK_CONFLICTS = ("runs", "items", "sizes", "n_seq_max")
#: where a run without `--out` writes: a quick report gets its own file, a full run writes nothing
#: (the full campaign's artifacts are the explicitly named `docs/evidence/e2_*.json`)
DEFAULT_OUT_NAME = "ggufone-bench-{suite}{suffix}.json"
QUICK_OUT_SUFFIX = "_quick"


class BenchError(UserError):
    """A benchmark problem the user can fix (exit code 2, like every other user error)."""

    def __init__(self, message: str, *, code: str = "E_BENCH_USAGE") -> None:
        super().__init__(message, code=code)


# ------------------------------------------------------------------ the soft time cap
class TimeBudget:
    """The `--max-seconds` soft cap: checked **between** measurements, never inside one.

    A measurement that started is always allowed to finish — a half-measured `p50` would be a
    fabricated number, and the whole point of the cap is that the report stays honest. What the cap
    buys is the tail: every measurement that never started is recorded in `skipped` so the report
    can list the unmeasured rows under `"truncated": true` and still exit 0.

    `max_seconds=None` (no flag) is a budget that never expires; `0` expires at the first
    checkpoint, which is the degenerate case the tests use to pin the shape.
    """

    def __init__(self, max_seconds: float | None = None) -> None:
        self.max_seconds = None if max_seconds is None else float(max_seconds)
        self._started = time.monotonic()
        self.skipped: list[dict[str, Any]] = []

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def expired(self) -> bool:
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds

    def skip(self, section: str, row: str) -> dict[str, Any]:
        """Record a measurement that never started (the report lists it verbatim)."""
        entry = {"section": section, "row": row,
                 "reason": f"--max-seconds {self.max_seconds:g} reached "
                           f"after {self.elapsed():.1f}s"}
        self.skipped.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {"max_seconds": self.max_seconds, "expired": self.expired(),
                "skipped": len(self.skipped)}


# ------------------------------------------------------------------ statistics
def percentile(values: Sequence[float], q: float) -> float:
    """The interpolated percentile of `values` (`q` in 0..100; raises on an empty sample)."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarise(values: Sequence[float]) -> dict[str, float | int | None]:
    """`n / min / p50 / p95 / max / mean` — the row shape every table shares."""
    clean = [float(value) for value in values]
    if not clean:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {"n": len(clean), "min": min(clean), "p50": percentile(clean, 50),
            "p95": percentile(clean, 95), "max": max(clean), "mean": sum(clean) / len(clean)}


def ratio_summarise(numerators: Sequence[float], denominators: Sequence[float]) -> dict[str, Any]:
    """tok/s per run: summarise the per-run ratio, never a ratio of two medians."""
    pairs = [(float(num), float(den)) for num, den in zip(numerators, denominators, strict=True)
             if float(den) > 0]
    if not pairs:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return summarise([num / den for num, den in pairs])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95 % Wilson score interval for a binomial proportion (0/0 -> the full range)."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denominator = 1.0 + (z * z) / n
    centre = (phat + (z * z) / (2 * n)) / denominator
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def reliability_bins(confidences: Sequence[float], correct: Sequence[bool],
                     *, n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Equal-width confidence bins with their accuracy, mean confidence and gap.

    `gap = accuracy - mean_confidence` (positive = under-confident); `ece` is their
    sample-weighted mean. Empty bins are reported with `n == 0` so a table keeps its shape.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    buckets: list[dict[str, Any]] = [
        {"lo": index / n_bins, "hi": (index + 1) / n_bins, "n": 0, "correct": 0,
         "mean_confidence": None, "accuracy": None, "gap": None}
        for index in range(n_bins)]
    sums = [0.0] * n_bins
    for confidence, hit in zip(confidences, correct, strict=True):
        value = min(max(float(confidence), 0.0), 1.0)
        index = min(int(value * n_bins), n_bins - 1)
        bucket = buckets[index]
        bucket["n"] += 1
        bucket["correct"] += int(bool(hit))
        sums[index] += value
    for bucket, total in zip(buckets, sums, strict=True):
        if bucket["n"]:
            bucket["mean_confidence"] = total / bucket["n"]
            bucket["accuracy"] = bucket["correct"] / bucket["n"]
            bucket["gap"] = bucket["accuracy"] - bucket["mean_confidence"]
    return buckets


def ece(bins: Sequence[Mapping[str, Any]]) -> float:
    """Expected calibration error: the sample-weighted mean of `|accuracy - confidence|`."""
    total = sum(int(entry["n"]) for entry in bins)
    if total == 0:
        return 0.0
    return sum(int(entry["n"]) * abs(float(entry["accuracy"]) - float(entry["mean_confidence"]))
               for entry in bins if entry["n"]) / total


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson's r, or None when it is undefined (n < 2 or a constant series)."""
    if len(xs) != len(ys):
        raise ValueError("pearson needs two series of the same length")
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(float(x) for x in xs) / n
    mean_y = sum(float(y) for y in ys) / n
    covariance = sum((float(x) - mean_x) * (float(y) - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((float(x) - mean_x) ** 2 for x in xs)
    variance_y = sum((float(y) - mean_y) ** 2 for y in ys)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


# ------------------------------------------------------------------ determinism
def strip_timings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A deep copy of a response body without the volatile fields (SPEC A5)."""
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: clean(item) for key, item in value.items()
                    if key not in VOLATILE_KEYS}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(dict(payload))


def canonical(payload: Any) -> str:
    """Canonical JSON: sorted keys, no insignificant whitespace, 6-significant rounding done."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the timings-stripped body — the byte-identity witness of A-E2-5."""
    return "sha256:" + hashlib.sha256(canonical(strip_timings(payload)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ host facts
def host_facts() -> dict[str, Any]:
    """Where a number came from: platform, CPUs *and* the cgroup quota that really bounds us."""
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "machine": platform.machine(),
    }
    with contextlib.suppress(OSError, ValueError):
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        facts["cgroup_cpu_max"] = quota if quota == "max" else int(quota) / int(period)
    memory_max = pathlib.Path("/sys/fs/cgroup/memory.max")
    if memory_max.exists():
        with contextlib.suppress(OSError, ValueError):
            raw = memory_max.read_text().strip()
            facts["cgroup_memory_bytes"] = None if raw == "max" else int(raw)
    return facts


# ------------------------------------------------------------------ the model seam
@dataclass(frozen=True)
class ModelSpec:
    """What one benchmarked model/backend combination needs."""

    path: str
    backend: str = CPU_BACKEND
    runtime_dir: str | None = None
    threads: int = 1
    kv_type: str = "auto"
    n_gpu_layers: int = 0

    def label(self) -> str:
        return f"{pathlib.Path(self.path).name}|{self.backend}|threads={self.threads}"


@dataclass(frozen=True)
class Placement:
    """The loader-facing placement: the minimal object `open_model` normalizes (`fit.coerce_plan`).

    A benchmark sets the placement explicitly (`--gpu-layers`) instead of consuming a fit plan:
    the published row must be reproducible from its flags on any host, and `0` (CPU) has to stay
    the default for the primary table. That is why this carries *only* the layer count — no
    `kv_type`, no estimates — and why `session.open_model` accepts it: what the ladder reads must
    have a default, not an `AttributeError` (card t_31b3943a).
    """

    n_gpu_layers: int = 0


@runtime_checkable
class ModelLike(Protocol):
    """The only surface the suites use (implemented by `LiveModel` and the test fake)."""

    spec: ModelSpec
    load_ms: float

    def load(self) -> float: ...

    def tokenize(self, text: str) -> list[int]: ...

    def session(self, *, n_ctx: int, n_seq_max: int,
                threads: int | None = ...) -> Any: ...

    def decide(self, request: Any, *, n_ctx: int | None = ..., n_seq_max: int | None = ...,
               threads: int | None = ...) -> Any: ...

    def close(self) -> None: ...


Factory = Callable[[ModelSpec], ModelLike]


def resolve_model_path(ref: str | None, *, env: Mapping[str, str] | None = None) -> str:
    """A benchmark model is a *file*, never an alias: the registry is out of bounds (A-E2-7)."""
    environ = os.environ if env is None else env
    candidate = ref or environ.get("GGUFONE_BENCH_MODEL")
    if not candidate:
        raise BenchError(
            "a benchmark needs --model <path.gguf> (or GGUFONE_BENCH_MODEL): benchmarks never "
            "read the model registry, so an alias cannot be resolved here",
            code="E_BENCH_MODEL")
    if candidate.startswith("@"):
        raise BenchError(f"--model takes a path, not {candidate!r}", code="E_BENCH_MODEL")
    path = pathlib.Path(os.path.expanduser(candidate))
    if not path.is_file():
        raise BenchError(
            f"{path} is not a file; a benchmark resolves models from the filesystem only "
            f"(the registry is never consulted, and nothing is ever downloaded)",
            code="E_BENCH_MODEL")
    return str(path)


def valid_suite(suite: str, *, available: Sequence[str] = SUITES) -> str:
    if suite not in available:
        raise BenchError(f"unknown benchmark suite {suite!r}; known suites: "
                         f"{', '.join(available)}", code="E_BENCH_SUITE")
    return suite


def runtime_roots(*, home: pathlib.Path | None = None,
                  env: Mapping[str, str] | None = None) -> list[pathlib.Path]:
    """Where local llama.cpp bundles are looked for (all offline, none of them the registry)."""
    candidates: list[pathlib.Path] = []
    environ = os.environ if env is None else env
    for name in ("GGUFONE_RUNTIME_DIR", "GGUFONE_BENCH_RUNTIME_DIR"):
        value = environ.get(name)
        if not value:
            continue
        entry = pathlib.Path(os.path.expanduser(value))
        if not entry.is_dir():
            continue
        candidates.append(entry)
        candidates.extend(sorted(child for child in entry.iterdir() if child.is_dir()))
    # lazy: the runtime finder imports the registry package at module level, and bench must not
    from ggufone.runtime import finder
    candidates.extend(finder.runtime_dirs(home))
    unique: list[pathlib.Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def classify_runtime(directory: pathlib.Path) -> str | None:
    """`vulkan | cuda | metal | cpu` from the ggml backends a bundle carries, else None."""
    from ggufone.runtime import finder
    if not (directory / finder.library_names()["llama"]).exists():
        return None
    for backend, library in BACKEND_LIBRARIES.items():
        if (directory / library).exists():
            return backend
    return CPU_BACKEND


def backend_runtimes(*, home: pathlib.Path | None = None,
                     env: Mapping[str, str] | None = None) -> dict[str, pathlib.Path]:
    """Classify every locally visible bundle: `{backend: runtime_dir}` (first hit wins).

    Every complete llama.cpp bundle also carries the CPU backend (`libggml-cpu.so`), so a bundle
    is registered under its accelerator *and* under `cpu`: a Vulkan or Metal bundle can always
    answer the CPU comparison, and a box with only an accelerator bundle is never "no backend".
    """
    found: dict[str, pathlib.Path] = {}
    explicit = (env or os.environ).get("GGUFONE_RUNTIME_DIR")
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(os.path.expanduser(explicit)))
    candidates.extend(runtime_roots(home=home, env=env))
    for directory in candidates:
        backend = classify_runtime(directory)
        if backend is None:
            continue
        found.setdefault(backend, directory)
        found.setdefault(CPU_BACKEND, directory)
    return found


def backend_unavailable_reason(backend: str) -> str:
    return (f"no local llama.cpp bundle carries {BACKEND_LIBRARIES.get(backend, backend)} "
            f"(benchmarks never download one: run `ggufone init --backend {backend}` or point "
            f"GGUFONE_BENCH_RUNTIME_DIR at extracted bundles)")


class LiveModel:
    """`ModelLike` over the pinned runtime: one model, one backend, explicit context sizing.

    Prefix states live in a temporary directory owned by the benchmark, so a bench run can
    measure the warm-cache path without ever touching (or polluting) the user's state cache.

    The placement (`n_gpu_layers`) is passed to the loader as a `Placement` instead of a fit
    plan: a benchmark must be reproducible from its flags alone, and a published row always
    prints the placement it used.
    """

    def __init__(self, spec: ModelSpec, *, states_home: pathlib.Path | None = None,
                 alias: str | None = None) -> None:
        self.spec = spec
        self.load_ms = 0.0
        self.alias = alias or f"bench-{pathlib.Path(spec.path).stem}-{spec.backend}"
        self._states_home = (pathlib.Path(states_home) if states_home
                             else pathlib.Path(tempfile.mkdtemp(prefix="ggufone-bench-states-")))
        self.handle: Any = None
        self.placement: dict[str, Any] = {}     # what the loader really did (after degradation)
        self._temp = states_home is None

    # ---- the seam
    def load(self) -> float:
        from ggufone.engine import session as session_module
        if self.handle is not None:                      # a re-load must not leak the old model
            self.handle.close()
            self.handle = None
        self.placement = {}
        self.handle = session_module.open_model(self.spec.path, runtime_dir=self.spec.runtime_dir,
                                                fit_plan=Placement(self.spec.n_gpu_layers))
        self.placement = self.handle.placement.to_dict()
        self.load_ms = float(self.handle.load_ms)
        return self.load_ms

    def tokenize(self, text: str) -> list[int]:
        return list(self._require_handle().tokenize(text))

    @contextlib.contextmanager
    def session(self, *, n_ctx: int, n_seq_max: int, threads: int | None = None) -> Iterator[Any]:
        from ggufone.engine import decide as decide_module
        from ggufone.engine import session as session_module
        plan = decide_module.ContextPlan(prefix_tokens=(), n_ctx=int(n_ctx),
                                        n_seq_max=int(n_seq_max),
                                        threads=int(self.spec.threads if threads is None
                                                    else threads),
                                        kv_type=self.spec.kv_type)
        with session_module.ModelSession(self._require_handle(), plan,
                                        backend=self.spec.backend,
                                        states_home=self._states_home) as live:
            yield live

    def decide(self, request: Any, *, n_ctx: int | None = None, n_seq_max: int | None = None,
               threads: int | None = None) -> Any:
        from ggufone.engine import decide as decide_module
        handle = self._require_handle()
        plan = decide_module.plan_context(request, handle)
        sequences = n_seq_max or request.options.n_seq_max or _needed_sequences(request)
        with self.session(n_ctx=n_ctx or plan.n_ctx, n_seq_max=sequences,
                          threads=threads) as live:
            live_plan = decide_module.plan_context(request, live)
            return decide_module.DecisionEngine(live).decide(request, plan=live_plan,
                                                             model_alias=self.alias)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self._temp:
            shutil.rmtree(self._states_home, ignore_errors=True)

    def __enter__(self) -> LiveModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- internals
    def _require_handle(self) -> Any:
        if self.handle is None:
            raise RuntimeMissingError(
                f"E_BENCH_STATE: {self.spec.path} was not loaded; call load() first")
        return self.handle


def _needed_sequences(request: Any) -> int:
    """`prefix + candidates` for the widest question, at least the engine's minimum of 3."""
    widest = max((len(question.options) for question in request.questions), default=1)
    return max(3, 1 + widest)


def placement_of(model: Any, spec: ModelSpec) -> dict[str, Any]:
    """The placement a row *really* ran with — the loader's own answer, not the requested flags.

    A benchmark names the placement it asked for (`--gpu-layers`), but a load that cannot offload
    that much degrades (`session.open_model`'s ladder) and the row has to say so: a table that
    keeps printing the request would report layers that never left the host. A model that never
    went through the loader (the model-free seam) reports the request and `used: None`.
    """
    used = getattr(model, "placement", None)
    return {"requested": f"n_gpu_layers={int(spec.n_gpu_layers)}",
            "used": dict(used) if isinstance(used, Mapping) and used else None}


def spec_for(config: BenchConfig, backend: str, *, runtimes: Mapping[str, pathlib.Path] | None
             = None) -> ModelSpec:
    """The spec for one backend, or a `RuntimeMissingError` naming the missing bundle."""
    found = dict(runtimes) if runtimes is not None else backend_runtimes(home=config.home)
    if backend not in found:
        raise RuntimeMissingError(f"E_RUNTIME_MISSING: {backend_unavailable_reason(backend)}")
    layers = config.gpu_layers if config.gpu_layers is not None else (
        0 if backend == CPU_BACKEND else -1)
    return ModelSpec(path=config.model_path or "", backend=backend,
                     runtime_dir=str(found[backend]), threads=config.threads or 1,
                     kv_type=config.kv_type, n_gpu_layers=layers)


@dataclass(frozen=True)
class BenchConfig:
    """One `ggufone bench` invocation (the published tables pin every field here)."""

    suite: str
    model_path: str | None = None
    backend: str = "auto"
    runs: int = DEFAULT_RUNS
    threads: int | None = None
    devset: str | None = None
    items: int | None = None
    n_seq_max: int | None = None
    kv_type: str = "auto"
    gpu_layers: int | None = None
    n_bins: int = N_BINS
    prefill_sizes: tuple[int, ...] = PREFILL_SIZES
    candidate_counts: tuple[int, ...] = CANDIDATE_COUNTS
    wave_scaling: tuple[int, ...] = WAVE_SCALING
    parts: tuple[str, ...] = ()
    home: pathlib.Path | None = None
    #: `--quick`: this run is a short preset, never a published table (`envelope` records it)
    quick: bool = False
    #: take at most N dev items *per question type* (stratified) instead of the first N items
    items_per_type: int | None = None
    #: how many repeats the determinism suite compares (A-E2-5's "3 repeats" is the default)
    determinism_repeats: int = 3
    #: run at most N backends (the quick preset resolves one); the rest are reported, not measured
    backend_limit: int | None = None
    #: `--max-seconds`: soft cap, checked between measurements (`TimeBudget`, never mid-measurement)
    max_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["home"] = str(self.home) if self.home else None
        for key in ("prefill_sizes", "candidate_counts", "wave_scaling", "parts"):
            payload[key] = list(payload[key])
        return payload

    def backends(self, *, available: Mapping[str, pathlib.Path] | None = None) -> list[str]:
        """The backend selection: `all` -> every documented one, `auto` -> what exists locally."""
        found = dict(available) if available is not None else backend_runtimes(home=self.home)
        if self.backend == "all":
            return list(DEFAULT_BACKENDS)
        if self.backend == "auto":
            return [backend for backend in DEFAULT_BACKENDS if backend in found] or [CPU_BACKEND]
        if self.backend not in DEFAULT_BACKENDS:
            raise BenchError(f"unknown backend {self.backend!r}; known: "
                             f"{', '.join(DEFAULT_BACKENDS)} ", code="E_BENCH_BACKEND")
        return [self.backend]


def quick_config(config: BenchConfig) -> BenchConfig:
    """`--quick`: the same config with every *scale* knob replaced by the short preset.

    Only scale changes. Model, backend, threads, placement (`--gpu-layers`), `--kv-type`, dev-set
    source and the soft cap are measurement *conditions*, not scale, and a quick run must still be
    comparable to the full one it previews. Passing a scale flag next to `--quick` is rejected
    before this function is reached (`QUICK_CONFLICTS`, `cli._cmd_bench`), so a quick config can
    never be a half-applied preset.
    """
    return dataclasses.replace(
        config, quick=True, runs=QUICK_RUNS, prefill_sizes=QUICK_PREFILL_SIZES,
        candidate_counts=QUICK_CANDIDATE_COUNTS, wave_scaling=QUICK_WAVE_SCALING,
        items=QUICK_ITEMS, items_per_type=QUICK_ITEMS_PER_TYPE,
        determinism_repeats=QUICK_DETERMINISM_REPEATS, backend_limit=QUICK_BACKENDS)


def quick_note() -> str:
    """The one note every quick report carries: what it is, and what it is not."""
    return (f"quick preset (card t_f46cec41): runs={QUICK_RUNS}, prefill sizes "
            f"{list(QUICK_PREFILL_SIZES)}, candidates {list(QUICK_CANDIDATE_COUNTS)}, waves "
            f"{list(QUICK_WAVE_SCALING)}, {QUICK_ITEMS} dev items ({QUICK_ITEMS_PER_TYPE} per "
            f"type), determinism {QUICK_DETERMINISM_REPEATS} repeats, {QUICK_BACKENDS} backend. "
            f"This is an iteration preset, not a published table: docs/BENCHMARKS.md is "
            f"full-campaign only and is never produced with --quick.")


def default_out_path(suite: str, *, quick: bool) -> str:
    """The default report path of a run without `--out` (`QUICK_OUT_SUFFIX` keeps it distinct).

    The name is *derived from the preset*, so no quick run can silently land on the file a full
    campaign wrote (`docs/evidence/e2_<suite>.json`). A full run without `--out` writes nothing —
    the published artifacts are named explicitly — so the two modes can never share a default.
    """
    return DEFAULT_OUT_NAME.format(suite=suite, suffix=QUICK_OUT_SUFFIX if quick else "")


def envelope(config: BenchConfig, *, model: Mapping[str, Any] | None = None,
             generated_at: str | None = None) -> dict[str, Any]:
    """The first keys of every report: what ran, on what box, on what model."""
    payload = {
        "schema": SCHEMA,
        "suite": config.suite,
        # a quick report is never mistakable for a published one: the flag, the note and the
        # effective config all travel with the numbers
        "quick": bool(config.quick),
        "generated_at": generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host_facts(),
        "config": config.to_dict(),
        "model": dict(model or {}),
        "notes": list([quick_note()]) if config.quick else [],
        "commands": {"reproduce": reproduce_command(config)},
    }
    return payload


def reproduce_command(config: BenchConfig) -> str:
    parts = ["uv run ggufone bench --suite", config.suite]
    if config.model_path:
        parts += ["--model", config.model_path]
    parts += ["--backend", config.backend]
    # `--quick` fixes the scale flags: re-stating them would be an E_BENCH_QUICK error, so the
    # command a report prints must not do it (`tests/test_bench_quick.py` runs the line back)
    if config.quick:
        parts += ["--quick"]
    else:
        parts += ["--runs", str(config.runs)]
    if config.threads:
        parts += ["--threads", str(config.threads)]
    if config.items and not config.quick:
        parts += ["--items", str(config.items)]
    if config.n_seq_max and not config.quick:
        parts += ["--n-seq-max", str(config.n_seq_max)]
    if config.devset:
        parts += ["--devset", config.devset]
    if config.gpu_layers is not None:
        parts += ["--gpu-layers", str(config.gpu_layers)]
    if config.max_seconds is not None:
        parts += ["--max-seconds", f"{config.max_seconds:g}"]
    parts += ["--json"]
    return " ".join(parts)


def model_facts(path: str) -> dict[str, Any]:
    """Header-only model facts (arch/quant/size) — no tensor data, no network.

    A path that is unreadable is reported as such instead of raising: `run_suite` must be able to
    describe *why* nothing could be measured.
    """
    facts: dict[str, Any] = {"path": path, "name": pathlib.Path(path).name, "bytes": None}
    if path and pathlib.Path(path).is_file():
        facts["bytes"] = pathlib.Path(path).stat().st_size
        with contextlib.suppress(Exception):    # a header problem must not kill a benchmark
            from ggufone.registry import gguf
            kv = gguf.parse_gguf_metadata(path)["kv"]
            facts["arch"] = gguf.arch_of(kv)
            facts["quant"] = gguf.quant_of(kv)
            facts["n_layer"] = kv.get(f"{facts['arch']}.block_count")
    return facts


# ------------------------------------------------------------------ report rendering
def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3f}" if abs(value) >= 1 else f"{value:.4f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _summary_row(label: str, summary: Mapping[str, Any], extra: Sequence[Any] = ()) -> str:
    cells = [label] + [_number(item) for item in extra] + [
        _number(summary.get("n")), _number(summary.get("p50")), _number(summary.get("p95")),
        _number(summary.get("min")), _number(summary.get("max"))]
    return "| " + " | ".join(cells) + " |"


def _table(title: str, header: Sequence[str], rows: Sequence[str]) -> list[str]:
    if not rows:
        return []
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend(rows)
    lines.append("")
    return lines


def render_report(report: Mapping[str, Any]) -> str:
    """The published markdown tables of one report — the same text `docs/BENCHMARKS.md` shows."""
    config = report.get("config", {})
    host = report.get("host", {})
    model = report.get("model", {})
    name = pathlib.Path(str(model.get("path") or model.get("name") or "model")).name
    placement = report.get("placement") or {}
    used = placement.get("used") or {}
    placement_line = ([f"- placement: requested {placement.get('requested')}, used "
                       f"n_gpu_layers={used.get('n_gpu_layers')} kv_type={used.get('kv_type')}"
                       + (" (degraded)" if used.get("degraded") else "")]
                      if used else [])
    # `--backend auto` resolved to more than one local bundle: this suite measured one of them
    selection = report.get("backend_selection") or {}
    chosen = selection.get("selected")
    available = list(selection.get("available") or [])
    selection_line = ([f"- backend selection: {chosen} of the local bundles "
                       f"({', '.join(available)}) — one suite run measures one backend"]
                      if chosen and len(available) > 1 else [])
    budget = report.get("budget") or {}
    skipped = list(report.get("skipped") or [])
    wall_ms = report.get("wall_ms")
    wall_line = [] if wall_ms is None else [
        f"- wall: {float(wall_ms) / 1000.0:.1f} s"
        + (f" (soft cap {budget.get('max_seconds')} s)" if budget.get("max_seconds") is not None
           else "")]
    truncated_line = [
        f"- TRUNCATED at --max-seconds {budget.get('max_seconds')} s: {len(skipped)} unmeasured "
        f"row(s) — " + ", ".join(f"{entry['section']}/{entry['row']}" for entry in skipped)] \
        if report.get("truncated") else []
    quick_line = [
        f"- preset: --quick (runs={config.get('runs')} sizes={config.get('prefill_sizes')} "
        f"candidates={config.get('candidate_counts')} waves={config.get('wave_scaling')} "
        f"items={config.get('items')} = {config.get('items_per_type')}/type, determinism "
        f"repeats={config.get('determinism_repeats')}, backends<="
        f"{config.get('backend_limit')}) — an iteration preset, never a published table"] \
        if report.get("quick") else []
    lines = [f"### {report.get('suite')} — {name}",
             "",
             f"- generated: {report.get('generated_at')}",
             f"- host: {host.get('platform')} · cpus {host.get('cpu_count')} · "
             f"cgroup quota {host.get('cgroup_cpu_max', 'n/a')}",
             f"- config: backend={config.get('backend')} runs={config.get('runs')} "
             f"threads={config.get('threads')}",
             f"- reproduce: `{report.get('commands', {}).get('reproduce')}`",
             *quick_line,
             *wall_line,
             *truncated_line,
             # what the row really ran with (a degradation offloads less than the flags asked)
             *placement_line,
             # which local bundle `auto` picked, when there was a choice
             *selection_line,
             ""]
    summary_header = ["n", "p50", "p95", "min", "max"]
    if report.get("suite") == "latency":
        lines += _table("model load (ms)", ["row"] + summary_header,
                        [_summary_row("model_load_ms", report.get("model_load", {}))])
        lines += _table("prefill", ["tokens"] + summary_header,
                        [_summary_row(str(row["tokens"]), row["ms"], []) for row in
                         report.get("prefill", [])])
        lines += _table("prefill throughput (tok/s)", ["tokens"] + summary_header,
                        [_summary_row(str(row["tokens"]), row["tok_per_s"])
                         for row in report.get("prefill", [])])
        lines += _table("per question", ["candidates", "waves", "forks"] + summary_header,
                        [_summary_row(str(row["candidates"]), row["ms"],
                                      [row["waves"], row["forks"]])
                         for row in report.get("per_question", [])])
        lines += _table("wave scaling (N questions)", ["questions", "waves"] + summary_header,
                        [_summary_row(str(row["questions"]), row["ms"], [row["waves"]])
                         for row in report.get("wave_scaling", [])])
        warm = report.get("warm_cache") or {}
        if warm:
            lines += _table("warm cache (state reuse)", ["row"] + summary_header,
                            [_summary_row("prefill_ms", warm.get("prefill_ms", {})),
                             _summary_row("questions_ms", warm.get("questions_ms", {}))])
        amortised = report.get("load_amortisation") or {}
        if amortised:
            lines += _table("load amortisation (serve vs one-shot)", ["path", "ms per request"],
                            [f"| serve (model already loaded) | "
                             f"{_number(amortised.get('serve_ms_per_request'))} |",
                             f"| one-shot CLI (load each call) | "
                             f"{_number(amortised.get('one_shot_ms_per_request'))} |",
                             f"| model_load_ms | {_number(amortised.get('model_load_ms'))} |"])
    elif report.get("suite") == "throughput":
        rows = []
        for row in report.get("backends", []):
            if row.get("measured"):
                rows.append("| " + " | ".join([
                    row["backend"], row.get("placement", ""),
                    _number(row.get("prefill_tok_per_s", {}).get("p50")),
                    _number(row.get("decision_ms", {}).get("p50")),
                    _number(row.get("load_ms", {}).get("p50")),
                    _number(row.get("decision_tok_per_s", {}).get("p50"))]) + " |")
            else:
                rows.append(f"| {row['backend']} | not measured | — | — | — | {row['reason']} |")
        lines += _table("backends", ["backend", "placement", "prefill tok/s (p50)",
                                     "decision ms (p50)", "load ms (p50)", "decision tok/s (p50)"],
                        rows)
    elif report.get("suite") in ("quality", "calibration"):
        rows = [f"| {qtype} | {row['n']} | {row['correct']} | {_number(row['agreement'])} | "
                f"{_number(row['ci'][0])} – {_number(row['ci'][1])} |"
                for qtype, row in sorted(report.get("per_type", {}).items())]
        overall = report.get("overall", {})
        lines += _table("exact-match agreement", ["type", "n", "correct", "agreement", "95% CI"],
                        rows + [f"| overall | {overall.get('n')} | {overall.get('correct')} | "
                                f"{_number(overall.get('agreement'))} | "
                                f"{_number((overall.get('ci') or [None, None])[0])} – "
                                f"{_number((overall.get('ci') or [None, None])[1])} |"])
    if report.get("suite") == "calibration":
        lines += _table("reliability bins", ["bin", "n", "mean confidence", "accuracy", "gap"],
                        [f"| {_number(entry['lo'])}–{_number(entry['hi'])} | {entry['n']} | "
                         f"{_number(entry['mean_confidence'])} | {_number(entry['accuracy'])} | "
                         f"{_number(entry['gap'])} |"
                         for entry in report.get("reliability", []) if entry["n"]])
        lines += _table("confidence modes", ["mode", "n", "ECE"],
                        [f"| {mode} | {row['n']} | {_number(row['ece'])} |"
                         for mode, row in sorted(report.get("modes", {}).items())])
        lines.append(f"- ECE (bins={report.get('n_bins')}): {_number(report.get('ece'))}")
        lines.append(f"- confidence/coverage correlation: "
                     f"{_number(report.get('confidence_coverage_correlation'))}")
        lines.append("")
    if report.get("suite") == "determinism":
        lines += _table("byte identity (timings stripped, 3 repeats)",
                        ["backend", "identical", "digest"],
                        [f"| {row['backend']} | {'yes' if row['identical'] else 'NO'} | "
                         f"`{row['digests'][0][:23]}…` |" if row.get("digests")
                         else f"| {row['backend']} | not measured | {row['reason']} |"
                         for row in report.get("backends", [])])
        lines.append(f"- repeats: {report.get('repeats')} · ok: {report.get('ok')}")
        lines.append("")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    if report.get("notes"):
        lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], path: str | os.PathLike[str]) -> str:
    """Write the JSON report (canonical bytes, trailing newline) and return the path."""
    target = pathlib.Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical(report) + "\n", encoding="utf-8")
    return str(target)
