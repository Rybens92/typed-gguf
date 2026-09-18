"""Fit plans: `llama-fit-params` + ggufone's own KV/n_seq_max math (SPEC 2.10, A-E1c-4/5/6).

Milestone: E1c.

A **fit plan** answers "how much of this model can this host hold, and with what context?" for
one `(model, host)` pair:

```json
{"n_gpu_layers": 0, "n_ctx": 4096, "kv_type": "f16", "n_seq_max": 8,
 "est_weights_bytes": ..., "est_kv_bytes": ..., "est_total_bytes": ...,
 "backend": "cpu", "source": "estimate | llama-fit-params"}
```

Two sources, in the order SPEC 2.10 gives them:

1. **`llama-fit-params`** — the bundle's own tool, run exactly as SPEC 2.10 pins it
   (`--fit on --fit-target MiB --fit-ctx N --fit-print on`, plus the plan's `-c`/`-ngl`/`-b`),
   and its memory table (device name, model, context, compute — MiB) is parsed. `source` is
   `"llama-fit-params"` whenever that run produced a table.
2. **`estimate`** — ggufone's own math, used when no runtime/binary is available (or the run
   failed). That path always carries `W_FIT_ESTIMATED`.

The plan is cached per `(model sha256, host fingerprint)` under `<data-home>/fit/`, and applied
on load unless `--no-fit`.

**KV accounting (two numbers, on purpose).** The *planner* of E1a
(`registry/recommend.py`) charges KV conservatively as 1 byte per element for both q8_0 and
q4_0 — SPEC 2.4 freezes that, and it is an upper bound. The *fit plan* must be a faithful memory
estimate instead (A-E1c-6 cross-checks it against measured RSS ±20%), so it uses the real ggml
element sizes: f16 = 2 B/element, q8_0 = 34/32 B, q4_0 = 18/32 B. With unified KV
(`kv_unified=True`, mandatory for the fork engine) the cache holds `n_ctx` cells in total, so

    est_kv_bytes = kv_bytes_per_token(...) x n_ctx          (NOT x n_seq_max)

which is why the fit estimate and the conservative planner differ by design. `n_seq_max` is a
*concurrency* bound (how many candidate branches can be in flight), not a memory multiplier.

**Over budget (A-E1c-5).** The ladder is fixed: `kv_type` moves f16 -> q8_0 -> q4_0 first, and
each step that happens is reported with `W_KV_TYPE_DOWNGRADE`. Only when q4_0 still does not fit
does `n_ctx` shrink (down to `--fit-ctx`, default 4096), and a plan whose *weights* alone exceed
the budget is reported (`insufficient` note) rather than silently truncated.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ggufone.errors import GgufCorruptError, ModelNotFoundError
from ggufone.registry import gguf, recommend, store
from ggufone.runtime import finder

FIT_SCHEMA = "ggufone.fit/v1"
FIT_FIELDS: tuple[str, ...] = ("n_gpu_layers", "n_ctx", "kv_type", "n_seq_max",
                               "est_weights_bytes", "est_kv_bytes", "est_total_bytes",
                               "backend", "source")
KV_DOWNGRADE_ORDER: tuple[str, ...] = ("f16", "q8_0", "q4_0")
#: real ggml block sizes for the KV types we support (see the module docstring)
KV_BYTES_PER_ELEMENT: dict[str, float] = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
OVERHEAD_BYTES = 512 * 1024 * 1024            # SPEC 2.4's `conservative_plan` overhead
DEFAULT_N_CTX = 4096
DEFAULT_N_SEQ_MAX = 8
DEFAULT_FIT_TARGET_MB = 1024
MIN_CTX_FLOOR = 512
DEFAULT_TIMEOUT = 300.0
MIB = 1024 * 1024

#: ggml_type -> (elements per block, bytes per block). b11026's enum; ids 4 and 5 are the
#: removed Q4_2/Q4_3. Unknown ids are an error (never guessed — see `UnknownTensorType`).
GGML_TYPE_BLOCKS: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_k", 256, 84), 11: ("q3_k", 256, 110), 12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210), 15: ("q8_k", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1), 25: ("i16", 1, 2),
    26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8), 29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2), 31: ("q4_0_4_4", 32, 18), 32: ("q4_0_4_8", 32, 18),
    33: ("q4_0_8_8", 32, 18), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    36: ("iq4_nl_4_4", 32, 18), 37: ("iq4_nl_4_8", 32, 18), 38: ("iq4_nl_8_8", 32, 18),
    39: ("mxFP4", 32, 17),
}


class UnknownTensorType(GgufCorruptError):
    """A tensor uses a ggml type this build does not know — the plan must not guess its size."""


# ------------------------------------------------------------------- model facts
@dataclass(frozen=True, slots=True)
class ModelFacts:
    """Everything a fit plan needs from a GGUF: shape, context window and weight bytes."""

    path: str
    sha256: str
    arch: str | None
    n_layer: int
    n_kv_head: int
    key_len: int
    value_len: int
    n_ctx_train: int
    weights_bytes: int
    file_size: int = 0

    @classmethod
    def read(cls, path: str | os.PathLike[str], *, sha256: str | None = None,
             want_sha256: bool = True) -> ModelFacts:
        """Read the header + tensor index; the tensor data itself is never touched."""
        model_path = pathlib.Path(path)
        if not model_path.exists():
            raise ModelNotFoundError(f"E_MODEL_NOT_FOUND: {model_path} does not exist")
        meta = gguf.parse_gguf_metadata(model_path)
        kv = meta["kv"]
        arch = gguf.arch_of(kv)
        tensors = read_tensor_index(model_path)
        weights = sum(size_of_tensor(dims, ttype) for _name, dims, ttype, _offset in tensors)
        return cls(
            path=str(model_path),
            sha256=sha256 or (gguf.sha256_file(model_path) if want_sha256 else ""),
            arch=arch,
            n_layer=_kv_int(kv, arch, "block_count") or 0,
            n_kv_head=_kv_int(kv, arch, "attention.head_count_kv")
            or _kv_int(kv, arch, "attention.head_count") or 1,
            key_len=_kv_int(kv, arch, "attention.key_length")
            or _kv_int(kv, arch, "embedding_length") or 0,
            value_len=_kv_int(kv, arch, "attention.value_length")
            or _kv_int(kv, arch, "embedding_length") or 0,
            n_ctx_train=_kv_int(kv, arch, "context_length") or 0,
            weights_bytes=weights,
            file_size=model_path.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "arch": self.arch,
                "n_layer": self.n_layer, "n_kv_head": self.n_kv_head,
                "key_len": self.key_len, "value_len": self.value_len,
                "n_ctx_train": self.n_ctx_train, "weights_bytes": self.weights_bytes,
                "file_size": self.file_size}

    @property
    def kv_per_token_f16(self) -> int:
        return kv_bytes_per_token(self.n_layer, self.n_kv_head, self.key_len, self.value_len, 2)


def _kv_int(kv: dict[str, Any], arch: str | None, suffix: str) -> int | None:
    for key in (f"{arch}.{suffix}" if arch else None, f"general.{suffix}"):
        if key and isinstance(kv.get(key), int):
            return int(kv[key])
    return None


def read_tensor_index(path: str | os.PathLike[str]) -> list[tuple[str, list[int], int, int]]:
    """Parse the GGUF tensor index: `(name, dims, ggml_type, offset)` for every tensor.

    The index sits between the KV block and the tensor data (GGUF v3), so this is still a
    header-only read — no tensor byte is loaded.
    """
    import struct

    rows: list[tuple[str, list[int], int, int]] = []
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != gguf.GGUF_MAGIC:
            raise GgufCorruptError(f"E_GGUF_CORRUPT: {path}: not a GGUF file")
        struct.unpack("<I", handle.read(4))[0]                  # GGUF version
        n_tensors = struct.unpack("<Q", handle.read(8))[0]
        n_kv = struct.unpack("<Q", handle.read(8))[0]

        def read_string() -> str:
            length = struct.unpack("<Q", handle.read(8))[0]
            return handle.read(length).decode("utf-8", "replace")

        def skip_value(kind: int) -> None:
            fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if kind in fixed:
                handle.read(fixed[kind])
                return
            if kind == 8:
                read_string()
                return
            if kind == 9:
                item = struct.unpack("<I", handle.read(4))[0]
                count = struct.unpack("<Q", handle.read(8))[0]
                for _ in range(count):
                    skip_value(item)
                return
            raise GgufCorruptError(
                f"E_GGUF_CORRUPT: {path}: unknown metadata value type {kind}")

        for _ in range(n_kv):
            read_string()
            skip_value(struct.unpack("<I", handle.read(4))[0])
        for _ in range(n_tensors):
            name = read_string()
            n_dims = struct.unpack("<I", handle.read(4))[0]
            dims = [struct.unpack("<Q", handle.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", handle.read(4))[0]
            offset = struct.unpack("<Q", handle.read(8))[0]
            rows.append((name, dims, ttype, offset))
    return rows


def size_of_tensor(dims: Iterable[int], ttype: int) -> int:
    """Bytes one tensor occupies (block-quantised; raises on an unknown ggml type)."""
    if ttype not in GGML_TYPE_BLOCKS:
        raise UnknownTensorType(
            f"E_GGUF_CORRUPT: unknown ggml tensor type {ttype}; ggufone cannot size it "
            f"(known types: {min(GGML_TYPE_BLOCKS)}..{max(GGML_TYPE_BLOCKS)})")
    _name, block, block_bytes = GGML_TYPE_BLOCKS[ttype]
    elements = 1
    for dim in dims:
        elements *= int(dim)
    if elements % block:
        elements += block - (elements % block)          # the last block is padded
    return elements // block * block_bytes


# -------------------------------------------------------------------- host facts
@dataclass(frozen=True, slots=True)
class HostFacts:
    """What the host offers, plus the fingerprint the cache keys on."""

    backend: str
    ram_bytes: int
    vram_bytes: int
    n_cpu: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "ram_bytes": self.ram_bytes,
                "vram_bytes": self.vram_bytes, "n_cpu": self.n_cpu,
                "fingerprint": self.fingerprint}

    @property
    def budget_bytes(self) -> int:
        """Device memory when the plan offloads, host RAM otherwise."""
        return self.vram_bytes if self.vram_bytes > 0 else self.ram_bytes


def host_facts(*, meminfo_path: pathlib.Path | None = None,
               vram_probe: Callable[[], int | None] | None = None,
               backend: str | None = None, n_cpu: int | None = None) -> HostFacts:
    """The host's memory + the backend the runtime proved (SPEC 2.2/A-E1a)."""
    budget = recommend.host_budget(meminfo_path=meminfo_path, vram_probe=vram_probe)
    from ggufone.engine import session as session_module

    resolved_backend = backend or session_module.runtime_backend()
    cpus = n_cpu if n_cpu is not None else (os.cpu_count() or 1)
    fingerprint = "|".join((
        f"backend={resolved_backend}",
        f"ram={budget.ram_bytes}",
        f"vram={budget.vram_bytes}",
        f"cpus={cpus}",
        f"arch={platform.machine()}",
        f"os={platform.system().lower()}",
    ))
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return HostFacts(backend=resolved_backend, ram_bytes=budget.ram_bytes,
                     vram_bytes=budget.vram_bytes, n_cpu=cpus, fingerprint=digest)


# ---------------------------------------------------------------------- the plan
@dataclass(frozen=True, slots=True)
class FitPlan:
    n_gpu_layers: int
    n_ctx: int
    kv_type: str
    n_seq_max: int
    est_weights_bytes: int
    est_kv_bytes: int
    est_total_bytes: int
    backend: str
    source: str
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    arch: str | None = None
    model_sha256: str = ""
    host_fingerprint: str = ""
    budget_bytes: int = 0
    created_at: str = ""

    @property
    def insufficient(self) -> bool:
        return any("insufficient" in note for note in self.notes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {field_name: getattr(self, field_name)
                                   for field_name in FIT_FIELDS}
        payload.update({"warnings": list(self.warnings), "notes": list(self.notes),
                        "arch": self.arch, "model_sha256": self.model_sha256,
                        "host_fingerprint": self.host_fingerprint,
                        "budget_bytes": self.budget_bytes, "created_at": self.created_at,
                        "schema": FIT_SCHEMA})
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FitPlan:
        return cls(
            n_gpu_layers=int(payload["n_gpu_layers"]), n_ctx=int(payload["n_ctx"]),
            kv_type=str(payload["kv_type"]), n_seq_max=int(payload["n_seq_max"]),
            est_weights_bytes=int(payload["est_weights_bytes"]),
            est_kv_bytes=int(payload["est_kv_bytes"]),
            est_total_bytes=int(payload["est_total_bytes"]), backend=str(payload["backend"]),
            source=str(payload["source"]), warnings=tuple(payload.get("warnings", ())),
            notes=tuple(payload.get("notes", ())), arch=payload.get("arch"),
            model_sha256=str(payload.get("model_sha256", "")),
            host_fingerprint=str(payload.get("host_fingerprint", "")),
            budget_bytes=int(payload.get("budget_bytes", 0)),
            created_at=str(payload.get("created_at", "")))


def kv_bytes_per_token(n_layer: int, n_kv_head: int, key_len: int, value_len: int,
                       type_bytes: float) -> int:
    """Per-token KV cost of an attention stack (SPEC 2.4's formula, float for real ratios)."""
    return int(round(n_layer * n_kv_head * (key_len + value_len) * type_bytes))


def estimate_plan(model: ModelFacts, host: HostFacts, *, n_ctx: int = DEFAULT_N_CTX,
                  n_seq_max: int = DEFAULT_N_SEQ_MAX, kv_type: str = "auto",
                  fit_target_mb: int = DEFAULT_FIT_TARGET_MB, min_ctx: int | None = None,
                  budget_bytes: int | None = None, overhead_bytes: int = OVERHEAD_BYTES
                  ) -> FitPlan:
    """ggufone's own fit math (chain step 2): no binary, `source="estimate"`."""
    floor = min_ctx if min_ctx is not None else DEFAULT_N_CTX
    budget = budget_bytes if budget_bytes is not None \
        else max(0, host.budget_bytes - fit_target_mb * MIB)
    ladder = list(KV_DOWNGRADE_ORDER if kv_type in ("auto", None) else
                  KV_DOWNGRADE_ORDER[KV_DOWNGRADE_ORDER.index(kv_type):])
    warnings: list[str] = []
    notes: list[str] = []
    requested_ctx = max(int(n_ctx), floor)
    chosen_kv, chosen_ctx = ladder[0], requested_ctx
    downgraded = False
    for index, candidate in enumerate(ladder):
        if _plan_bytes(model, candidate, requested_ctx, overhead_bytes) <= budget:
            chosen_kv, chosen_ctx = candidate, requested_ctx
            downgraded = index > 0
            break
        if index < len(ladder) - 1:
            continue                     # A-E1c-5: the KV type moves down the ladder first
        chosen_kv = candidate            # last rung: shrink the context instead
        per_token = kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                       model.value_len, KV_BYTES_PER_ELEMENT[candidate])
        room = budget - model.weights_bytes - overhead_bytes
        shrunk = int(room // per_token) if per_token > 0 and room > 0 else 0
        chosen_ctx = max(floor, min(requested_ctx, shrunk))
        downgraded = index > 0
        if chosen_ctx < requested_ctx:
            notes.append(
                f"n_ctx shrunk {requested_ctx} -> {chosen_ctx} to fit the budget "
                f"({budget / MIB:.0f} MiB); raise --fit-target, lower --fit-ctx or use a "
                f"smaller quant")
    if downgraded:
        warnings.append("W_KV_TYPE_DOWNGRADE")
    kv_bytes = kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                  model.value_len, KV_BYTES_PER_ELEMENT[chosen_kv]) * chosen_ctx
    total_bytes = model.weights_bytes + kv_bytes + overhead_bytes
    if model.weights_bytes + overhead_bytes > budget:
        notes.append(
            f"insufficient device memory: weights {model.weights_bytes / MIB:.0f} MiB + "
            f"overhead {overhead_bytes / MIB:.0f} MiB exceed the budget "
            f"{budget / MIB:.0f} MiB; loading will spill or fail")
    n_gpu_layers = _gpu_layers(model, host, kv_bytes, budget, overhead_bytes)
    warnings.append("W_FIT_ESTIMATED")
    return FitPlan(n_gpu_layers=n_gpu_layers, n_ctx=chosen_ctx, kv_type=chosen_kv,
                   n_seq_max=int(n_seq_max), est_weights_bytes=model.weights_bytes,
                   est_kv_bytes=kv_bytes, est_total_bytes=total_bytes, backend=host.backend,
                   source="estimate", warnings=tuple(warnings), notes=tuple(notes),
                   arch=model.arch, model_sha256=model.sha256,
                   host_fingerprint=host.fingerprint, budget_bytes=budget,
                   created_at=_timestamp())


def _plan_bytes(model: ModelFacts, kv_type: str, n_ctx: int, overhead_bytes: int) -> int:
    return model.weights_bytes + overhead_bytes + kv_bytes_per_token(
        model.n_layer, model.n_kv_head, model.key_len, model.value_len,
        KV_BYTES_PER_ELEMENT[kv_type]) * n_ctx


def _gpu_layers(model: ModelFacts, host: HostFacts, kv_bytes: int, budget: int,
                overhead_bytes: int) -> int:
    """Full offload when it fits, else the per-layer split that does, else CPU (0)."""
    if host.vram_bytes <= 0 or model.n_layer <= 0:
        return 0
    room = budget - kv_bytes - overhead_bytes
    if room <= 0:
        return 0
    per_layer = model.weights_bytes / model.n_layer
    layers = int(room // per_layer)
    if layers <= 0:
        return 0
    return min(model.n_layer, layers)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------- the binary (chain step 1)
@dataclass(frozen=True, slots=True)
class FitTableRow:
    name: str
    model_bytes: int
    context_bytes: int
    compute_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.model_bytes + self.context_bytes + self.compute_bytes


def parse_fit_table(text: str) -> list[FitTableRow]:
    """Parse `llama-fit-params --fit-print on` stdout: `name model context compute` (MiB)."""
    rows: list[FitTableRow] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        numbers = parts[-3:]
        if not all(part.isdigit() for part in numbers):
            continue
        rows.append(FitTableRow(name=" ".join(parts[:-3]) or "Host",
                                model_bytes=int(numbers[0]) * MIB,
                                context_bytes=int(numbers[1]) * MIB,
                                compute_bytes=int(numbers[2]) * MIB))
    return rows


def fit_binary(runtime_dir: str | os.PathLike[str] | None) -> pathlib.Path | None:
    if runtime_dir is None:
        return None
    tool = finder.layout(runtime_dir).tools.get("llama-fit-params")
    return tool if tool and tool.exists() else None


def run_llama_fit_params(model: ModelFacts, host: HostFacts, *,
                         runtime_dir: str | os.PathLike[str] | None,
                         n_ctx: int, n_seq_max: int, fit_target_mb: int = DEFAULT_FIT_TARGET_MB,
                         min_ctx: int = DEFAULT_N_CTX, budget_bytes: int | None = None,
                         runner: Callable[[list[str]], str] | None = None,
                         timeout: float = DEFAULT_TIMEOUT) -> FitPlan | None:
    """Run the bundle's own tool (SPEC 2.10 flags); `None` when it cannot run/parse."""
    binary = fit_binary(runtime_dir)
    if runner is None and binary is None:
        return None
    batch = max(512, int(n_ctx))
    argv = [str(binary) if binary else "llama-fit-params", "-m", model.path,
            "--fit", "on", "--fit-target", str(int(fit_target_mb)),
            "--fit-ctx", str(int(max(min_ctx, MIN_CTX_FLOOR))), "--fit-print", "on",
            "-c", str(int(n_ctx)), "-b", str(batch), "-ub", str(min(512, batch)),
            "-ngl", str(0 if host.vram_bytes <= 0 else model.n_layer)]
    table = runner(argv) if runner is not None else _run(argv, timeout)
    if table is None:
        return None
    rows = parse_fit_table(table)
    if not rows:
        return None
    return plan_from_binary(model, host, table=table, n_ctx=n_ctx, n_seq_max=n_seq_max,
                            runtime_dir=runtime_dir, budget_bytes=budget_bytes)


def _run(argv: list[str], timeout: float) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,  # noqa: S603
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def plan_from_binary(model: ModelFacts, host: HostFacts, *, table: str, n_ctx: int,
                     n_seq_max: int, runtime_dir: str | os.PathLike[str] | None,
                     budget_bytes: int | None = None, kv_type: str = "auto") -> FitPlan:
    """Turn a parsed table into a plan whose `est_*` numbers are the binary's own."""
    rows = parse_fit_table(table)
    weights = sum(row.model_bytes for row in rows)
    context = sum(row.context_bytes for row in rows)
    compute = sum(row.compute_bytes for row in rows)
    budget = budget_bytes if budget_bytes is not None \
        else max(0, host.budget_bytes - DEFAULT_FIT_TARGET_MB * MIB)
    chosen_kv, kv_bytes = _kv_from_budget(model, n_ctx, budget, context, kv_type)
    warnings: list[str] = []
    if chosen_kv != "f16":
        warnings.append("W_KV_TYPE_DOWNGRADE")
    notes = [f"memory table from {pathlib.Path(str(runtime_dir or 'llama-fit-params')).name}/"
             f"llama-fit-params (model {weights / MIB:.0f} MiB, context "
             f"{context / MIB:.0f} MiB, compute {compute / MIB:.0f} MiB)"]
    return FitPlan(n_gpu_layers=0 if host.vram_bytes <= 0 else model.n_layer, n_ctx=int(n_ctx),
                   kv_type=chosen_kv, n_seq_max=int(n_seq_max), est_weights_bytes=weights,
                   est_kv_bytes=kv_bytes, est_total_bytes=weights + kv_bytes + compute,
                   backend=host.backend, source="llama-fit-params", warnings=tuple(warnings),
                   notes=tuple(notes), arch=model.arch, model_sha256=model.sha256,
                   host_fingerprint=host.fingerprint, budget_bytes=budget,
                   created_at=_timestamp())


def _kv_from_budget(model: ModelFacts, n_ctx: int, budget: int, binary_context: int,
                    kv_type: str) -> tuple[str, int]:
    """Which KV type fits, and what to report as `est_kv_bytes` for it."""
    if kv_type not in ("auto", None):
        return kv_type, kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                           model.value_len, KV_BYTES_PER_ELEMENT[kv_type]) * n_ctx
    for candidate in KV_DOWNGRADE_ORDER:
        kv_bytes = kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                      model.value_len, KV_BYTES_PER_ELEMENT[candidate]) * n_ctx
        if model.weights_bytes + kv_bytes + OVERHEAD_BYTES <= budget:
            # f16 is what the binary measured: keep its number, not our formula's
            return candidate, (binary_context if candidate == "f16" else kv_bytes)
    return KV_DOWNGRADE_ORDER[-1], kv_bytes_per_token(
        model.n_layer, model.n_kv_head, model.key_len, model.value_len,
        KV_BYTES_PER_ELEMENT[KV_DOWNGRADE_ORDER[-1]]) * n_ctx


# -------------------------------------------------------------------------- cache
def cache_path(sha256: str, fingerprint: str, home: pathlib.Path | None = None) -> pathlib.Path:
    return (home or store.data_home()) / "fit" / f"{sha256}.{fingerprint}.json"


def load_cached(sha256: str, fingerprint: str, home: pathlib.Path | None = None
                ) -> FitPlan | None:
    path = cache_path(sha256, fingerprint, home)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != FIT_SCHEMA:
        return None
    try:
        return FitPlan.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def store_plan(plan: FitPlan, home: pathlib.Path | None = None) -> pathlib.Path:
    if not plan.model_sha256:
        raise ValueError("a cached plan needs the model sha256 (the cache key)")
    path = cache_path(plan.model_sha256, plan.host_fingerprint, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(plan.to_dict(), indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def plan_for_model(model: ModelFacts, host: HostFacts, *, home: pathlib.Path | None = None,
                   runtime_dir: str | os.PathLike[str] | None = None, use_cache: bool = True,
                   runner: Callable[[list[str]], str] | None = None,
                   budget_bytes: int | None = None, n_ctx: int = DEFAULT_N_CTX,
                   n_seq_max: int = DEFAULT_N_SEQ_MAX, kv_type: str = "auto",
                   fit_target_mb: int = DEFAULT_FIT_TARGET_MB,
                   min_ctx: int | None = None) -> FitPlan:
    """The A-E1c-4 entry point: cache -> binary -> estimate, in that order."""
    if use_cache and model.sha256:
        cached = load_cached(model.sha256, host.fingerprint, home)
        if cached is not None:
            return cached
    fallback_kv = kv_type if kv_type not in ("auto", None) else "f16"
    preliminary = estimate_plan(model, host, n_ctx=n_ctx, n_seq_max=n_seq_max,
                                kv_type=fallback_kv, budget_bytes=budget_bytes,
                                fit_target_mb=fit_target_mb, min_ctx=min_ctx)
    plan = run_llama_fit_params(model, host, runtime_dir=runtime_dir, n_ctx=preliminary.n_ctx,
                                n_seq_max=n_seq_max, fit_target_mb=fit_target_mb,
                                min_ctx=min_ctx or n_ctx, budget_bytes=budget_bytes,
                                runner=runner) or preliminary
    if use_cache and model.sha256:
        store_plan(plan, home)
    return plan


def plan_for_path(path: str | os.PathLike[str], *, host: HostFacts | None = None,
                  home: pathlib.Path | None = None, sha256: str | None = None,
                  **kwargs: Any) -> FitPlan:
    """:func:`plan_for_model` for a plain path (the CLI's `fit <model>`)."""
    model = ModelFacts.read(path, sha256=sha256)
    return plan_for_model(model, host or host_facts(), home=home, **kwargs)


# ------------------------------------------------------------- apply on load (E1c-5)
def session_overrides(plan: FitPlan, *, n_ctx: int | None, kv_type: str, n_seq_max: int | None
                      ) -> dict[str, Any]:
    """What the plan contributes to a load: the request always wins where it spoke."""
    return {
        "n_ctx": plan.n_ctx if n_ctx is None else n_ctx,
        "kv_type": plan.kv_type if kv_type in ("auto", None) else kv_type,
        "n_seq_max": plan.n_seq_max if n_seq_max is None else n_seq_max,
        "n_gpu_layers": plan.n_gpu_layers,
    }


# ------------------------------------------------------------- RSS cross-check (E1c-6)
def rss_ratio(estimated_bytes: int, measured_bytes: int) -> float:
    """`(measured - estimated) / estimated` — the A-E1c-6 cross-check number."""
    if estimated_bytes <= 0:
        return float("inf")
    return (measured_bytes - estimated_bytes) / estimated_bytes


def within_tolerance(estimated_bytes: int, measured_bytes: int, tolerance: float = 0.20) -> bool:
    return abs(rss_ratio(estimated_bytes, measured_bytes)) <= tolerance


def measured_rss_bytes(pid: int | None = None) -> int | None:
    """Current RSS of `pid` (default: this process) from `/proc/<pid>/status`, or None."""
    target = pid or os.getpid()
    status = pathlib.Path(f"/proc/{target}/status")
    if not status.exists():
        return None
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None
