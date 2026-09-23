"""Fit plans: `llama-fit-params` + typed-gguf's own KV/n_seq_max math (SPEC 2.10, A-E1c-4/5/6).

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
2. **`estimate`** — typed-gguf's own math, used when no runtime/binary is available (or the run
   failed). That path always carries `W_FIT_ESTIMATED`.

The plan is cached per `(model sha256, host fingerprint)` under `<data-home>/fit/`, and applied
on load unless `--no-fit`.

**KV accounting (two numbers, on purpose).** The *planner* of E1a
(`registry/recommend.py`) charges KV conservatively as 1 byte per element for both q8_0 and
q4_0 — SPEC 2.4 freezes that, and it is an upper bound. The *fit plan* must be a faithful memory
estimate instead (A-E1c-6 cross-checks it against measured RSS ±20%), so it uses the real ggml
element sizes: f16 = 2 B/element, q8_0 = 34/32 B, q4_0 = 18/32 B. With unified KV
(`kv_unified=True`, mandatory for the fork engine) the cache holds `n_ctx` cells in total, so

    est_kv_bytes = kv_bytes(model, n_ctx, kv_type)          (NOT x n_seq_max)

which is why the fit estimate and the conservative planner differ by design. `n_seq_max` is a
*concurrency* bound (how many candidate branches can be in flight), not a memory multiplier.

**Sliding-window attention (SPEC-context-v2 §3).** `kv_bytes` models what the runtime really
allocates: for a model that declares `attention.sliding_window`, llama.cpp's `llama_kv_cache_iswa`
holds `n_ctx` cells on the global layers and `window + n_ubatch` cells on the SWA layers
(measured; 1 024 at `-ub 512`, 768 at 256), so the SWA part is a constant that does not grow with
the context. The all-layer formula (`kv_bytes_per_token`) over-charges such a model ~4x — it is
kept, unchanged, for models without a window and for the SPEC 2.4 oracle path.

**Context policy v2 (SPEC-context-v2 §5, ratified 2026-09-23).** A plan with no explicit `--n-ctx`
aims at `STANDARD_N_CTX = 32768`, grows to the largest context the box holds at the top rung that
can reach the standard (window-capped) and shrinks gracefully — KV ladder first, then context below
the standard with `W_CTX_BELOW_STANDARD` — when it cannot. An explicit `--n-ctx N` is a *pin*
(`min(N, cap)`, never grown). `DEFAULT_N_CTX = 4096` stays the shrink **floor** (`--fit-ctx`) and
the low-level default of an explicit `estimate_plan(..., n_ctx=N)`.

**Over budget (A-E1c-5, v2 §5.3).** The ladder is fixed: `kv_type` moves f16 -> q8_0 -> q4_0 first,
and each step that happens is reported with `W_KV_TYPE_DOWNGRADE`. Only when the last rung still
cannot hold the target does `n_ctx` shrink (down to `--fit-ctx`, default 4096, and never above the
model's own window), reported as `W_CTX_BELOW_STANDARD`; a plan whose *weights* alone exceed the
budget is reported (`insufficient` note) rather than silently truncated.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from typed_gguf.errors import BackendOomError, GgufCorruptError, ModelNotFoundError
from typed_gguf.registry import gguf, recommend, store
from typed_gguf.runtime import finder

FIT_SCHEMA = "typed_gguf.fit/v1"
FIT_FIELDS: tuple[str, ...] = ("n_gpu_layers", "n_ctx", "kv_type", "n_seq_max",
                               "est_weights_bytes", "est_kv_bytes", "est_total_bytes",
                               "backend", "source", "standard_n_ctx", "ctx_limit")
KV_DOWNGRADE_ORDER: tuple[str, ...] = ("f16", "q8_0", "q4_0")
#: real ggml block sizes for the KV types we support (see the module docstring)
KV_BYTES_PER_ELEMENT: dict[str, float] = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
OVERHEAD_BYTES = 512 * 1024 * 1024            # SPEC 2.4's `conservative_plan` overhead
#: The shrink **floor** (`--fit-ctx` default) and the low-level default of an explicit
#: `estimate_plan(..., n_ctx=N)`. It is NOT the target of a policy plan — that is
#: `STANDARD_N_CTX` (SPEC-context-v2 §5.1: one standard constant only).
DEFAULT_N_CTX = 4096
#: SPEC-context-v2 §5.1/§5.2: the context a plan aims at when the user pins nothing. It stays a
#: *plan* default: a request may use anything up to the loaded context, and grows/shrinks are
#: reported (`standard_n_ctx`, `ctx_limit`, `W_CTX_BELOW_STANDARD`).
STANDARD_N_CTX = 32768
#: What `max_fit_n_ctx` answers for a model whose every layer is SWA: the cache does not grow
#: with the context at all, so only the model's own window (or this) bounds it.
UNBOUNDED_CTX = 1_048_576
DEFAULT_N_SEQ_MAX = 8
DEFAULT_FIT_TARGET_MB = 1024
MIN_CTX_FLOOR = 512
DEFAULT_TIMEOUT = 300.0
MIB = 1024 * 1024
#: `n_ubatch` the SWA cache sizing assumes (`session.DEFAULT_N_BATCH`; SPEC-context-v2 §3)
DEFAULT_N_UBATCH = 512

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
    """Everything a fit plan needs from a GGUF: shape, context window and weight bytes.

    The three trailing SWA fields are SPEC-context-v2 §5.1/§7.1: `attention.sliding_window` and
    its bool `sliding_window_pattern` are *optional* — absent means "no sliding window" and every
    formula stays byte-identical to the pre-v2 one (AC-7).
    """

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
    #: the window in tokens (512 on spark2_5); 0 = this model does not use SWA
    sliding_window: int = 0
    #: layers whose cache is SWA-sized (the `True` entries of the pattern; all of them when the
    #: model declares a window without a pattern)
    n_swa_layers: int = 0
    #: layers whose cache holds `n_ctx` cells
    n_global_layers: int = 0

    @property
    def has_swa(self) -> bool:
        """Whether the SWA cache model applies to this model at all."""
        return self.sliding_window > 0 and self.n_swa_layers > 0

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
        n_layer = _kv_int(kv, arch, "block_count") or 0
        window = _kv_int(kv, arch, "attention.sliding_window") or 0
        n_swa = 0
        if window > 0:
            pattern = _kv_bools(kv, arch, "attention.sliding_window_pattern")
            n_swa = sum(1 for flag in pattern if flag) if pattern else n_layer
        return cls(
            path=str(model_path),
            sha256=sha256 or (gguf.sha256_file(model_path) if want_sha256 else ""),
            arch=arch,
            n_layer=n_layer,
            n_kv_head=_kv_int(kv, arch, "attention.head_count_kv")
            or _kv_int(kv, arch, "attention.head_count") or 1,
            key_len=_kv_int(kv, arch, "attention.key_length")
            or _kv_int(kv, arch, "embedding_length") or 0,
            value_len=_kv_int(kv, arch, "attention.value_length")
            or _kv_int(kv, arch, "embedding_length") or 0,
            n_ctx_train=_kv_int(kv, arch, "context_length") or 0,
            weights_bytes=weights,
            file_size=model_path.stat().st_size,
            sliding_window=window,
            n_swa_layers=max(0, min(n_swa, n_layer)),
            n_global_layers=max(0, n_layer - max(0, min(n_swa, n_layer))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "arch": self.arch,
                "n_layer": self.n_layer, "n_kv_head": self.n_kv_head,
                "key_len": self.key_len, "value_len": self.value_len,
                "n_ctx_train": self.n_ctx_train, "weights_bytes": self.weights_bytes,
                "file_size": self.file_size, "sliding_window": self.sliding_window,
                "n_swa_layers": self.n_swa_layers, "n_global_layers": self.n_global_layers}

    @property
    def kv_per_token_f16(self) -> int:
        return kv_bytes_per_token(self.n_layer, self.n_kv_head, self.key_len, self.value_len, 2)


def _kv_int(kv: dict[str, Any], arch: str | None, suffix: str) -> int | None:
    for key in (f"{arch}.{suffix}" if arch else None, f"general.{suffix}"):
        if key and isinstance(kv.get(key), int):
            return int(kv[key])
    return None


def _kv_bools(kv: dict[str, Any], arch: str | None, suffix: str) -> list[bool]:
    """The `[T,T,T,F,…]` SWA pattern as written in the GGUF (empty when the key is absent)."""
    for key in (f"{arch}.{suffix}" if arch else None, f"general.{suffix}"):
        value = kv.get(key) if key else None
        if isinstance(value, list):
            return [bool(item) for item in value]
    return []


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
            f"E_GGUF_CORRUPT: unknown ggml tensor type {ttype}; typed-gguf cannot size it "
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
    #: What the driver says is free *right now*. 0 = not reported (fall back to `vram_bytes`).
    vram_free_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "ram_bytes": self.ram_bytes,
                "vram_bytes": self.vram_bytes, "vram_free_bytes": self.vram_free_bytes,
                "n_cpu": self.n_cpu, "fingerprint": self.fingerprint}

    @property
    def budget_bytes(self) -> int:
        """Device memory a plan may spend: the FREE number when the driver reports one.

        `vram_bytes` is the nominal size and is what the *fingerprint* keys on (a plan must not be
        re-planned from scratch every time the desktop takes another 200 MiB); `vram_free_bytes`
        is what the planner and the load-time re-validation must respect (card t_8cb0a05e: the
        operator's box reported 8192 MiB total and 1112 MiB free).
        """
        if self.vram_bytes > 0:
            if 0 < self.vram_free_bytes < self.vram_bytes:
                return self.vram_free_bytes
            return self.vram_bytes
        return self.ram_bytes


def host_facts(*, meminfo_path: pathlib.Path | None = None,
               vram_probe: Callable[[], int | None] | None = None,
               device_probe: Callable[[], recommend.DeviceMemory | None] | None = None,
               backend: str | None = None, n_cpu: int | None = None) -> HostFacts:
    """The host's memory + the backend the runtime proved (SPEC 2.2/A-E1a).

    One driver query answers both numbers (`recommend.device_memory`). An injected `device_probe`
    owns the answer completely — the real driver is then never consulted, which is what keeps a
    test honest on a box that has a GPU.
    """
    if device_probe is None and vram_probe is None:
        device_probe = recommend.device_memory
    memory = device_probe() if device_probe is not None else None
    if memory is None:
        # A total-only probe (the E1a `vram_probe=` API): free is unknown, not zero.
        total = int(vram_probe() or 0) if vram_probe is not None else 0
        memory = recommend.DeviceMemory(total_bytes=total, free_bytes=0, source="injected")
    budget = recommend.host_budget(meminfo_path=meminfo_path,
                                   vram_probe=lambda: memory.total_bytes or None)
    from typed_gguf.engine import session as session_module

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
                     vram_bytes=budget.vram_bytes, n_cpu=cpus, fingerprint=digest,
                     vram_free_bytes=int(memory.free_bytes))


def fit_budget(host: HostFacts, fit_target_mb: int = DEFAULT_FIT_TARGET_MB) -> int:
    """The bytes a plan may spend: the host budget minus the `--fit-target` margin.

    `--fit-target MiB` is a margin the plan must leave free (llama.cpp's own `--fit-target`
    semantics: "target margin per device", default 1024). It bounds every source of a plan —
    the estimate *and* the `llama-fit-params` table — and it can take the budget to zero, which
    is the correct answer on a desktop that holds the whole device (1112 MiB free, 5200 MiB
    target -> 0 -> CPU).
    """
    return max(0, host.budget_bytes - int(fit_target_mb) * MIB)


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
    #: SPEC-context-v2 §5.1: the standard in force for this plan (`0` = none applies: a payload
    #: written before v2, or a plan from a caller that knows no policy)
    standard_n_ctx: int = 0
    #: why the context is what it is: "standard" | "grown" | "shrunk" | "pinned" | "window" | ""
    ctx_limit: str = ""

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
            created_at=str(payload.get("created_at", "")),
            standard_n_ctx=int(payload.get("standard_n_ctx", 0)),
            ctx_limit=str(payload.get("ctx_limit", "")))


def kv_bytes_per_token(n_layer: int, n_kv_head: int, key_len: int, value_len: int,
                       type_bytes: float) -> int:
    """Per-token KV cost of an attention stack (SPEC 2.4's formula, float for real ratios)."""
    return int(round(n_layer * n_kv_head * (key_len + value_len) * type_bytes))


def kv_bytes(model: ModelFacts, n_ctx: int, kv_type: str, *,
             n_ubatch: int = DEFAULT_N_UBATCH) -> int:
    """The bytes the KV cache really holds for `n_ctx` at `kv_type` (SPEC-context-v2 §3).

    Without sliding-window attention this is SPEC 2.4's formula, byte for byte (AC-7). With it,
    the model is llama.cpp's `llama_kv_cache_iswa`: `n_global_layers` hold `n_ctx` cells and
    `n_swa_layers` hold `window + n_ubatch` cells, so the SWA part is a constant and only the
    global layers price the growth (36 864 B/token at f16 for the 4B — the measured layout).
    """
    type_bytes = KV_BYTES_PER_ELEMENT[kv_type]
    if not model.has_swa:
        return kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                  model.value_len, type_bytes) * int(n_ctx)
    per_layer = int(round(model.n_kv_head * (model.key_len + model.value_len) * type_bytes))
    cells = max(0, int(model.sliding_window)) + max(0, int(n_ubatch))
    return (model.n_global_layers * per_layer * int(n_ctx)
            + model.n_swa_layers * per_layer * cells)


def _kv_per_layer(model: ModelFacts, kv_type: str) -> int:
    return int(round(model.n_kv_head * (model.key_len + model.value_len)
                     * KV_BYTES_PER_ELEMENT[kv_type]))


def max_fit_n_ctx(model: ModelFacts, kv_type: str, budget: int, *,
                  overhead_bytes: int = OVERHEAD_BYTES,
                  n_ubatch: int = DEFAULT_N_UBATCH) -> int:
    """The largest `n_ctx` this budget holds at `kv_type` (0 when even the weights do not fit).

    The inverse of :func:`kv_bytes`, which is what AC-2 asserts: a grown plan's `n_ctx` *is* this
    number at its rung. `UNBOUNDED_CTX` answers a model whose every layer is SWA (its cache does
    not grow with the context at all).
    """
    room = int(budget) - model.weights_bytes - int(overhead_bytes)
    if room <= 0:
        return 0
    if model.has_swa:
        growth = model.n_global_layers * _kv_per_layer(model, kv_type)
        constant = model.n_swa_layers * _kv_per_layer(model, kv_type) * (
            model.sliding_window + max(0, int(n_ubatch)))
        if growth <= 0:
            return UNBOUNDED_CTX
        return max(0, (room - constant) // growth)
    per_token = kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                   model.value_len, KV_BYTES_PER_ELEMENT[kv_type])
    return max(0, room // per_token) if per_token > 0 else UNBOUNDED_CTX


def model_window(model: ModelFacts) -> int | None:
    """The model's own context ceiling, or `None` when the GGUF declares none (§5.2)."""
    return model.n_ctx_train if model.n_ctx_train > 0 else None


def policy_target(model: ModelFacts) -> int:
    """§5.2's target for an unpinned plan: the standard, capped by a smaller model window."""
    window = model_window(model)
    return STANDARD_N_CTX if window is None else min(STANDARD_N_CTX, window)


def ctx_limit_for(chosen: int, *, target: int, model: ModelFacts, pinned: bool = False) -> str:
    """§5.3.3/§5.4: the label for *how* the chosen context was reached."""
    window = model_window(model)
    if pinned and chosen == target:
        return "pinned"
    if window is not None and chosen == window and chosen != STANDARD_N_CTX:
        return "window"
    if chosen == STANDARD_N_CTX:
        return "standard"
    if chosen > target:
        return "grown"
    return "shrunk"


def policy_note(target: int, chosen: int, *, kv_type: str, budget: int,
                fit_target_mb: int) -> str | None:
    """§5.3.3's arithmetic note, or `None` when there is nothing to explain."""
    if chosen > target:
        return (f"n_ctx {target} -> {chosen}: the box holds more "
                f"(fit-target {int(fit_target_mb)} MiB kept free, kv_type {kv_type})")
    if chosen < target:
        return (f"n_ctx shrunk {target} -> {chosen} to fit the budget "
                f"({budget / MIB:.0f} MiB); raise --fit-target, lower --fit-ctx or use a "
                f"smaller quant")
    return None


def estimate_plan(model: ModelFacts, host: HostFacts, *, n_ctx: int | None = None,
                  n_seq_max: int = DEFAULT_N_SEQ_MAX, kv_type: str = "auto",
                  fit_target_mb: int = DEFAULT_FIT_TARGET_MB, min_ctx: int | None = None,
                  budget_bytes: int | None = None, overhead_bytes: int = OVERHEAD_BYTES
                  ) -> FitPlan:
    """typed-gguf's own fit math (chain step 2): no binary, `source="estimate"`.

    `n_ctx=None` (the v2 default) applies SPEC-context-v2 §5.2-5.4: the plan aims at
    `STANDARD_N_CTX`, grows into the room the box really has at the top rung that can reach the
    standard, and shrinks with `W_CTX_BELOW_STANDARD` when nothing above the floor does. An int
    `n_ctx` is a **pin** (§5.5): `min(n_ctx, cap)`, never grown — every pre-v2 caller passes one.
    `min_ctx` stays the shrink floor (`--fit-ctx`, default `DEFAULT_N_CTX`).
    """
    floor = min_ctx if min_ctx is not None else DEFAULT_N_CTX
    window = model_window(model)
    if window is not None:
        floor = min(floor, window)
    budget = budget_bytes if budget_bytes is not None else fit_budget(host, fit_target_mb)
    ladder = list(KV_DOWNGRADE_ORDER if kv_type in ("auto", None) else
                  KV_DOWNGRADE_ORDER[KV_DOWNGRADE_ORDER.index(kv_type):])
    pinned = n_ctx is not None
    target = max(1, int(n_ctx)) if pinned else policy_target(model)
    if pinned and window is not None:
        target = min(target, window)
    warnings: list[str] = []
    notes: list[str] = []
    chosen_kv, chosen_ctx = ladder[0], target
    downgraded = False
    for index, candidate in enumerate(ladder):
        cap = max_fit_n_ctx(model, candidate, budget, overhead_bytes=overhead_bytes)
        if window is not None:
            cap = min(cap, window)
        if cap >= target:
            chosen_kv = candidate
            chosen_ctx = cap if not pinned else min(int(n_ctx), cap)
            downgraded = index > 0
            break
        if index == len(ladder) - 1:
            # §5.3.4: nothing above the floor reaches the target — the last rung shrinks the
            # context instead, and growth is not attempted on a rung chosen this way.
            chosen_kv = candidate
            chosen_ctx = max(floor, min(target, cap))
            downgraded = index > 0
    if downgraded:
        warnings.append("W_KV_TYPE_DOWNGRADE")
    if not pinned and chosen_ctx < policy_target(model):
        # §5.4: a plan below the *standard* — when a smaller model window is the reason, §5.2 is
        # explicit that nothing was degraded (no warning, the window simply is the ceiling).
        warnings.append("W_CTX_BELOW_STANDARD")
    if window is not None and window < STANDARD_N_CTX and not pinned:
        notes.append(f"the model's own window is {window} tokens (standard {STANDARD_N_CTX}); "
                     f"a plan cannot go above it")
    note = policy_note(target, chosen_ctx, kv_type=chosen_kv, budget=budget,
                       fit_target_mb=fit_target_mb)
    if note is not None:
        notes.append(note)
    limit = ctx_limit_for(chosen_ctx, target=target, model=model, pinned=pinned)
    kv_total = kv_bytes(model, chosen_ctx, chosen_kv)
    total_bytes = model.weights_bytes + kv_total + overhead_bytes
    if model.weights_bytes + overhead_bytes > budget:
        notes.append(
            f"insufficient device memory: weights {model.weights_bytes / MIB:.0f} MiB + "
            f"overhead {overhead_bytes / MIB:.0f} MiB exceed the budget "
            f"{budget / MIB:.0f} MiB; loading will spill or fail")
    n_gpu_layers = _gpu_layers(model, host, kv_total, budget, overhead_bytes)
    if budget_bytes is None and host.vram_bytes > 0 and 0 < host.vram_free_bytes \
            < host.vram_bytes:
        # The desktop holds part of the device: say out loud that the plan is smaller than the
        # nominal host could take (card t_8cb0a05e — W_FIT_DOWNGRADE is the machine-readable form).
        nominal_budget = fit_budget(dataclasses.replace(host, vram_free_bytes=0), fit_target_mb)
        nominal_layers = _gpu_layers(model, host, kv_total, nominal_budget, overhead_bytes)
        if n_gpu_layers < nominal_layers:
            warnings.append("W_FIT_DOWNGRADE")
            notes.append(
                f"planned against free device memory: {host.vram_free_bytes / MIB:.0f} MiB free "
                f"of {host.vram_bytes / MIB:.0f} MiB, so {n_gpu_layers} instead of "
                f"{nominal_layers} layer(s) are offloaded")
    warnings.append("W_FIT_ESTIMATED")
    return FitPlan(n_gpu_layers=n_gpu_layers, n_ctx=chosen_ctx, kv_type=chosen_kv,
                   n_seq_max=int(n_seq_max), est_weights_bytes=model.weights_bytes,
                   est_kv_bytes=kv_total, est_total_bytes=total_bytes, backend=host.backend,
                   source="estimate", warnings=tuple(warnings), notes=tuple(notes),
                   arch=model.arch, model_sha256=model.sha256,
                   host_fingerprint=host.fingerprint, budget_bytes=budget,
                   created_at=_timestamp(), standard_n_ctx=STANDARD_N_CTX, ctx_limit=limit)


def _plan_bytes(model: ModelFacts, kv_type: str, n_ctx: int, overhead_bytes: int) -> int:
    return model.weights_bytes + overhead_bytes + kv_bytes(model, n_ctx, kv_type)


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


# ------------------------------------------------- re-planning against live free memory (E1c FIX)
def plan_device_bytes(plan: FitPlan, model: ModelFacts) -> int:
    """How many bytes of *device* memory this plan asks for.

    Full offload owns the weights, the KV cache and the runtime overhead; a partial offload owns
    the offloaded share of the weights plus the same KV/overhead (llama.cpp keeps the KV of an
    offloaded layer on that device, which is the conservative reading `_gpu_layers` already uses).
    A CPU plan (`n_gpu_layers == 0`) asks the device for nothing; a *negative* count means every
    layer (`planned_layers`), i.e. the full weight footprint.
    """
    layers = planned_layers(plan, model)
    if layers <= 0 or model.n_layer <= 0:
        return 0
    share = min(1.0, layers / model.n_layer)
    weights = int(plan.est_weights_bytes * share) if share < 1.0 else plan.est_weights_bytes
    return weights + plan.est_kv_bytes


def _kv_bytes_for(model: ModelFacts, kv_type: str, n_ctx: int) -> int:
    """KV bytes for one rung of a plan (SWA-aware — see :func:`kv_bytes`)."""
    return kv_bytes(model, n_ctx, kv_type)


# ------------------------------------------------- placements, not just plans (E2 FIX t_31b3943a)
@runtime_checkable
class PlacementLike(Protocol):
    """The minimum a caller must carry to be let through the ladder.

    `typed-gguf bench` passes exactly this — a layer count and nothing else — because a published
    row has to be reproducible from its flags, not from a fit plan written on another box.
    """

    n_gpu_layers: int


def kv_start(kv_type: str | None) -> str:
    """The rung a KV ladder starts at: a known type keeps its place, anything else starts on top.

    `auto` (the request default) and an unknown word both mean "nothing was pinned", which is the
    top rung — the rule `estimate_plan` and `session._kv_ladder` already use, kept in one function
    so the load-time ladder cannot disagree with them.
    """
    value = str(kv_type) if kv_type else ""
    return value if value in KV_DOWNGRADE_ORDER else KV_DOWNGRADE_ORDER[0]


def coerce_plan(plan: Any) -> FitPlan:
    """Normalize a *placement-like* object into the `FitPlan` the loader's ladder documents.

    Not every caller of `session.open_model` holds a fit plan: `typed-gguf bench` names it
    explicitly (`--gpu-layers`), and that object carries one field (card t_31b3943a). Every field
    the ladder reads must have an answer, so the ones a minimal placement cannot know get the
    honest default instead of an `AttributeError`:

    * `kv_type = "auto"` — nothing was pinned; a load only needs the layer count and the context
      init resolves the rung (`session._kv_ladder`).
    * `n_ctx`, `est_*`, `budget_bytes = 0` — a load sizes no cache, so there is nothing to size.

    A real `FitPlan` is returned unchanged (identity preserved: callers compare plans).
    """
    if isinstance(plan, FitPlan):
        return plan
    return FitPlan(
        n_gpu_layers=int(getattr(plan, "n_gpu_layers", 0) or 0),
        n_ctx=int(getattr(plan, "n_ctx", 0) or 0),
        kv_type=str(getattr(plan, "kv_type", None) or "auto"),
        n_seq_max=int(getattr(plan, "n_seq_max", 0) or 0),
        est_weights_bytes=int(getattr(plan, "est_weights_bytes", 0) or 0),
        est_kv_bytes=int(getattr(plan, "est_kv_bytes", 0) or 0),
        est_total_bytes=int(getattr(plan, "est_total_bytes", 0) or 0),
        backend=str(getattr(plan, "backend", "") or ""),
        source=str(getattr(plan, "source", "") or ""),
        warnings=tuple(getattr(plan, "warnings", ()) or ()),
        notes=tuple(getattr(plan, "notes", ()) or ()),
        arch=getattr(plan, "arch", None),
        model_sha256=str(getattr(plan, "model_sha256", "") or ""),
        host_fingerprint=str(getattr(plan, "host_fingerprint", "") or ""),
        budget_bytes=int(getattr(plan, "budget_bytes", 0) or 0),
        created_at=str(getattr(plan, "created_at", "") or ""))


def planned_layers(plan: FitPlan, model: ModelFacts) -> int:
    """The layer count a plan really asks the device for: a negative count means "all of them".

    llama.cpp reads `n_gpu_layers < 0` as "offload every layer" — which is what the benchmark's
    GPU default passes — so `-1` is the *largest* footprint a plan can have, not an empty one: the
    ladder has to be able to reduce from it (card t_31b3943a).
    """
    if plan.n_gpu_layers < 0:
        return max(0, int(model.n_layer))
    return max(0, int(plan.n_gpu_layers))


def degrade_ladder(plan: FitPlan | PlacementLike, model: ModelFacts) -> tuple[FitPlan, ...]:
    """The documented degradation ladder: fewer layers -> smaller kv_type -> CPU-only.

    Ordered by decreasing device footprint, ending at a plan that asks the device for nothing, so
    the last rung is always available. Each step carries `W_FIT_DOWNGRADE` (the plan was reduced)
    and, when the KV type moved, `W_KV_TYPE_DOWNGRADE`. A caller walks it until a load succeeds.

    The input may be any placement-like object: it is normalized first (`coerce_plan`), so a
    minimal `n_gpu_layers`-only placement and a `kv_type: auto` plan are walked like any other —
    never an `AttributeError`, never a `KeyError` (card t_31b3943a).
    """
    plan = coerce_plan(plan)
    steps: list[FitPlan] = []
    start = kv_start(plan.kv_type)           # `auto`/unknown -> the top rung, like the planner
    rungs = list(KV_DOWNGRADE_ORDER[KV_DOWNGRADE_ORDER.index(start):])

    def emit(n_gpu_layers: int, kv_type: str) -> None:
        kv_bytes = _kv_bytes_for(model, kv_type, plan.n_ctx)
        warnings = list(plan.warnings)
        if "W_FIT_DOWNGRADE" not in warnings:
            warnings.append("W_FIT_DOWNGRADE")
        if kv_type != start and "W_KV_TYPE_DOWNGRADE" not in warnings:
            warnings.append("W_KV_TYPE_DOWNGRADE")
        notes = list(plan.notes)
        if n_gpu_layers == 0:
            notes.append("CPU only: no weights are offloaded to the device")
        steps.append(dataclasses.replace(
            plan, n_gpu_layers=int(n_gpu_layers), kv_type=kv_type, est_kv_bytes=kv_bytes,
            est_total_bytes=plan.est_weights_bytes + kv_bytes
            + (plan.est_total_bytes - plan.est_weights_bytes - plan.est_kv_bytes),
            warnings=tuple(warnings), notes=tuple(notes)))

    layers = planned_layers(plan, model)
    if layers > 0:
        emit(max(1, layers // 2), start)
        emit(0, start)
    for rung in rungs:
        if rung != start:
            emit(0, rung)
    return tuple(steps)


def replan_for_host(model: ModelFacts, plan: FitPlan, host: HostFacts, *,
                    fit_target_mb: int = DEFAULT_FIT_TARGET_MB, min_ctx: int | None = None,
                    overhead_bytes: int = OVERHEAD_BYTES) -> FitPlan:
    """Re-validate a (possibly cached) plan against the device memory free *now*.

    A cache keyed on the host fingerprint cannot see the desktop: the fingerprint is host identity
    (nominal VRAM included), while availability changes between runs. So every plan is re-checked
    against a fresh `fit_budget` and shrunk down the ladder until its device footprint fits; when
    even the CPU rung cannot hold the weights the plan is still returned, with the `insufficient`
    note (a CPU path that will page is better evidence than a hard failure).
    """
    budget = fit_budget(host, fit_target_mb)
    if plan_device_bytes(plan, model) <= budget:
        return plan if plan.budget_bytes == budget else dataclasses.replace(
            plan, budget_bytes=budget)
    for candidate in degrade_ladder(plan, model):
        if plan_device_bytes(candidate, model) <= budget:
            notes = list(candidate.notes)
            notes.append(
                f"re-planned for free device memory: {budget / MIB:.0f} MiB budget "
                f"(driver reported {host.vram_free_bytes / MIB:.0f} MiB free of "
                f"{host.vram_bytes / MIB:.0f} MiB, --fit-target {int(fit_target_mb)} MiB); "
                f"n_gpu_layers {plan.n_gpu_layers} -> {candidate.n_gpu_layers}, kv_type "
                f"{plan.kv_type} -> {candidate.kv_type}")
            return dataclasses.replace(candidate, budget_bytes=budget, notes=tuple(notes))
    # Nothing above the CPU rung can be afforded: place the whole model on the host CPU. The plan
    # is returned (not raised): a CPU decision is a working answer, a hard failure is not.
    cpu = dataclasses.replace(plan, n_gpu_layers=0, budget_bytes=budget)
    notes = list(cpu.notes)
    notes.append(
        f"CPU only: {budget / MIB:.0f} MiB of device memory is available after --fit-target "
        f"{int(fit_target_mb)} MiB (driver reported {host.vram_free_bytes / MIB:.0f} MiB free of "
        f"{host.vram_bytes / MIB:.0f} MiB); weights stay on the host")
    if host.ram_bytes > 0 and cpu.est_total_bytes > host.ram_bytes:
        notes.append(
            f"insufficient host memory too: the CPU plan needs {cpu.est_total_bytes / MIB:.0f} "
            f"MiB but the host has {host.ram_bytes / MIB:.0f} MiB")
    return dataclasses.replace(cpu, warnings=tuple(cpu.warnings + ("W_FIT_DOWNGRADE",))
                              if "W_FIT_DOWNGRADE" not in cpu.warnings else cpu.warnings,
                              notes=tuple(notes))


# ------------------------------------------- backend allocation failures (E1c FIX, req. 3 + 4)
#: What a ggml/driver backend prints when it cannot get device memory. The reference is the
#: operator's own tail (card t_8cb0a05e):
#:     ggml_vulkan: Device memory allocation of size 1058982400 failed.
#:     ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
#:     alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1058982400
#:     llama_model_load: error loading model: unable to allocate Vulkan0 buffer
OOM_SIGNATURES: tuple[str, ...] = (
    "erroroutofdevicememory", "out of device memory", "device memory allocation of size",
    "failed to allocate", "unable to allocate", "cannot allocate memory",
    "out of memory", "cudamalloc failed", "hipmalloc failed",
)
#: ... and what a real architecture failure looks like (never retried as if it were memory).
ARCH_SIGNATURES: tuple[str, ...] = (
    "unknown model architecture", "unsupported model architecture",
    "invalid model architecture", "e_model_arch_unsupported",
    "architecture is not supported", "architecture not supported",
)
ALLOCATION_SIZE_RE = re.compile(r"(?:allocation|buffer) of size (\d+)", re.IGNORECASE)


def classify_load_failure(text: str) -> str:
    """`"oom"` | `"arch"` | `"unknown"` for a captured backend log.

    OOM wins over arch on purpose: the operator's tail contains *both* (typed-gguf's own
    `E_MODEL_ARCH_UNSUPPORTED` line was the last line of an allocation failure), and reading it as
    an architecture problem is exactly the misclassification the card reports.
    """
    lowered = (text or "").lower()
    if any(signature in lowered for signature in OOM_SIGNATURES):
        return "oom"
    if any(signature in lowered for signature in ARCH_SIGNATURES):
        return "arch"
    return "unknown"


def allocation_bytes_from_log(text: str) -> int | None:
    """The biggest allocation size the backend named, in bytes (0/None when the log is silent)."""
    sizes = [int(match.group(1)) for match in ALLOCATION_SIZE_RE.finditer(text or "")]
    return max(sizes) if sizes else None


def oom_log_line(text: str) -> str:
    """The first line that looks like the allocation failure, for the error message."""
    for line in (text or "").splitlines():
        if any(signature in line.lower() for signature in OOM_SIGNATURES):
            return line.strip()[:200]
    return ""


def backend_oom_error(plan: FitPlan | None, *, free_bytes: int | None = None,
                      needed_bytes: int | None = None, log_tail: str = "",
                      attempts: Iterable[str] = ()) -> BackendOomError:
    """E_BACKEND_OOM carrying the plan, the free/needed numbers and the hints (requirement 4)."""
    layers = "n/a" if plan is None else plan.n_gpu_layers
    kv_type = "n/a" if plan is None else plan.kv_type
    free_text = "unknown" if free_bytes is None else f"{free_bytes / MIB:.0f} MiB"
    needed_text = "unknown" if needed_bytes is None else f"{needed_bytes / MIB:.0f} MiB"
    parts = [f"E_BACKEND_OOM: llama.cpp could not allocate device memory for the fit plan "
             f"(n_gpu_layers={layers}, kv_type={kv_type}, needed ~{needed_text}); the driver "
             f"reports {free_text} free"]
    if plan is not None and plan.budget_bytes and free_bytes is None:
        parts.append(f"and the plan's own budget was {plan.budget_bytes / MIB:.0f} MiB")
    attempted = list(attempts)
    if attempted:
        parts.append(f"tried {len(attempted)} placement(s) down to CPU-only, none fit: "
                     + "; ".join(attempted))
    if needed_bytes is not None:
        parts.append(f"the backend asked for a {needed_bytes / MIB:.0f} MiB allocation")
    line = oom_log_line(log_tail)
    if line:
        parts.append(f"backend log: '{line}'")
    parts.append("fix: `--no-fit` runs on the CPU, `--fit-target <MiB>` leaves that much device "
                 "memory free for the rest of the desktop, or use a smaller quant")
    return BackendOomError("; ".join(parts))


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
                         timeout: float = DEFAULT_TIMEOUT, pinned: bool = False) -> FitPlan | None:
    """Run the bundle's own tool (SPEC 2.10 flags); `None` when it cannot run/parse.

    `-ngl` is the layer count the current budget can actually hold (never an unconditional full
    offload: on a busy desktop that asks the tool about a placement that cannot exist, which is
    how `--fit-target` came to be ignored — card t_8cb0a05e). `pinned` says how the caller derived
    `n_ctx` (`--n-ctx` vs the v2 policy); it only labels the plan's `ctx_limit` (§5.1).
    """
    binary = fit_binary(runtime_dir)
    if runner is None and binary is None:
        return None
    batch = max(512, int(n_ctx))
    budget = budget_bytes if budget_bytes is not None else fit_budget(host, fit_target_mb)
    layers = _gpu_layers(model, host, kv_bytes=_kv_bytes_for(model, "f16", n_ctx), budget=budget,
                         overhead_bytes=OVERHEAD_BYTES)
    argv = [str(binary) if binary else "llama-fit-params", "-m", model.path,
            "--fit", "on", "--fit-target", str(int(fit_target_mb)),
            "--fit-ctx", str(int(max(min_ctx, MIN_CTX_FLOOR))), "--fit-print", "on",
            "-c", str(int(n_ctx)), "-b", str(batch), "-ub", str(min(512, batch)),
            "-ngl", str(layers)]
    table = runner(argv) if runner is not None else _run(argv, timeout)
    if table is None:
        return None
    rows = parse_fit_table(table)
    if not rows:
        return None
    return plan_from_binary(model, host, table=table, n_ctx=n_ctx, n_seq_max=n_seq_max,
                            runtime_dir=runtime_dir, budget_bytes=budget,
                            fit_target_mb=fit_target_mb, pinned=pinned)


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
                     budget_bytes: int | None = None, kv_type: str = "auto",
                     fit_target_mb: int = DEFAULT_FIT_TARGET_MB,
                     pinned: bool = False) -> FitPlan:
    """Turn a parsed table into a plan whose `est_*` numbers are the binary's own.

    The caller's `--fit-target` bounds the plan here too (the table itself is measured for a full
    offload, so without this the plan would claim device memory the target forbids) and
    `n_gpu_layers` is the floored per-layer split that actually fits the budget.

    The context *decision* stays the caller's (`n_ctx` is the resolved v2 policy answer or the
    pin), so this is also where the plan gets its `standard_n_ctx` / `ctx_limit` label and the
    policy's arithmetic note (§5.1/§5.3.3).
    """
    rows = parse_fit_table(table)
    weights = sum(row.model_bytes for row in rows)
    context = sum(row.context_bytes for row in rows)
    compute = sum(row.compute_bytes for row in rows)
    budget = budget_bytes if budget_bytes is not None else fit_budget(host, fit_target_mb)
    chosen_kv, kv_total = _kv_from_budget(model, n_ctx, budget, context, kv_type)
    window = model_window(model)
    target = min(int(n_ctx), window) if pinned and window is not None else (
        int(n_ctx) if pinned else policy_target(model))
    warnings: list[str] = []
    if chosen_kv != "f16" and chosen_kv != kv_type:
        warnings.append("W_KV_TYPE_DOWNGRADE")
    notes = [f"memory table from {pathlib.Path(str(runtime_dir or 'llama-fit-params')).name}/"
             f"llama-fit-params (model {weights / MIB:.0f} MiB, context "
             f"{context / MIB:.0f} MiB, compute {compute / MIB:.0f} MiB)"]
    note = policy_note(target, int(n_ctx), kv_type=chosen_kv, budget=budget,
                       fit_target_mb=fit_target_mb)
    if note is not None:
        notes.append(note)
    layers = _gpu_layers(model, host, kv_bytes=kv_total, budget=budget,
                         overhead_bytes=compute or OVERHEAD_BYTES)
    if host.vram_bytes > 0 and layers < model.n_layer:
        warnings.append("W_FIT_DOWNGRADE")
        notes.append(
            f"device memory bound: offloading {layers}/{model.n_layer} layers within "
            f"{budget / MIB:.0f} MiB (--fit-target {int(fit_target_mb)} MiB, "
            f"{host.vram_free_bytes / MIB:.0f} MiB free of {host.vram_bytes / MIB:.0f} MiB)")
    return FitPlan(n_gpu_layers=layers, n_ctx=int(n_ctx),
                   kv_type=chosen_kv, n_seq_max=int(n_seq_max), est_weights_bytes=weights,
                   est_kv_bytes=kv_total, est_total_bytes=weights + kv_total + compute,
                   backend=host.backend, source="llama-fit-params", warnings=tuple(warnings),
                   notes=tuple(notes), arch=model.arch, model_sha256=model.sha256,
                   host_fingerprint=host.fingerprint, budget_bytes=budget,
                   created_at=_timestamp(), standard_n_ctx=STANDARD_N_CTX,
                   ctx_limit=ctx_limit_for(int(n_ctx), target=target, model=model,
                                           pinned=pinned))


def _kv_from_budget(model: ModelFacts, n_ctx: int, budget: int, binary_context: int,
                    kv_type: str) -> tuple[str, int]:
    """Which KV type fits, and what to report as `est_kv_bytes` for it (SWA-aware, §3)."""
    if kv_type not in ("auto", None):
        return kv_type, kv_bytes(model, n_ctx, kv_type)
    for candidate in KV_DOWNGRADE_ORDER:
        kv_total = kv_bytes(model, n_ctx, candidate)
        if model.weights_bytes + kv_total + OVERHEAD_BYTES <= budget:
            # f16 is what the binary measured: keep its number, not our formula's
            return candidate, (binary_context if candidate == "f16" else kv_total)
    return KV_DOWNGRADE_ORDER[-1], kv_bytes(model, n_ctx, KV_DOWNGRADE_ORDER[-1])


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
                   budget_bytes: int | None = None, n_ctx: int | None = None,
                   n_seq_max: int = DEFAULT_N_SEQ_MAX, kv_type: str = "auto",
                   fit_target_mb: int = DEFAULT_FIT_TARGET_MB,
                   min_ctx: int | None = None) -> FitPlan:
    """The A-E1c-4 entry point: cache -> binary -> estimate, in that order.

    `n_ctx=None` (the v2 default) is the policy of §5.2-5.4; an int is the pin of §5.5. The
    binary is asked about the **policy's** context, never about a shrunken estimate's (§3.1: the
    old seed made the binary confirm the 4x-over-charged answer instead of answering for itself);
    the rung it reports is still re-derived from the budget.

    A cache hit is a *candidate*, not an answer: the cached plan is re-validated against the
    device memory free right now (`replan_for_host`) and the shrunken plan is written back, so a
    plan that was honest when it was written cannot OOM the box after the desktop grew
    (card t_8cb0a05e; the fresh reading comes from the caller's `host`). A plan computed from
    fresh probe numbers was already planned against them.
    """
    if use_cache and model.sha256:
        cached = load_cached(model.sha256, host.fingerprint, home)
        if cached is not None:
            fresh = replan_for_host(model, cached, host, fit_target_mb=fit_target_mb,
                                    min_ctx=min_ctx)
            if fresh.to_dict() != cached.to_dict():
                store_plan(fresh, home)
            return fresh
    preliminary = estimate_plan(model, host, n_ctx=n_ctx, n_seq_max=n_seq_max,
                                kv_type=kv_type, budget_bytes=budget_bytes,
                                fit_target_mb=fit_target_mb, min_ctx=min_ctx)
    pinned = n_ctx is not None
    plan = run_llama_fit_params(model, host, runtime_dir=runtime_dir, n_ctx=preliminary.n_ctx,
                                n_seq_max=n_seq_max, fit_target_mb=fit_target_mb,
                                min_ctx=min_ctx if min_ctx is not None else (
                                    int(n_ctx) if pinned else DEFAULT_N_CTX),
                                budget_bytes=budget_bytes, runner=runner,
                                pinned=pinned) or preliminary
    plan = replan_for_host(model, plan, host, fit_target_mb=fit_target_mb, min_ctx=min_ctx)
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
