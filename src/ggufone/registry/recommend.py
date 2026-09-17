"""recommend_quant(vram, ram, n_ctx, n_seq_max) — conservative planner

Milestone: E1a.

Contract + executed reference values: docs/verify_runtime_contract.py
section C.

The plan is a deliberate upper bound (SPEC S-4): KV is charged `n_ctx * n_seq_max`
(non-unified worst case) until E1a measures the real unified-cache footprint
(A-E1a-8). The recommender therefore stays pessimistic on purpose.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ggufone.errors import AmbiguousQuantError, ModelNotFoundError, UserError

# per-element KV storage cost (SPEC 2.4, mirrored by the oracle)
GGML_TYPE_BYTES: dict[str, int] = {"f16": 2, "q8_0": 1, "q4_0": 1}
KV_TYPES: tuple[str, ...] = ("f16", "q8_0", "q4_0")
OVERHEAD_BYTES = 512 * 1024 * 1024
VRAM_MARGIN = 0.10
RAM_MARGIN = 0.20

DEFAULT_N_CTX = 4096
DEFAULT_N_SEQ_MAX = 8

KiB = 1024.0
GIB = 1024 ** 3


def kv_bytes_per_token(n_layer: int, n_kv_head: int, key_len: int, value_len: int,
                       type_bytes: int) -> int:
    """Per token, per sequence KV cost of an attention stack (SPEC 2.4)."""
    return n_layer * n_kv_head * (key_len + value_len) * type_bytes


def plan_memory(weights_bytes: int, kv_per_token: int, n_ctx: int, n_seq_max: int,
                overhead_bytes: int = OVERHEAD_BYTES) -> dict[str, int]:
    """Conservative upper bound for one (model, host) plan (SPEC 2.4)."""
    kv = kv_per_token * n_ctx * n_seq_max
    return {"weights": weights_bytes, "kv": kv, "overhead": overhead_bytes,
            "total": weights_bytes + kv + overhead_bytes}


def recommend_quant(candidates: list[tuple[str, int]], *, vram_bytes: int, ram_bytes: int,
                    kv_per_token_f16: int, n_ctx: int, n_seq_max: int,
                    vram_margin: float = VRAM_MARGIN, ram_margin: float = RAM_MARGIN,
                    overhead_bytes: int = OVERHEAD_BYTES) -> dict[str, Any]:
    """Largest quant, then largest KV type, whose conservative plan fits.

    GPU first (VRAM margin 10%), then CPU (RAM margin 20%), else `insufficient`.
    Mirrors the oracle's reference implementation exactly (SPEC 2.7).
    """
    ordered = sorted(candidates, key=lambda c: -c[1])
    for label, weights in ordered:
        for kv_type in KV_TYPES:
            kvt = kv_per_token_f16 // 2 * GGML_TYPE_BYTES[kv_type]
            plan = plan_memory(weights, kvt, n_ctx, n_seq_max, overhead_bytes)
            if plan["total"] <= int(vram_bytes * (1 - vram_margin)):
                return {"quant": label, "kv_type": kv_type, "placement": "gpu", **plan}
    for label, weights in ordered:
        plan = plan_memory(weights, kv_per_token_f16, n_ctx, n_seq_max, overhead_bytes)
        if plan["total"] <= int(ram_bytes * (1 - ram_margin)):
            return {"quant": label, "kv_type": "f16", "placement": "cpu", **plan}
    return {"quant": None, "kv_type": None, "placement": "insufficient",
            "warning": "no quant fits vram or ram at this ctx/n_seq_max"}


# ---------------------------------------------------------------- host budget
@dataclass(frozen=True)
class Budget:
    vram_bytes: int
    ram_bytes: int


def _query_nvidia_smi() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    best = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            best = max(best or 0, int(line) * 1024 * 1024)
    return best


def _meminfo_total(path: pathlib.Path) -> int:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return 0


def host_budget(*, meminfo_path: pathlib.Path | None = None,
                nvidia_smi: Callable[[], int | None] | None = None,
                vram_probe: Callable[[], int | None] | None = None) -> Budget:
    """VRAM (best effort) + total RAM of this host, in bytes."""
    probe = vram_probe or nvidia_smi or _query_nvidia_smi
    vram = probe() or 0
    ram = _meminfo_total(meminfo_path or pathlib.Path("/proc/meminfo"))
    return Budget(vram_bytes=int(vram), ram_bytes=int(ram))


# ------------------------------------------------------------ file selection
_QUANT_RE = re.compile(r"(?:^|[-_.])(?P<q>[A-Za-z]{1,2}\d{1,2}(?:_[A-Za-z0-9]{1,4})*)(?:[-_.]|$)")


@dataclass(frozen=True)
class FileChoice:
    file: dict[str, Any]
    quant: str | None
    reason: str


def normalize_quant(quant: str) -> str:
    q = quant.strip().upper()
    if q.startswith("MOSTLY_"):
        q = q[len("MOSTLY_"):]
    if q in ("FP16", "F16", "HALF"):
        return "F16"
    if q in ("FP32", "F32"):
        return "F32"
    if q in ("BF16", "BFLOAT16"):
        return "BF16"
    return q


def is_gguf(entry: dict[str, Any]) -> bool:
    return str(entry.get("path", "")).lower().endswith(".gguf")


def quant_token_of(path: str) -> str | None:
    """Extract the quant token from a GGUF file name, e.g. `...-4B-Q8_0.gguf` -> `Q8_0`."""
    stem = pathlib.PurePosixPath(path).name
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    found = None
    for match in _QUANT_RE.finditer(stem):
        token = match.group("q")
        if re.match(r"^(?:Q|IQ|F|BF)\d", token, re.IGNORECASE):
            found = normalize_quant(token)
    return found


def _url_basename(path: str) -> str:
    return pathlib.PurePosixPath(path).name


def select_file(files: Iterable[dict[str, Any]], *, quant: str | None = None,
                explicit_file: str | None = None, repo: str | None = None,
                lock: Any = None, vram_bytes: int | None = None,
                ram_bytes: int | None = None, n_ctx: int = DEFAULT_N_CTX,
                n_seq_max: int = DEFAULT_N_SEQ_MAX,
                kv_per_token_f16: int | None = None) -> FileChoice:
    """Pick exactly one GGUF file out of a repo tree (SPEC 2.7 / A-E1a-5)."""
    ggufs = [f for f in files if is_gguf(f)]
    if not ggufs:
        raise ModelNotFoundError(
            f"repo {repo or '<unknown>'} publishes no .gguf file"
            + (f" (looking for {quant})" if quant else ""))

    if explicit_file:
        for f in ggufs:
            if explicit_file in (f["path"], _url_basename(f["path"])):
                return FileChoice(file=f, quant=quant_token_of(f["path"]) or quant,
                                  reason="explicit")
        raise ModelNotFoundError(
            f"--file {explicit_file} is not in {(repo or 'the repo')}: "
            f"available: {', '.join(sorted(_url_basename(f['path']) for f in ggufs))}")

    if quant:
        want = normalize_quant(quant)
        # 1. the pinned alternates are authoritative for the default model (e.g. the F16
        #    file carries no quant token at all: `Spark-X2.5-4B.gguf`).
        if lock is not None and repo:
            default = getattr(lock, "default_model", None)
            if default is not None and repo == default.repo:
                alt = default.alternates.get(want) or default.alternates.get(quant.upper())
                if alt:
                    for f in ggufs:
                        if _url_basename(f["path"]) == alt["file"]:
                            return FileChoice(file=f, quant=want, reason="pinned-alternate")
        # 2. token match on the file name
        matches = [f for f in ggufs if quant_token_of(f["path"]) == want]
        if len(matches) == 1:
            return FileChoice(file=matches[0], quant=want, reason="quant")
        if len(matches) > 1:
            names = ", ".join(sorted(_url_basename(f["path"]) for f in matches))
            raise AmbiguousQuantError(
                f"E_AMBIGUOUS_QUANT: {repo or 'repo'}:{quant} matches {len(matches)} files "
                f"({names}); pass --file NAME to choose one")
        # 3. a quant equal to the whole stem (repos that name files after the quant)
        exact = [f for f in ggufs
                 if quant_token_of(f["path"]) is None
                 and normalize_quant(pathlib.PurePosixPath(f["path"]).stem) == want]
        if len(exact) == 1:
            return FileChoice(file=exact[0], quant=want, reason="quant")
        # 4. unquantized-file shorthand: F16/BF16/F32 files often have no token
        if want in ("F16", "BF16", "F32"):
            bare = [f for f in ggufs if quant_token_of(f["path"]) is None]
            if len(bare) == 1:
                return FileChoice(file=bare[0], quant=want, reason="quant-no-suffix")
            if len(bare) > 1:
                names = ", ".join(sorted(_url_basename(f["path"]) for f in bare))
                raise AmbiguousQuantError(
                    f"E_AMBIGUOUS_QUANT: {repo or 'repo'}:{quant} matches {len(bare)} files "
                    f"without a quant suffix ({names}); pass --file NAME")
        available = ", ".join(sorted({quant_token_of(f["path"]) or _url_basename(f["path"])
                                     for f in ggufs}))
        raise ModelNotFoundError(
            f"quant {quant} is not published by {repo or 'the repo'}; available: {available}")

    # bare repo -> recommend_quant picks the largest quant that fits this host
    budget = (Budget(vram_bytes or 0, ram_bytes or 0) if vram_bytes is not None or ram_bytes
              else host_budget())
    kvpt = kv_per_token_f16 or DEFAULT_KV_PER_TOKEN_F16
    cands = [(_url_basename(f["path"]), int(f.get("size") or 0)) for f in ggufs]
    plan = recommend_quant(cands, vram_bytes=budget.vram_bytes, ram_bytes=budget.ram_bytes,
                           kv_per_token_f16=kvpt, n_ctx=n_ctx, n_seq_max=n_seq_max)
    if plan["quant"] is None:
        raise UserError(
            f"no quant of {repo or 'the repo'} fits this host at n_ctx={n_ctx}, "
            f"n_seq_max={n_seq_max} (vram {budget.vram_bytes / GIB:.1f} GiB, "
            f"ram {budget.ram_bytes / GIB:.1f} GiB); pass an explicit quant "
            f"(e.g. :Q4_K_M) or lower n_ctx/n_seq_max",
            code="E_MODEL_NOT_FOUND")
    chosen = next(f for f in ggufs if _url_basename(f["path"]) == plan["quant"])
    return FileChoice(
        file=chosen, quant=quant_token_of(chosen["path"]),
        reason=f"recommend_quant({plan['placement']}, kv={plan['kv_type']}, "
               f"total={plan['total'] / GIB:.2f} GiB)")


# Spark-X2.5-4B KV/token at f16, from the executed GGUF header (SPEC 2.7). Used as the
# planning constant when a caller has not read the model's own header yet.
DEFAULT_KV_PER_TOKEN_F16 = 147_456

# the executed reference table of SPEC 2.7 (A-E1a-7) — `models recommend-quant --table`
PINNED_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"label": "8 GiB VRAM / ctx 4096 / n_seq_max 8", "vram_gib": 8, "ram_gib": 31,
     "n_ctx": 4096, "n_seq_max": 8},
    {"label": "8 GiB VRAM / ctx 2048 / n_seq_max 4", "vram_gib": 8, "ram_gib": 31,
     "n_ctx": 2048, "n_seq_max": 4},
    {"label": "8 GiB VRAM / ctx 32768 / n_seq_max 16", "vram_gib": 8, "ram_gib": 31,
     "n_ctx": 32768, "n_seq_max": 16},
)
