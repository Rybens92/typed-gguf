"""GGUF header/metadata reader (header only, never tensor data)

Milestone: E1a.

Contract mirrored by docs/verify_runtime_contract.py:
parse_gguf_metadata + sha256_file.

The reader reads the fixed header and the KV block, stops, and never touches tensor data
(SPEC 2.7). Every structural problem raises `E_GGUF_CORRUPT` with the path and the reason,
so a truncated download can never be mistaken for a usable model.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import struct
from typing import IO, Any

from ggufone.errors import GgufCorruptError

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (1, 2, 3)

# gguf metadata value types (gguf spec)
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32 = 0, 1, 2, 3, 4, 5, 6
T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 7, 8, 9, 10, 11, 12
_TYPE_FMT: dict[int, str] = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}
# guards against absurd counts in corrupt files (metadata only == small)
MAX_N_KV = 1 << 20
MAX_ARRAY_LEN = 1 << 22
MAX_ARRAY_DEPTH = 2
MAX_STRING_LEN = 1 << 24

# llama_ftype enum, include/llama.h @ b11026 (captured 2026-09-17; the oracle pins 7 and 15)
FTYPE_NAMES: dict[int, str] = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
    19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S",
    22: "MOSTLY_IQ3_XS",
    23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S",
    25: "MOSTLY_IQ4_NL",
    26: "MOSTLY_IQ3_S",
    27: "MOSTLY_IQ3_M",
    28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M",
    30: "MOSTLY_IQ4_XS",
    31: "MOSTLY_IQ1_M",
    32: "MOSTLY_BF16",
    36: "MOSTLY_TQ1_0",
    37: "MOSTLY_TQ2_0",
    38: "MOSTLY_MXFP4_MOE",
    39: "MOSTLY_NVFP4",
    40: "MOSTLY_Q1_0",
    41: "MOSTLY_Q2_0",
    1024: "GUESSED",
}


def quant_label(file_type: int) -> str:
    """llama_ftype -> the short quant name used in registry entries."""
    name = FTYPE_NAMES.get(int(file_type))
    if name is None:
        return f"FTYPE_{int(file_type)}"
    for prefix in ("MOSTLY_", "ALL_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class _Reader:
    """Bounds-checked little-endian reader over a binary stream."""

    def __init__(self, fh: IO[bytes], label: str) -> None:
        self.fh = fh
        self.label = label

    def read(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GgufCorruptError(
                f"E_GGUF_CORRUPT: {self.label}: unexpected end of file "
                f"(wanted {n} more bytes, got {len(data)})")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > MAX_STRING_LEN:
            raise GgufCorruptError(
                f"E_GGUF_CORRUPT: {self.label}: string length {n} exceeds the metadata cap")
        return self.read(n).decode("utf-8", "replace")

    def value(self, type_id: int, depth: int = 0) -> Any:
        if type_id == T_STRING:
            return self.string()
        if type_id == T_ARRAY:
            if depth >= MAX_ARRAY_DEPTH:
                raise GgufCorruptError(
                    f"E_GGUF_CORRUPT: {self.label}: array nesting deeper than {MAX_ARRAY_DEPTH}")
            elem_type = self.u32()
            count = self.u64()
            if count > MAX_ARRAY_LEN:
                raise GgufCorruptError(
                    f"E_GGUF_CORRUPT: {self.label}: array length {count} exceeds the cap")
            return [self.value(elem_type, depth + 1) for _ in range(count)]
        fmt = _TYPE_FMT.get(type_id)
        if fmt is None:
            raise GgufCorruptError(
                f"E_GGUF_CORRUPT: {self.label}: unknown metadata value type {type_id}")
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


def parse_gguf_metadata(source: str | os.PathLike[str] | IO[bytes]) -> dict[str, Any]:
    """Header + KV metadata of a GGUF file (never tensor data).

    Accepts a path or an open binary stream. Raises `E_GGUF_CORRUPT` for anything that is
    not a readable GGUF v1..v3 header.
    """
    if hasattr(source, "read"):
        fh = source  # type: ignore[assignment]
        label = getattr(source, "name", "<stream>")
        return _parse(_Reader(fh, str(label)), str(label))
    path = pathlib.Path(source)
    try:
        with open(path, "rb") as handle:
            return _parse(_Reader(handle, str(path)), str(path))
    except FileNotFoundError as exc:
        raise GgufCorruptError(f"E_GGUF_CORRUPT: {path}: file not found") from exc
    except IsADirectoryError as exc:
        raise GgufCorruptError(f"E_GGUF_CORRUPT: {path}: is a directory") from exc
    except PermissionError as exc:
        raise GgufCorruptError(f"E_GGUF_CORRUPT: {path}: permission denied") from exc


def _parse(reader: _Reader, label: str) -> dict[str, Any]:
    magic = reader.read(4)
    if magic != GGUF_MAGIC:
        raise GgufCorruptError(
            f"E_GGUF_CORRUPT: {label}: not a GGUF file (magic {magic!r})")
    version = reader.u32()
    if version not in SUPPORTED_VERSIONS:
        raise GgufCorruptError(
            f"E_GGUF_CORRUPT: {label}: GGUF version {version} is not supported "
            f"(this reader handles v1..v3)")
    n_tensors = reader.u64()
    n_kv = reader.u64()
    if n_kv > MAX_N_KV:
        raise GgufCorruptError(
            f"E_GGUF_CORRUPT: {label}: n_kv {n_kv} exceeds the metadata cap")
    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key = reader.string()
        kv[key] = reader.value(reader.u32())
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "kv": kv}


# oracle-compatible alias (docs/verify_runtime_contract.py calls it parse_gguf_metadata)
read_metadata = parse_gguf_metadata


def arch_of(kv: dict[str, Any]) -> str | None:
    value = kv.get("general.architecture")
    return str(value) if value else None


def file_type_of(kv: dict[str, Any]) -> int | None:
    value = kv.get("general.file_type")
    return int(value) if isinstance(value, int) else None


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
