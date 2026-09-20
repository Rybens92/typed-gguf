"""Mutation-hardening round (Tier M): boundary cases the first pass showed were untested.

The survivors of the scoped mutmut pass clustered on (a) *exact* budget boundaries in
`recommend_quant` and (b) the structural caps of the GGUF reader — both data-integrity paths,
so they get tests rather than a "metric mutant" note.
"""
from __future__ import annotations

import struct

import pytest

from typed_gguf.errors import TypedGgufError
from typed_gguf.registry import gguf, recommend


# --------------------------------------------------------------- budget boundaries
def test_gpu_budget_boundary_is_inclusive() -> None:
    """total == int(vram * (1 - margin)) must fit; the `<` variant would drop a quant."""
    got = recommend.recommend_quant(
        [("m.gguf", 8_000_000)], vram_bytes=10_000_000, ram_bytes=1,
        kv_per_token_f16=1_000_000, n_ctx=1, n_seq_max=1, overhead_bytes=0)
    assert got["placement"] == "gpu"
    assert got["kv_type"] == "f16"          # the largest KV type in the chain
    assert got["total"] == 9_000_000        # exactly int(10_000_000 * 0.90)


def test_gpu_budget_rejects_when_even_the_smallest_kv_does_not_fit() -> None:
    got = recommend.recommend_quant(
        [("m.gguf", 9_000_001)], vram_bytes=10_000_000, ram_bytes=1,
        kv_per_token_f16=1_000_000, n_ctx=1, n_seq_max=1, overhead_bytes=0)
    assert got["placement"] == "insufficient"


def test_gpu_margin_is_a_fraction_of_the_budget() -> None:
    """total 9.5 MB must not fit a 9.0 MB budget: `* (1 + margin)` / `/(1 - margin)` must fail."""
    got = recommend.recommend_quant(
        [("m.gguf", 8_500_000)], vram_bytes=10_000_000, ram_bytes=1,
        kv_per_token_f16=1_000_000, n_ctx=1, n_seq_max=1, overhead_bytes=0)
    assert got["placement"] == "gpu"
    assert got["kv_type"] == "q8_0"         # f16 needs 9.5 MB > 9.0 MB budget
    assert got["total"] == 9_000_000


def test_ram_budget_boundary_and_margin() -> None:
    # GPU can hold nothing, so the CPU branch decides — and it always uses f16 KV (the oracle's
    # reference semantics): budget = 10 MB * 0.80 = 8.0 MB exactly, weights 7.0 MB + 1.0 MB KV.
    got = recommend.recommend_quant(
        [("m.gguf", 7_000_000)], vram_bytes=1, ram_bytes=10_000_000,
        kv_per_token_f16=1_000_000, n_ctx=1, n_seq_max=1, overhead_bytes=0)
    assert got["placement"] == "cpu"
    assert got["kv_type"] == "f16"
    assert got["total"] == 8_000_000


def test_ram_rejects_a_candidate_above_the_margin() -> None:
    # 9.5 MB + 1.0 MB = 10.5 MB: fits a 10 MB RAM only if the margin is applied as
    # `*(1 + margin)` or `/(1 - margin)` instead of `*(1 - margin)`.
    got = recommend.recommend_quant(
        [("m.gguf", 9_500_000)], vram_bytes=1, ram_bytes=10_000_000,
        kv_per_token_f16=1_000_000, n_ctx=1, n_seq_max=1, overhead_bytes=0)
    assert got["placement"] == "insufficient"


def test_overhead_is_charged() -> None:
    """The 512 MiB overhead is not decorative: dropping it would flip this decision."""
    got = recommend.recommend_quant(
        [("m.gguf", 8_000_000)], vram_bytes=10_000_000, ram_bytes=1,
        kv_per_token_f16=1_000_000, n_ctx=1, n_seq_max=1, overhead_bytes=512 * 1024 * 1024)
    assert got["placement"] == "insufficient"


# --------------------------------------------------------------- GGUF reader caps
def gstr(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def header(kvs: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
            + b"".join(kvs))


def kv_str(key: str, value: str) -> bytes:
    return gstr(key) + struct.pack("<I", 8) + gstr(value)


def arr(elem_type: int, items: list[bytes]) -> bytes:
    return struct.pack("<IQ", elem_type, len(items)) + b"".join(items)


def write(tmp_path, blob: bytes, name: str = "s.gguf"):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


def test_string_at_the_cap_is_accepted_and_one_over_is_rejected(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "MAX_STRING_LEN", 8)
    ok = write(tmp_path, header([kv_str("k", "12345678")]), "ok.gguf")
    assert gguf.parse_gguf_metadata(ok)["kv"]["k"] == "12345678"
    bad = write(tmp_path, header([kv_str("k", "123456789")]), "bad.gguf")
    with pytest.raises(TypedGgufError) as exc:
        gguf.parse_gguf_metadata(bad)
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_n_kv_at_the_cap_is_accepted_and_one_over_is_rejected(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "MAX_N_KV", 1)
    ok = write(tmp_path, header([kv_str("k", "v")]), "ok.gguf")
    assert gguf.parse_gguf_metadata(ok)["n_kv"] == 1
    bad = write(tmp_path, header([kv_str("a", "1"), kv_str("b", "2")]), "bad.gguf")
    with pytest.raises(TypedGgufError) as exc:
        gguf.parse_gguf_metadata(bad)
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_array_length_at_the_cap_is_accepted_and_one_over_is_rejected(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "MAX_ARRAY_LEN", 2)
    ok = write(tmp_path, header([gstr("t") + struct.pack("<I", 9)
                                 + arr(8, [gstr("a"), gstr("b")])]), "ok.gguf")
    assert gguf.parse_gguf_metadata(ok)["kv"]["t"] == ["a", "b"]
    bad = write(tmp_path, header([gstr("t") + struct.pack("<I", 9)
                                  + arr(8, [gstr("a"), gstr("b"), gstr("c")])]), "bad.gguf")
    with pytest.raises(TypedGgufError) as exc:
        gguf.parse_gguf_metadata(bad)
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_three_level_nesting_is_rejected(tmp_path) -> None:
    """depth is charged per nesting level: a deeper array must hit MAX_ARRAY_DEPTH."""
    inner = arr(8, [gstr("x")])
    middle = arr(9, [inner])
    outer = arr(9, [middle])
    blob = header([gstr("t") + struct.pack("<I", 9) + outer])
    with pytest.raises(TypedGgufError) as exc:
        gguf.parse_gguf_metadata(write(tmp_path, blob))
    assert exc.value.code == "E_GGUF_CORRUPT"
    assert "nesting" in str(exc.value)


def test_two_level_nesting_is_fine(tmp_path) -> None:
    inner = arr(8, [gstr("x"), gstr("y")])
    blob = header([gstr("t") + struct.pack("<I", 9) + arr(9, [inner])])
    assert gguf.parse_gguf_metadata(write(tmp_path, blob))["kv"]["t"] == [["x", "y"]]
