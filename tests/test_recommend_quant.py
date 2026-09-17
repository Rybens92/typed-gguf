"""recommend_quant + quant/file selection (SPEC 2.7, A-E1a-5, A-E1a-7).

The executed reference table lives in docs/verify_runtime_contract.py section C and in
SPEC 2.7; these tests re-assert it through ggufone's own implementation.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ggufone.errors import GgufoneError
from ggufone.registry.recommend import (
    GGML_TYPE_BYTES,
    OVERHEAD_BYTES,
    FileChoice,
    host_budget,
    kv_bytes_per_token,
    plan_memory,
    recommend_quant,
    select_file,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPARK_Q8 = ("Spark-X2.5-4B-Q8_0.gguf", 4_375_021_152)
SPARK_Q4 = ("Spark-X2.5-4B-Q4_K_M.gguf", 2_600_224_352)
SPARK_F16 = ("Spark-X2.5-4B.gguf", 8_229_920_352)
SPARK_KVPT_F16 = 147_456
GIB = 1024 ** 3


def cands() -> list[tuple[str, int]]:
    return [SPARK_Q8, SPARK_Q4, SPARK_F16]


# ---------------------------------------------------------------- KV math
def test_kv_bytes_per_token_matches_executed_table() -> None:
    assert kv_bytes_per_token(36, 4, 256, 256, GGML_TYPE_BYTES["f16"]) == 147_456
    assert kv_bytes_per_token(36, 4, 256, 256, GGML_TYPE_BYTES["q8_0"]) == 73_728
    assert kv_bytes_per_token(36, 4, 256, 256, GGML_TYPE_BYTES["q4_0"]) == 73_728
    assert kv_bytes_per_token(24, 8, 128, 128, 2) == 24 * 8 * 256 * 2


def test_plan_memory_is_conservative_upper_bound() -> None:
    plan = plan_memory(1_000_000_000, 147_456, 4096, 8, OVERHEAD_BYTES)
    assert plan["kv"] == 147_456 * 4096 * 8
    assert plan["total"] == 1_000_000_000 + plan["kv"] + OVERHEAD_BYTES
    assert OVERHEAD_BYTES == 512 * 1024 * 1024


# ------------------------------------------------- pinned scenarios (SPEC 2.7)
def test_recommend_quant_8gib_ctx4096_seq8_picks_q8_0_weights_and_q8_0_kv_on_gpu() -> None:
    got = recommend_quant(cands(), vram_bytes=8 * GIB, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=4096, n_seq_max=8)
    assert got["quant"] == SPARK_Q8[0]
    assert got["kv_type"] == "q8_0"
    assert got["placement"] == "gpu"
    # 4 375 021 152 + 73 728*4096*8 + 512 MiB
    assert got["total"] == 4_375_021_152 + 73_728 * 4096 * 8 + OVERHEAD_BYTES
    assert got["total"] < 8 * GIB * 0.90


def test_recommend_quant_8gib_ctx2048_seq4_allows_f16_kv() -> None:
    got = recommend_quant(cands(), vram_bytes=8 * GIB, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=2048, n_seq_max=4)
    assert got["quant"] == SPARK_Q8[0]
    assert got["kv_type"] == "f16"
    assert got["placement"] == "gpu"
    assert got["total"] == 4_375_021_152 + 147_456 * 2048 * 4 + OVERHEAD_BYTES
    assert got["total"] == 6_119_851_616  # 6.12 GB in the executed table


def test_recommend_quant_huge_ctx_is_insufficient_with_warning() -> None:
    got = recommend_quant(cands(), vram_bytes=8 * GIB, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=32768, n_seq_max=16)
    assert got["placement"] == "insufficient"
    assert got["quant"] is None and got["kv_type"] is None
    assert "warning" in got and got["warning"]


def test_recommend_quant_falls_back_to_cpu_when_vram_is_too_small() -> None:
    got = recommend_quant(cands(), vram_bytes=1024 ** 3, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=2048, n_seq_max=4)
    assert got["placement"] == "cpu"
    assert got["kv_type"] == "f16"  # CPU fallback keeps the largest KV type
    assert got["quant"] == SPARK_F16[0]  # largest quant that fits RAM, per SPEC 2.7
    assert got["total"] <= int(31 * GIB * 0.80)


def test_recommend_quant_orders_by_size_not_by_input_order() -> None:
    shuffled = [SPARK_Q4, SPARK_F16, SPARK_Q8]
    got = recommend_quant(shuffled, vram_bytes=8 * GIB, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=4096, n_seq_max=8)
    assert got["quant"] == SPARK_Q8[0]


def test_recommend_quant_drops_a_quant_when_no_kv_type_fits() -> None:
    # 5 GiB VRAM: Q8_0 needs >= 5.52 GB even with q8_0 KV (> 4.5 GB budget), so the next
    # quant down is considered; Q4_K_M fits with the largest KV type (f16).
    got = recommend_quant(cands(), vram_bytes=5 * GIB, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=2048, n_seq_max=4)
    assert got["placement"] == "gpu"
    assert got["quant"] == SPARK_Q4[0]
    assert got["kv_type"] == "f16"
    assert got["total"] == 2_600_224_352 + 147_456 * 2048 * 4 + OVERHEAD_BYTES
    assert got["total"] <= int(5 * GIB * 0.90)


def test_recommend_quant_keeps_q8_0_weights_and_downgrades_kv() -> None:
    # 6 GiB VRAM: Q8_0 + f16 KV (6.12 GB) does not fit, Q8_0 + q8_0 KV (5.52 GB) does.
    got = recommend_quant(cands(), vram_bytes=6 * GIB, ram_bytes=31 * GIB,
                          kv_per_token_f16=SPARK_KVPT_F16, n_ctx=2048, n_seq_max=4)
    assert got["quant"] == SPARK_Q8[0]
    assert got["kv_type"] == "q8_0"
    assert got["placement"] == "gpu"


# ------------------------------------------------------------ host budget
def test_host_budget_reads_meminfo(tmp_path: pathlib.Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32761996 kB\nMemFree:  100 kB\n")
    budget = host_budget(meminfo_path=meminfo, nvidia_smi=lambda: None)
    assert budget.ram_bytes == 32_761_996 * 1024
    assert budget.vram_bytes == 0  # probe reported no GPU


def test_host_budget_takes_vram_from_query(tmp_path: pathlib.Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1000 kB\n")
    budget = host_budget(meminfo_path=meminfo, nvidia_smi=lambda: 8 * GIB)
    assert budget.vram_bytes == 8 * GIB


# ------------------------------------------------------- quant selection
def hf_files(*paths: str) -> list[dict[str, object]]:
    sizes = {SPARK_Q8[0]: SPARK_Q8[1], SPARK_Q4[0]: SPARK_Q4[1], SPARK_F16[0]: SPARK_F16[1]}
    return [{"path": p, "size": sizes.get(p, 100), "oid": None if p not in sizes else "ab" * 32}
            for p in paths]


def test_select_file_by_quant_token() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0], SPARK_F16[0], "README.md")
    choice = select_file(files, quant="Q8_0")
    assert isinstance(choice, FileChoice)
    assert choice.file["path"] == SPARK_Q8[0]
    assert choice.quant == "Q8_0"


def test_select_file_quant_match_is_case_insensitive() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0])
    assert select_file(files, quant="q4_k_m").file["path"] == SPARK_Q4[0]


def test_select_file_f16_matches_bare_gguf_without_quant_token() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0], SPARK_F16[0])
    assert select_file(files, quant="F16").file["path"] == SPARK_F16[0]


def test_select_file_ambiguous_quant_lists_candidates() -> None:
    files = hf_files("Spark-X2.5-4B-Q8_0.gguf", "Spark-X2.5-4B-Q8_0-imatrix.gguf")
    with pytest.raises(GgufoneError) as exc:
        select_file(files, quant="Q8_0")
    assert exc.value.code == "E_AMBIGUOUS_QUANT"
    msg = str(exc.value)
    assert "Spark-X2.5-4B-Q8_0.gguf" in msg and "Spark-X2.5-4B-Q8_0-imatrix.gguf" in msg


def test_select_file_unknown_quant_is_actionable() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0])
    with pytest.raises(GgufoneError) as exc:
        select_file(files, quant="Q3_K_XL")
    assert exc.value.code == "E_MODEL_NOT_FOUND"
    assert "Q3_K_XL" in str(exc.value)
    assert "Q8_0" in str(exc.value)  # lists what IS available


def test_select_file_bare_repo_uses_recommend_quant() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0], SPARK_F16[0])
    choice = select_file(files, quant=None, vram_bytes=8 * GIB, ram_bytes=31 * GIB,
                         n_ctx=4096, n_seq_max=8)
    assert choice.file["path"] == SPARK_Q8[0]
    assert choice.reason.startswith("recommend")


def test_select_file_explicit_file_wins() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0])
    choice = select_file(files, explicit_file=SPARK_Q4[0])
    assert choice.file["path"] == SPARK_Q4[0]
    assert choice.reason == "explicit"


def test_select_file_explicit_file_missing_is_an_error() -> None:
    with pytest.raises(GgufoneError) as exc:
        select_file(hf_files(SPARK_Q8[0]), explicit_file="nope.gguf")
    assert exc.value.code == "E_MODEL_NOT_FOUND"


def test_select_file_ignores_non_gguf_entries() -> None:
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0]) + [{"path": "images/x.png", "size": 12}]
    assert select_file(files, quant="Q8_0").file["path"] == SPARK_Q8[0]


def test_select_file_default_repo_uses_pinned_alternates() -> None:
    """The pinned F16 file is named Spark-X2.5-4B.gguf (no quant token): the lock
    records the alternates, so the selector must not need to guess."""
    lock = json.loads((ROOT / "runtime.lock").read_text())
    files = hf_files(SPARK_Q8[0], SPARK_Q4[0], SPARK_F16[0])
    choice = select_file(files, quant="F16", repo=lock["default_model"]["repo"], lock=lock)
    assert choice.file["path"] == lock["default_model"]["alternates"]["F16"]["file"]
