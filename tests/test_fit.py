"""A-E1c-4/5/6: `ggufone fit` — the plan, its source, its cache and the downgrade order.

Offline half: synthetic GGUFs (header + tensor index, no tensor data), fake hosts and a fake
`llama-fit-params` executable, so every branch is pinned without a runtime or a GPU. The live
numbers (the binary on this box, RSS cross-check) are in `tests/test_fit_live.py` and
`docs/evidence/e1c_*.md`.
"""
from __future__ import annotations

import json
import os
import pathlib
import stat
import struct

import pytest

from ggufone import cli
from ggufone.errors import WARNING_CODES
from ggufone.runtime import fit

GIB = 1024 ** 3
MIB = 1024 ** 2
T_STRING = 8
GGUF_MAGIC = b"GGUF"


# --------------------------------------------------------------- synthetic GGUF
def gstr(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, type_id: int, payload: bytes) -> bytes:
    return gstr(key) + struct.pack("<I", type_id) + payload


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def write_gguf(path: pathlib.Path, *, arch: str = "spark2_5", n_layer: int = 4,
               n_kv_head: int = 4, key_len: int = 256, value_len: int = 256,
               n_ctx_train: int = 32768, tensors: list[tuple[str, list[int], int]] | None = None,
               data_bytes: int = 4096) -> pathlib.Path:
    """A structurally valid GGUF: header, KV block, tensor index, then tensor data."""
    tensors = tensors or [("token_embd.weight", [64, 32], 8),
                          ("blk.0.attn_norm.weight", [64], 0)]
    kvs = [
        kv("general.architecture", T_STRING, gstr(arch)),
        kv(f"{arch}.block_count", 4, u32(n_layer)),
        kv(f"{arch}.attention.head_count_kv", 4, u32(n_kv_head)),
        kv(f"{arch}.attention.key_length", 4, u32(key_len)),
        kv(f"{arch}.attention.value_length", 4, u32(value_len)),
        kv(f"{arch}.context_length", 4, u32(n_ctx_train)),
    ]
    blob = (GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))
    index = b""
    for name, dims, ttype in tensors:
        index += gstr(name) + u32(len(dims))
        for dim in dims:
            index += struct.pack("<Q", dim)
        index += u32(ttype) + struct.pack("<Q", 0)
    path.write_bytes(blob + index + b"\0" * data_bytes)
    return path


def facts(path: pathlib.Path) -> fit.ModelFacts:
    return fit.ModelFacts.read(path)


def cpu_host(ram_gib: float = 31.0) -> fit.HostFacts:
    return fit.HostFacts(backend="cpu", ram_bytes=int(ram_gib * GIB), vram_bytes=0, n_cpu=24,
                         fingerprint="cpu:test")


# matplotlib-free "TinyPlan" helpers -----------------------------------------
def tiny_model() -> fit.ModelFacts:
    """36 layers × 4 kv heads × 512 → 147456 B/token at f16 (SPEC 2.4's reference number)."""
    return fit.ModelFacts(path="/fake/model.gguf", sha256="a" * 64, arch="spark2_5", n_layer=36,
                          n_kv_head=4, key_len=256, value_len=256, n_ctx_train=32768,
                          weights_bytes=4 * GIB)


# ----------------------------------------------------------- A-E1c-4: the fields
def test_the_plan_carries_every_documented_field() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(), fit_target_mb=1024)
    payload = plan.to_dict()
    for field in fit.FIT_FIELDS:
        assert field in payload, field
    assert payload["kv_type"] in ("f16", "q8_0", "q4_0")
    assert payload["source"] == "estimate"
    assert "W_FIT_ESTIMATED" in payload["warnings"]
    assert payload["warnings"][0] in WARNING_CODES
    assert payload["backend"] == "cpu"
    assert payload["n_gpu_layers"] == 0


def test_the_kv_estimate_is_the_unified_cache_not_the_worst_case() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(), n_ctx=4096, n_seq_max=8)
    per_token = fit.kv_bytes_per_token(36, 4, 256, 256, 2)
    assert per_token == 147456                      # SPEC 2.4, executed
    assert plan.est_kv_bytes == per_token * 4096    # kv_unified: n_ctx cells, not ctx × seq
    assert plan.est_total_bytes == plan.est_weights_bytes + plan.est_kv_bytes + fit.OVERHEAD_BYTES
    assert plan.est_total_bytes < 4 * GIB + per_token * 4096 * 8 + fit.OVERHEAD_BYTES


def test_the_estimate_is_deterministic_and_json_safe() -> None:
    first = fit.estimate_plan(tiny_model(), cpu_host()).to_dict()
    second = fit.estimate_plan(tiny_model(), cpu_host()).to_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_plan_never_exceeds_the_budget_it_was_given() -> None:
    host = cpu_host(ram_gib=6.0)
    plan = fit.estimate_plan(tiny_model(), host, fit_target_mb=1024)
    assert plan.est_total_bytes <= host.ram_bytes - 1024 * MIB or plan.notes


# ------------------------------------------------- A-E1c-5: the downgrade order
def test_an_over_budget_plan_downgrades_kv_type_in_the_documented_order() -> None:
    model, host = tiny_model(), cpu_host()
    roomy = fit.estimate_plan(model, host, n_ctx=4096)
    assert roomy.kv_type == "f16"
    # a budget that is 16 MiB short of the f16 plan
    budget = model.weights_bytes + roomy.est_kv_bytes + fit.OVERHEAD_BYTES - 16 * MIB
    plan = fit.estimate_plan(model, host, budget_bytes=budget, n_ctx=4096)
    assert plan.kv_type == "q8_0"
    assert "W_KV_TYPE_DOWNGRADE" in plan.warnings
    assert plan.est_kv_bytes < roomy.est_kv_bytes


def test_a_tighter_budget_walks_the_ladder_down_to_q4_0() -> None:
    model, host = tiny_model(), cpu_host()
    roomy = fit.estimate_plan(model, host, n_ctx=4096)
    budget = model.weights_bytes + roomy.est_kv_bytes + fit.OVERHEAD_BYTES - 16 * MIB
    q8_0 = fit.estimate_plan(model, host, budget_bytes=budget, n_ctx=4096)
    tighter = model.weights_bytes + q8_0.est_kv_bytes + fit.OVERHEAD_BYTES - 16 * MIB
    plan = fit.estimate_plan(model, host, budget_bytes=tighter, n_ctx=4096)
    assert (roomy.kv_type, q8_0.kv_type, plan.kv_type) == ("f16", "q8_0", "q4_0")
    assert "W_KV_TYPE_DOWNGRADE" in plan.warnings


def test_an_explicit_kv_type_is_never_silently_upgraded() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(), kv_type="q4_0")
    assert plan.kv_type == "q4_0"
    assert plan.warnings == ("W_FIT_ESTIMATED",)          # no downgrade warning: not downgraded


def test_when_even_q4_0_is_over_budget_the_context_shrinks_to_the_minimum() -> None:
    model, host = tiny_model(), cpu_host()
    roomy = fit.estimate_plan(model, host, n_ctx=8192)
    per_token_q4 = fit.kv_bytes_per_token(model.n_layer, model.n_kv_head, model.key_len,
                                          model.value_len, fit.KV_BYTES_PER_ELEMENT["q4_0"])
    budget = model.weights_bytes + fit.OVERHEAD_BYTES + per_token_q4 * 8192 // 2
    plan = fit.estimate_plan(model, host, budget_bytes=budget, n_ctx=8192, min_ctx=1024)
    assert plan.kv_type == "q4_0"
    assert plan.n_ctx < 8192
    assert plan.n_ctx >= 1024
    assert any("n_ctx" in note for note in plan.notes)
    assert roomy.kv_type == "f16"


def test_a_budget_that_cannot_hold_the_weights_is_reported_not_hidden() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host(ram_gib=2.0), budget_bytes=1 * GIB)
    assert any("insufficient" in note for note in plan.notes)
    assert plan.kv_type == "q4_0" and plan.n_ctx >= fit.MIN_CTX_FLOOR


def test_over_budget_downgrade_is_an_e1c5_gate_not_a_silent_resize() -> None:
    """The warning is present exactly when the kv type moved down the documented ladder."""
    small = fit.estimate_plan(tiny_model(), cpu_host())
    tight = fit.estimate_plan(tiny_model(), cpu_host(),
                              budget_bytes=4 * GIB + 300 * MIB + fit.OVERHEAD_BYTES)
    assert "W_KV_TYPE_DOWNGRADE" not in small.warnings
    assert "W_KV_TYPE_DOWNGRADE" in tight.warnings
    assert fit.KV_DOWNGRADE_ORDER == ("f16", "q8_0", "q4_0")


# --------------------------------------------------- A-E1c-4: the binary's path
def test_the_binary_table_is_parsed_into_rows() -> None:
    table = ("Vulkan0 100 50 200\n"
             "Host 400 60 100\n")
    rows = fit.parse_fit_table(table)
    assert [row.name for row in rows] == ["Vulkan0", "Host"]
    assert rows[0].model_bytes == 100 * MIB and rows[0].context_bytes == 50 * MIB
    assert rows[0].compute_bytes == 200 * MIB
    assert rows[1].total_bytes == (400 + 60 + 100) * MIB
    assert fit.parse_fit_table("garbage\n\n") == []


def test_running_the_binary_yields_source_llama_fit_params() -> None:
    plan = fit.plan_from_binary(tiny_model(), cpu_host(), table=(
        "Host 4096 512 256\n"), n_ctx=4096, n_seq_max=8, runtime_dir="/fake/rt")
    assert plan.source == "llama-fit-params"
    assert plan.warnings == ()
    assert plan.est_weights_bytes == 4096 * MIB
    assert plan.est_kv_bytes == 512 * MIB
    assert plan.est_total_bytes == (4096 + 512 + 256) * MIB


def test_the_binary_invocation_uses_the_spec_flags(monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: pathlib.Path) -> None:
    runtime = tmp_path / "rt"
    runtime.mkdir()
    (runtime / "libllama.so").write_bytes(b"")
    (runtime / "libggml.so").write_bytes(b"")
    script = runtime / "llama-fit-params"
    script.write_text("#!/bin/sh\nprintf 'Host 100 20 30\\n'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    seen: dict[str, list[str]] = {}

    def runner(argv: list[str]) -> str:
        seen["argv"] = argv
        return "Host 100 20 30\n"

    plan = fit.run_llama_fit_params(tiny_model(), cpu_host(), runtime_dir=runtime,
                                    n_ctx=4096, n_seq_max=8, fit_target_mb=1024, min_ctx=4096,
                                    runner=runner)
    assert plan is not None and plan.source == "llama-fit-params"
    assert plan.est_weights_bytes == 100 * MIB
    argv = seen["argv"]
    assert argv[0].endswith("llama-fit-params")
    for flag in ("--fit", "on", "--fit-target", "1024", "--fit-ctx", "4096", "--fit-print", "on"):
        assert flag in argv, flag
    assert "-c" in argv and "4096" in argv


def test_a_failing_binary_falls_back_to_the_estimate(tmp_path: pathlib.Path) -> None:
    runtime = tmp_path / "rt"
    runtime.mkdir()
    script = runtime / "llama-fit-params"
    script.write_text("#!/bin/sh\necho boom >&2\nexit 1\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    plan = fit.run_llama_fit_params(tiny_model(), cpu_host(), runtime_dir=runtime,
                                    n_ctx=4096, n_seq_max=8, runner=None, timeout=30)
    assert plan is None                       # the caller falls back to `estimate`


def test_no_binary_at_all_is_not_an_error() -> None:
    assert fit.run_llama_fit_params(tiny_model(), cpu_host(), runtime_dir=None, n_ctx=4096,
                                    n_seq_max=8) is None


# ---------------------------------------------------------------- A-E1c-4: cache
def test_the_plan_is_cached_per_model_sha_and_host_fingerprint(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    model, host = tiny_model(), cpu_host()
    calls: list[str] = []

    def runner(argv: list[str]) -> str:
        calls.append("ran")
        return "Host 4096 512 256\n"

    first = fit.plan_for_model(model, host, home=home, runtime_dir="/fake/rt", runner=runner)
    assert first.source == "llama-fit-params"
    assert calls == ["ran"]
    second = fit.plan_for_model(model, host, home=home, runtime_dir="/fake/rt", runner=runner)
    assert second.to_dict() == first.to_dict()
    assert calls == ["ran"]                   # a cache hit must not run the binary again
    path = fit.cache_path(model.sha256, host.fingerprint, home)
    assert path.exists() and json.loads(path.read_text())["schema"] == fit.FIT_SCHEMA
    assert str(path).endswith(f"{model.sha256}.{host.fingerprint}.json")


def test_the_estimate_path_is_cached_too(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    model, host = tiny_model(), cpu_host()
    first = fit.plan_for_model(model, host, home=home)
    assert first.source == "estimate"
    assert fit.load_cached(model.sha256, host.fingerprint, home).to_dict() == first.to_dict()


def test_a_different_host_fingerprint_is_a_cache_miss(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    model = tiny_model()
    fit.plan_for_model(model, cpu_host(), home=home)
    other = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB, n_cpu=24,
                          fingerprint="vulkan:other")
    assert fit.load_cached(model.sha256, other.fingerprint, home) is None


def test_no_cache_bypasses_the_read_but_still_writes(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    model, host = tiny_model(), cpu_host()
    fit.plan_for_model(model, host, home=home)
    assert fit.load_cached(model.sha256, host.fingerprint, home) is not None
    fresh = fit.plan_for_model(model, host, home=home, use_cache=False)
    assert fresh.source == "estimate"
    assert fit.load_cached(model.sha256, host.fingerprint, home) is not None


# ----------------------------------------------------------- host fingerprint
def test_the_host_fingerprint_is_stable_and_carries_the_budget(tmp_path: pathlib.Path) -> None:
    host = fit.host_facts(meminfo_path=tmp_path / "meminfo", vram_probe=lambda: 8 * GIB)
    assert host.backend in ("cpu", "vulkan", "cuda", "metal")
    again = fit.host_facts(meminfo_path=tmp_path / "meminfo", vram_probe=lambda: 8 * GIB)
    assert host.fingerprint == again.fingerprint
    other = fit.host_facts(meminfo_path=tmp_path / "meminfo", vram_probe=lambda: None)
    assert other.fingerprint != host.fingerprint or other.vram_bytes == host.vram_bytes


def test_a_host_with_vram_offloads_layers() -> None:
    gpu = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=24 * GIB, n_cpu=24,
                        fingerprint="vulkan:big")
    plan = fit.estimate_plan(tiny_model(), gpu, budget_bytes=24 * GIB)
    assert plan.n_gpu_layers > 0
    cpu = fit.estimate_plan(tiny_model(), cpu_host())
    assert cpu.n_gpu_layers == 0


# -------------------------------------------------------------- A-E1c-6: RSS
def test_the_rss_ratio_is_the_documented_check() -> None:
    assert fit.rss_ratio(1_000, 1_100) == pytest.approx(0.10)
    assert fit.rss_ratio(1_000, 900) == pytest.approx(-0.10)
    assert fit.within_tolerance(1_000, 1_150, tolerance=0.20) is True
    assert fit.within_tolerance(1_000, 1_250, tolerance=0.20) is False
    assert fit.rss_ratio(0, 1_000) == float("inf")


def test_measured_rss_reads_this_process(tmp_path: pathlib.Path) -> None:
    measured = fit.measured_rss_bytes()
    assert measured is None or measured > 0


# --------------------------------------------------------- applied on load / CLI
def test_session_overrides_let_the_request_win() -> None:
    plan = fit.estimate_plan(tiny_model(), cpu_host())
    overrides = fit.session_overrides(plan, n_ctx=2048, kv_type="f16", n_seq_max=4)
    assert overrides["n_ctx"] == 2048 and overrides["kv_type"] == "f16"
    assert overrides["n_seq_max"] == 4 and overrides["n_gpu_layers"] == plan.n_gpu_layers
    default = fit.session_overrides(plan, n_ctx=None, kv_type="auto", n_seq_max=None)
    assert default["n_ctx"] == plan.n_ctx and default["kv_type"] == plan.kv_type


def test_the_cli_fit_command_returns_the_documented_json(tmp_path: pathlib.Path,
                                                         capsys: pytest.CaptureFixture[str],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    model = write_gguf(tmp_path / "synthetic.gguf")
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    code = cli.main(["fit", str(model), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    for field in fit.FIT_FIELDS:
        assert field in payload, field
    assert payload["source"] == "estimate"
    assert payload["warnings"] == ["W_FIT_ESTIMATED"]
    assert payload["arch"] == "spark2_5"


def test_the_cli_fit_command_runs_a_binary_when_one_exists(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    model = write_gguf(tmp_path / "synthetic.gguf")
    runtime = tmp_path / "rt"
    runtime.mkdir()
    (runtime / "libllama.so").write_bytes(b"")
    script = runtime / "llama-fit-params"
    script.write_text("#!/bin/sh\nprintf 'Host 96 12 24\\n'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GGUFONE_RUNTIME_DIR", str(runtime))
    code = cli.main(["fit", str(model), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "llama-fit-params"
    assert payload["est_weights_bytes"] == 96 * MIB


def test_fit_never_touches_the_network(monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: pathlib.Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    import socket

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network call attempted by `fit`")

    model = write_gguf(tmp_path / "synthetic.gguf")
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert cli.main(["fit", str(model), "--json"]) == 0
    capsys.readouterr()


def test_fit_reports_a_missing_model_with_the_pinned_code(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    code = cli.main(["fit", str(tmp_path / "nope.gguf")])
    assert code == 2
    assert "E_MODEL_NOT_FOUND" in capsys.readouterr().err


def test_the_fit_cache_lives_under_the_data_home(tmp_path: pathlib.Path) -> None:
    path = fit.cache_path("b" * 64, "fp", tmp_path / "home")
    assert path.parent == tmp_path / "home" / "fit"


def test_an_unknown_ggml_tensor_type_is_reported_not_guessed(tmp_path: pathlib.Path) -> None:
    model = write_gguf(tmp_path / "weird.gguf",
                       tensors=[("blk.0.foo", [64], 250)])
    with pytest.raises(fit.UnknownTensorType) as excinfo:
        fit.ModelFacts.read(model)
    assert "250" in str(excinfo.value) or "unknown" in str(excinfo.value).lower()


def test_model_facts_read_the_real_metadata_of_a_gguf(tmp_path: pathlib.Path) -> None:
    path = write_gguf(tmp_path / "facts.gguf", arch="qwen35", n_layer=24, n_kv_head=2,
                      key_len=256, value_len=256, n_ctx_train=262144,
                      tensors=[("token_embd.weight", [64, 32], 8)])
    model = fit.ModelFacts.read(path)
    assert model.arch == "qwen35"
    assert model.n_layer == 24 and model.n_kv_head == 2
    assert model.weights_bytes == 64 * 32 // 32 * 34        # one q8_0 tensor
    assert model.sha256 and len(model.sha256) == 64
    assert model.n_ctx_train == 262144


def test_estimate_uses_the_real_file_when_given_a_path(tmp_path: pathlib.Path) -> None:
    path = write_gguf(tmp_path / "one.gguf", tensors=[("token_embd.weight", [32, 32], 8)])
    plan = fit.plan_for_path(path, host=cpu_host(), home=tmp_path / "home")
    assert plan.est_weights_bytes == 32 * 32 // 32 * 34
    assert plan.source == "estimate"
