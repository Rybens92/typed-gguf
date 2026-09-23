"""E1c FIX (card t_8cb0a05e): plan against FREE device memory, bound by `--fit-target`.

Offline half of the card's requirements 1 and 2. The operator's box reported 8192 MiB *total* but
only 1112 MiB *free* at run time (the desktop held ~6.8 GB); the plan was still built from the
nominal 8 GiB and asked the device for a 1.06 GB allocation that could not exist. Every test here
is hermetic: a synthetic GGUF, fake host facts and — where the driver is read at all — a fake
`nvidia-smi` on a REPLACED `PATH` (see tests/test_host_purity.py for why replacing beats
prepending).
"""
from __future__ import annotations

import json
import pathlib
import stat

import pytest

from tests.test_fit import GIB, MIB, tiny_model, write_gguf
from typed_gguf import cli
from typed_gguf.registry import recommend
from typed_gguf.runtime import fit

# --------------------------------------------------------------- a fake driver on PATH
NVIDIA_TOTAL_MIB = 8192
NVIDIA_FREE_MIB = 1112


def fake_driver(tmp_path: pathlib.Path, *, total_mib: int = NVIDIA_TOTAL_MIB,
                free_mib: int = NVIDIA_FREE_MIB) -> pathlib.Path:
    """A `nvidia-smi` shim that answers the real query shape (`--query-gpu=...`, csv, nounits).

    The operator's busy-desktop numbers: 8192 MiB total, 1112 MiB free. `PATH` is *replaced* by
    the caller (`monkeypatch.setenv`), so a real driver on the box cannot answer instead.
    """
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    tool = shim / "nvidia-smi"
    tool.write_text(
        "#!/bin/sh\n"
        "for arg in \"$@\"; do\n"
        "  case \"$arg\" in\n"
        f"    *memory.total,memory.free*) echo '{total_mib}, {free_mib}'; exit 0;;\n"
        f"    *memory.total*) printf '{total_mib}\\n'; exit 0;;\n"
        f"    *memory.free*) printf '{free_mib}\\n'; exit 0;;\n"
        "  esac\n"
        "done\n"
        f"printf '{total_mib}, {free_mib}\\n'\n",
        encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim


# ------------------------------------------------------------ R1: the driver is asked for FREE
def test_the_driver_query_carries_both_total_and_free_memory(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    shim = fake_driver(tmp_path)
    monkeypatch.setenv("PATH", str(shim))

    memory = recommend.device_memory()

    assert memory.total_bytes == NVIDIA_TOTAL_MIB * MIB
    assert memory.free_bytes == NVIDIA_FREE_MIB * MIB
    assert memory.source == "nvidia-smi"


def test_the_nvidia_query_is_one_call_asking_for_both_numbers(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """One driver round trip: `--query-gpu=memory.total,memory.free`, csv, no units."""
    shim = tmp_path / "bin"
    shim.mkdir()
    seen = tmp_path / "argv.txt"
    tool = shim / "nvidia-smi"
    tool.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARGV_FILE\"\nprintf '8192, 1112\\n'\n",
                    encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim))
    monkeypatch.setenv("ARGV_FILE", str(seen))

    memory = recommend.device_memory()

    assert memory.free_bytes == 1112 * MIB
    argv = seen.read_text(encoding="utf-8").splitlines()
    assert any("memory.total,memory.free" in part for part in argv), argv
    assert "--format=csv,noheader,nounits" in argv, argv


def test_an_injected_probe_never_reaches_the_real_driver(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The seam owns the answer: an injected (total, free) pair is not a hint to ask the box."""
    def tripwire() -> object:
        raise AssertionError("host access leaked: recommend._query_nvidia_smi_memory()")

    monkeypatch.setattr(recommend, "_query_nvidia_smi_memory", tripwire)

    memory = recommend.device_memory(probe=lambda: (8 * GIB, 1112 * MIB))

    assert (memory.total_bytes, memory.free_bytes, memory.source) == (8 * GIB, 1112 * MIB,
                                                                     "injected")


def test_an_empty_injected_probe_stays_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def tripwire() -> object:
        raise AssertionError("host access leaked: recommend._query_nvidia_smi_memory()")

    monkeypatch.setattr(recommend, "_query_nvidia_smi_memory", tripwire)

    memory = recommend.device_memory(probe=lambda: None)

    assert (memory.total_bytes, memory.free_bytes) == (0, 0)


def test_amdgpu_sysfs_is_the_fallback_when_there_is_no_nvidia_smi(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A Vulkan-only box still answers: `mem_info_vram_total`/`_used` are the generic DRM facts."""
    monkeypatch.setenv("PATH", str(tmp_path / "no-drivers"))
    drm = tmp_path / "drm"
    device = drm / "card0" / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text(str(8 * GIB) + "\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text(str(6 * GIB) + "\n", encoding="utf-8")

    memory = recommend.device_memory(drm_root=drm)

    assert memory.total_bytes == 8 * GIB
    assert memory.free_bytes == 2 * GIB
    assert memory.source == "amdgpu-sysfs"


def test_an_unknown_device_is_zero_bytes_not_an_error(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "no-drivers"))
    memory = recommend.device_memory(drm_root=tmp_path / "no-drm")
    assert (memory.total_bytes, memory.free_bytes, memory.source) == (0, 0, "unknown")


# ------------------------------------------------- R1: the fit budget comes from the free number
def test_host_facts_plan_against_free_memory_not_the_nominal_size(
        tmp_path: pathlib.Path) -> None:
    """8192 MiB total / 1112 MiB free -> the budget is 1112 MiB, not 8 GiB."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32761996 kB\n", encoding="utf-8")

    host = fit.host_facts(meminfo_path=meminfo, backend="vulkan", n_cpu=8,
                          device_probe=lambda: recommend.DeviceMemory(
                              total_bytes=8 * GIB, free_bytes=1112 * MIB,
                              source="injected"))

    assert host.vram_bytes == 8 * GIB          # the identity fact stays nominal
    assert host.vram_free_bytes == 1112 * MIB
    assert host.budget_bytes == 1112 * MIB      # what a plan may spend


def test_the_free_reading_does_not_change_the_host_fingerprint(
        tmp_path: pathlib.Path) -> None:
    """Free memory moves every minute; the cache key must not churn with it."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32761996 kB\n", encoding="utf-8")
    roomy = fit.host_facts(meminfo_path=meminfo, backend="vulkan", n_cpu=8,
                           device_probe=lambda: recommend.DeviceMemory(8 * GIB, 7 * GIB,
                                                                       "injected"))
    busy = fit.host_facts(meminfo_path=meminfo, backend="vulkan", n_cpu=8,
                          device_probe=lambda: recommend.DeviceMemory(8 * GIB, 1112 * MIB,
                                                                      "injected"))
    assert roomy.fingerprint == busy.fingerprint
    assert roomy.budget_bytes != busy.budget_bytes


def test_an_injected_total_only_probe_still_works(tmp_path: pathlib.Path) -> None:
    """`vram_probe=` (E1a API) injects a total and nothing else: the budget stays conservative.

    Updated by card t_287e0d18 (requirement d): a total with **no** free reading used to be read as
    "the whole device is spendable". It is not — the device is shared with everything else, and the
    repro that card fixes planned 5482 MiB against a nominal 8 GiB while the desktop held ~1.5 GB
    of it. The nominal size stays the *identity* fact (`vram_bytes`, which the fingerprint keys on);
    the budget is the conservative share of it.
    """
    host = fit.host_facts(meminfo_path=tmp_path / "meminfo", vram_probe=lambda: 6 * GIB,
                          backend="vulkan", n_cpu=2)
    assert host.vram_bytes == 6 * GIB and host.vram_free_bytes == 0
    assert host.free_is_known is False
    assert host.budget_bytes == int(6 * GIB * (1.0 - fit.UNKNOWN_FREE_RESERVE))
    assert host.budget_bytes < 6 * GIB


def test_the_target_is_subtracted_from_free_memory(tmp_path: pathlib.Path) -> None:
    host = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=1112 * MIB, n_cpu=8, fingerprint="vulkan:busy")
    assert fit.fit_budget(host) == max(0, 1112 * MIB - fit.DEFAULT_FIT_TARGET_MB * MIB)
    assert fit.fit_budget(host, fit_target_mb=5200) == 0
    assert fit.fit_budget(host, fit_target_mb=100) == (1112 - 100) * MIB


def test_an_estimate_on_a_busy_desktop_never_promises_a_plan_it_cannot_hold() -> None:
    """The operator's numbers, through the estimate path: no 36-layer / 5.3 GB plan."""
    host = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=1112 * MIB, n_cpu=8, fingerprint="vulkan:busy")
    model = tiny_model()
    plan = fit.estimate_plan(model, host, n_ctx=4096, n_seq_max=8)
    assert plan.n_gpu_layers == 0                       # nothing can be offloaded into 88 MiB
    assert fit.plan_device_bytes(plan, model) <= plan.budget_bytes
    assert plan.insufficient is True or plan.notes


# ----------------------------------------------- R2: `--fit-target` bounds the binary's plan too
def test_the_binary_plan_is_bounded_by_the_fit_target() -> None:
    """`--fit-target 5200` on the 8 GiB box leaves at most 2992 MiB — full offload is impossible."""
    model = tiny_model()
    host = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=8 * GIB, n_cpu=8, fingerprint="vulkan:roomy")
    plan = fit.plan_from_binary(model, host, table="Vulkan0 4096 512 128\n", n_ctx=4096,
                                n_seq_max=8, runtime_dir="/rt/llama-b11026-vulkan",
                                fit_target_mb=5200)
    assert plan.budget_bytes == 8 * GIB - 5200 * MIB
    assert plan.n_gpu_layers < model.n_layer
    # (2992 MiB budget − 162 MiB q4_0 KV − 128 MiB compute) // (4096 MiB / 36 layers) = 23
    assert plan.n_gpu_layers == 23
    assert fit.plan_device_bytes(plan, model) <= plan.budget_bytes
    assert "W_FIT_DOWNGRADE" in plan.warnings


def test_the_binary_plan_still_offloads_everything_when_the_target_allows_it() -> None:
    model = tiny_model()
    host = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=8 * GIB, n_cpu=8, fingerprint="vulkan:roomy")
    plan = fit.plan_from_binary(model, host, table="Vulkan0 4032 512 128\n", n_ctx=4096,
                                n_seq_max=8, runtime_dir="/rt/llama-b11026-vulkan",
                                fit_target_mb=1024)
    assert plan.n_gpu_layers == model.n_layer
    assert plan.budget_bytes == 8 * GIB - fit.DEFAULT_FIT_TARGET_MB * MIB
    assert "W_FIT_DOWNGRADE" not in plan.warnings


def test_the_binary_argv_asks_for_the_layer_count_the_budget_can_hold() -> None:
    """The tool is asked about the placement we intend, not about full offload."""
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> str:
        seen.append(list(argv))
        return "Host 4096 512 128\n"

    host = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=1112 * MIB, n_cpu=8, fingerprint="vulkan:busy")
    fit.run_llama_fit_params(tiny_model(), host, runtime_dir=None, n_ctx=4096, n_seq_max=8,
                             fit_target_mb=1024, runner=runner)
    argv = seen[0]
    layers = argv[argv.index("-ngl") + 1]
    assert layers == "0"                     # 88 MiB of budget cannot hold a single 4 GiB/36 layer


# --------------------------------------------- R1: a stale cached plan is re-planned, not trusted
def test_a_cached_plan_is_replanned_when_free_memory_dropped(tmp_path: pathlib.Path) -> None:
    """The operator's sequence: cache written on a roomy box, then the desktop takes ~6.8 GB."""
    home = tmp_path / "home"
    model = tiny_model()
    roomy = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                          vram_free_bytes=8 * GIB, n_cpu=8, fingerprint="vulkan:identical")
    busy = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=1112 * MIB, n_cpu=8, fingerprint="vulkan:identical")
    first = fit.plan_for_model(model, roomy, home=home)
    assert first.n_gpu_layers == model.n_layer
    assert fit.load_cached(model.sha256, roomy.fingerprint, home) is not None

    second = fit.plan_for_model(model, busy, home=home)

    assert second.n_gpu_layers == 0                    # re-planned for the free reading
    assert fit.plan_device_bytes(second, model) == 0   # a CPU plan asks the device for nothing
    assert second.budget_bytes == max(0, 1112 * MIB - fit.DEFAULT_FIT_TARGET_MB * MIB)
    assert "W_FIT_DOWNGRADE" in second.warnings
    assert any("free" in note for note in second.notes), second.notes
    stored = json.loads(fit.cache_path(model.sha256, busy.fingerprint, home).read_text())
    assert stored["n_gpu_layers"] == 0                 # the cache now holds the honest plan
    assert stored["warnings"] == list(second.warnings)


def test_a_fitting_cached_plan_is_returned_untouched(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    model = tiny_model()
    host = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                         vram_free_bytes=8 * GIB, n_cpu=8, fingerprint="vulkan:roomy")
    first = fit.plan_for_model(model, host, home=home)
    second = fit.plan_for_model(model, host, home=home)
    assert second.to_dict() == first.to_dict()
    assert "W_FIT_DOWNGRADE" not in second.warnings


def test_the_cli_fit_command_reports_the_free_number_it_planned_against(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    model = write_gguf(tmp_path / "synthetic.gguf")
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TYPED_GGUF_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(fit, "host_facts", lambda **kwargs: fit.HostFacts(
        backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
        vram_free_bytes=NVIDIA_FREE_MIB * MIB, n_cpu=8, fingerprint="vulkan:busy"))
    code = cli.main(["fit", str(model), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host"]["vram_free_bytes"] == NVIDIA_FREE_MIB * MIB
    assert payload["host"]["vram_bytes"] == 8 * GIB
    assert payload["budget_bytes"] == max(0, NVIDIA_FREE_MIB * MIB
                                          - fit.DEFAULT_FIT_TARGET_MB * MIB)
