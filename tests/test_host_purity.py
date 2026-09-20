"""The mapping is a pure function of the probes it is handed (E1a FIX, card t_1b4632de).

The bug this file pins down is invisible in a GPU-less sandbox and obvious on the operator's
RTX 3060 Ti: detection consulted the REAL machine even when explicit facts were injected.
There, `detect_backend(system="linux", dri_nodes=["/dev/dri/renderD128"])` answered ``cuda``
(``shutil.which("nvidia-smi")`` found the host binary) and
`host_variant("auto", system="linux", machine="x86_64")` planned the CUDA bundle for a caller
who had described a CPU box — 7 tests failed on the host and passed in the sandbox.

`simulate_host()` therefore builds the *real-host* worlds at the OS level — a real `nvidia-smi`
shim on `PATH`, a real DRM render node, a real Vulkan ICD directory — so the production readers
(``shutil.which``, ``platform.*``, the ``/dev/dri`` glob, the ICD stat) see a GPU box on a
GPU-less machine. `PATH` is *replaced*, never extended: extending it would still find the
operator's real ``nvidia-smi`` in the GPU-absent world, which is the very leak this file exists
to catch.

The B1–B3 pins (card t_83ee1eed, closing the survivors of the adversarial duel t_0fc576df) extend
the same rule to the surfaces the duel reached: `capability.backends(system=…)` (m11 — the
injected platform must answer the glob, not `finder.library_glob()`'s host default), an unnamed
`machine` (m08 — a caller error, never `platform.machine()`), omitted `dri_nodes` next to a
supplied ICD (m09 — absent, never the real `/dev/dri` listing), and
`registry.recommend.host_budget`'s vram seam (m10 — it lives outside the detection modules, so it
is pinned here as well as in `test_recommend_quant.py`).
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from typed_gguf.errors import RuntimeMissingError
from typed_gguf.registry import recommend
from typed_gguf.runtime import capability, install, pins

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "runtime.lock"

# Environment a host-fact reader could read INSTEAD of a probe. Detection must not consult any of
# them (the installer's `TYPED_GGUF_OFFLINE_CACHE` is a cache knob, not a host fact).
DEVICE_ENV = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
              "GPU_DEVICE_ORDINAL", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_LOADER_LAYERS_ENABLE", "DISPLAY",
              "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "TYPED_GGUF_BACKEND", "TYPED_GGUF_DEVICE",
              "TYPED_GGUF_ACCELERATOR", "TYPED_GGUF_GPU")
# Names that could carry a host fact, by prefix: a probe default smuggled through the environment
# has to be read by one of these.
HOST_FACT_PREFIXES = ("TYPED_GGUF_", "CUDA", "NVIDIA", "VK_", "DRI", "HIP_", "ROCR_", "DISPLAY",
                      "WAYLAND", "XDG_SESSION_TYPE", "LIBGL", "MESA", "GBM_", "VULKAN", "NEO_",
                      "INTEL_VK", "RADV", "AMD_VULKAN", "WSL", "WSLENV")


def host_fact_reads(reads: list[str]) -> list[str]:
    """Keys of an environment-read log that could name a host fact, deduplicated and sorted."""
    return sorted({key for key in reads
                   if key in DEVICE_ENV or key.startswith(HOST_FACT_PREFIXES)})


def simulate_host(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, *, world: str) -> None:
    """Install a synthetic machine for the real-host readers (`current_host()`).

    world=cpu:    no nvidia-smi, no DRM render node, no Vulkan ICD
    world=vulkan: no nvidia-smi, but a DRM render node *and* a Vulkan ICD (a Vulkan-only box)
    world=cuda:   nvidia-smi on PATH (the operator's RTX box), plus the DRM/ICD facts

    The three worlds differ ONLY in what the production readers can see, so an assertion that
    holds in all three is an assertion about the code, not about the box it runs on.
    """
    assert world in ("cpu", "vulkan", "cuda")
    shim = tmp_path / f"bin-{world}"
    shim.mkdir()
    if world == "cuda":
        nvidia_smi = shim / "nvidia-smi"
        nvidia_smi.write_text("#!/bin/sh\necho 'NVIDIA GeForce RTX 3060 Ti, 8192 MiB'\n")
        nvidia_smi.chmod(0o755)
    # Replaced, not extended: on the operator's GPU host an extended PATH would still answer
    # `shutil.which("nvidia-smi")` in the GPU-absent world.
    monkeypatch.setenv("PATH", str(shim))
    monkeypatch.setattr(pins.platform, "system", lambda: "linux")
    monkeypatch.setattr(pins.platform, "machine", lambda: "x86_64")
    dri = tmp_path / f"dev-dri-{world}"
    dri.mkdir()
    icd = tmp_path / f"vulkan-icd-{world}"
    icd.mkdir()
    if world in ("vulkan", "cuda"):
        (dri / "renderD128").write_bytes(b"")
        (icd / "nvidia_icd.json").write_text("{}")
    # pre-fix `pins` has no such constants (they are part of the fix); `raising=False` keeps this
    # helper usable on both trees for the RED demonstration.
    monkeypatch.setattr(pins, "DRI_DIR", dri, raising=False)
    monkeypatch.setattr(pins, "ICD_DIR", icd if world != "cpu" else tmp_path / "no-icd.d",
                        raising=False)


def injected_facts(tmp_path: pathlib.Path) -> dict[str, object]:
    """A caller describing a Vulkan-capable Linux x86_64 box, without naming nvidia-smi."""
    icd = tmp_path / "caller-icd.d"
    icd.mkdir(exist_ok=True)
    return {"system": "linux", "dri_nodes": ["/dev/dri/renderD128"], "icd_dir": str(icd)}


# ------------------------------------------------------------------ the real-host worlds
@pytest.mark.parametrize(("world", "backend"), [("cpu", "cpu"), ("vulkan", "vulkan"),
                                                ("cuda", "cuda")])
def test_each_simulated_world_really_looks_like_that_box(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: pathlib.Path, world: str,
                                                         backend: str) -> None:
    """Non-vacuous worlds: `current_host()` (the production reader) must see each machine."""
    simulate_host(monkeypatch, tmp_path, world=world)

    host = pins.current_host()

    assert host.system == "linux" and host.machine == "x86_64"
    assert host.has_nvidia_smi is (world == "cuda")
    assert bool(host.dri_nodes) is (world != "cpu")
    assert host.detect_backend() == backend
    assert pins.detect_backend() == backend          # the no-argument production path


# ------------------------------------------------------------------ purity of the mapping
@pytest.mark.parametrize("world", ["cpu", "vulkan", "cuda"])
def test_the_mapping_matrix_does_not_move_with_the_real_host(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, world: str) -> None:
    """Facts in, answers out: detection -> variant -> pinned asset -> plan, in every world.

    The cuda world is the operator's RTX box: pre-fix, the facts below planned
    ``linux-x64-cuda-12.8`` there (and the CPU bundle elsewhere), which is the bug.
    """
    simulate_host(monkeypatch, tmp_path, world=world)

    detected = pins.detect_backend(**injected_facts(tmp_path))  # type: ignore[arg-type]
    variant = pins.host_variant("auto", system="linux", machine="x86_64")
    plan = install.plan_install("auto", home=tmp_path / "home", lock=pins.load_lock(LOCK),
                                system="linux", machine="x86_64")

    assert detected == "vulkan"
    assert (variant, plan.variant) == ("linux-x64-cpu", "linux-x64-cpu")
    assert plan.asset == "llama-b11026-bin-ubuntu-x64.tar.gz"
    assert plan.backend == "cpu"
    assert plan.host["has_nvidia_smi"] is False      # decided from the facts, not the box
    assert plan.host["backend"] == "cpu"


def test_the_gpu_absent_world_never_yields_cuda(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: pathlib.Path) -> None:
    """The card's clause: a GPU-absent box must not answer ``cuda`` — with or without probes."""
    simulate_host(monkeypatch, tmp_path, world="cpu")

    assert pins.detect_backend() == "cpu"
    assert pins.host_variant("auto") == "linux-x64-cpu"
    assert install.plan_install("auto", home=tmp_path / "home",
                                lock=pins.load_lock(LOCK)).variant == "linux-x64-cpu"
    assert pins.detect_backend(system="linux", machine="x86_64") == "cpu"
    assert pins.detect_backend(probes=pins.fake_host(system="linux", machine="x86_64")) == "cpu"


def test_the_vulkan_world_with_supplied_probes_answers_vulkan(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The card's first claim, on a box that has a DRM node: supplied probes win."""
    simulate_host(monkeypatch, tmp_path, world="cuda")   # the GPU box is the adversarial case

    assert pins.detect_backend(**injected_facts(tmp_path)) == "vulkan"  # type: ignore[arg-type]
    assert pins.host_variant("vulkan", system="linux", machine="x86_64") == "linux-x64-vulkan"
    assert pins.host_variant("auto", system="linux", machine="x86_64") == "linux-x64-cpu"


# ------------------------------------------------------------------ the environment
def test_an_environment_naming_a_gpu_does_not_move_the_mapping(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """`CUDA_VISIBLE_DEVICES=0` and friends are not probes: the facts stay authoritative."""
    simulate_host(monkeypatch, tmp_path, world="cuda")
    for key in DEVICE_ENV:
        monkeypatch.setenv(key, "0")

    assert pins.detect_backend(system="linux", has_nvidia_smi=False, dri_nodes=[]) == "cpu"
    assert pins.detect_backend(system="linux", machine="x86_64") == "cpu"
    assert pins.host_variant("auto", system="linux", machine="x86_64") == "linux-x64-cpu"
    assert pins.host_variant("auto", probes=pins.fake_host(machine="x86_64")) == "linux-x64-cpu"


class EnvSpy(dict):  # type: ignore[type-arg]
    """`os.environ`, recording every key the code under test looks up."""

    def __init__(self, real: os._Environ[str]) -> None:
        super().__init__(real)
        self.reads: list[str] = []

    def get(self, key: str, default: object = None) -> object:
        self.reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> str:
        self.reads.append(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.reads.append(str(key))
        return super().__contains__(key)


def test_the_injected_path_reads_no_host_fact_from_the_environment(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Detection supply-side purity: no env fallback for a fact that was not supplied.

    The assertion is "no key that could name a host fact", not "no read at all": the test
    harness reads the environment itself (a mutation run adds e.g. ``MUTANT_UNDER_TEST`` and
    re-reads the mapping), which says nothing about the code under test.
    """
    simulate_host(monkeypatch, tmp_path, world="cuda")
    spy = EnvSpy(os.environ)
    monkeypatch.setattr(os, "environ", spy)

    assert pins.detect_backend(**injected_facts(tmp_path)) == "vulkan"  # type: ignore[arg-type]
    assert pins.host_variant("auto", system="linux", machine="x86_64") == "linux-x64-cpu"
    assert pins.host_variant("auto", probes=pins.fake_host(machine="x86_64")) == "linux-x64-cpu"

    assert host_fact_reads(spy.reads) == [], f"host-fact env read: {spy.reads}"


def test_the_injected_path_reads_no_device_environment_and_runs_no_nvidia_smi(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The installer path may read its own cache knob — never a device one, never a subprocess."""
    simulate_host(monkeypatch, tmp_path, world="cuda")
    spy = EnvSpy(os.environ)
    monkeypatch.setattr(os, "environ", spy)
    spawned: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        spawned.append(repr(args))
        raise AssertionError("the injected path spawned a subprocess (nvidia-smi probe?)")

    for name in ("run", "Popen", "check_output", "check_call", "call"):
        monkeypatch.setattr(subprocess, name, forbidden)

    plan = install.plan_install("auto", home=tmp_path / "home", lock=pins.load_lock(LOCK),
                                system="linux", machine="x86_64")

    assert plan.variant == "linux-x64-cpu"
    assert spawned == []
    assert not (set(spy.reads) & set(DEVICE_ENV)), f"device env read: {set(spy.reads)}"


# ------------------------------------------- the pins of the duel's survivors (card t_83ee1eed)
def distractor_bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    """A bundle carrying one backend per platform: the Linux CPU lib and the Windows Vulkan dll.

    A single-platform bundle cannot tell the caller's platform from this host's, so every glob
    (``libggml-*.so``, ``*ggml-*.dll``) finds the same file. The distractor makes the two answers
    differ, which is what turns the injected `system` into an observable fact.
    """
    runtime = tmp_path / "runtime-cross-platform"
    runtime.mkdir()
    (runtime / "libggml-cpu.so").write_bytes(b"")
    (runtime / "ggml-vulkan.dll").write_bytes(b"")
    return runtime


def test_backends_answers_the_system_the_caller_named(tmp_path: pathlib.Path) -> None:
    """B1 (duel t_0fc576df, m11): `capability.backends(system=…)` is a pure function of `system`.

    m11 dropped the argument (`finder.library_glob()` — `platform.system()`), so a caller stating
    "windows" got this host's Linux-shaped glob: `['vulkan']` became `['cpu']`. Both directions
    are asserted because either one alone is host-dependent: only the pair differs from the host
    platform in *every* world.
    """
    runtime = distractor_bundle(tmp_path)

    assert capability.backends(runtime, system="linux") == ["cpu"]
    assert capability.backends(runtime, system="windows") == ["vulkan"]


def test_backends_never_asks_this_host_for_a_system_the_caller_supplied(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """B1 as a tripwire: with `system=` given, `platform.system()` must not be read at all.

    `pins.platform` is the process-wide `platform` module `finder` also imports, so this makes the
    m11 fall-through loud instead of host-dependent (it would otherwise pass on Windows hosts).
    """
    runtime = distractor_bundle(tmp_path)

    def tripwire(*args: object, **kwargs: object) -> str:
        raise AssertionError("host access leaked: platform.system()")

    monkeypatch.setattr(pins.platform, "system", tripwire)

    assert capability.backends(runtime, system="linux") == ["cpu"]
    assert capability.backends(runtime, system="windows") == ["vulkan"]


def test_an_unnamed_machine_is_a_caller_error_not_a_platform_machine_read(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """B2 (duel t_0fc576df, m08): a synthetic world without an arch is the caller's mistake to fix.

    m08 filled the omitted `machine` from `platform.machine()`, so a caller who described only the
    OS silently got a plan for the real box's arch. The tripwire makes the read loud; HEAD answers
    the caller error (`host_variant`'s "no pinned bundle for platform linux-") instead of planning.
    """
    simulate_host(monkeypatch, tmp_path, world="cuda")   # the GPU box is the adversarial case
    read: list[str] = []

    def tripwire(*args: object, **kwargs: object) -> str:
        read.append("platform.machine()")
        raise AssertionError("host access leaked: platform.machine()")

    monkeypatch.setattr(pins.platform, "machine", tripwire)

    with pytest.raises(RuntimeMissingError) as excinfo:
        pins.host_variant("auto", system="linux")

    assert read == []
    assert "no pinned llama.cpp bundle for platform linux-" in str(excinfo.value)


def test_omitted_dri_nodes_never_list_the_real_dev_dri(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: pathlib.Path) -> None:
    """B2 (duel t_0fc576df, m09): a supplied ICD without `dri_nodes` does not make this box Vulkan.

    m09 listed the real `/dev/dri` when `dri_nodes` was omitted — the file's own "facts not
    supplied count as absent" clause. The simulated world really does carry a render node, so the
    mutant answers `vulkan`; the Trap additionally makes any such listing loud.
    """
    simulate_host(monkeypatch, tmp_path, world="vulkan")   # a render node IS present on this box
    icd = tmp_path / "caller-icd.d"
    icd.mkdir()

    assert pins.detect_backend(system="linux", has_nvidia_smi=False, icd_dir=str(icd)) == "cpu"

    class Trap:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"host access leaked: /dev/dri.{name}")

    monkeypatch.setattr(pins, "DRI_DIR", Trap())
    assert pins.detect_backend(system="linux", has_nvidia_smi=False, icd_dir=str(icd)) == "cpu"


def test_an_empty_injected_vram_probe_never_reaches_the_real_driver(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """B3 (duel t_0fc576df, m10): the `nvidia_smi=` seam owns vram — an empty answer stays empty.

    m10 fell through to the real driver when the injected probe answered empty (`vram 0 ->
    8589934592` on the operator's RTX box; on a GPU-less CI box the same fall-through is a hidden
    host read). The tripwire keeps this pin host-independent, so it fails on the mutant here too.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32761996 kB\n")

    def tripwire() -> int | None:
        raise AssertionError("host access leaked: recommend._query_nvidia_smi()")

    monkeypatch.setattr(recommend, "_query_nvidia_smi", tripwire)

    budget = recommend.host_budget(meminfo_path=meminfo, nvidia_smi=lambda: None)

    assert budget.vram_bytes == 0
    assert budget.ram_bytes == 32_761_996 * 1024
