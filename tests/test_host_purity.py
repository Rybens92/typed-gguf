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
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from ggufone.runtime import install, pins

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "runtime.lock"

# Environment a host-fact reader could read INSTEAD of a probe. Detection must not consult any of
# them (the installer's `GGUFONE_OFFLINE_CACHE` is a cache knob, not a host fact).
DEVICE_ENV = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
              "GPU_DEVICE_ORDINAL", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_LOADER_LAYERS_ENABLE", "DISPLAY",
              "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "GGUFONE_BACKEND", "GGUFONE_DEVICE",
              "GGUFONE_ACCELERATOR", "GGUFONE_GPU")
# Names that could carry a host fact, by prefix: a probe default smuggled through the environment
# has to be read by one of these.
HOST_FACT_PREFIXES = ("GGUFONE_", "CUDA", "NVIDIA", "VK_", "DRI", "HIP_", "ROCR_", "DISPLAY",
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
