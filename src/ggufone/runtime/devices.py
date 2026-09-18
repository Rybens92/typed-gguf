"""Which device did the engine *really* use? — read back from llama.cpp's own log.

Milestone: E2 FIX (card t_603a35a0). The bench tables label rows by backend, and on a host with
more than one bundle installed the label can lie in both directions:

* a row labelled `cpu` measured 587.9 tok/s of prefill on the operator host — op offload ran the
  graph on the Vulkan device while the weights stayed on the host (the placement note said
  "CPU only");
* `--backend all` across a CPU bundle and the Vulkan bundle produced a `vulkan` row whose log
  showed `CPU_Mapped model buffer size = 4167.21 MiB` and nine `CPU compute buffer size` lines
  with not one `Vulkan0` line (9.11 tok/s, host-class);
* the honest single-bundle Vulkan run shows `Vulkan0 compute buffer size` / `Vulkan_Host …`.

This module turns that log into counts per device. Two rules keep it honest:

* **Compute buffers are the evidence.** `llama_context`/`sched_reserve` print
  `<device> compute buffer size` when the graph scheduler reserves memory for a device — that is
  the device the work runs on. Model buffers say where the *weights* live, which op offload
  deliberately separates from the compute path.
* **`load_tensors: offloaded N/M layers to GPU` is a request, not a measurement.** The mixed-bundle
  run printed `offloaded 37/37 layers to GPU` immediately before walking every layer onto the CPU,
  so that line is deliberately not parsed here.

Pure stdlib, no runtime import: the engine (for a session's own report) and the bench harness (for
a row's attribution) both read it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: `sched_reserve:        CPU compute buffer size =   166.26 MiB` (creation) and
#: `~llama_context:    Vulkan0 compute buffer size is 545.3125 MiB, …` (destruction)
_BUFFER_LINE = re.compile(r"(?P<device>[A-Za-z][A-Za-z0-9_.-]*)\s+"
                          r"(?P<kind>compute|model|KV)\s+buffer size\b")
#: `load_tensors: layer   3 assigned to device CPU, is_swa = 0`
_LAYER_LINE = re.compile(r"assigned to device (?P<device>[A-Za-z][A-Za-z0-9_.-]*)")
#: backend *kinds* a device name carries: `Vulkan0`/`Vulkan_Host` -> vulkan, `CPU_Mapped` -> cpu
_DEVICE_SUFFIXES = ("_host", "_mapped", "_repack", "_host_mapped", "_shared")
#: backends that do not run the graph on the box, so they never name a compute path
_NON_COMPUTE = ("rpc",)
BACKEND_KINDS = ("cpu", "vulkan", "cuda", "metal")
_KIND_TO_KINDS = {"compute": "compute_buffers", "model": "model_buffers", "KV": "kv_buffers"}


def backend_of(device: str) -> str:
    """`Vulkan_Host` / `Vulkan0` -> `vulkan`, `CPU_Mapped` -> `cpu`, `CUDA0` -> `cuda`.

    Strips the trailing device index (`Vulkan1`), the buffer-type suffix (`_Host`, `_Mapped`,
    `_Repack`) and lowercases — the ggml backend name behind the device name.
    """
    stem = re.split(r"\d", str(device), maxsplit=1)[0].rstrip("_").lower()
    for suffix in _DEVICE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or "unknown"


@dataclass(frozen=True, slots=True)
class DeviceUsage:
    """Per-device buffer counts read out of one engine log (counts, never a request)."""

    compute_buffers: dict[str, int] = field(default_factory=dict)
    model_buffers: dict[str, int] = field(default_factory=dict)
    kv_buffers: dict[str, int] = field(default_factory=dict)
    layers: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"compute_buffers": dict(self.compute_buffers),
                "model_buffers": dict(self.model_buffers),
                "kv_buffers": dict(self.kv_buffers),
                "layers": dict(self.layers)}

    @property
    def devices(self) -> tuple[str, ...]:
        """Every device name the log mentions (sorted) — what the engine touched."""
        names: set[str] = set(self.compute_buffers) | set(self.model_buffers) | set(self.kv_buffers)
        names |= set(self.layers)
        return tuple(sorted(names))

    @property
    def compute_backends(self) -> tuple[str, ...]:
        """The backends behind the *compute* buffers (sorted, de-duplicated)."""
        return tuple(sorted({backend_of(device) for device in self.compute_buffers}))

    @property
    def effective(self) -> str | None:
        """The backend that computed, from the compute buffers alone — else None (unknown).

        `cpu+vulkan` when the graph really ran on both (a partial offload), `None` when the log
        carries no compute-buffer line at all: a claim that cannot be corroborated is reported as
        unknown, never repeated as if it were measured.
        """
        backends = [backend for backend in self.compute_backends if backend not in _NON_COMPUTE]
        return "+".join(backends) if backends else None


def parse_device_usage(log_text: str | None) -> DeviceUsage:
    """Count `compute`/`model`/`KV` buffer lines per device in one engine log."""
    buckets: dict[str, dict[str, int]] = {"compute_buffers": {}, "model_buffers": {},
                                          "kv_buffers": {}, "layers": {}}
    for line in (log_text or "").splitlines():
        match = _BUFFER_LINE.search(line)
        if match:
            bucket = buckets[_KIND_TO_KINDS[match.group("kind")]]
            device = match.group("device")
            bucket[device] = bucket.get(device, 0) + 1
            continue
        assigned = _LAYER_LINE.search(line)
        if assigned:
            device = assigned.group("device")
            buckets["layers"][device] = buckets["layers"].get(device, 0) + 1
    return DeviceUsage(**buckets)


def contradicts(claimed: str, usage: DeviceUsage) -> bool:
    """Does the evidence refute the claim? (an *uncorroborated* accelerator claim does)

    `cpu` is refuted by any accelerator in the compute buffers (op offload computes on the device
    while the weights stay on the host), and an accelerator is refuted when its own backend never
    appears there — *including* the case where the log carries no compute-buffer line at all: a
    positive device claim the engine cannot corroborate is not publishable (card t_603a35a0, the
    mixed-bundle run where the second bundle's rows are captured with nothing). An empty set does
    not refute a `cpu` claim: nothing was computed anywhere else.
    """
    evidence = [backend for backend in usage.compute_backends if backend not in _NON_COMPUTE]
    if claimed == "cpu":
        return bool(evidence) and evidence != ["cpu"]
    return claimed not in evidence


__all__ = ["BACKEND_KINDS", "DeviceUsage", "backend_of", "contradicts", "parse_device_usage"]
