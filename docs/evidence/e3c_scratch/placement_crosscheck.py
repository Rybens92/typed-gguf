"""E3c scratch: the placement cross-check — the same item measured on the CPU and on the device.

The sweep runs CPU-only (the GPU was held by a sibling campaign); the card's gates ask for
`--backend vulkan --gpu-layers 7`. This re-runs one dev item on the device and prints, per shape,
the verdict and the `bare` coverage side by side, plus what each run's engine log proves about the
device that computed.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / ".e3c" / name).read_text())


for name in ("occamy_c01.json", "occamy_c01_vulkan.json"):
    record = load(name)
    item = record["items"][0]
    print(f"== {name}")
    print(f"   claim {record['backend_claim']} ({record['backend_source']}) · "
          f"gpu_layers {record['gpu_layers']} · placement {record['placement'].get('n_gpu_layers')} "
          f"layers, kv {record['placement'].get('kv_type')}")
    print(f"   load {record['load_ms'] / 1000:.1f} s · prefill {item['prefill_ms']:.0f} ms")
    print(f"   device log tail: {item['device_log_tail'][-160:]!r}")

cpu, gpu = load("occamy_c01.json")["items"][0], load("occamy_c01_vulkan.json")["items"][0]
print("\nper shape: verdict (closer, mass) and `bare` coverage, CPU vs device")
for shape in cpu["shapes"]:
    left, right = cpu["shapes"][shape]["readout"], gpu["shapes"][shape]["readout"]
    print(f"  {shape:22s} cpu refused={left['cue']['refused']!s:5s} mass={left['cue']['mass']:.6f} "
          f"cov={left['labels']['bare']['coverage']:.3e} | "
          f"gpu refused={right['cue']['refused']!s:5s} mass={right['cue']['mass']:.6f} "
          f"cov={right['labels']['bare']['coverage']:.3e}")
for key in cpu["ranked"]:
    left, right = cpu["ranked"][key], gpu["ranked"][key]
    print(f"  ranked {key:22s} cpu got={left['got']!s:10s} rel={left['reliability']!s:9s} | "
          f"gpu got={right['got']!s:10s} rel={right['reliability']}")

print("\nwhere each run's two-step readout actually sits (the advance row and its top tokens)")
for shape in ("two_step_shipped", "two_step_answer_is"):
    for name, item in (("cpu", cpu), ("gpu", gpu)):
        shape_block = item["shapes"][shape]
        advance = shape_block.get("advance")
        top = ", ".join(f"{token['piece']!r} {token['p_full']:.4f}"
                        for token in shape_block["readout"]["top_tokens"][:3])
        print(f"  {shape:20s} {name}: advanced by {advance['piece']!r} "
              f"(token {advance['token']}) at row {shape_block['readout']['position']} -> {top}")
