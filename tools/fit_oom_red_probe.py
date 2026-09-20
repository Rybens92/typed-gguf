#!/usr/bin/env python3
"""Before/after probe for card t_8cb0a05e — uses ONLY APIs that exist before the fix.

That is the point: the same script runs against the pre-fix tree and the fixed tree, so the
operator's failure mode can be *demonstrated* (RED) instead of asserted from a changelog.

    python3 tools/fit_oom_red_probe.py --model <gguf> [--busy-mib 1112] [--fake-runtime DIR]
                                                  [--fake-oom-all]

Two worlds:

1. **busy desktop** (needs no runtime): the host is described as the operator's box — 8192 MiB
   total, `--busy-mib` free — and the script prints the plan the CURRENT tree produces for it.
   On the pre-fix tree the plan asks for full offload (the 1.06 GB allocation that could not
   exist); on the fixed tree it is CPU-only and `--fit-target` narrows it further.
2. **fake allocation failure** (`--fake-runtime`, a bundle built from tools/fixtures/
   fit_oom_bundle.c): loads a full-offload plan and prints the error code. Pre-fix that code is
   `E_MODEL_ARCH_UNSUPPORTED` (the misclassification); fixed it is `E_BACKEND_OOM` — and with
   `--fake-oom-all` unset the loader instead degrades to CPU and succeeds.

Exit code 0 when the run matches the FIXED contract, 3 when it matches the pre-fix one (so a
pre-fix run is visibly "not fixed" rather than just noisy).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.engine import session as session_module  # noqa: E402
from typed_gguf.errors import TypedGgufError  # noqa: E402
from typed_gguf.registry import recommend  # noqa: E402
from typed_gguf.runtime import fit  # noqa: E402

SCHEMA = "typed_gguf.evidence.fit-oom-red/v1"
GIB = 1024 ** 3
MIB = 1024 ** 2


def busy_plan(model_path: pathlib.Path, free_mib: int) -> dict[str, object]:
    """The plan this tree builds for an 8 GiB box whose desktop holds everything but `free_mib`.

    `host_plan` is the number the card is about: on the pre-fix tree it is a full-offload plan
    built from the NOMINAL 8 GiB (the operator's `n_gpu_layers: 36`, `budget_bytes` ≈ 7168 MiB,
    a 1.06 GB allocation that could not exist); on the fixed tree the same call is bounded by the
    free reading and lands on the CPU.
    """
    model = fit.ModelFacts.read(model_path, want_sha256=False)
    facts = fit.host_facts()
    report: dict[str, object] = {
        "host_budget_bytes": facts.budget_bytes,
        "host_vram_bytes": facts.vram_bytes,
        "host_vram_free_bytes": getattr(facts, "vram_free_bytes", None),
        "free_probe_mib": free_mib,
    }
    host_plan = fit.estimate_plan(model, facts, n_ctx=4096, n_seq_max=8)
    report["host_plan"] = {key: host_plan.to_dict()[key] for key in fit.FIT_FIELDS}
    report["host_plan_budget_bytes"] = host_plan.budget_bytes
    report["host_plan_warnings"] = list(host_plan.warnings)
    report["host_plan_notes"] = list(host_plan.notes)
    nominal = fit.HostFacts(backend=facts.backend, ram_bytes=facts.ram_bytes,
                            vram_bytes=8 * GIB, n_cpu=facts.n_cpu, fingerprint=facts.fingerprint)
    plain = fit.estimate_plan(model, nominal, n_ctx=4096, n_seq_max=8, budget_bytes=8 * GIB)
    report["nominal_plan"] = {key: plain.to_dict()[key] for key in fit.FIT_FIELDS}
    tight = fit.estimate_plan(model, nominal, n_ctx=4096, n_seq_max=8, fit_target_mb=5200)
    report["fit_target_5200"] = {key: tight.to_dict()[key] for key in fit.FIT_FIELDS}
    report["fit_target_narrowed_the_plan"] = (tight.n_gpu_layers != plain.n_gpu_layers)
    return report


def oom_world(model_path: pathlib.Path, runtime: pathlib.Path, free_mib: int,
              all_rungs: bool) -> dict[str, object]:
    """What code does the CURRENT tree raise when the backend cannot allocate?"""
    model = fit.ModelFacts.read(model_path, want_sha256=False)
    roomy = fit.HostFacts(backend="vulkan", ram_bytes=31 * GIB, vram_bytes=8 * GIB,
                          n_cpu=4, fingerprint="probe:red")
    plan = fit.estimate_plan(model, roomy, n_ctx=4096, n_seq_max=8, budget_bytes=8 * GIB)
    report: dict[str, object] = {"plan": {key: plan.to_dict()[key] for key in fit.FIT_FIELDS}}
    try:
        handle = session_module.open_model(model_path, runtime_dir=runtime, fit_plan=plan)
    except TypedGgufError as exc:
        report["raised"] = {"code": exc.code, "message": str(exc)[:400]}
        return report
    try:
        report["loaded"] = {"n_gpu_layers": int(getattr(handle, "n_gpu_layers", -1)),
                            "placement": getattr(getattr(handle, "placement", None), "note",
                                                 "<no placement surface pre-fix>")}
    finally:
        handle.close()
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--busy-mib", type=int, default=1112)
    parser.add_argument("--fake-runtime", default=None)
    parser.add_argument("--fake-oom-all", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    model_path = pathlib.Path(args.model)
    memory = getattr(recommend, "device_memory", None)
    receipt: dict[str, object] = {
        "schema": SCHEMA, "model": str(model_path),
        "device_memory": (memory().to_dict() if memory is not None
                          else {"note": "this tree has no recommend.device_memory()"}),
        "busy_plan": busy_plan(model_path, args.busy_mib)}
    if args.fake_runtime:
        receipt["oom_world"] = oom_world(model_path, pathlib.Path(args.fake_runtime),
                                         args.busy_mib, args.fake_oom_all)
    busy = receipt["busy_plan"]
    fixed_busy = bool("W_FIT_DOWNGRADE" in (busy.get("host_plan_warnings") or []))
    fixed_error = True
    if args.fake_runtime:
        world = receipt["oom_world"]
        if "raised" in world:
            fixed_error = world["raised"]["code"] == "E_BACKEND_OOM"
        else:
            fixed_error = world.get("loaded", {}).get("n_gpu_layers") == 0
    receipt["verdict"] = {"busy_desktop_plan_is_bounded": fixed_busy,
                          "oom_is_classified_as_backend_oom": fixed_error,
                          "fixed": fixed_busy and fixed_error}
    print(json.dumps(receipt, indent=1))
    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    return 0 if (fixed_busy and fixed_error) else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
