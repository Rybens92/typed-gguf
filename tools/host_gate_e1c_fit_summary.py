#!/usr/bin/env python3
"""Fold a `tools/host_gate_e1c_fit.sh` log directory into one JSON evidence block.

    python3 tools/host_gate_e1c_fit_summary.py <log-dir>   # -> <log-dir>/host_gate_e1c_fit.json

Every number comes from a file the gate wrote (exit codes, elapsed seconds, raw stdout) or from
the JSON the run itself produced; nothing is computed by hand. The report also states its own
VEHICLE (`is_host_run`, `fake_driver`, `model`) so a sandbox rehearsal can never be read as an
operator-host run — the failure mode card t_8cb0a05e's predecessor was rejected for.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

# step -> (command, what the exit code means)
STEPS: dict[str, tuple[str, str]] = {
    "pytest_before": ("uv run pytest -q", "0 = offline suite green on this host"),
    "fit_plan": ("uv run ggufone fit <model> --no-cache --json",
                 "0 = the plan + the free-VRAM reading it was bounded by"),
    "fit_target_bounded": ("uv run ggufone fit <model> --no-cache --json --fit-target 5200",
                           "0 = --fit-target narrowed the plan (see plan_delta)"),
    "run_busy_desktop": ("uv run ggufone ask ... <model> --no-fit-cache",
                         "0 = the decision ran; see placement/warnings (degraded is OK)"),
    "run_no_fit": ("uv run ggufone ask ... <model> --no-fit",
                   "0 = CPU-only run; engine.placement must say so"),
    "fake_oom_degrade": ("python3 tools/fit_oom_probe.py --runtime <fake bundle>",
                         "0 = an allocation failure was survived by degrading to CPU"),
    "fake_oom_all_rungs": ("GGUFONE_FAKE_OOM_ALL=1 python3 tools/fit_oom_probe.py",
                           "0 = nothing fits and the answer was E_BACKEND_OOM"),
    "bench_placement": ("uv run ggufone bench --suite latency --model <model> --gpu-layers 36",
                        "0 = the bench placement went through the loader's ladder; the row must "
                        "print requested AND used (E2 FIX t_31b3943a)"),
    "bench_placement_oom": ("GGUFONE_FAKE_OOM_ALL=1 uv run ggufone bench --suite throughput "
                            "--model <gguf> --gpu-layers 4",
                            "1 = nothing fits at any rung; the row must name E_BACKEND_OOM, "
                            "never an AttributeError (E_INTERNAL)"),
}
TAIL = 1200
FREE_RE = re.compile(r"free_vram_(before|after): (.+)")


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def tail(text: str, limit: int = TAIL) -> str:
    return text[-limit:]


def step_block(log_dir: pathlib.Path, name: str) -> dict[str, object]:
    out = read(log_dir / f"{name}.out")
    err = read(log_dir / f"{name}.err")
    raw_exit = read(log_dir / f"{name}.exit").strip()
    elapsed = read(log_dir / f"{name}.elapsed").strip()
    command, meaning = STEPS.get(name, ("?", ""))
    block: dict[str, object] = {
        "cmd": command,
        "exit": int(raw_exit) if raw_exit.lstrip("-").isdigit() else None,
        "elapsed_s": int(elapsed) if elapsed.isdigit() else None,
        "meaning": meaning,
        "stdout_tail": tail(out),
    }
    if err.strip():
        block["stderr_tail"] = tail(err)
    return block


def read_json(path: pathlib.Path) -> dict:
    try:
        payload = json.loads(read(path))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def bench_view(path: pathlib.Path) -> dict[str, object]:
    """A bench row's placement: what the flags asked for vs what the loader really did (E2 FIX).

    `requested` is `spec.n_gpu_layers`; `used_*` is the loader's own placement (a degraded retry
    offloads fewer layers than the flags asked for), and the throughput world reports a row reason
    instead of a load.
    """
    payload = read_json(path)
    placement = payload.get("placement") if isinstance(payload.get("placement"), dict) else {}
    used = placement.get("used") if isinstance(placement.get("used"), dict) else {}
    backends = payload.get("backends") if isinstance(payload.get("backends"), list) else []
    first = backends[0] if backends and isinstance(backends[0], dict) else {}
    return {
        "requested": placement.get("requested"),
        "used_n_gpu_layers": used.get("n_gpu_layers"),
        "used_kv_type": used.get("kv_type"),
        "degraded": used.get("degraded"),
        "note": used.get("note"),
        "attempts": used.get("attempts"),
        "model_load_n": (payload.get("model_load") or {}).get("n"),
        "row_measured": first.get("measured"),
        "row_reason": first.get("reason"),
    }


def plan_view(path: pathlib.Path) -> dict[str, object]:
    """The plan fields this gate is about, plus the free VRAM the fit command reported.

    `_first` rather than `a or b`: a legitimate `0` (`budget_bytes` for `--fit-target 5200`) must
    not be mistaken for "absent" — the truthiness bug this tool exists to catch, caught in itself.
    """
    payload = read_json(path)
    host = payload.get("host") if isinstance(payload.get("host"), dict) else {}
    engine = payload.get("engine") if isinstance(payload.get("engine"), dict) else {}
    fit_block = engine.get("fit") if isinstance(engine.get("fit"), dict) else {}

    def _first(key: str) -> object:
        for source in (payload, fit_block, engine):
            if source.get(key) is not None:
                return source[key]
        return None

    return {
        "budget_bytes": _first("budget_bytes"),
        "n_gpu_layers": _first("n_gpu_layers"),
        "kv_type": _first("kv_type"),
        "est_total_bytes": _first("est_total_bytes"),
        "source": _first("source"),
        "vram_total_bytes": host.get("vram_bytes"),
        "vram_free_bytes": host.get("vram_free_bytes"),
        "vram_source": host.get("vram_source"),
        "warnings": _first("warnings"),
        "placement": engine.get("placement"),
        "notes": _first("notes"),
    }


def main(argv: list[str]) -> int:
    log_dir = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    facts = read(log_dir / "host_facts.txt")
    fake_driver = ""
    model = ""
    for line in facts.splitlines():
        if line.startswith("fake_driver: "):
            fake_driver = line.split(": ", 1)[1].strip()
        if line.startswith("model: "):
            model = line.split(": ", 1)[1].split(" (", 1)[0].strip()
    free_before = [match.group(2) for match in FREE_RE.finditer(facts)]
    free_after = [match.group(2) for match in FREE_RE.finditer(read(log_dir / "after.txt"))]

    plan = plan_view(log_dir / "fit_plan.out")
    bounded = plan_view(log_dir / "fit_target_bounded.out")
    busy = plan_view(log_dir / "run_busy_desktop.json")
    no_fit = plan_view(log_dir / "run_no_fit.json")
    degrade = read_json(log_dir / "fake_oom_degrade.json")
    all_rungs = read_json(log_dir / "fake_oom_all_rungs.json")
    bench = bench_view(log_dir / "bench_placement.json")
    bench_oom = bench_view(log_dir / "bench_placement_oom.json")

    report: dict[str, object] = {
        "schema": "ggufone.evidence.e1c.fit.host_gate/v1",
        "is_host_run": not fake_driver and fake_driver != "<none>",
        "vehicle": ("OPERATOR HOST: the real driver, the real bundle, the real desktop share"
                    if not fake_driver or fake_driver == "<none>" else
                    f"SANDBOX REHEARSAL: the GPU world is simulated by the fake driver at "
                    f"{fake_driver}; the compilation of the fake-OOM bundle and the GGUF reads are "
                    f"real. The host run of tools/host_gate_e1c_fit.sh produces this same file "
                    f"with is_host_run true."),
        "log_dir": str(log_dir),
        "model": model,
        "fake_driver": fake_driver or None,
        "free_vram_before": free_before,
        "free_vram_after": free_after,
        "host_facts": facts,
        "steps": {name: step_block(log_dir, name) for name in STEPS},
        "fit_plan": plan,
        "fit_target_bounded": bounded,
        "fit_target_delta": {
            "budget_before": plan.get("budget_bytes"),
            "budget_after": bounded.get("budget_bytes"),
            "layers_before": plan.get("n_gpu_layers"),
            "layers_after": bounded.get("n_gpu_layers"),
            "changed": (plan.get("budget_bytes"), plan.get("n_gpu_layers"))
            != (bounded.get("budget_bytes"), bounded.get("n_gpu_layers")),
        },
        "run_busy_desktop": busy,
        "run_no_fit": no_fit,
        "fake_oom_degrade": degrade,
        "fake_oom_all_rungs": all_rungs,
        "bench_placement": bench,
        "bench_placement_oom": bench_oom,
        "checks": _checks(plan, bounded, busy, no_fit, degrade, all_rungs, bench, bench_oom),
    }
    target = log_dir / "host_gate_e1c_fit.json"
    target.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(json.dumps(report["checks"], indent=1))
    return 0


def _checks(plan: dict, bounded: dict, busy: dict, no_fit: dict, degrade: dict,
            all_rungs: dict, bench: dict, bench_oom: dict) -> dict[str, object]:
    """The card's requirements, as boolean facts — a reader should not have to eyeball the tails."""
    result = degrade.get("result") or {}
    error = all_rungs.get("error") or {}
    free = plan.get("vram_free_bytes")
    budget = plan.get("budget_bytes")
    placement = busy.get("placement") or {}
    bench_reason = str(bench_oom.get("row_reason") or "")
    return {
        "fit_plan_reports_free_vram": isinstance(free, int),
        "fit_plan_budget_leq_free_minus_target": (isinstance(free, int) and isinstance(budget, int)
                                                  and budget <= max(0, free - 1024 * 1024)),
        "fit_target_bound_the_plan": bool(bounded.get("budget_bytes") is not None
                                          and (bounded.get("budget_bytes") != budget
                                               or bounded.get("n_gpu_layers")
                                               != plan.get("n_gpu_layers"))),
        "busy_desktop_run_succeeded": busy.get("n_gpu_layers") is not None,
        "busy_desktop_placement_named": bool(isinstance(placement, dict) and placement.get("note")),
        "no_fit_says_cpu_only": bool((no_fit.get("placement") or {}).get("note", "")
                                     .lower().find("fit disabled") >= 0),
        "fake_oom_degraded_to_cpu": bool(result.get("ok") and result.get("n_gpu_layers") == 0),
        "fake_oom_reported_W_BACKEND_OOM": "W_BACKEND_OOM" in (result.get("warnings") or []),
        "all_rungs_error_is_backend_oom": error.get("code") == "E_BACKEND_OOM",
        "all_rungs_error_has_hints": bool("--no-fit" in str(error.get("message", ""))
                                          and "--fit-target" in str(error.get("message", ""))),
        # E2 FIX t_31b3943a: the bench placement through the loader's ladder
        "bench_run_loaded_a_model": isinstance(bench.get("model_load_n"), int)
        and int(bench["model_load_n"]) >= 1,
        "bench_row_prints_the_placement_used": isinstance(bench.get("used_n_gpu_layers"), int),
        "bench_row_keeps_the_request_next_to_it": isinstance(bench.get("requested"), str),
        "bench_retry_row_is_a_typed_error": ("E_BACKEND_OOM" in bench_reason
                                             and "AttributeError" not in bench_reason
                                             and "E_INTERNAL" not in bench_reason),
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv))
