"""`tools/matrix_oom_row.py` — the two fake-OOM worlds of the linux-cpu step, pinned as data.

The step `Engine smoke — the placement retry answers a typed row, never E_INTERNAL` used to assert
`'3 placement(s)' in reason` against a *bench* row. That number belongs to `fit_oom_probe`'s ladder;
a bench row resolves to the `cpu` backend, which is CPU-pinned (card t_55de5779) and has no rungs —
so the assertion could never pass, and nobody noticed because the fixture was refused one gate
earlier (`E_RUNTIME_SYMBOLS`, job 108153215436, card t_8dab8b3a).

The fixtures below are the *measured* shapes: the reason texts are the real ones from this card's
runs (the bench row's `E_BACKEND_OOM` with one rung, the probe's three-rung ladder), and the fields
are named the way the product writes them.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_oom_row.py"
#: The real reason the fixed bench row carries (card t_8dab8b3a, `/tmp/placement-oom-green.json`).
BENCH_REASON = (
    "BackendOomError: E_BACKEND_OOM: llama.cpp could not allocate device memory for the fit plan "
    "(n_gpu_layers=4, kv_type=auto, needed ~1010 MiB); the driver reports 6577 MiB free; tried 1 "
    "placement(s) down to CPU-only, none fit: n_gpu_layers=0 -> oom; the backend asked for a 1010 "
    "MiB allocation; backend log: 'ggml_vulkan: Device memory allocation of size 1058982400 "
    "failed.'; fix: `--no-fit` runs on the CPU")
#: … and the real ladder answer the same bundle produces through `fit_oom_probe.py`.
LADDER_MESSAGE = (
    "E_BACKEND_OOM: llama.cpp could not allocate device memory for the fit plan (n_gpu_layers=0, "
    "kv_type=f16, needed ~1010 MiB); the driver reports 1112 MiB free; tried 3 placement(s) down "
    "to CPU-only, none fit: n_gpu_layers=36 -> oom; n_gpu_layers=18 -> oom; n_gpu_layers=0 -> "
    "oom; the backend asked for a 1010 MiB allocation")
#: The pre-fix shape (the E2 crash this step exists for): what the parent-rev control produced.
CRASH_REASON = "AttributeError: 'Placement' object has no attribute 'kv_type'"


def load_tool():
    spec = importlib.util.spec_from_file_location("matrix_oom_row", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bench_report(reason: str = BENCH_REASON, *, measured: bool = False,
                 backend: str = "cpu") -> dict:
    return {"suite": "throughput", "backends": [
        {"backend": backend, "measured": measured, "runtime_dir": "/tmp/fake-bundle",
         "placement": "n_gpu_layers=4 (cpu compute pinned)", "threads": 1, "reason": reason}]}


def probe_receipt(message: str = LADDER_MESSAGE, code: str = "E_BACKEND_OOM",
                  world: str = "oom-all") -> dict:
    return {"schema": "typed_gguf.evidence.fit-oom/v1", "world": world,
            "result": {"ok": False},
            "error": {"code": code, "exit_code": 3, "message": message}}


# ------------------------------------------------------------------------- the pinned truth
def test_both_worlds_pass_the_judge() -> None:
    module = load_tool()
    problems, facts = module.judge_bench(bench_report(), exit_code=1)
    assert problems == [], problems
    assert facts["placements"] == 1 and facts["backend"] == "cpu"
    ladder_problems, ladder_facts = module.judge_ladder(probe_receipt())
    assert ladder_problems == [], ladder_problems
    assert ladder_facts["placements"] == 3 and ladder_facts["code"] == "E_BACKEND_OOM"


# --------------------------------------------------------------------------- the RED directions
def test_the_crash_reason_this_step_exists_for_is_refused() -> None:
    """The parent-rev control's reason: the step must never go green on it again."""
    module = load_tool()
    problems, _ = module.judge_bench(bench_report(CRASH_REASON), exit_code=4)
    assert problems, "an AttributeError reason is exactly what this step was written to catch"
    assert any("AttributeError" in problem for problem in problems), problems


def test_a_reason_without_the_typed_code_is_refused() -> None:
    module = load_tool()
    reason = BENCH_REASON.replace("E_BACKEND_OOM", "E_RUNTIME_MISSING")
    problems, _ = module.judge_bench(bench_report(reason), exit_code=1)
    assert problems and any("E_BACKEND_OOM" in problem for problem in problems), problems


def test_a_row_that_measured_something_is_refused() -> None:
    """An OOM row measured nothing: `measured: true` means the row is not the fake world."""
    module = load_tool()
    problems, _ = module.judge_bench(bench_report(measured=True), exit_code=1)
    assert problems and any("measured" in problem for problem in problems), problems


def test_the_stale_three_rung_expectation_on_a_bench_row_is_refused() -> None:
    """The assertion this card is fixing: a CPU-pinned row has one rung, not three."""
    module = load_tool()
    reason = BENCH_REASON.replace("tried 1 placement(s)", "tried 3 placement(s)")
    problems, _ = module.judge_bench(bench_report(reason), exit_code=1)
    assert problems and any("3 placement(s)" in problem for problem in problems), problems


def test_a_bench_row_that_is_not_cpu_pinned_is_refused() -> None:
    """The count above is only true because the row is cpu + pinned: say so when it drifts."""
    module = load_tool()
    report = bench_report(backend="vulkan")
    report["backends"][0]["placement"] = "n_gpu_layers=4"
    problems, _ = module.judge_bench(report, exit_code=1)
    assert problems and any("CPU-pinned" in problem for problem in problems), problems


def test_an_exit_code_that_drifted_is_refused() -> None:
    module = load_tool()
    problems, _ = module.judge_bench(bench_report(), exit_code=0)
    assert problems and any("exited 0" in problem for problem in problems), problems


def test_a_ladder_that_answered_another_code_is_refused() -> None:
    module = load_tool()
    problems, _ = module.judge_ladder(probe_receipt(code="E_MODEL_ARCH_UNSUPPORTED"))
    assert problems and any("E_MODEL_ARCH_UNSUPPORTED" in problem for problem in problems), problems


def test_a_ladder_that_no_longer_reaches_cpu_only_is_refused() -> None:
    module = load_tool()
    message = LADDER_MESSAGE.replace("tried 3 placement(s) down to CPU-only",
                                     "tried 2 placement(s)")
    problems, _ = module.judge_ladder(probe_receipt(message))
    assert problems, "a ladder that stops short of CPU-only is not the pinned world"


def test_a_probe_receipt_from_another_world_is_refused() -> None:
    module = load_tool()
    problems, _ = module.judge_ladder(probe_receipt(world="degrade-to-cpu"))
    assert problems and any("world" in problem for problem in problems), problems


# --------------------------------------------------------------------------------- the wiring
def test_main_judges_the_two_files_and_writes_a_verdict(tmp_path: pathlib.Path) -> None:
    row = tmp_path / "placement-oom.json"
    receipt = tmp_path / "oom-probe.json"
    row.write_text(json.dumps(bench_report()), encoding="utf-8")
    receipt.write_text(json.dumps(probe_receipt()), encoding="utf-8")
    module = load_tool()
    out = tmp_path / "verdict.json"
    code = module.main(["--bench-row", str(row), "--bench-exit", "1",
                        "--probe-receipt", str(receipt), "--json", str(out)])
    assert code == 0, out.read_text(encoding="utf-8")
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["ok"] is True and verdict["facts"]["ladder"]["placements"] == 3


def test_main_refuses_a_row_it_cannot_read(tmp_path: pathlib.Path) -> None:
    row = tmp_path / "placement-oom.json"
    row.write_text("not json", encoding="utf-8")
    receipt = tmp_path / "oom-probe.json"
    receipt.write_text(json.dumps(probe_receipt()), encoding="utf-8")
    module = load_tool()
    code = module.main(["--bench-row", str(row), "--bench-exit", "1",
                        "--probe-receipt", str(receipt)])
    assert code == 1


def test_the_tool_exists_where_the_workflow_looks_for_it() -> None:
    assert TOOL.is_file()
    assert load_tool().SCHEMA.startswith("typed_gguf.matrix.")
