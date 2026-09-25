"""`tools/matrix_windows_doctor.py` — the Windows doctor truth, pinned instead of assumed (AC2).

AC2 asks the Windows job to run `doctor` against the flattened pinned bundle via
`TYPED_GGUF_RUNTIME_DIR`. What `doctor` *says* there is not a free choice: `runtime.lock` pins
`required_files` as the Linux SONAMEs (`libllama.so`, `libggml.so`, `libggml-base.so` — `test_pins`
pins that too), while the pinned Windows zip carries `llama.dll` / `ggml.dll` / `ggml-base.dll` at
the archive root and a `llama-cli.exe` (not `llama-cli`, which is the name `build_number` looks
for). So on Windows the bundle is *found* and classified, and the report still fails on the two
places the distribution layer is Linux-shaped.

The simulation below is that runner: a bundle directory with those DLL names, `platform.system()`
patched to `Windows` and `shutil.which("nvidia-smi")` silenced (a CPU runner), then the product's
own `cli.doctor_checks()`. Its report is what the tool must accept — and every direction that would
make the job quietly green (a bundle that was never found, an exit code that drifted, a product fix
that makes doctor pass) is refused *by name*, so the finding this card reports cannot be lost.

This is a report-level pin, not a claim about Windows: the job itself runs the real doctor on the
real runner (the card's own truth-pinning rule).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from typed_gguf import cli
from typed_gguf.runtime import pins

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_windows_doctor.py"
#: The DLL names the pinned win-cpu zip carries at its root (verified against the real asset).
DLLS = ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cpu-x64.dll")
#: The Linux SONAMEs `runtime.lock` demands, which is what makes the Windows report red.
LINUX_SONAMES = ("libllama.so", "libggml.so", "libggml-base.so")


def load_tool():
    if not TOOL.exists():
        raise AssertionError(
            f"{TOOL.relative_to(ROOT)} is missing: the Windows job runs it to pin what `doctor` "
            "reports there, and without it the job would either assert nothing or assert a "
            "linux-shaped green that cannot happen (card t_f96fed7f)")
    spec = importlib.util.spec_from_file_location("matrix_windows_doctor", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def windows_bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    """The flattened pinned zip, as the workflow's `Expand-Archive` leaves it."""
    bundle = tmp_path / "typed-gguf-rt"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in DLLS:
        (bundle / name).write_bytes(b"")
    (bundle / "llama-cli.exe").write_bytes(b"")
    (bundle / "llama-fit-params.exe").write_bytes(b"")
    return bundle


def simulated_windows_report(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
                             bundle: pathlib.Path | None = None) -> dict:
    """`cli.doctor_checks()` as a Windows CPU runner would answer it."""
    bundle = bundle if bundle is not None else windows_bundle(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(bundle))
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pins.platform, "system", lambda: "Windows")
    monkeypatch.setattr(pins.shutil, "which", lambda *args, **kwargs: None)  # no nvidia-smi
    return cli.doctor_checks()


def checks_of(report: dict) -> dict[str, dict]:
    return {check["id"]: check for check in report["checks"]}


# -------------------------------------------------------------------- the pinned Windows truth
def test_the_simulated_windows_report_is_the_pinned_truth(tmp_path: pathlib.Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture itself: the bundle is found and classified, and the Linux names are what fail."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    checks = checks_of(report)
    assert report["exit_code"] == 1 and report["status"] == "failures", report["status"]
    assert checks["runtime.present"]["status"] == "ok", checks["runtime.present"]
    assert checks["runtime.files"]["status"] == "fail", checks["runtime.files"]
    for soname in LINUX_SONAMES:
        assert soname in checks["runtime.files"]["detail"], checks["runtime.files"]
    assert checks["runtime.backends"]["status"] == "ok", checks["runtime.backends"]
    assert "cpu" in checks["runtime.backends"]["detail"], checks["runtime.backends"]


def test_the_tool_accepts_the_report_and_explains_the_gap(tmp_path: pathlib.Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    report = simulated_windows_report(tmp_path, monkeypatch)
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems == [], problems
    annotation = module.annotation(report)
    assert annotation.startswith("::warning"), annotation
    for soname in LINUX_SONAMES:
        assert soname in annotation, annotation
    assert "llama.dll" in annotation, "the gap names the bundle that *is* there"


# --------------------------------------------------------------------------- the RED directions
def test_a_bundle_that_was_never_found_is_refused(tmp_path: pathlib.Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2's first half is TYPED_GGUF_RUNTIME_DIR actually resolving: an empty dir is not green."""
    empty = tmp_path / "empty"
    empty.mkdir()
    report = simulated_windows_report(tmp_path, monkeypatch, bundle=empty)
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems, "a doctor that never saw the bundle must fail this step"
    assert any("present" in problem for problem in problems), problems


def test_a_green_doctor_is_refused_because_this_card_reports_the_gap(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the product is ever fixed for Windows, this pin must be updated deliberately."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    checks_of(report)["runtime.files"]["status"] = "ok"
    checks_of(report)["runtime.files"]["detail"] = "all required libraries present: llama.dll"
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems, "the pinned gap is gone: the tool must say so, not pass quietly"


def test_an_exit_code_that_drifted_is_refused(tmp_path: pathlib.Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """`doctor`'s exit code is the contract (A-E1a-3: 1 = broken): 0 here means something moved."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    module = load_tool()
    problems = module.judge(report, exit_code=0)
    assert problems and any("exit" in problem for problem in problems), problems


def test_a_report_without_the_bundle_checks_is_refused(tmp_path: pathlib.Path) -> None:
    module = load_tool()
    problems = module.judge({"status": "ok", "exit_code": 1, "checks": []}, exit_code=1)
    assert problems, "an empty report is not the pinned truth"


# --------------------------------------------------------------------------------- the wiring
def test_main_runs_doctor_and_reports_its_truth(tmp_path: pathlib.Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """The step is one command: the tool runs `doctor --json` itself and judges what it printed."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    module = load_tool()
    calls: list[list[str]] = []

    def fake_doctor(prefix, cwd):
        calls.append(list(prefix))
        return json.dumps(report), report["exit_code"]

    monkeypatch.setattr(module, "run_doctor", fake_doctor)
    out = tmp_path / "verdict.json"
    code = module.main(["--cli", "uv run typed-gguf", "--json", str(out)])
    assert code == 0, (code, out.read_text(encoding="utf-8"))
    assert calls == [["uv", "run", "typed-gguf"]], calls
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_main_refuses_a_doctor_that_could_not_run(tmp_path: pathlib.Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_tool()
    monkeypatch.setattr(module, "run_doctor", lambda prefix, cwd: ("not json at all", 2))
    code = module.main(["--cli", "uv run typed-gguf"])
    assert code == 1


def test_the_dll_names_the_workflow_asserts_come_from_the_finder() -> None:
    """The flatten step checks the names the product itself looks for, not invented ones."""
    from typed_gguf.runtime import finder

    assert finder.library_names("windows")["llama"] == "llama.dll"
    assert finder.library_names("windows")["ggml"] == "ggml.dll"
    assert finder.library_names("windows")["ggml_base"] == "ggml-base.dll"
    assert finder.library_glob("windows") == "*ggml-*.dll"


def test_the_tool_is_pure_over_the_report_it_is_handed() -> None:
    """The judge reads a report, never a machine: that is what makes the pin drivable here."""
    module = load_tool()
    assert module.SCHEMA.startswith("typed_gguf.matrix."), module.SCHEMA
    assert module.judge({"status": "failures", "exit_code": 1, "checks": [
        {"id": "runtime.present", "status": "ok", "detail": "/somewhere"},
    ]}, exit_code=1), "a report with only `present` cannot be the pinned truth"
