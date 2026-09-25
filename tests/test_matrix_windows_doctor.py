"""`tools/matrix_windows_doctor.py` — the Windows doctor truth, pinned instead of assumed.

`t_f96fed7f` pinned the *gap*: on Windows `doctor` failed `runtime.files`/`runtime.loadable` on a
complete bundle (the lock's Linux SONAMEs) and could not read a build number. `t_8dab8b3a` closed
it — the distribution check uses the platform's names (`finder.required_files`), `build_number`
runs the platform's CLI (`llama-cli.exe`) and the arch scan reads the ABI's own decoration. What
this file pins is the *new* truth, in the same style: the product's own `cli.doctor_checks()` under
a patched platform, the tool's verdict on that report, and every direction that would make the step
quietly green (a bundle that was never found, a `fail` check left anywhere, an exit code that
drifted, a build number that was not really read) refused by name.

The simulation is a *report-level* pin, not a claim about Windows: the job runs the real doctor on
the real runner, and the live run is the acceptance (the card's own rule). Two substitutions are
stated rather than hidden:

* `platform.system()` is patched to `Windows` (the same monkeypatch the old pin used);
* the deep symbol probe is switched off (`TYPED_GGUF_DEEP_PROBE=0`), because a PE cannot be
  dlopened on this box — the report then carries the honest `runtime.symbols` warning, which the
  tool must *report* and must not fail on. `test_the_dlopen_seam_resolves_every_symbol` shows the
  same report with the probe injected as fully resolved: the fixed gates do not depend on it;
* `llama-cli.exe` is a shell script on this box that prints the banner the real b11026 CLI prints
  (captured from the pinned Linux twin of the same release: `version: 0.4.1-dev (build 11026,
  commit b49650adb)`). The *file name* is the zip's real one, which is the fact under test.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import stat
import sys

import pytest

from typed_gguf import cli
from typed_gguf.runtime import capability, finder, isolated, pins

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_windows_doctor.py"
#: The DLL names the pinned win-cpu zip carries at its root (verified against the real asset).
DLLS = ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cpu-x64.dll")
#: The CLI banner the pinned b11026 CLI really prints.
REAL_CLI_BANNER = "version: 0.4.1-dev (build 11026, commit b49650adb)"
#: `runtime.lock`'s tag — what `doctor` has to read out of that bundle on Windows.
PINNED_TAG = "b11026"


def load_tool():
    if not TOOL.exists():
        raise AssertionError(
            f"{TOOL.relative_to(ROOT)} is missing: the Windows job runs it to pin what `doctor` "
            "reports there, and without it the job would assert nothing (card t_8dab8b3a)")
    spec = importlib.util.spec_from_file_location("matrix_windows_doctor", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def windows_bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    """The flattened pinned zip, as the workflow's `Expand-Archive` leaves it."""
    bundle = tmp_path / "typed-gguf-rt"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in DLLS:
        (bundle / name).write_bytes(b"")
    _script(bundle / "llama-cli.exe", f"#!/bin/sh\necho '{REAL_CLI_BANNER}' >&2\n")
    _script(bundle / "llama-fit-params.exe", "#!/bin/sh\necho usage\nexit 0\n")
    return bundle


def simulated_windows_report(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
                             bundle: pathlib.Path | None = None,
                             *, deep: bool = False) -> dict:
    """`cli.doctor_checks()` as a Windows CPU runner would answer it."""
    bundle = bundle if bundle is not None else windows_bundle(tmp_path)
    monkeypatch.setenv("TYPED_GGUF_RUNTIME_DIR", str(bundle))
    monkeypatch.setenv("TYPED_GGUF_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pins.platform, "system", lambda: "Windows")
    monkeypatch.setattr(pins.shutil, "which", lambda *args, **kwargs: None)  # no nvidia-smi
    if not deep:
        # a PE cannot be dlopened here: the honest "symbols not probed" answer (see the module doc)
        monkeypatch.setenv("TYPED_GGUF_DEEP_PROBE", "0")
    else:
        monkeypatch.delenv("TYPED_GGUF_DEEP_PROBE", raising=False)
        monkeypatch.setattr(capability, "DEFAULT_SCAN",
                            lambda *args, **kwargs: isolated.ProbeScan())
    return cli.doctor_checks()


def checks_of(report: dict) -> dict[str, dict]:
    return {check["id"]: check for check in report["checks"]}


# -------------------------------------------------------------------- the pinned Windows truth
def test_the_simulated_windows_report_is_the_new_pinned_truth(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundle validates: the distribution check names the DLLs and the build is really read."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    checks = checks_of(report)
    assert report["exit_code"] in (0, 2) and report["status"] in ("ok", "warnings"), \
        report["status"]
    assert not [c for c in report["checks"] if c["status"] == "fail"], report["checks"]
    assert checks["runtime.present"]["status"] == "ok", checks["runtime.present"]
    assert checks["runtime.files"]["status"] == "ok", checks["runtime.files"]
    for dll in ("llama.dll", "ggml.dll", "ggml-base.dll"):
        assert dll in checks["runtime.files"]["detail"], checks["runtime.files"]
    assert "libllama.so" not in checks["runtime.files"]["detail"], checks["runtime.files"]
    assert checks["runtime.build"]["status"] == "ok", checks["runtime.build"]
    assert PINNED_TAG in checks["runtime.build"]["detail"], checks["runtime.build"]
    assert checks["runtime.backends"]["status"] == "ok", checks["runtime.backends"]
    assert "cpu" in checks["runtime.backends"]["detail"], checks["runtime.backends"]
    assert "runtime.loadable" not in checks, (
        "a complete Windows bundle must not report E_RUNTIME_MISSING at all: " + str(checks))


def test_the_dlopen_seam_resolves_every_symbol(tmp_path: pathlib.Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed gates do not rest on the skipped probe: with the ABI resolved they still hold."""
    report = simulated_windows_report(tmp_path, monkeypatch, deep=True)
    checks = checks_of(report)
    assert checks["runtime.symbols"]["status"] == "ok", checks["runtime.symbols"]
    assert checks["runtime.files"]["status"] == "ok", checks["runtime.files"]
    assert checks["runtime.build"]["status"] == "ok", checks["runtime.build"]
    module = load_tool()
    assert module.judge(report, exit_code=report["exit_code"]) == []


def test_the_tool_accepts_the_report_and_reports_the_rest_loudly(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = simulated_windows_report(tmp_path, monkeypatch)
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems == [], problems
    warns = [check["id"] for check in report["checks"] if check["status"] == "warn"]
    assert warns, "this simulation has honest warnings (no runtime.json, no registry model)"
    annotation = module.annotation(report)
    assert annotation.startswith("::warning::"), annotation
    for name in warns:
        assert name in annotation, (name, annotation)
    assert "llama.dll" in annotation, "the annotation names the bundle that is there"
    assert "b11026" in annotation, annotation


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


def test_a_distribution_check_that_went_back_to_linux_names_is_refused(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap this card closed must never come back quietly."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    files = checks_of(report)["runtime.files"]
    files["status"], files["detail"] = "fail", "missing libllama.so, libggml.so, libggml-base.so"
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems and any("files" in problem for problem in problems), problems


def test_a_report_with_any_fail_check_is_refused(tmp_path: pathlib.Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A `fail` check anywhere is a broken bundle (A-E1a-3): the step fails, it never annotates."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    entry = checks_of(report)["runtime.symbols"]
    entry["status"], entry["detail"] = "fail", "0/34 required symbols"
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems and any("symbols" in problem for problem in problems), problems


def test_an_unknown_build_number_is_refused(tmp_path: pathlib.Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """`build_number` reads the number or the step is red — never "unknown" and green."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    entry = checks_of(report)["runtime.build"]
    entry["status"], entry["detail"] = "fail", "cannot determine the build number"
    module = load_tool()
    problems = module.judge(report, exit_code=report["exit_code"])
    assert problems and any("build" in problem for problem in problems), problems


def test_an_exit_code_that_drifted_is_refused(tmp_path: pathlib.Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """`doctor`'s exit codes are the contract (A-E1a-3): 1 means a `fail` check was reported."""
    report = simulated_windows_report(tmp_path, monkeypatch)
    module = load_tool()
    problems = module.judge(report, exit_code=1)
    assert problems and any("exit" in problem for problem in problems), problems


def test_a_report_without_the_bundle_checks_is_refused() -> None:
    module = load_tool()
    problems = module.judge({"status": "ok", "exit_code": 0, "checks": []}, exit_code=0)
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
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["ok"] is True
    assert verdict["annotation"] and verdict["annotation"].startswith("::warning::")
    assert verdict["warnings"], verdict


def test_main_refuses_a_doctor_that_could_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_tool()
    monkeypatch.setattr(module, "run_doctor", lambda prefix, cwd: ("not json at all", 2))
    code = module.main(["--cli", "uv run typed-gguf"])
    assert code == 1


def test_the_dll_names_the_workflow_asserts_come_from_the_finder() -> None:
    """The flatten step checks the names the product itself looks for, not invented ones."""
    assert finder.library_names("windows")["llama"] == "llama.dll"
    assert finder.library_names("windows")["ggml"] == "ggml.dll"
    assert finder.library_names("windows")["ggml_base"] == "ggml-base.dll"
    assert finder.library_glob("windows") == "*ggml-*.dll"
    assert finder.tool_names("windows") == ("llama-cli.exe", "llama-fit-params.exe",
                                            "llama-tokenize.exe")


def test_the_tool_is_pure_over_the_report_it_is_handed() -> None:
    """The judge reads a report, never a machine: that is what makes the pin drivable here."""
    module = load_tool()
    assert module.SCHEMA.startswith("typed_gguf.matrix."), module.SCHEMA
    assert module.judge({"status": "failures", "exit_code": 1, "checks": [
        {"id": "runtime.present", "status": "ok", "detail": "/somewhere"},
    ]}, exit_code=1), "a report with only `present` cannot be the pinned truth"
