""".github/workflows/runtime-matrix.yml — the live platform matrix, pinned (card t_f96fed7f).

The workflow is the acceptance harness for SPEC A1/A6 and milestone E1a: it unpacks the *pinned*
llama.cpp bundle per platform and runs the product against it. Its first-ever run (36151935399,
2026-09-25) is this card's whole context: both Linux jobs failed on an assertion that had never
been exercised ("skips are failures" vs two skips that are inherent on a runner), and the Windows
job downloaded a zip without running anything against it ("once E1a lands" — E1a landed long ago).

These gates are the workflow's *shape*, in the order of the card's acceptance criteria. They read
the YAML (and the tools it names) rather than trusting a comment: which commands each platform job
runs, that the skip policy is the named filter's, that no job can quietly download the 4.4 GB Spark
model, that every job is time-bounded, and that the runner/action pins `test_release_publish.py`
mirrors are spelled the same way here.

A workflow gate is not a substitute for the run — the coordinator dispatches the matrix after this
branch lands, and that run is the live proof. This file is what makes the shape reviewable.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from tests.test_release_publish import CHECKOUT, SETUP_UV

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW_PATH = WORKFLOWS / "runtime-matrix.yml"
WORKFLOW = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
CI_PATH = WORKFLOWS / "ci.yml"
#: The five jobs the card's OUTCOME names (a sixth macOS job would double a 0.38 GB download).
JOBS = {
    "linux-cpu": "ubuntu-24.04",
    "linux-vulkan": "ubuntu-24.04",
    "windows-cpu": "windows-2022",
    "macos-metal": "macos-14",
    "macos-engine-smoke": "macos-14",
}
#: The only model any of these jobs may download (the card: 0.5B smoke model, NO Spark).
SMOKE_GGUF = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
#: The bundle the matrix pins (`runtime.lock` / the workflow's own URL).
BUNDLE_TAG = "b11026"
#: The skip filter (AC1) and the two tools the serve steps call (AC2/AC3).
SKIP_FILTER = "tools/matrix_oracle_skips.py"
SERVE_PROBE = "tools/matrix_serve_probe.py"
WARM_WINDOW = "tools/matrix_warm_window.py"
WINDOWS_DOCTOR = "tools/matrix_windows_doctor.py"
OOM_ROW = "tools/matrix_oom_row.py"
REGISTER = "tools/matrix_register_model.py"
SDK_CLIENT = "tools/host_gate_serve_client.py"


def job(name: str) -> dict:
    jobs = WORKFLOW.get("jobs") or {}
    assert name in jobs, f"the matrix lost its {name!r} job: {sorted(jobs)}"
    return jobs[name]


def steps(name: str) -> list[dict]:
    return job(name).get("steps") or []


def runs(name: str) -> list[tuple[str, str]]:
    """`(step name, its shell text)` for every `run:` step of a job."""
    return [(str(step.get("name") or ""), str(step.get("run") or ""))
            for step in steps(name) if "run" in step]


def step_containing(name: str, needle: str) -> str:
    hits = [text for _, text in runs(name) if needle in text]
    assert hits, f"{name} has no run step mentioning {needle!r}:\n" + "\n".join(
        f"  {step_name or '<unnamed>'}" for step_name, _ in runs(name))
    return "\n".join(hits)


def uses(name: str) -> list[str]:
    return [str(step["uses"]) for step in steps(name) if "uses" in step]


# ------------------------------------------------------------------ the harness's own shape
def test_the_five_jobs_are_still_the_five_platforms() -> None:
    assert sorted(WORKFLOW["jobs"]) == sorted(JOBS), sorted(WORKFLOW["jobs"])
    for name, runner in JOBS.items():
        assert job(name).get("runs-on") == runner, (name, job(name).get("runs-on"))


def test_every_job_is_time_bounded() -> None:
    """The card's constraint: "jobs bounded". An unbounded runner burn is a 6 h failure."""
    for name in JOBS:
        assert job(name).get("timeout-minutes"), f"{name} has no timeout-minutes"


def test_the_weekly_and_manual_split_stays_out_of_ci() -> None:
    """The live harness is manual/scheduled on purpose: it downloads a bundle and a model."""
    triggers = WORKFLOW.get("on") or WORKFLOW.get(True) or {}
    assert "workflow_dispatch" in triggers, "the re-run path must stay manual-dispatched"
    assert triggers.get("schedule") == [{"cron": "17 4 * * 1"}], triggers.get("schedule")
    assert set(triggers) == {"workflow_dispatch", "schedule"}, sorted(triggers)
    ci = CI_PATH.read_text(encoding="utf-8")
    assert "llama.cpp/releases/download" not in ci, (
        "the live harness must not be folded into ci.yml (offline suite + lint only): no bundle "
        "download belongs in the trigger that runs on every push")
    assert "workflow_run" not in ci, "ci.yml must not chain onto the live harness"


def test_the_runner_and_action_pins_match_the_rest_of_the_repository() -> None:
    """`test_release_publish.py` owns the pin rationale; the matrix mirrors it, never floats."""
    for name in JOBS:
        assert CHECKOUT in uses(name) and SETUP_UV in uses(name), (name, uses(name))
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "actions/checkout@v4" not in text and "setup-uv@v" not in text.replace(SETUP_UV, "")


def test_the_live_harness_runs_no_pytest() -> None:
    """Offline suite in ci.yml, live probes here: the separation the SPEC keeps."""
    assert "pytest" not in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_the_only_model_downloaded_is_the_small_smoke_one() -> None:
    """The card's constraint in one gate: no 4.4 GB Spark download, ever, in CI."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Spark-X2.5-4B" not in text and ".hermes/models" not in text, (
        "the Spark model is not on a runner and must not be fetched by CI")
    urls = re.findall(r"https?://[^\s\"']+\.gguf", text)
    assert urls, "the smoke GGUF download is what the engine steps need"
    for url in urls:
        assert url.endswith(SMOKE_GGUF), url


def test_every_tool_the_workflow_names_exists() -> None:
    """A workflow that calls a tool that is not in the tree is a red run nobody can explain."""
    for name in JOBS:
        for _, body in runs(name):
            for token in body.split():
                if token.startswith("tools/") and token.endswith(".py"):
                    assert (ROOT / token).is_file(), f"{name} calls {token}, which is missing"


# ------------------------------------------------------------------- AC1: the skip filter
def test_both_linux_jobs_filter_skips_by_name_instead_of_by_substring() -> None:
    bodies = "\n".join(body for name in JOBS for _, body in runs(name))
    assert 'grep -q "SKIP"' not in bodies and "grep -q SKIP" not in bodies, (
        "the assertion that failed run 36151935399 ('skips are failures' over the whole oracle "
        "output) must be gone from the jobs — the skip policy is the named filter's now")
    for name in ("linux-cpu", "linux-vulkan"):
        body = step_containing(name, "verify_runtime_contract.py")
        assert SKIP_FILTER in body, (name, body)
        assert "verify_runtime_contract.py" in body, body
        assert '"$out"' in body or "$out" in body, (
            "the oracle's captured stdout is the filter's input")


def test_the_filter_is_the_tool_this_card_gates() -> None:
    """AC1's RED replay lives in `tests/test_matrix_oracle_skips.py`; this is the wiring."""
    assert (ROOT / SKIP_FILTER).is_file()
    body = step_containing("linux-cpu", "verify_runtime_contract.py")
    assert "python3 " + SKIP_FILTER in body, body


# --------------------------------------------------------- AC2: Windows becomes a real smoke
def test_the_windows_bundle_is_flattened_and_checked_for_the_dlls_the_finder_wants() -> None:
    body = step_containing("windows-cpu", "llama-b11026-bin-win-cpu-x64.zip")
    assert "Expand-Archive" in body, body
    for dll in ("llama.dll", "ggml.dll", "ggml-base.dll"):
        assert dll in body, f"the flatten step must prove {dll} landed at the bundle root"
    assert "TYPED_GGUF_RUNTIME_DIR" in body, (
        "the bundle is consumed through TYPED_GGUF_RUNTIME_DIR (SPEC 2.7)")
    assert "GITHUB_ENV" in body, "later steps need the resolved bundle path"
    assert "/tmp/" not in body, "a Windows runner's /tmp is C:\\tmp: use $env:RUNNER_TEMP"
    assert "once E1a lands" not in body, "the stale comment the first run carried"


def test_the_windows_job_runs_doctor_and_pins_what_it_reports() -> None:
    body = step_containing("windows-cpu", WINDOWS_DOCTOR)
    assert "doctor" in body.lower(), body
    assert (ROOT / WINDOWS_DOCTOR).is_file()
    assert "--json" in body, "the verdict travels as a file the receipt can cite"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "cannot pass on Windows" not in text, (
        "the matrix still claims doctor cannot pass on Windows: card t_8dab8b3a made the "
        "distribution check + build_number platform-aware, so the claim has to follow the code "
        "(tests/test_matrix_windows_doctor.py pins the closed truth)")


def test_the_fake_oom_step_judges_both_worlds_with_their_own_counts() -> None:
    """Card t_8dab8b3a F1: the ladder's `3 placement(s)` was asserted on a CPU-pinned bench row."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "'3 placement(s)' in reason" not in text, (
        "the stale assertion is back: a bench row resolves to the cpu backend, which is CPU-pinned "
        "(card t_55de5779) and has exactly one rung — that assertion could never pass")
    body = step_containing("linux-cpu", OOM_ROW)
    assert (ROOT / OOM_ROW).is_file(), "the step's judge must exist"
    assert "fit_oom_bundle.c" in body, "the step builds the fake-OOM bundle"
    assert "--bench-row" in body and "--bench-exit" in body, body
    assert "--probe-receipt" in body, "the ladder world is what carries the 3-rung claim"
    assert "fit_oom_probe.py" in body, body


def test_the_windows_job_runs_the_engine_end_to_end() -> None:
    body = step_containing("windows-cpu", "bench --suite latency")
    assert "typed-gguf bench" in body, body
    assert SMOKE_GGUF in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "model_load" in step_containing("windows-cpu", "bench --suite latency"), (
        "the report must be read: a bench that prints a table and exits 0 proves nothing")


def test_the_windows_job_boots_a_real_server_and_asserts_the_keep_fallback() -> None:
    """AC2: a real `serve` on Windows, a real HTTP request, and SPEC 2.12's inline fallback."""
    body = step_containing("windows-cpu", SERVE_PROBE)
    assert "Start-Process" in body, "the server must be a real process, not an import"
    assert "typed-gguf" in body and '"serve"' in body, body
    assert "--keep-alive" in body, (
        "a non-zero window is what makes the unsupported-platform branch run (keep-alive 0 "
        "returns before the platform check)")
    assert SERVE_PROBE in body and "--expect inline" in body, body
    assert REGISTER in body, (
        "the served route resolves aliases: the smoke model must be in the registry")
    assert "taskkill" in body or "Stop-Process" in body, "the server must be torn down"
    assert "$LASTEXITCODE" in body or "exit " in body, (
        "a PowerShell step ends on the last command's code: the probe's verdict has to be carried")


# ------------------------------------------------------- AC3: macOS gains the serve story
def test_the_macos_job_installs_the_pinned_sdk_and_runs_the_real_client() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "typesafe-sdk==0.7.1" in text, "the gate measures the pinned SDK wire (SPEC 2.9)"
    body = step_containing("macos-engine-smoke", SDK_CLIENT)
    assert "TYPESAFE_API_KEY" in body, (
        "the SDK insists on a non-empty API key; the server never reads it")
    assert "--base-url" in body, body


def test_the_macos_serve_step_proves_the_warm_host_and_a_clean_keep_stop() -> None:
    body = step_containing("macos-engine-smoke", "typed-gguf serve")
    assert REGISTER in body and "smoke" in body, "the served alias has to exist"
    assert SERVE_PROBE in body and "--expect host" in body, body
    assert "keep stop" in body, "AC3 asks for a clean `keep stop`"
    assert "keep status" in body, "and for the ledger to say the host is gone"
    assert "--server-log" in body, (
        "who answered is a local fact (the TypeSafe projection drops `engine.keep`): the "
        "server's own stream is the evidence")


# ------------------------------------------------------------- AC4: the warm window
def test_the_linux_job_proves_the_warm_window_with_a_short_keep_alive() -> None:
    body = step_containing("linux-cpu", WARM_WINDOW)
    assert "--keep-alive" in body, body
    window = body.split("--keep-alive", 1)[1].split()[0].strip('"')
    assert 15 <= float(window) <= 30, (
        f"the card pins a 15-30 s window (never burn 10 minutes of runner), got {window!r}")
    assert "keep stop" not in body, (
        "the host has to exit by itself: a `keep stop` in this step makes the claim a tautology")


def test_the_warm_window_step_uses_the_smoke_model_the_job_already_downloaded() -> None:
    body = step_containing("linux-cpu", WARM_WINDOW)
    assert "/tmp/smoke.gguf" in body, body
    assert "--json" in body, "the verdict travels as a file the receipt can cite"


@pytest.mark.parametrize("name", sorted(JOBS))
def test_each_job_is_a_list_of_steps(name: str) -> None:
    """A `jobs:` entry without steps is a job that would fail in "Set up job"."""
    assert steps(name), f"{name} has no steps"
    assert all("uses" in step or "run" in step for step in steps(name)), steps(name)
