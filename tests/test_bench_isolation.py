"""E2 FIX (card t_dd62ec29): two bundles, one process — `--backend all` must not abort.

The operator host carries **two** bundles (the pinned CPU one under `TYPED_GGUF_RUNTIME_DIR` and the
installed Vulkan b11026 one under `TYPED_GGUF_BENCH_RUNTIME_DIR`). `bench --suite throughput
--backend all` measures one row per bundle *in one process*, and the process then dies at teardown:
the whole `typed_gguf.bench/v1` report is on stdout, followed by

    double free or corruption (!prev)                                       exit 134

(`.e2e/t_603a35a0-backend-attribution/logs/{before,after}_mixed.raw`, both trees). Single-bundle
runs exit 0/1 normally. The neighbouring card t_603a35a0 fixed *what the report says* (a row whose
own engine log refutes its label is flagged); it left the abort alone, so a CI step that trusts the
exit code reads a complete, publishable report as a crash.

What this file pins (offline: two bundle *directories* on disk plus the fake bench seam, no model):

* **the decision** (`isolation.isolation_needed`): two *distinct* bundle directories cannot share a
  process; one bundle answering two labels, or a single measured backend, still runs in-process.
* **the child command** is this CLI with one backend (`python -m typed_gguf bench … --backend <one>
  --out <report.json> --json`), so the isolated measurement is the documented single-bundle path.
* **the child verdict** (`isolation.run_backend_child`): a child is usable only when it wrote a
  report whose exit code, backend, bundle and scale flags corroborate what the parent asked for. A
  child that aborts *after* writing a complete report — the exact shape of this defect — is not
  usable, and neither is one that never wrote a report, one whose exit code contradicts its own
  report, or one that measured another bundle.
* **the suite wiring**: with two local bundles and the live factory, *every* measured row is
  produced by a child and the parent process never dlopens a library (`test_ctypes_binding`'s
  E1a rule, kept for the bench stack); a broken child fails the report (`ok: false`, a named note)
  instead of a silent `ok: true`.
* **the exit code** stays the report's: `cli._bench` returns 0 when `ok`, 1 when flagged — never
  the 134 the kernel killed the old process with.
* the **determinism suite** (the other suite that measures every backend) isolates the same way.

The live half — the real command, two real bundles — is `tests/test_bench_isolation_live.py`
(`model`-marked) and the operator-box re-run is `docs/evidence/e2_fix_t_dd62ec29_*.md`.
"""
from __future__ import annotations

import json
import pathlib
import signal
import sys
from typing import Any

import pytest

from tests.test_bench import bench_factory
from typed_gguf import cli
from typed_gguf.bench import harness, suites
from typed_gguf.runtime import ctypes_binding

try:
    from typed_gguf.bench import isolation
except ImportError:   # pragma: no cover - a tree that predates this card has no such module:
    isolation = None  # type: ignore[assignment]  # every gate below must fail *there*, per test,
    # not at collection time (the RED of this card's requirement 3 — see `red_pretest.txt`).
else:
    #: The real seam, kept before any test patches `isolation.run_backend_child` with a fake.
    REAL_RUN_BACKEND_CHILD = isolation.run_backend_child

#: The operator's two bundles, as directories: one CPU, one carrying the Vulkan shim.
CPU_DIR = "/work/t603-runtime/b11026-linux-x64-cpu"
VULKAN_DIR = "/var/home/rybens/.local/share/typed-gguf/runtime/b11026-linux-x64-vulkan"


def two_bundle_runtimes(**kwargs: Any) -> dict[str, pathlib.Path]:
    """The mapping `harness.backend_runtimes` returns on the operator's mixed install."""
    return {"cpu": pathlib.Path(CPU_DIR), "vulkan": pathlib.Path(VULKAN_DIR)}


def one_bundle_runtimes(**kwargs: Any) -> dict[str, pathlib.Path]:
    """The single-bundle host: one directory answers both `cpu` and `vulkan`."""
    return {"cpu": pathlib.Path(VULKAN_DIR), "vulkan": pathlib.Path(VULKAN_DIR)}


# ------------------------------------------------------------------- the child seam (a fake CLI)
def cli_options(command: list[str]) -> dict[str, str]:
    """The `--flag value` pairs of a child command, after `[python, -m, typed-gguf, bench]`."""
    options: dict[str, str] = {}
    tokens = list(command[4:])
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token[2:]] = tokens[index + 1]
    return options


def echo_config(command: list[str]) -> dict[str, Any]:
    """The `config` block the real child's report carries: the argv it was handed, echoed back."""
    options = cli_options(command)
    return {
        "suite": options["suite"],
        "model_path": options["model"],
        "backend": options["backend"],
        "runs": int(options["runs"]) if "runs" in options else 5,
        "threads": int(options["threads"]) if "threads" in options else None,
        "kv_type": options.get("kv-type", "auto"),
        "gpu_layers": int(options["gpu-layers"]) if "gpu-layers" in options else None,
        "prefill_sizes": [int(size) for size in options.get("sizes", "256,2048,8192").split(",")],
        "determinism_repeats": 3,
    }


class FakeChild:
    """A `runner` for `isolation.run_backend_child`: writes a report, returns an exit code.

    Emulates the real child closely enough to exercise the parent's verification: the report file
    is written from the command line (like the CLI would), the exit code is the caller's choice,
    and `mutate` can bend either one away from what a well-behaved child would do.
    """

    def __init__(self, *, exit_code: int = 0, row: dict[str, Any] | None = None,
                 ok: bool | None = None, write_report: bool = True,
                 mutate: Any = None, stderr: str = "") -> None:
        self.exit_code = exit_code
        self.row = row
        self.ok = ok
        self.write_report = write_report
        self.mutate = mutate
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(command))
        options = cli_options(command)
        stdout = ""
        if self.write_report:
            backend = options["backend"]
            row = self.row or {
                "backend": backend, "measured": True,
                "runtime_dir": str(CPU_DIR if backend == "cpu" else VULKAN_DIR),
                "warnings": [], "devices": ["CPU"], "device_buffers": {"CPU": 3},
                "effective_backend": "cpu",
            }
            report: dict[str, Any] = {
                "schema": harness.SCHEMA, "suite": options["suite"], "ok": True,
                "backends": [row], "config": echo_config(command),
            }
            if self.ok is not None:
                report["ok"] = self.ok
            if self.mutate is not None:
                self.mutate(report, command)
            stdout = json.dumps(report)          # the real CLI prints the report, then writes it
            pathlib.Path(options["out"]).write_text(stdout)
        return type("Completed", (), {"returncode": self.exit_code, "stdout": stdout,
                                      "stderr": self.stderr})()


def throughput_config(**kwargs: Any) -> harness.BenchConfig:
    fields: dict[str, Any] = {"suite": "throughput", "model_path": "/tmp/fake.gguf",
                              "backend": "all", "runs": 1, "threads": 4, "prefill_sizes": (64,)}
    fields.update(kwargs)
    return harness.BenchConfig(**fields)


@pytest.fixture(autouse=True)
def _two_fake_bundles(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may touch a real bundle (or the network): two directories on disk are enough."""
    monkeypatch.setattr(harness, "backend_runtimes", two_bundle_runtimes)


# ------------------------------------------------------------------------------ the decision
def test_two_distinct_bundle_directories_cannot_share_a_process() -> None:
    assert isolation.isolation_needed(["cpu", "vulkan"], two_bundle_runtimes()) is True


def test_one_bundle_under_two_labels_stays_in_process() -> None:
    """A host with one bundle (`cpu` and `vulkan` the same directory) never needs a child."""
    assert isolation.isolation_needed(["cpu", "vulkan"], one_bundle_runtimes()) is False


def test_a_single_measured_backend_stays_in_process() -> None:
    assert isolation.isolation_needed(["cpu"], two_bundle_runtimes()) is False


def test_a_backend_without_a_bundle_is_not_a_reason_to_isolate() -> None:
    """`--backend all` asks for cuda too: a backend with no local bundle is never loaded."""
    assert isolation.isolation_needed(["cpu", "cuda"], two_bundle_runtimes()) is False


def test_the_same_directory_under_a_symlink_is_still_one_bundle(tmp_path: pathlib.Path) -> None:
    """`_LOADED` keys on the resolved directory; the decision has to agree with the loader."""
    real = tmp_path / "b11026-linux-x64-cpu"
    real.mkdir()
    link = tmp_path / "b11026-linux-x64-vulkan"
    link.symlink_to(real)
    assert isolation.isolation_needed(["cpu", "vulkan"], {"cpu": real, "vulkan": link}) is False


# ------------------------------------------------------------------------------ the child command
def test_the_child_command_is_this_cli_with_one_backend(tmp_path: pathlib.Path) -> None:
    """The child is not a private protocol: it is `bench --backend <one>`, the documented path."""
    config = throughput_config(kv_type="q8_0", gpu_layers=0)
    command = isolation.child_command(config, "vulkan", python="/usr/bin/python3",
                                      out_path=tmp_path / "row.json")
    assert command == [
        "/usr/bin/python3", "-m", "typed_gguf", "bench",
        "--suite", "throughput", "--model", "/tmp/fake.gguf", "--backend", "vulkan",
        "--runs", "1", "--threads", "4", "--sizes", "64",
        "--kv-type", "q8_0", "--gpu-layers", "0",
        "--out", str(tmp_path / "row.json"), "--json",
    ]


def test_the_child_command_omits_what_the_parent_never_set(tmp_path: pathlib.Path) -> None:
    """No `--threads`/`--kv-type`/`--gpu-layers` when the run left them to the defaults."""
    config = throughput_config(threads=None)
    command = isolation.child_command(config, "cpu", python="python",
                                      out_path=tmp_path / "row.json")
    assert "--threads" not in command
    assert "--kv-type" not in command
    assert "--gpu-layers" not in command
    assert command[command.index("--sizes") + 1] == "64"
    model_less = harness.BenchConfig(suite="throughput", backend="all")
    assert "--model" not in isolation.child_command(model_less, "cpu", python="python",
                                                    out_path=tmp_path / "row.json")


def test_the_child_command_lists_every_prefill_size(tmp_path: pathlib.Path) -> None:
    """A multi-size run must be re-measured with its own sizes, comma-joined."""
    command = isolation.child_command(throughput_config(prefill_sizes=(64, 128)), "cpu",
                                      python="python", out_path=tmp_path / "row.json")
    assert command[command.index("--sizes") + 1] == "64,128"


def test_a_run_without_a_model_is_range_checked_before_a_child_starts(
        tmp_path: pathlib.Path) -> None:
    """A child resolves `TYPED_GGUF_BENCH_MODEL` alone: never let it measure an unnamed model."""
    child = isolation.run_backend_child(harness.BenchConfig(suite="throughput", backend="all"),
                                        "cpu", out_dir=tmp_path, python="python",
                                        runner=lambda *args, **kwargs: pytest.fail("spawned"))
    assert child.ok is False
    assert "--model" in child.detail and "TYPED_GGUF_BENCH_MODEL" in child.detail


def test_the_child_command_keeps_the_quick_preset(tmp_path: pathlib.Path) -> None:
    """A quick run states `--quick`: re-stating its scale flags is an `E_BENCH_QUICK` error."""
    config = harness.quick_config(throughput_config())
    command = isolation.child_command(config, "cpu", python="python",
                                      out_path=tmp_path / "row.json")
    assert "--quick" in command
    assert "--runs" not in command
    assert "--sizes" not in command


def test_the_determinism_child_asks_for_the_suite_it_was_told(tmp_path: pathlib.Path) -> None:
    config = harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf", backend="all",
                                 determinism_repeats=3)
    command = isolation.child_command(config, "vulkan", python="python",
                                      out_path=tmp_path / "row.json")
    assert command[command.index("--suite") + 1] == "determinism"
    assert "--sizes" not in command


# ------------------------------------------------------------------------------ the child verdict
def _run(config: harness.BenchConfig, backend: str, child: FakeChild, tmp_path: pathlib.Path,
         runtime_dir: str | None = None) -> isolation.ChildRun:
    """`isolation.run_backend_child` against the fake CLI — through the *real* seam, unpatched."""
    return REAL_RUN_BACKEND_CHILD(
        config, backend, runner=child, out_dir=tmp_path, python="python",
        runtime_dir=runtime_dir if runtime_dir is not None
        else str(two_bundle_runtimes()[backend]))


def test_a_well_behaved_child_is_usable_and_carries_its_report(tmp_path: pathlib.Path) -> None:
    config = throughput_config()
    child = _run(config, "cpu", FakeChild(), tmp_path)
    assert child.ok is True and child.detail is None
    assert child.exit_code == 0
    assert child.row is not None and child.row["backend"] == "cpu"
    assert child.report is not None and child.report["schema"] == harness.SCHEMA
    assert str(tmp_path / "row-cpu.json") in child.command
    assert child.stderr_tail == ""                     # nothing on stderr: nothing is invented
    assert isolation.process_block(child)["isolated"] is True


def test_a_child_that_reports_a_failure_is_still_a_verified_child(tmp_path: pathlib.Path) -> None:
    """A child whose own report is flagged (`ok: false`) exits 1 (SPEC 2.5) — that is not a crash.

    The row it produced is the *child's* row: it is accepted, warnings and all, and the parent's
    own mismatch aggregation is what fails the report (`W_BACKEND_MISMATCH` below).
    """
    config = throughput_config()
    flagged = {"backend": "cpu", "measured": True, "runtime_dir": CPU_DIR,
               "warnings": ["W_BACKEND_MISMATCH"], "effective_backend": "vulkan",
               "devices": ["Vulkan0"], "device_buffers": {"Vulkan0": 3}}
    child = _run(config, "cpu", FakeChild(exit_code=1, ok=False, row=flagged), tmp_path)
    assert child.ok is True and child.detail is None
    assert isolation.process_block(child) == {"isolated": True, "exit_code": 1, "ok": True,
                                              "detail": None}
    assert child.row is not None and child.row["warnings"] == ["W_BACKEND_MISMATCH"]


def test_a_child_that_aborts_after_writing_its_report_is_not_usable(tmp_path: pathlib.Path) -> None:
    """The defect's own shape: a complete report, then `double free`, exit 134 (SIGABRT)."""
    config = throughput_config()
    child = _run(config, "vulkan", FakeChild(exit_code=-signal.SIGABRT,
                                            stderr="double free or corruption (!prev)"), tmp_path)
    assert child.ok is False
    assert child.report is not None                  # the report *was* written: that is the defect
    assert child.row is None                         # …and the parent still publishes nothing
    assert "-6 (SIGABRT; 134 in a shell)" in child.detail
    assert "double free" in child.detail
    assert isolation.process_block(child) == {"isolated": True, "exit_code": -signal.SIGABRT,
                                              "ok": False, "detail": child.detail}


def test_a_child_with_no_report_is_a_failure_not_a_silent_zero(tmp_path: pathlib.Path) -> None:
    child = _run(throughput_config(), "vulkan", FakeChild(write_report=False), tmp_path)
    assert child.ok is False and child.row is None and child.report is None
    assert "exited 0 without writing a report" in child.detail
    assert child.stderr_tail == "" and "stderr tail" not in child.detail
    assert not (tmp_path / "row-vulkan.stdout").exists()   # nothing was on stdout: nothing written


def test_a_child_whose_exit_code_contradicts_its_report_is_not_usable(
        tmp_path: pathlib.Path) -> None:
    """Exit 1 with `ok: true` is exactly the inconsistency a CI step must not read as a pass."""
    child = _run(throughput_config(), "cpu", FakeChild(exit_code=1, ok=True), tmp_path)
    assert child.ok is False
    assert "exited 1 while its own report says `ok: True`" in child.detail
    assert "contradict" in child.detail


def test_a_child_that_names_another_backend_is_not_usable(tmp_path: pathlib.Path) -> None:
    wrong = {"backend": "cpu", "measured": True, "runtime_dir": str(VULKAN_DIR), "warnings": []}
    child = _run(throughput_config(), "vulkan", FakeChild(row=wrong), tmp_path)
    assert child.ok is False
    assert "no row for backend 'vulkan'" in child.detail


def test_a_child_that_measured_another_bundle_is_not_usable(tmp_path: pathlib.Path) -> None:
    """The row must name the bundle the parent selected: a different one is a different run."""
    elsewhere = {"backend": "vulkan", "measured": True, "runtime_dir": "/somewhere/else",
                 "warnings": []}
    child = _run(throughput_config(), "vulkan", FakeChild(row=elsewhere), tmp_path)
    assert child.ok is False and child.row is not None      # the row was read, then withheld
    assert "/somewhere/else" in child.detail and VULKAN_DIR in child.detail
    row = isolation.isolated_row(child, gap={"backend": "vulkan", "measured": False})
    assert row["measured"] is False and row["backend"] == "vulkan"
    assert "devices" not in row            # a withheld row publishes nothing the child claimed


@pytest.mark.parametrize("suite, field, value, expected", [
    ("throughput", "runs", 5, "runs"),
    ("throughput", "prefill_sizes", [256, 2048, 8192], "prefill_sizes"),
    ("determinism", "determinism_repeats", 2, "determinism_repeats"),
    ("determinism", "runs", 1, "runs"),
])
def test_a_child_that_ran_other_scale_flags_is_not_usable(
        tmp_path: pathlib.Path, suite: str, field: str, value: Any, expected: str) -> None:
    """The report is the run the parent asked for, or it is not published at all."""
    fields: dict[str, Any] = {"suite": suite, "model_path": "/tmp/fake.gguf", "backend": "all"}
    if suite == "throughput":
        fields.update(runs=1, prefill_sizes=(64,))
    else:
        fields.update(runs=5)
    config = harness.BenchConfig(**fields)
    child = _run(config, "cpu",
                 FakeChild(mutate=lambda report, _cmd: report["config"].update({field: value})),
                 tmp_path)
    assert child.ok is False
    assert expected in child.detail
    assert f"{field}={value}," in child.detail          # the echo that disagreed is quoted


def test_the_child_report_echoing_the_asked_run_stays_usable(tmp_path: pathlib.Path) -> None:
    config = harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf", backend="all")
    child = _run(config, "cpu", FakeChild(), tmp_path)
    assert child.ok is True


def test_the_child_report_lives_in_a_temp_directory_the_parent_removes(
        tmp_path: pathlib.Path) -> None:
    """Scratch by default: only an evidence run that names `out_dir` keeps the child's raws."""
    child = REAL_RUN_BACKEND_CHILD(throughput_config(), "cpu", runner=FakeChild(), python="python")
    out_path = pathlib.Path(cli_options(list(child.command))["out"])
    assert not out_path.exists() and not out_path.parent.exists()
    kept = _run(throughput_config(), "cpu", FakeChild(), tmp_path)
    assert pathlib.Path(cli_options(list(kept.command))["out"]).is_file()


def test_a_broken_child_keeps_its_scratch_directory_and_its_logs(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A crash's evidence is the child's own log: keep it and name it in the row's reason."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    child = REAL_RUN_BACKEND_CHILD(
        throughput_config(), "vulkan", python="python",
        runner=FakeChild(exit_code=-signal.SIGABRT, stderr="free(): double free (!prev)"))
    out_path = pathlib.Path(cli_options(list(child.command))["out"])
    assert out_path.parent.is_dir()
    assert "free(): double free (!prev)" in child.detail
    assert str(out_path.parent) in child.detail
    stderr = out_path.parent / "row-vulkan.stderr"
    assert stderr.read_text(encoding="utf-8").strip() == "free(): double free (!prev)"
    # the child's stdout — the report it printed — is kept with it
    assert "typed_gguf.bench" in (out_path.parent / "row-vulkan.stdout").read_text(encoding="utf-8")

    # a child with nothing on stdout gets no stdout file: the kept scratch is the evidence, not
    # a placeholder for evidence that does not exist
    quiet = REAL_RUN_BACKEND_CHILD(throughput_config(), "cpu", python="python",
                                   runner=FakeChild(write_report=False, stderr="boom"))
    quiet_out = pathlib.Path(cli_options(list(quiet.command))["out"])
    assert (quiet_out.parent / "row-cpu.stderr").is_file()
    assert not (quiet_out.parent / "row-cpu.stdout").exists()


def test_a_child_runs_the_interpreter_the_parent_was_started_with(tmp_path: pathlib.Path) -> None:
    """`python=None` (production) means *this* interpreter; an explicit one is passed through."""
    default = REAL_RUN_BACKEND_CHILD(throughput_config(), "cpu", runner=FakeChild(), python=None,
                                     out_dir=tmp_path)
    assert default.command[0] == sys.executable
    explicit = REAL_RUN_BACKEND_CHILD(throughput_config(), "cpu", runner=FakeChild(),
                                      python="/custom/python", out_dir=tmp_path)
    assert explicit.command[0] == "/custom/python"


def test_a_report_that_is_not_an_object_is_not_a_report(tmp_path: pathlib.Path) -> None:
    """`--out` holds the report object; a JSON list is not one, and the row is withheld."""
    def list_report(command: list[str], **kwargs: Any) -> Any:
        pathlib.Path(cli_options(command)["out"]).write_text("[1, 2]", encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    child = _run(throughput_config(), "cpu", list_report, tmp_path)
    assert child.ok is False and child.report is None
    assert "without writing a report" in child.detail


def test_a_child_that_never_started_is_a_failure(tmp_path: pathlib.Path) -> None:
    """A missing interpreter (or a runner that cannot exec) is a withheld row, not an exception."""
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("no such file or directory: 'python'")

    child = _run(throughput_config(), "cpu", refuse, tmp_path)
    assert child.ok is False and child.exit_code is None
    assert "E_BENCH_CHILD" in child.detail and "never started" in child.detail


def test_a_child_report_without_a_config_block_cannot_be_checked(tmp_path: pathlib.Path) -> None:
    """The parent verifies the run it asked for: a report that does not echo it proves nothing."""
    child = _run(throughput_config(), "cpu",
                 FakeChild(mutate=lambda report, _cmd: report.pop("config")), tmp_path)
    assert child.ok is False
    assert "the report carries no `config` block" in child.detail
    assert child.detail.endswith("(the child's run cannot be checked)")


def test_a_long_child_log_is_tailed_into_the_reason(tmp_path: pathlib.Path) -> None:
    """The report carries the *end* of the child's log — a glibc abort line is the evidence."""
    child = _run(throughput_config(), "cpu",
                 FakeChild(stderr="x" * 500 + "\ndouble free or corruption (!prev)\n"), tmp_path)
    assert child.ok is True and len(child.stderr_tail) == 201      # 200 chars + the ellipsis
    assert child.stderr_tail.startswith("…") and child.stderr_tail.endswith("(!prev)")


def test_a_child_log_at_the_cap_is_kept_whole(tmp_path: pathlib.Path) -> None:
    """Exactly `STDERR_TAIL_CHARS` is the whole tail: the ellipsis is for what was cut."""
    child = _run(throughput_config(), "cpu", FakeChild(stderr="y" * 200), tmp_path)
    assert child.stderr_tail == "y" * 200 and not child.stderr_tail.startswith("…")


def test_the_note_of_a_child_that_never_started_says_so(tmp_path: pathlib.Path) -> None:
    child = REAL_RUN_BACKEND_CHILD(harness.BenchConfig(suite="throughput", backend="all"), "cpu",
                                   out_dir=tmp_path, python="python",
                                   runner=lambda *args, **kwargs: pytest.fail("spawned"))
    row = isolation.isolated_row(child, gap={"backend": "cpu", "measured": False})
    note = isolation.isolation_note(row)
    assert "that child exited never started:" in note
    assert "cpu" in note and "withheld" in note


def test_the_child_stderr_travels_with_the_verdict(tmp_path: pathlib.Path) -> None:
    child = _run(throughput_config(), "cpu", FakeChild(stderr="last words\n"), tmp_path)
    assert child.stderr_tail.strip() == "last words"


def test_the_row_published_for_an_isolated_child_carries_its_process_block(
        tmp_path: pathlib.Path) -> None:
    child = _run(throughput_config(), "cpu", FakeChild(), tmp_path)
    row = isolation.isolated_row(child, gap={"backend": "cpu", "measured": False})
    assert row["measured"] is True and row["runtime_dir"] == CPU_DIR
    assert row["process"]["isolated"] is True and row["process"]["ok"] is True


def test_the_gap_row_withholds_everything_the_child_did_not_prove(tmp_path: pathlib.Path) -> None:
    child = _run(throughput_config(), "vulkan", FakeChild(write_report=False), tmp_path)
    row = isolation.isolated_row(child, gap={"backend": "vulkan", "measured": False})
    assert row["measured"] is False and row["backend"] == "vulkan"
    assert "without writing a report" in row["reason"]
    assert row["process"]["ok"] is False
    assert "devices" not in row and "effective_backend" not in row


# ------------------------------------------------------------------------------ the suite wiring
def test_a_two_bundle_run_puts_every_measured_row_in_its_own_process(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    child = FakeChild()
    monkeypatch.setattr(isolation, "run_backend_child",
                        lambda config, backend, **kwargs: _run(config, backend, child, tmp_path))
    monkeypatch.setattr(
        suites, "live_factory",
        lambda spec: pytest.fail("the parent process loaded a model: it must not touch a bundle"))

    report = suites.run_suite(throughput_config())

    # `--backend all` also asks for cuda: no local bundle, so it is reported, never loaded
    assert [row["backend"] for row in report["backends"]] == ["cpu", "vulkan", "cuda"]
    assert [row["backend"] for row in report["backends"] if row.get("measured")] == \
        ["cpu", "vulkan"]
    assert all(row["process"]["isolated"] for row in report["backends"] if row.get("measured"))
    assert report["ok"] is True
    assert report["isolation"]["one_bundle_per_process"] is True
    assert report["isolation"]["bundles"] == {"cpu": CPU_DIR, "vulkan": VULKAN_DIR}
    assert report["isolation"]["suite"] == "throughput"
    assert report["isolation"]["reason"] == isolation.BUNDLE_ISOLATION_REASON
    assert any("one bundle per process" in note for note in report["notes"])
    assert any("over 2 distinct local bundle directories" in note for note in report["notes"])
    assert [cli_options(command)["backend"] for command in child.calls] == ["cpu", "vulkan"]


def test_the_parent_process_never_dlopens_a_bundle_when_it_isolates(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """E1a's rule for the bench stack: `--backend all` with two bundles loads both elsewhere."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the command process dlopened a bundle; the child must do that")

    monkeypatch.setattr(ctypes_binding, "load_libraries", refuse)
    monkeypatch.setattr(isolation, "run_backend_child",
                        lambda config, backend, **kwargs: _run(config, backend, FakeChild(),
                                                               tmp_path))
    report = suites.run_suite(throughput_config())
    assert report["ok"] is True


def test_a_broken_child_fails_the_report_and_names_the_backend(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(isolation, "run_backend_child",
                        lambda config, backend, **kwargs: _run(
                            config, backend,
                            FakeChild(exit_code=-signal.SIGABRT, stderr="double free"),
                            tmp_path) if backend == "vulkan" else _run(
                            config, backend, FakeChild(), tmp_path))
    report = suites.run_suite(throughput_config())
    rows = {row["backend"]: row for row in report["backends"]}
    assert rows["cpu"]["measured"] is True
    assert rows["vulkan"]["measured"] is False
    assert report["ok"] is False
    assert any("vulkan" in note and "SIGABRT" in note for note in report["notes"])


def test_a_mismatch_the_child_itself_reports_still_fails_the_parent_report(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The child's flagged report is *verified* (exit 1 + `ok: false`) and merged as it stands."""
    second = {"backend": "vulkan", "measured": True, "runtime_dir": VULKAN_DIR,
              "warnings": ["W_BACKEND_MISMATCH"], "effective_backend": "cpu",
              "devices": ["CPU"], "device_buffers": {"CPU": 3}}
    monkeypatch.setattr(isolation, "run_backend_child",
                        lambda config, backend, **kwargs: _run(
                            config, backend,
                            FakeChild(exit_code=1, ok=False, row=second) if backend == "vulkan"
                            else FakeChild(), tmp_path))
    report = suites.run_suite(throughput_config())
    assert report["ok"] is False
    assert any("W_BACKEND_MISMATCH" in note for note in report["notes"])
    flagged = [row for row in report["backends"] if row["backend"] == "vulkan"][0]
    assert flagged["measured"] is True and flagged["process"]["exit_code"] == 1
    assert flagged["process"]["ok"] is True          # verified, *not* withheld


def test_a_single_bundle_host_keeps_measuring_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing changes for the common host: one bundle, one process, the fake seam as before.

    The mismatch gate still fires there (both rows load the *same* library set), which is what the
    neighbour card t_603a35a0 built this path on.
    """
    monkeypatch.setattr(harness, "backend_runtimes", one_bundle_runtimes)
    monkeypatch.setattr(isolation, "run_backend_child",
                        lambda *args, **kwargs: pytest.fail("one bundle does not need a child"))
    report = suites.run_suite(throughput_config(), factory=bench_factory())
    assert [row["backend"] for row in report["backends"] if row.get("measured")] == \
        ["cpu", "vulkan"]
    assert "isolation" not in report
    assert all("process" not in row for row in report["backends"])


def test_the_quick_preset_resolves_one_backend_and_never_isolates(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`--quick` cuts the measured list to one backend before the decision: still one process."""
    quick = harness.quick_config(throughput_config())
    assert quick.backend_limit == 1, "the preset's own cut is what keeps the quick path in-process"
    monkeypatch.setattr(isolation, "run_backend_child",
                        lambda *args, **kwargs: pytest.fail("--quick measures one backend"))
    report = suites.run_suite(quick, factory=bench_factory())
    assert report["ok"] is True
    assert [row["backend"] for row in report["backends"] if row["measured"]] == ["cpu"]
    assert "isolation" not in report


def test_the_determinism_suite_isolates_its_backends_too(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:

    def child(config: harness.BenchConfig, backend: str, **kwargs: Any):
        row = {"backend": backend, "runtime_dir": str(CPU_DIR if backend == "cpu" else VULKAN_DIR),
               "threads": 1, "digests": ["sha256:aa", "sha256:aa", "sha256:aa"], "identical": True,
               "ok": True, "repeats": 3, "devices": ["CPU"], "device_buffers": {"CPU": 1},
               "effective_backend": "cpu", "warnings": []}
        return _run(config, backend, FakeChild(row=row), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    config = harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf", backend="all")
    report = suites.run_suite(config)
    assert [row["backend"] for row in report["backends"]] == ["cpu", "vulkan"]
    assert all(row["process"]["isolated"] for row in report["backends"])
    assert all(row["identical"] for row in report["backends"])
    assert report["ok"] is True


def test_a_broken_determinism_child_fails_the_report(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:

    def child(config: harness.BenchConfig, backend: str, **kwargs: Any):
        if backend == "vulkan":
            return _run(config, backend, FakeChild(write_report=False), tmp_path)
        row = {"backend": "cpu", "runtime_dir": CPU_DIR, "threads": 1, "digests": ["sha256:aa"],
               "identical": True, "ok": True, "repeats": 1, "warnings": []}
        return _run(config, backend, FakeChild(row=row), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    config = harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf", backend="all")
    report = suites.run_suite(config)
    assert report["ok"] is False
    assert any("vulkan" in note for note in report["notes"])


# ------------------------------------------------------------------------------ the exit code
def test_the_cli_exit_code_is_the_report_not_the_signal(monkeypatch: pytest.MonkeyPatch,
                                                        tmp_path: pathlib.Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """A withheld row is a flagged report: exit 1, never the 134 the process used to die with."""
    model = tmp_path / "model.gguf"
    model.write_text("not a real model: the CLI only checks that the path exists")
    crashed = {"schema": harness.SCHEMA, "suite": "throughput", "ok": False, "backends": [],
               "notes": []}
    monkeypatch.setattr(suites, "run_suite", lambda config, **kwargs: crashed)
    code = cli._cmd_bench(["--suite", "throughput", "--model", str(model), "--backend", "all",
                           "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_a_clean_two_bundle_report_exits_zero(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: pathlib.Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("not a real model: the CLI only checks that the path exists")
    clean = {"schema": harness.SCHEMA, "suite": "throughput", "ok": True, "backends": [],
             "notes": []}
    monkeypatch.setattr(suites, "run_suite", lambda config, **kwargs: clean)
    assert cli._cmd_bench(["--suite", "throughput", "--model", str(model), "--backend", "all",
                           "--json"]) == 0
