"""E2 FIX (card t_57cc0179): a child that SIGSEGVs at teardown must not cost the row.

Found while landing t_dd62ec29 and explicitly left unfixed there
(`.e2e/t_dd62ec29-mixed-bundle-teardown/logs/after_mixed.raw`): a **single** Vulkan-bundle child
can die with **exit -11 (SIGSEGV)** *after* writing a complete `typed_gguf.bench/v1` report, while
own report says `ok: true`. The device was memory-starved at the time (the operator host runs the
E3 campaign on the same box: `vram_before_controls.txt` reads ~3.2 GiB free of 8 GiB); the same
bundle alone exits 0 on a free device. The row is withheld, `ok: false`, exit 1 — the containment
of t_dd62ec29, kept here — but the measurement is lost and the reader cannot tell *why*.

What this file pins (offline: the fake child CLI of `tests/test_bench_isolation.py`, no model):

* **the shape** (`isolation.teardown_crash`): a fatal signal (-11/-6) *plus* a report the child did
  write is a crash at teardown; a clean exit, a flagged child (exit 1), a child that wrote nothing
  and a killing signal that is not a crash of the engine (-9) are not.
* **the starvation reading** (`isolation.device_memory`, `…starved`): the driver's free/total
  device memory, and whether the model's weights plus the `--fit-target` margin (1 GiB, SPEC 2.4)
  could fit at all. Unknown when there is no device or no model size — never guessed as zero.
* **the one retry**: a teardown crash is re-measured **once**, in a second child, with a degraded
  placement taken from the loader's own `fit.degrade_ladder` (fewer layers, or a smaller KV rung
  when the weights are already on the host). The retry is the same documented `bench --backend
  <one>` command with the same scale flags — only the placement moves — and it is verified exactly
  like the first attempt.
* **the named warning** (`W_BACKEND_CRASHED_AT_TEARDOWN`): when the retry crashes too, the row
  stays withheld, carries the warning name with the child's exit code and the free-VRAM reading,
  and the sentence reaches the **rendered table** (a table cell), not only the report's `notes`.
* **no silent row loss**: every backend of a report keeps a line in the rendered table.

The live half — the real SIGSEGV under a starving device and the repro recipe — is
`docs/evidence/e2_fix_t_57cc0179_vulkan_teardown_crash.md`.
"""
from __future__ import annotations

import pathlib
import signal
from typing import Any

import pytest

from tests.test_bench import bench_factory
from tests.test_bench_isolation import (
    CPU_DIR,
    VULKAN_DIR,
    FakeChild,
    cli_options,
    throughput_config,
    two_bundle_runtimes,
)
from typed_gguf.bench import harness, isolation, suites

GIB = 1024 ** 3
MIB = 1024 ** 2
#: the operator box: an 8 GiB board, ~3.2 GiB free while the E3 campaign holds the rest
BOARD = 8 * GIB
STARVED = 3 * GIB
#: the 4B model's weights (the header read of the real file)
WEIGHTS = 2572253184
#: the real seam, kept before any test patches `isolation.run_backend_child` with a fake
REAL_RUN_BACKEND_CHILD = isolation.run_backend_child


class Fact:
    """A `fit.ModelFacts`-like answer: the two fields the retry placement reads."""

    def __init__(self, n_layer: int = 32, weights_bytes: int = WEIGHTS) -> None:
        self.n_layer = n_layer
        self.weights_bytes = weights_bytes


@pytest.fixture(autouse=True)
def _two_fake_bundles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "backend_runtimes", two_bundle_runtimes)


def roomy() -> tuple[int, int]:
    return (BOARD, 7 * GIB)


def starving() -> tuple[int, int]:
    return (BOARD, STARVED)


class ScriptedChild:
    """A runner that plays one `FakeChild` per spawn (the last one repeats)."""

    def __init__(self, *children: FakeChild) -> None:
        self.children = list(children)
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(command))
        child = self.children[min(len(self.calls) - 1, len(self.children) - 1)]
        return child(command, **kwargs)


def run_child(config: harness.BenchConfig, backend: str, runner: Any, tmp_path: pathlib.Path,
              **kwargs: Any) -> isolation.ChildRun:
    """`run_backend_child` through the real seam, with the device probe + model facts injected."""
    kwargs.setdefault("probe", starving)
    kwargs.setdefault("facts", Fact())
    return REAL_RUN_BACKEND_CHILD(
        config, backend, runner=runner, out_dir=tmp_path, python="python",
        runtime_dir=str(two_bundle_runtimes()[backend]), **kwargs)


def attempts_of(child: isolation.ChildRun) -> list[dict[str, Any]]:
    """The `Attempt` records of a verdict, as the report's `process.attempts` carries them."""
    return [attempt.to_dict() for attempt in child.attempts]


def crashing(exit_code: int = -signal.SIGSEGV, stderr: str = "") -> FakeChild:
    """The defect's shape: a complete report, then a fatal signal."""
    return FakeChild(exit_code=exit_code, stderr=stderr or "ggml_vulkan: device lost")


# ------------------------------------------------------------------ the device reading
def test_device_memory_reads_the_driver_probe() -> None:
    memory = isolation.device_memory(probe=starving)
    assert (memory.total_bytes, memory.free_bytes, memory.known) == (BOARD, STARVED, True)
    assert memory.sentence() == "3072 MiB free of 8192 MiB"


def test_no_device_is_unknown_never_zero() -> None:
    """A box without a device (or a driver that answers nothing) has an *unknown* reading."""
    memory = isolation.device_memory(probe=lambda: None)
    assert memory.known is False
    assert "unknown" in memory.sentence()
    assert isolation.device_memory(probe=lambda: (BOARD, 0)).known is True


# ------------------------------------------------------------------ the shape
def test_a_fatal_signal_with_a_complete_report_is_the_teardown_crash() -> None:
    memory = isolation.device_memory(probe=starving)
    crash = isolation.teardown_crash(-signal.SIGSEGV, report={"ok": True}, memory=memory,
                                     weights_bytes=WEIGHTS)
    assert crash is not None
    assert crash.exit_code == -signal.SIGSEGV
    assert crash.starved is True
    assert "3072 MiB free of 8192 MiB at the attempt's start" in crash.sentence()
    assert "starved" in crash.sentence()
    assert crash.to_dict()["signal"] == "SIGSEGV"


@pytest.mark.parametrize("exit_code, report", [
    (0, {"ok": True}),          # a clean child: nothing to recover
    (1, {"ok": False}),         # a flagged child: the child answered, and its row is published
    (1, {"ok": True}),          # the exit-contradicts-report case: a different defect
    (-signal.SIGKILL, {"ok": True}),   # the OOM killer / a human, not an engine crash
])
def test_other_failures_are_not_the_teardown_crash(exit_code: int, report: dict[str, Any]) -> None:
    memory = isolation.device_memory(probe=starving)
    assert isolation.teardown_crash(exit_code, report=report, memory=memory,
                                    weights_bytes=WEIGHTS) is None


def test_a_child_that_wrote_no_report_is_not_this_shape() -> None:
    """`exit -11` with nothing on `--out` is a crash *before* the measurement, not at teardown."""
    memory = isolation.device_memory(probe=starving)
    assert isolation.teardown_crash(-signal.SIGSEGV, report=None, memory=memory,
                                    weights_bytes=WEIGHTS) is None


def test_the_crash_is_starved_only_when_the_weights_cannot_fit() -> None:
    """The shape's "low free VRAM at start": the model's weights + the fit margin (SPEC 2.4)."""
    crashing_child = -signal.SIGSEGV
    starved = isolation.teardown_crash(
        crashing_child, report={"ok": True}, memory=isolation.device_memory(probe=starving),
        weights_bytes=WEIGHTS)
    roomy_child = isolation.teardown_crash(
        crashing_child, report={"ok": True}, memory=isolation.device_memory(probe=roomy),
        weights_bytes=WEIGHTS)
    unknown = isolation.teardown_crash(
        crashing_child, report={"ok": True}, memory=isolation.device_memory(probe=lambda: None),
        weights_bytes=WEIGHTS)
    assert starved is not None and starved.starved is True
    assert roomy_child is not None and roomy_child.starved is False
    assert unknown is not None and unknown.starved is None
    assert "weights" in unknown.sentence()


# ------------------------------------------------------------------ the retry placement
def test_the_retry_halves_the_layers_that_really_ran() -> None:
    """`--gpu-layers` unset on a GPU backend means "all of them" (llama.cpp's `< 0`)."""
    placement = isolation.retry_placement(throughput_config(), "vulkan", facts=Fact(n_layer=32))
    assert placement.n_gpu_layers == 16
    assert placement.degraded is True
    assert "16" in placement.note and "32" in placement.note


def facts_for(n_layer: int = 32) -> Any:
    """A complete `fit.ModelFacts` (no model file needed): the ladder reads every field."""
    from typed_gguf.runtime import fit

    return fit.ModelFacts(path="/tmp/fake.gguf", sha256="", arch="qwen35", n_layer=n_layer,
                          n_kv_head=4, key_len=128, value_len=128, n_ctx_train=4096,
                          weights_bytes=WEIGHTS, file_size=WEIGHTS)


@pytest.mark.parametrize("n_layer", [32, 7, 1])
def test_the_retry_placement_is_the_loaders_own_degrade_rung(n_layer: int) -> None:
    """The number is not invented here: it is the rung `fit.degrade_ladder` walks to next."""
    from types import SimpleNamespace

    from typed_gguf.runtime import fit

    facts = facts_for(n_layer)
    config = harness.BenchConfig(suite="throughput", model_path="/tmp/fake.gguf", backend="vulkan")
    rungs = fit.degrade_ladder(fit.coerce_plan(SimpleNamespace(n_gpu_layers=-1, kv_type="auto")),
                               facts)
    assert rungs[0].n_gpu_layers == max(1, n_layer // 2)
    assert isolation.retry_placement(config, "vulkan", facts=facts).n_gpu_layers == \
        max(0, n_layer // 2)
    # the KV rung is the ladder's too, for a placement that already asks for no weights
    host = harness.BenchConfig(suite="throughput", model_path="/tmp/fake.gguf", backend="vulkan",
                               gpu_layers=0)
    kv_rungs = fit.degrade_ladder(fit.coerce_plan(SimpleNamespace(n_gpu_layers=0, kv_type="auto")),
                                  facts)
    assert isolation.retry_placement(host, "vulkan", facts=facts).kv_type == kv_rungs[0].kv_type


def test_the_retry_reads_what_really_ran_out_of_the_childs_own_report() -> None:
    """The first child wrote a complete report: its own placement is the rung to degrade from."""
    row = {"placement": "n_gpu_layers=20", "placement_used": {"used": {"n_gpu_layers": 20}}}
    placement = isolation.retry_placement(throughput_config(), "vulkan", first_row=row,
                                          facts=Fact(n_layer=32))
    assert placement.n_gpu_layers == 10


def test_the_retry_downgrades_the_kv_type_when_the_weights_are_already_on_the_host() -> None:
    """At `n_gpu_layers=0` there are no layers left to drop: the ladder's next rung is the KV."""
    row = {"placement": "n_gpu_layers=0", "placement_used": {"used": {"n_gpu_layers": 0}}}
    placement = isolation.retry_placement(throughput_config(), "vulkan", first_row=row,
                                          facts=Fact())
    assert (placement.n_gpu_layers, placement.kv_type, placement.degraded) == (0, "q8_0", True)


def test_a_placement_already_at_the_bottom_rung_is_retried_unchanged() -> None:
    """0 layers + the smallest KV rung: nothing left to degrade, and the card asks for one retry."""
    row = {"placement_used": {"used": {"n_gpu_layers": 0}}}
    placement = isolation.retry_placement(throughput_config(kv_type="q4_0"), "vulkan",
                                          first_row=row, facts=Fact())
    assert placement.degraded is False
    assert (placement.n_gpu_layers, placement.kv_type) == (0, "q4_0")
    assert "already" in placement.note


def test_an_unreadable_model_still_gets_a_degraded_retry() -> None:
    """No header to count the layers from: the rung that asks the device for nothing."""
    placement = isolation.retry_placement(throughput_config(), "vulkan", facts=None)
    assert placement.degraded is True
    assert (placement.n_gpu_layers, placement.kv_type) == (0, "q8_0")


def test_the_retry_placement_travels_as_a_bench_config() -> None:
    """The retry is the documented `bench --backend <one>` command: a config, not a private argv."""
    config = throughput_config()
    placement = isolation.retry_placement(config, "vulkan", facts=Fact())
    retry = isolation.retry_config(config, "vulkan", placement)
    options = cli_options(isolation.child_command(retry, "vulkan", python="python",
                                                 out_path=pathlib.Path("/tmp/row.json")))
    assert options["gpu-layers"] == "16" and options["backend"] == "vulkan"
    assert options["suite"] == "throughput" and options["runs"] == "1"
    assert options["sizes"] == "64" and options["threads"] == "4"
    assert config.gpu_layers is None and config.kv_type == "auto"    # the run itself is untouched


# ------------------------------------------------------------------ the retry
def test_a_teardown_crash_is_retried_once_with_the_degraded_placement(
        tmp_path: pathlib.Path) -> None:
    config = throughput_config()
    child = ScriptedChild(crashing(-signal.SIGSEGV, stderr="ggml_vulkan: device lost"),
                          FakeChild())
    result = run_child(config, "vulkan", child, tmp_path)

    assert result.ok is True and result.detail is None
    assert result.row is not None and result.row["backend"] == "vulkan"
    assert len(child.calls) == 2
    first, retry = (cli_options(call) for call in child.calls)
    assert "gpu-layers" not in first                                     # the run's own placement
    assert retry["gpu-layers"] == "16"                                   # the ladder's first rung
    assert retry.get("kv-type", "auto") == "auto"                        # `auto` stays unpinned
    assert retry["backend"] == "vulkan" and retry["suite"] == "throughput"
    assert retry["out"] != first["out"]                                  # the first report is kept

    walked = attempts_of(result)
    assert [attempt["exit_code"] for attempt in walked] == [-signal.SIGSEGV, 0]
    assert [attempt["ok"] for attempt in walked] == [False, True]
    assert walked[0]["free_bytes"] == STARVED
    assert walked[0]["total_bytes"] == BOARD
    assert walked[0]["placement"]["n_gpu_layers"] is None       # what the run asked for
    assert walked[1]["placement"]["n_gpu_layers"] == 16         # what the retry asked for
    block = isolation.process_block(result)
    assert block["ok"] is True and block["exit_code"] == 0
    assert [attempt["exit_code"] for attempt in block["attempts"]] == [-signal.SIGSEGV, 0]


def test_a_recovered_row_is_a_measured_row_with_no_warning(tmp_path: pathlib.Path) -> None:
    """The retry measured it: the row is published, and the crash travels as its `process` block."""
    child = ScriptedChild(crashing(), FakeChild())
    result = run_child(throughput_config(), "vulkan", child, tmp_path)
    row = isolation.isolated_row(result, gap={"backend": "vulkan", "measured": False})
    assert row["measured"] is True
    assert row.get("warnings") == [] or "warnings" not in row
    assert "crash" in row["process"]
    assert row["process"]["crash"]["exit_code"] == -signal.SIGSEGV


def test_a_clean_child_is_never_retried(tmp_path: pathlib.Path) -> None:
    child = ScriptedChild(FakeChild())
    result = run_child(throughput_config(), "vulkan", child, tmp_path)
    assert result.ok is True
    assert len(child.calls) == 1
    assert len(result.attempts) == 1
    assert "attempts" not in isolation.process_block(result)


@pytest.mark.parametrize("child", [
    FakeChild(write_report=False),                                # never wrote a report
    FakeChild(exit_code=0, mutate=lambda report, _cmd: report.update(ok=False)),  # 0 + ok: false
    FakeChild(exit_code=1, ok=True),                              # exit contradicts the report
    FakeChild(exit_code=-signal.SIGKILL),                         # killed, not an engine crash
])
def test_a_failure_that_is_not_the_shape_is_not_retried(tmp_path: pathlib.Path,
                                                        child: FakeChild) -> None:
    scripted = ScriptedChild(child, FakeChild())
    result = run_child(throughput_config(), "vulkan", scripted, tmp_path)
    assert result.ok is False
    assert len(scripted.calls) == 1, "only the teardown-crash shape is worth a second child"


def test_the_retry_happens_exactly_once(tmp_path: pathlib.Path) -> None:
    child = ScriptedChild(crashing(-signal.SIGSEGV), crashing(-signal.SIGABRT))
    result = run_child(throughput_config(), "vulkan", child, tmp_path)
    assert result.ok is False
    assert len(child.calls) == 2, "the card asks for one retry, not a loop"
    assert [attempt["exit_code"] for attempt in attempts_of(result)] == [-signal.SIGSEGV,
                                                                        -signal.SIGABRT]


def test_a_failed_retry_keeps_the_containment_and_names_the_warning(
        tmp_path: pathlib.Path) -> None:
    config = throughput_config()
    child = ScriptedChild(crashing(-signal.SIGSEGV, stderr="ggml_vulkan: device lost"),
                          crashing(-signal.SIGABRT, stderr="free(): invalid pointer"))
    result = run_child(config, "vulkan", child, tmp_path)

    assert result.ok is False and result.row is None
    assert result.warning == isolation.W_BACKEND_CRASHED_AT_TEARDOWN
    assert "-11 (SIGSEGV; 139 in a shell)" in result.detail
    assert "-6 (SIGABRT; 134 in a shell)" in result.detail
    assert "3072 MiB free of 8192 MiB" in result.detail      # the reading of the failed attempt
    row = isolation.isolated_row(result, gap={"backend": "vulkan", "measured": False})
    assert row["measured"] is False and row["backend"] == "vulkan"
    assert row["warnings"] == [isolation.W_BACKEND_CRASHED_AT_TEARDOWN]
    assert "devices" not in row                       # nothing the child claimed is published
    assert isolation.broken_rows([row]) == [row]
    # the containment's own shape is kept: a withheld row's block is `detail` + the exit code, and
    # the whole story (both attempts, both readings) is the sentence the table renders
    assert set(row["process"]) == {"isolated", "exit_code", "ok", "detail"}
    note = isolation.isolation_note(row)
    assert isolation.W_BACKEND_CRASHED_AT_TEARDOWN in note and "withheld" in note


def test_the_retry_child_is_verified_against_the_retry_placement(tmp_path: pathlib.Path) -> None:
    """A retry child that echoes the *original* placement is not the run it was asked for."""
    child = ScriptedChild(crashing(),
                          FakeChild(mutate=lambda report, _cmd: report["config"].update(
                              gpu_layers=-1)))
    result = run_child(throughput_config(), "vulkan", child, tmp_path)
    assert result.ok is False and result.warning == isolation.W_BACKEND_CRASHED_AT_TEARDOWN
    assert "gpu_layers=-1, this run asked 16" in result.detail


def test_a_crash_without_a_readable_model_header_retries_with_the_cpu_only_rung(
        tmp_path: pathlib.Path) -> None:
    """No `facts` (a model file the header read cannot reach): the rung that asks for nothing.

    `/tmp/fake.gguf` is the fixture's model: nothing is there to count the layers from, and the
    retry must still be a degraded, verified child rather than a second identical attempt.
    """
    child = ScriptedChild(crashing(), FakeChild())
    result = isolation.run_backend_child(
        throughput_config(), "vulkan", runner=child, out_dir=tmp_path, python="python",
        runtime_dir=VULKAN_DIR, facts=None, probe=starving)
    assert result.ok is True
    retry = cli_options(child.calls[1])
    assert retry["gpu-layers"] == "0" and retry["kv-type"] == "q8_0"


# ------------------------------------------------------------------ the report + the table
def test_a_failed_retry_fails_the_report_with_the_named_warning(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    def child(config: harness.BenchConfig, backend: str, **kwargs: Any) -> isolation.ChildRun:
        scripted = ScriptedChild(crashing(-signal.SIGSEGV), crashing(-signal.SIGABRT))
        return run_child(config, backend, scripted, tmp_path) if backend == "vulkan" else \
            run_child(config, backend, ScriptedChild(FakeChild()), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    report = suites.run_suite(throughput_config())
    rows = {row["backend"]: row for row in report["backends"]}
    assert rows["cpu"]["measured"] is True
    assert rows["vulkan"]["measured"] is False
    assert rows["vulkan"]["warnings"] == [isolation.W_BACKEND_CRASHED_AT_TEARDOWN]
    assert report["ok"] is False
    assert any(isolation.W_BACKEND_CRASHED_AT_TEARDOWN in note for note in report["notes"])


def test_a_recovered_row_keeps_the_report_ok(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    def child(config: harness.BenchConfig, backend: str, **kwargs: Any) -> isolation.ChildRun:
        if backend == "vulkan":
            return run_child(config, backend, ScriptedChild(crashing(), FakeChild()), tmp_path)
        return run_child(config, backend, ScriptedChild(FakeChild()), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    report = suites.run_suite(throughput_config())
    rows = {row["backend"]: row for row in report["backends"]}
    assert report["ok"] is True, "a recovered row is a measurement, not a failure"
    assert rows["vulkan"]["measured"] is True
    assert rows["vulkan"]["process"]["attempts"][0]["exit_code"] == -signal.SIGSEGV
    assert rows["vulkan"]["process"]["crash"]["starved"] is True
    # the crash is still named in the report — as a recovery, not as the withheld warning
    note = [note for note in report["notes"] if "RECOVERED_AFTER_TEARDOWN_CRASH" in note]
    assert note and "vulkan" in note[0] and "-11 (SIGSEGV; 139 in a shell)" in note[0]
    assert isolation.W_BACKEND_CRASHED_AT_TEARDOWN not in harness.render_report(report)


def test_the_warning_reaches_the_rendered_table_not_only_the_notes(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The card's requirement 2: the sentence a reader needs is in the table cell itself."""
    def child(config: harness.BenchConfig, backend: str, **kwargs: Any) -> isolation.ChildRun:
        if backend == "vulkan":
            return run_child(config, backend,
                             ScriptedChild(crashing(-signal.SIGSEGV), crashing(-signal.SIGABRT)),
                             tmp_path)
        return run_child(config, backend, ScriptedChild(FakeChild()), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    report = suites.run_suite(throughput_config())
    markdown = harness.render_report(report)
    cells = [line for line in markdown.splitlines() if line.startswith("| vulkan |")]
    assert len(cells) == 1, "the withheld row must be visible in the table"
    cell = cells[0]
    assert isolation.W_BACKEND_CRASHED_AT_TEARDOWN in cell
    assert "-11 (SIGSEGV; 139 in a shell)" in cell
    assert "3072 MiB free of 8192 MiB" in cell


def test_no_backend_row_can_disappear_from_the_rendered_table(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Requirement 4: a published table never silently loses a row — withheld rows keep a line."""
    def child(config: harness.BenchConfig, backend: str, **kwargs: Any) -> isolation.ChildRun:
        if backend == "vulkan":
            return run_child(config, backend,
                             ScriptedChild(crashing(-signal.SIGSEGV), crashing(-signal.SIGABRT)),
                             tmp_path)
        return run_child(config, backend, ScriptedChild(FakeChild()), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    report = suites.run_suite(throughput_config())
    markdown = harness.render_report(report)
    table = [line for line in markdown.splitlines()
             if line.startswith("| ") and not line.startswith("| backend ")]
    rendered = {line.split(" |")[0].lstrip("| ") for line in table}
    assert rendered == {row["backend"] for row in report["backends"]}
    withheld = [row for row in report["backends"] if not row["measured"]]
    for row in withheld:
        line = [line for line in table if line.startswith(f"| {row['backend']} |")][0]
        assert "not measured" in line and row["reason"].split(";")[0][:40] in line


def test_the_determinism_suite_recovers_and_never_calls_the_crash_a_mismatch(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The other isolating suite: the warning must not re-enter W_BACKEND_MISMATCH accounting."""
    def child(config: harness.BenchConfig, backend: str, **kwargs: Any) -> isolation.ChildRun:
        if backend == "vulkan":
            return run_child(config, backend,
                             ScriptedChild(crashing(-signal.SIGSEGV), crashing(-signal.SIGABRT)),
                             tmp_path)
        row = {"backend": "cpu", "runtime_dir": CPU_DIR, "threads": 1, "digests": ["sha256:aa"],
               "identical": True, "ok": True, "repeats": 1, "warnings": []}
        return run_child(config, backend, ScriptedChild(FakeChild(row=row)), tmp_path)

    monkeypatch.setattr(isolation, "run_backend_child", child)
    config = harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf", backend="all")
    report = suites.run_suite(config)
    assert report["ok"] is False
    assert any(isolation.W_BACKEND_CRASHED_AT_TEARDOWN in note for note in report["notes"])
    assert not any("W_BACKEND_MISMATCH" in note for note in report["notes"])
    rows = {row["backend"]: row for row in report["backends"]}
    assert rows["vulkan"]["measured"] is False and rows["vulkan"]["ok"] is False


def test_a_crash_that_gets_no_retry_says_so(tmp_path: pathlib.Path) -> None:
    """`attempts=1` is the API's own "no retry": the withheld row must say the retry never ran."""
    child = ScriptedChild(crashing(-signal.SIGSEGV), FakeChild())
    result = run_child(throughput_config(), "vulkan", child, tmp_path, attempts=1)
    assert result.ok is False and result.warning == isolation.W_BACKEND_CRASHED_AT_TEARDOWN
    assert len(child.calls) == 1
    assert "no retry was attempted" in result.detail
    assert result.detail.count("-11 (SIGSEGV; 139 in a shell)") == 1


def test_a_driver_that_raises_reads_as_an_unknown_device(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing driver query is "unknown", never an exception in the middle of a benchmark."""
    from typed_gguf.registry import recommend

    def refuse() -> Any:
        raise RuntimeError("nvidia-smi: the driver answered nothing")

    monkeypatch.setattr(recommend, "device_memory", refuse)
    memory = isolation.device_memory()
    assert memory.known is False and memory.source == "unknown"
    crash = isolation.teardown_crash(-signal.SIGSEGV, report={"ok": True}, memory=memory,
                                     weights_bytes=WEIGHTS)
    assert crash is not None and crash.starved is None
    assert "weights are unknown" in crash.sentence()


def test_the_device_records_serialise_to_the_reports_json() -> None:
    memory = isolation.device_memory(probe=roomy)
    assert memory.to_dict() == {"total_bytes": BOARD, "free_bytes": 7 * GIB, "source": "probe"}
    crash = isolation.teardown_crash(-signal.SIGSEGV, report={"ok": True}, memory=memory,
                                     weights_bytes=WEIGHTS)
    assert crash is not None
    assert crash.to_dict() == {"exit_code": -signal.SIGSEGV, "signal": "SIGSEGV",
                               "free_bytes": 7 * GIB, "total_bytes": BOARD,
                               "weights_bytes": WEIGHTS, "starved": False,
                               "free_source": "probe"}
    assert "would fit" in crash.sentence()
    placement = isolation.retry_placement(throughput_config(), "vulkan", facts=Fact())
    assert placement.to_dict() == {"n_gpu_layers": 16, "kv_type": "auto", "degraded": True,
                                   "note": placement.note}


def test_the_model_size_is_read_from_the_file_and_never_invented(tmp_path: pathlib.Path) -> None:
    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")
    assert isolation.model_bytes(str(empty)) is None            # 0 bytes is not a size
    assert isolation.model_bytes(str(tmp_path / "gone.gguf")) is None
    assert isolation.model_bytes(None) is None
    real = tmp_path / "model.gguf"
    real.write_bytes(b"x" * 1234)
    assert isolation.model_bytes(str(real)) == 1234


def test_the_model_header_is_the_facts_seam_and_a_bad_header_is_none(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """`model_facts` resolves the real header, and an unreadable one only means "safe rung"."""
    from typed_gguf.runtime import fit

    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real gguf")
    assert isolation.model_facts(str(model)) is None            # the read really did fail
    assert isolation.model_facts(None) is None
    sentinel = Fact(n_layer=7)
    monkeypatch.setattr(fit.ModelFacts, "read", classmethod(lambda cls, *a, **k: sentinel))
    assert isolation.model_facts(str(model)) is sentinel
    assert isolation.retry_placement(throughput_config(), "vulkan",
                                     facts=isolation.model_facts(str(model))).n_gpu_layers == 3


def test_a_single_bundle_host_never_pays_for_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-process run touches no device probe: the seam is only on the child path."""
    def refuse(**kwargs: Any) -> Any:
        raise AssertionError("the in-process path asked the driver for free device memory")

    monkeypatch.setattr(isolation, "device_memory", refuse)
    monkeypatch.setattr(harness, "backend_runtimes",
                        lambda **kwargs: {"cpu": pathlib.Path(CPU_DIR),
                                          "vulkan": pathlib.Path(CPU_DIR)})
    report = suites.run_suite(throughput_config(), factory=bench_factory())
    assert [row["backend"] for row in report["backends"] if row.get("measured")] == \
        ["cpu", "vulkan"]
    assert "isolation" not in report                    # one bundle: the fake seam stays in-process
    assert all("process" not in row for row in report["backends"])
