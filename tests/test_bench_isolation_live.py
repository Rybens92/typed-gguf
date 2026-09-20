"""E2 FIX (card t_dd62ec29), live half: the two-bundle command, on the box that aborts.

`bench --backend all` on a host with **two** local bundles used to write its whole report and then
die at teardown (`double free or corruption (!prev)`, exit **134**) — see
`.e2e/t_603a35a0-backend-attribution/logs/{before,after}_mixed.raw` and this card's
`docs/evidence/e2_fix_t_dd62ec29_mixed_bundle_isolation.md`. This test is the executable form of the
card's requirement 1+2: the *real* command, two real bundles on disk, the exit code the report's.

    TYPED_GGUF_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu \\
    TYPED_GGUF_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/typed-gguf/runtime \\
    TYPED_GGUF_BENCH_MODEL=/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \\
      uv run --frozen pytest -q --run-network tests/test_bench_isolation_live.py -s

(`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` in a container, so the second bundle's own
process can see the device; without it that row still runs, on the host CPU, and is flagged — the
assertions below read the row's own evidence rather than assuming a device.)

It skips, with the reason, on a box that has fewer than two bundles or no benchmarkable GGUF: this
gate is about a host shape, and a missing host shape is a skip, never a silent pass.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

import typed_gguf
from typed_gguf.bench import harness

#: The known local GGUFs (`tests/test_bench_live.py` uses the same two names): a small one for a
#: fast gate, the operator's Spark 4B as the fallback.
QUICK_MODEL = pathlib.Path("/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
SPARK = pathlib.Path("/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf")


def two_bundles() -> tuple[str, str, str]:
    """`(cpu_dir, accel_backend, accel_dir)` for a box the bench would load two bundles on."""
    runtimes = harness.backend_runtimes()
    cpu = runtimes.get("cpu")
    accelerators = [(backend, runtimes[backend]) for backend in harness.BACKEND_LIBRARIES
                    if backend in runtimes]
    if cpu is None or not accelerators:
        pytest.skip("this box has fewer than two local llama.cpp bundles "
                    "(set TYPED_GGUF_RUNTIME_DIR and TYPED_GGUF_BENCH_RUNTIME_DIR)")
    backend, directory = accelerators[0]
    if pathlib.Path(cpu).resolve() == pathlib.Path(directory).resolve():
        pytest.skip(f"one bundle answers `cpu` and `{backend}`: `--backend all` stays in one "
                    f"process here, so there is nothing to isolate")
    return str(cpu), backend, str(directory)


def benchmarkable_model() -> pathlib.Path:
    explicit = os.environ.get("TYPED_GGUF_BENCH_MODEL")
    candidates = ([pathlib.Path(explicit)] if explicit else []) + [QUICK_MODEL, SPARK]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("no benchmarkable GGUF on this box (set TYPED_GGUF_BENCH_MODEL)")


def child_env() -> dict[str, str]:
    """The CLI child's env: the caller's, plus this checkout's `src/` on `PYTHONPATH`."""
    env = dict(os.environ)
    src_root = str(pathlib.Path(typed_gguf.__file__).resolve().parents[1])
    parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    if src_root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src_root, *parts])
    return env


@pytest.mark.model
def test_the_two_bundle_command_does_not_abort_and_its_exit_code_is_the_report(
        tmp_path: pathlib.Path) -> None:
    """The card's gate: no SIGABRT, no silent 0, every measured row proved by its own child."""
    cpu, accel_backend, accel_dir = two_bundles()
    model = benchmarkable_model()
    report_path = tmp_path / "throughput-all.json"
    command = [sys.executable, "-m", "typed_gguf", "bench", "--suite", "throughput",
               "--model", str(model), "--backend", "all", "--runs", "1", "--threads", "4",
               "--sizes", "64", "--out", str(report_path), "--json"]
    completed = subprocess.run(command, capture_output=True, text=True, env=child_env(),
                               timeout=1800)
    tail = (completed.stderr or "")[-1500:]
    # requirement 1: the process ends with the CLI's own codes, never with the kernel's signal
    assert completed.returncode in (0, 1), (
        f"`--backend all` exited {completed.returncode} on {cpu} + {accel_dir}\n{tail}")
    assert report_path.is_file(), f"the report was never written\n{tail}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == harness.SCHEMA
    # requirement 2: the exit code is the report's (SPEC 2.5) — never 134, never a silent 0
    assert completed.returncode == (0 if report["ok"] else 1), (
        f"exit {completed.returncode} contradicts report ok={report['ok']}\n{tail}")
    # `--json` stays parseable: the report the operator reads on stdout is the file's
    assert json.loads(completed.stdout) == report
    assert report["isolation"]["one_bundle_per_process"] is True
    assert report["isolation"]["bundles"] == {"cpu": cpu, accel_backend: accel_dir}
    assert any("one bundle per process" in note for note in report["notes"])

    rows = {row["backend"]: row for row in report["backends"]}
    assert {"cpu", accel_backend} <= set(rows), f"rows: {sorted(rows)}\n{tail}"
    measured = {backend: row for backend, row in rows.items() if row.get("measured")}
    # every measured row is a row its own child produced and the parent verified
    assert "cpu" in measured, f"the CPU bundle must always run here: {rows['cpu']}\n{tail}"
    for backend, row in measured.items():
        process = row["process"]
        assert process == {"isolated": True, "exit_code": 0, "ok": True, "detail": None}, (
            f"{backend}: {process}")
        # the second bundle's engine used to emit no line at all (t_603a35a0's F2): the row now
        # carries the device set its *own* process really touched
        assert row["devices"], f"{backend}: no engine device log in the row: {row}"
        if row["effective_backend"]:
            assert "W_BACKEND_MISMATCH" not in row["warnings"], (
                f"{backend}: {row['effective_backend']} computed but the row is flagged")
        else:
            assert "W_BACKEND_MISMATCH" in row["warnings"], (
                f"{backend}: an uncorroborated claim must be flagged, not published")
    # a row that did not run says why — the box's own states included: the device may be too busy
    # *right now* (this box shares it with the E3 campaign) or a child may have died at teardown,
    # and both are reported rows here instead of a killed command (the card's requirements 1+2)
    for backend, row in rows.items():
        if not row.get("measured"):
            assert row.get("reason"), f"{backend}: an unmeasured row must carry its reason"
    withheld = [row for row in rows.values() if (row.get("process") or {}).get("ok") is False]
    for row in withheld:
        print(f"[live] the {row['backend']} row was withheld: {str(row['reason'])[:220]}")
        assert any("ISOLATED_CHILD_FAILED" in note and row["backend"] in note
                   for note in report["notes"]), row["backend"]
    assert report["ok"] == (bool(measured)
                            and not any(row["warnings"] for row in measured.values()))
