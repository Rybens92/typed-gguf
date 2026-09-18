"""E2 (card t_f46cec41): `bench --quick` (short preset) and `--max-seconds` (soft cap).

Offline and model-free, like `tests/test_bench.py`: the preset is a *configuration* contract, so
every assertion here runs against the deterministic fake seam. The live wall-clock budget lives in
`tests/test_bench_live.py` (`model`-marked) and the measured numbers in
`docs/evidence/e2_t_f46cec41_bench_quick.md`.

What is pinned:

* the quick preset's **effective config** (exact values, and the non-scale fields it must keep);
* one suite per row shape: latency sizes/candidates/waves, throughput backends, quality
  stratification (2/2/2), calibration bins "as available" + the three confidence modes,
  determinism repeats;
* report integrity: `"quick": true`, the `reproduce:` command is *runnable* (`--quick` never
  re-states the scale flags it fixes), the default `--out` of a quick run is its own file and
  cannot land on a full-campaign JSON;
* `--max-seconds` semantics: checked between measurements, the current one finishes, the unmeasured
  rows are listed under `"truncated": true`, and the exit code stays 0;
* `--quick` + an explicit scale flag (`--runs`, `--items`, `--sizes`, `--n-seq-max`) is a typed
  `E_BENCH_QUICK` error — the preset fixes those knobs, and a run that ignored half of them would
  be a report that lies about which preset it is.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from ggufone import cli
from ggufone.bench import harness, suites
from tests.test_bench import bench_factory, devset_script

#: the effective config `--quick` must resolve to (exact values, card t_f46cec41)
QUICK_FIELDS = {
    "runs": 1,
    "prefill_sizes": (256,),
    "candidate_counts": (2, 4),
    "wave_scaling": (1, 2),
    "items": 6,
    "items_per_type": 2,
    "determinism_repeats": 2,
    "backend_limit": 1,
    "quick": True,
}


@pytest.fixture(autouse=True)
def _local_cpu_bundle(monkeypatch, tmp_path):
    """Pretend one local CPU bundle exists: no test here may need (or download) a real one."""
    monkeypatch.setattr(harness, "backend_runtimes",
                        lambda **kwargs: {"cpu": tmp_path / "b11026-linux-x64-cpu"})


def quick_config(suite: str, **kwargs) -> harness.BenchConfig:
    return harness.quick_config(
        harness.BenchConfig(suite=suite, model_path="/tmp/fake.gguf", **kwargs))


def model_file(tmp_path: pathlib.Path) -> pathlib.Path:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\x00" * 16)
    return model


# --------------------------------------------------------------------------- the preset itself
def test_the_quick_preset_pins_the_effective_config():
    full = harness.BenchConfig(suite="latency", model_path="/tmp/m.gguf", backend="cpu", threads=4,
                               gpu_layers=0, kv_type="q8_0", n_seq_max=6, max_seconds=30.0)
    quick = harness.quick_config(full)
    for field, expected in QUICK_FIELDS.items():
        assert getattr(quick, field) == expected, field
    # everything that is not a *scale* survives untouched: same model, backend, threads, placement
    assert (quick.suite, quick.model_path, quick.backend, quick.threads, quick.gpu_layers,
            quick.kv_type, quick.n_seq_max, quick.max_seconds, quick.home) == \
           (full.suite, full.model_path, full.backend, full.threads, full.gpu_layers,
            full.kv_type, full.n_seq_max, full.max_seconds, full.home)
    assert full.runs == harness.DEFAULT_RUNS           # the input config is never mutated
    assert quick.to_dict()["prefill_sizes"] == [256]   # JSON-clean, like every report field
    assert harness.QUICK_TARGET_SECONDS == 180.0       # the card's <= ~3 minutes
    assert set(harness.QUICK_CONFLICTS) == {"runs", "items", "sizes", "n_seq_max"}


def test_a_quick_report_carries_the_flag_the_preset_and_the_effective_config():
    report = suites.run_suite(quick_config("latency", threads=2), factory=bench_factory())
    assert report["quick"] is True
    assert report["config"]["quick"] is True
    assert report["config"]["runs"] == 1
    assert report["config"]["prefill_sizes"] == [256]
    assert report["config"]["backend_limit"] == 1
    assert any("quick" in note for note in report["notes"])
    assert report["truncated"] is False
    assert report["skipped"] == []
    assert report["wall_ms"] > 0
    assert report["budget"]["max_seconds"] is None
    assert json.loads(json.dumps(report)) == report


def test_a_full_report_is_not_marked_quick():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=2), factory=bench_factory())
    assert report["quick"] is False
    assert report["config"]["quick"] is False
    assert [row["tokens"] for row in report["prefill"]] == [256, 2048, 8192]


def test_the_quick_reproduce_command_never_re_states_the_scale_it_fixes():
    """The `reproduce:` line is runnable: `--quick --runs 1` would be an E_BENCH_QUICK error."""
    line = harness.reproduce_command(quick_config("latency", threads=4))
    assert line.startswith("uv run ggufone bench --suite latency")
    assert "--quick" in line and "--threads 4" in line
    for flag in ("--runs", "--items", "--n-seq-max"):
        assert flag not in line, line
    full = harness.reproduce_command(harness.BenchConfig(suite="latency", runs=3, items=6))
    assert "--quick" not in full and "--runs 3" in full and "--items 6" in full


# --------------------------------------------------------------------------- one row per suite
def test_a_quick_latency_run_measures_the_short_preset():
    report = suites.run_suite(quick_config("latency"), factory=bench_factory())
    assert report["model_load"]["n"] == 1
    assert [row["tokens"] for row in report["prefill"]] == [256]
    assert [row["candidates"] for row in report["per_question"]] == [2, 4]
    assert [row["questions"] for row in report["wave_scaling"]] == [1, 2]
    assert report["warm_cache"]["prefill_reused"] is True
    assert report["load_amortisation"]["calls"] == 1
    assert report["wave_accounting"]["waves"] >= 1
    for row in report["prefill"] + report["per_question"] + report["wave_scaling"]:
        assert row["ms"]["n"] == 1


def test_a_quick_throughput_run_measures_one_backend_and_names_the_others():
    report = suites.run_suite(quick_config("throughput", backend="all"), factory=bench_factory())
    assert [row["backend"] for row in report["backends"]] == ["cpu", "vulkan", "cuda"]
    measured = [row for row in report["backends"] if row.get("measured")]
    assert [row["backend"] for row in measured] == ["cpu"]
    assert measured[0]["load_ms"]["n"] == 1
    assert measured[0]["decision_ms"]["n"] == 1
    for row in report["backends"][1:]:
        assert row["measured"] is False
        assert "--quick" in row["reason"]
    assert report["ok"] is True


def test_a_quick_quality_run_takes_two_items_of_each_type():
    report = suites.run_suite(quick_config("quality"),
                              factory=bench_factory(script=devset_script(right=True)))
    assert report["devset"]["items"] == 6
    assert report["devset"]["counts"] == {"choice": 2, "score": 2, "noul": 2}
    assert report["devset"]["measured"] == 6
    assert [row["type"] for row in report["items"]] == ["choice", "choice", "score", "score",
                                                       "noul", "noul"]
    assert report["overall"]["n"] == 6
    assert report["overall"]["agreement"] == pytest.approx(1.0)


def test_stratified_selection_degrades_to_what_the_dev_set_has():
    """A custom dev set with one type only is not an error: take what is there, in type order."""
    from ggufone.bench import devset as devset_module

    noul = [item for item in dev_items() if item.type == "noul"]
    picked = devset_module.stratify(noul, per_type=2)
    assert [item.type for item in picked] == ["noul", "noul"]
    assert devset_module.stratify(dev_items(), per_type=2)[0].type == "choice"
    assert devset_module.stratify(dev_items(), per_type=0) == []
    # per_type=1 is the "one of each type" case, not the empty one (a `<` for `<=` here would
    # still return the right list for 0 and `[]` for 1 — pinned by the mutation pass)
    assert [item.type for item in devset_module.stratify(dev_items(), per_type=1)] == \
        ["choice", "score", "noul"]
    assert devset_module.stratify(dev_items(), per_type=-1) == []


def test_a_quick_calibration_run_keeps_three_modes_with_bins_as_available():
    report = suites.run_suite(quick_config("calibration"),
                              factory=bench_factory(script=devset_script(right=True)))
    assert report["n"] == 6
    assert report["n_bins"] == 6                      # never more bins than samples under --quick
    assert len(report["reliability"]) == 6
    assert sum(entry["n"] for entry in report["reliability"]) == 6
    assert 0.0 <= report["ece"] <= 1.0
    assert set(report["modes"]) == {"normalized_peak", "entropy", "margin"}
    for row in report["modes"].values():
        assert row["n"] == 6 and len(row["bins"]) == 6
    full = suites.run_suite(harness.BenchConfig(suite="calibration", model_path="/tmp/fake.gguf"),
                            factory=bench_factory(script=devset_script(right=True)))
    assert full["n_bins"] == 10                       # the full suite still bins ten ways


def test_a_quick_determinism_run_repeats_one_request_twice():
    report = suites.run_suite(quick_config("determinism", backend="cpu"), factory=bench_factory())
    assert report["repeats"] == 2
    assert len(report["request"]["questions"]) == 3   # choice + score + noul: the same one request
    assert len(report["backends"]) == 1
    assert len(report["backends"][0]["digests"]) == 2
    assert report["backends"][0]["identical"] is True
    assert report["ok"] is True


# --------------------------------------------------------------------------- the CLI surface
@pytest.mark.parametrize("flag,value", [("--runs", "5"), ("--items", "60"), ("--sizes", "256,2048"),
                                        ("--n-seq-max", "4")])
def test_quick_refuses_an_explicit_scale_flag(flag, value, tmp_path, capsys):
    code = cli.main(["bench", "--suite", "latency", "--quick", "--model", str(model_file(tmp_path)),
                     flag, value])
    err = capsys.readouterr().err
    assert code == 2, err
    assert "E_BENCH_QUICK" in err
    assert flag in err
    assert "--quick" in err


def test_the_same_scale_flags_still_work_without_quick(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    code = cli.main(["bench", "--suite", "latency", "--model", str(model_file(tmp_path)),
                     "--runs", "2", "--sizes", "128", "--n-seq-max", "3", "--json"])
    assert code == 0, capsys.readouterr().err
    report = json.loads(capsys.readouterr().out)
    assert report["quick"] is False
    assert report["config"]["runs"] == 2
    assert [row["tokens"] for row in report["prefill"]] == [128]


def test_bench_help_documents_quick_and_the_soft_cap(capsys):
    code = cli.main(["bench", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--quick" in out and "--max-seconds" in out
    assert "--runs" in out and "--items" in out        # the frozen flags stay documented
    assert "between measurements" in out


def test_a_quick_run_writes_its_own_report_and_never_a_full_campaign_file(tmp_path, monkeypatch,
                                                                         capsys):
    model = model_file(tmp_path)
    campaign = tmp_path / "e2_latency.json"
    campaign.write_text('{"full": true}', encoding="utf-8")
    other = tmp_path / "ggufone-bench-latency.json"
    other.write_text('{"full": true}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    code = cli.main(["bench", "--suite", "latency", "--quick", "--model", str(model), "--json"])
    assert code == 0, capsys.readouterr().err
    assert cli._bench_out_path({}, suite="latency", quick=False) is None
    assert cli._bench_out_path({}, suite="latency", quick=True) == \
        "ggufone-bench-latency_quick.json"
    # the *name* both branches produce, pinned: a quick report never lands on the full one, and the
    # full name is spelled out rather than only "different from the quick one"
    assert harness.default_out_path("latency", quick=False) == "ggufone-bench-latency.json"
    assert harness.default_out_path("latency", quick=True) == \
        "ggufone-bench-latency_quick.json"
    assert harness.default_out_path("latency", quick=True) != \
        harness.default_out_path("latency", quick=False)
    quick = tmp_path / "ggufone-bench-latency_quick.json"
    assert quick.is_file()
    report = json.loads(quick.read_text(encoding="utf-8"))
    assert report["quick"] is True and report["suite"] == "latency"
    assert campaign.read_text(encoding="utf-8") == '{"full": true}'
    assert other.read_text(encoding="utf-8") == '{"full": true}'


def test_a_full_run_without_out_still_writes_nothing_and_an_explicit_out_wins(tmp_path, monkeypatch,
                                                                             capsys):
    model = model_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    assert cli.main(["bench", "--suite", "latency", "--model", str(model), "--runs", "1",
                     "--sizes", "128", "--json"]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["m.gguf"]
    chosen = tmp_path / "mine.json"
    # --items conflicts with --quick: the preset fixes it, so the CLI refuses the combination
    assert cli.main(["bench", "--suite", "quality", "--quick", "--model", str(model),
                     "--items", "6", "--out", str(chosen), "--json"]) == 2
    capsys.readouterr()
    assert cli.main(["bench", "--suite", "quality", "--quick", "--model", str(model), "--out",
                     str(chosen), "--json"]) == 0
    assert chosen.is_file() and json.loads(chosen.read_text())["quick"] is True
    assert not (tmp_path / "ggufone-bench-quality_quick.json").exists()


# --------------------------------------------------------------------------- the soft cap
def test_max_seconds_rejects_a_value_that_is_not_a_duration(tmp_path, capsys):
    for value in ("-1", "abc", "nan", ""):
        code = cli.main(["bench", "--suite", "latency", "--model", str(model_file(tmp_path)),
                         "--max-seconds", value])
        err = capsys.readouterr().err
        assert code == 2, (value, err)
        assert "E_BENCH_USAGE" in err


@pytest.mark.parametrize("value,expected", [("30", 30.0), ("0", 0.0), ("2.5", 2.5)])
def test_max_seconds_is_a_float_that_may_be_fractional(value, expected):
    assert cli._bench_max_seconds(value) == expected
    assert cli._bench_max_seconds(None) is None


def test_the_soft_cap_is_checked_between_measurements(monkeypatch):
    """One budget check per measurement: a started unit finishes, the rest is listed unmeasured."""
    monkeypatch.setattr(harness, "TimeBudget", ClockedBudget)
    report = suites.run_suite(quick_config("latency", max_seconds=25.0), factory=bench_factory())
    assert report["truncated"] is True
    assert report["budget"] == {"max_seconds": 25.0, "expired": True, "skipped": 6}
    assert [row["tokens"] for row in report["prefill"]] == [256]      # it was started: it finished
    assert report["per_question"] == [] and report["wave_scaling"] == []
    assert report["warm_cache"] == {} and report["load_amortisation"] == {}
    skipped = {(entry["section"], entry["row"]) for entry in report["skipped"]}
    assert ("model_load", "load#1") not in skipped
    assert ("prefill", "tokens=256") not in skipped
    assert ("per_question", "candidates=2") in skipped
    assert ("wave_scaling", "questions=1") in skipped
    assert ("warm_cache", "state reuse") in skipped
    assert all("--max-seconds 25" in entry["reason"] for entry in report["skipped"])


def test_a_truncated_run_exits_zero_with_the_unmeasured_rows_listed(tmp_path, monkeypatch, capsys):
    model = model_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    code = cli.main(["bench", "--suite", "latency", "--quick", "--model", str(model),
                     "--max-seconds", "0", "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    report = json.loads(captured.out)
    assert report["truncated"] is True
    assert report["ok"] is True                       # truncation is not a gate failure
    assert report["model_load"]["n"] == 0
    assert report["prefill"] == [] and report["per_question"] == []
    listed = {(entry["section"], entry["row"]) for entry in report["skipped"]}
    assert ("model_load", "load#1") in listed
    assert ("prefill", "tokens=256") in listed
    assert ("per_question", "candidates=4") in listed
    assert ("wave_scaling", "questions=2") in listed


def test_a_truncated_quality_run_lists_the_items_it_never_asked(tmp_path, monkeypatch, capsys):
    model = model_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suites, "live_factory",
                        bench_factory(script=devset_script(right=True)))
    code = cli.main(["bench", "--suite", "quality", "--quick", "--model", str(model),
                     "--max-seconds", "0", "--json"])
    assert code == 0, capsys.readouterr().err
    report = json.loads(capsys.readouterr().out)
    assert report["devset"]["items"] == 6 and report["devset"]["measured"] == 0
    assert report["overall"]["n"] == 0
    from ggufone.bench import devset as devset_module
    expected = [item.id for item in devset_module.stratify(dev_items(), per_type=2)]
    listed = [(entry["section"], entry["row"]) for entry in report["skipped"]]
    assert ("model_load", "load#1") in listed
    assert [row for section, row in listed if section == "items"] == expected
    assert expected != [item.id for item in dev_items()[:6]]     # stratified, not the first six


def test_a_fully_truncated_report_is_still_ok_and_keeps_its_shape(tmp_path, monkeypatch, capsys):
    model = model_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    for suite, extra in (("throughput", ["--backend", "all"]),
                         ("calibration", []), ("determinism", ["--backend", "cpu"])):
        code = cli.main(["bench", "--suite", suite, "--quick", "--model", str(model),
                         "--max-seconds", "0", "--json", *extra])
        assert code == 0, (suite, capsys.readouterr().err)
        report = json.loads(capsys.readouterr().out)
        assert report["truncated"] is True and report["ok"] is True, suite
        assert report["skipped"], suite
        assert json.loads(json.dumps(report)) == report
    assert json.loads((tmp_path / "ggufone-bench-throughput_quick.json").read_text())["truncated"]


def test_a_measured_gate_failure_still_exits_one_even_when_the_run_was_truncated(
        tmp_path, monkeypatch, capsys):
    """The soft cap excuses *incompleteness*, never a gate that ran and failed."""
    model = model_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness, "backend_runtimes",
                        lambda **kwargs: {"cpu": tmp_path / "rt", "vulkan": tmp_path / "rt"})
    monkeypatch.setattr(harness, "TimeBudget", ClockedBudget)
    script = {"The billing dashboard is blank for every user after login since 09:12.": "billing"}
    monkeypatch.setattr(suites, "live_factory",
                        bench_factory(script=script, nondeterministic=True))
    code = cli.main(["bench", "--suite", "determinism", "--model", str(model), "--backend", "all",
                     "--max-seconds", "15", "--json"])
    assert code == 1, capsys.readouterr().err
    report = json.loads(capsys.readouterr().out)
    assert report["truncated"] is True
    assert report["ok"] is False                       # the row that ran was not identical
    assert [row["backend"] for row in report["backends"]] == ["cpu"]
    assert [entry["row"] for entry in report["skipped"]] == ["vulkan"]


def test_the_rendered_report_shows_the_wall_time_the_preset_and_the_truncation():
    quick = suites.run_suite(quick_config("latency"), factory=bench_factory())
    markdown = harness.render_report(quick)
    assert "- wall:" in markdown
    assert "--quick" in markdown
    truncated = suites.run_suite(quick_config("latency", max_seconds=0.0), factory=bench_factory())
    text = harness.render_report(truncated)
    assert "- wall:" in text and "truncated" in text.lower()
    assert "tokens=256" in text                    # the unmeasured row is named in the report head


def test_a_quick_run_without_json_prints_the_preset_table_and_names_its_report(
        tmp_path, monkeypatch, capsys):
    """The user story of the card, verbatim: a short run from the terminal, table + report path."""
    model = model_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    code = cli.main(["bench", "--suite", "latency", "--quick", "--model", str(model),
                     "--threads", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "latency" in out and "p50" in out
    assert "- preset: --quick" in out
    assert "- wall:" in out
    assert "- reproduce: `uv run ggufone bench --suite latency" in out
    assert "report: ggufone-bench-latency_quick.json" in out
    assert (tmp_path / "ggufone-bench-latency_quick.json").is_file()


def test_the_reproduce_command_names_a_custom_dev_set_and_a_soft_cap():
    """Both flags a reader needs to replay a report exactly (`--devset`, `--max-seconds`)."""
    line = harness.reproduce_command(harness.BenchConfig(
        suite="quality", runs=1, devset="/tmp/mine.jsonl", max_seconds=42.5, threads=4))
    assert "--devset /tmp/mine.jsonl" in line
    assert "--max-seconds 42.5" in line
    assert "--quick" not in line


# --------------------------------------------------------------------------- the reproduce tool
def test_the_reproduce_tool_mirrors_the_quick_preset(tmp_path):
    module = reproduce_tool()
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\x00" * 16)
    parser = module.make_parser()
    args = parser.parse_args(["--suite", "latency", "--quick", "--model", str(model),
                              "--threads", "4", "--max-seconds", "5"])
    config = module.build_config(args, "latency")
    assert config.quick is True and config.runs == 1
    assert config.prefill_sizes == (256,) and config.candidate_counts == (2, 4)
    assert config.wave_scaling == (1, 2) and config.backend_limit == 1
    assert config.threads == 4 and config.max_seconds == 5.0
    full = module.build_config(
        parser.parse_args(["--suite", "latency", "--model", str(model)]), "latency")
    assert full.quick is False and full.runs == harness.DEFAULT_RUNS
    with pytest.raises(SystemExit):
        module.build_config(parser.parse_args(["--suite", "latency", "--model", str(model),
                                               "--max-seconds", "-1"]), "latency")


def test_the_reproduce_tool_refuses_quick_plus_an_explicit_scale_flag(capsys):
    module = reproduce_tool()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--suite", "latency", "--quick", "--runs", "5", "--model", "/tmp/f.gguf"])
    assert excinfo.value.code == 2
    assert "--quick" in capsys.readouterr().err


def test_the_reproduce_tool_never_writes_a_quick_report_over_a_full_one():
    module = reproduce_tool()
    parser = module.make_parser()
    quick = module.resolve_out(parser.parse_args(["--suite", "latency", "--quick"]), "latency")
    full = module.resolve_out(parser.parse_args(["--suite", "latency"]), "latency")
    assert quick == "ggufone-bench-latency_quick.json"
    assert full is None
    out_dir = module.resolve_out(
        parser.parse_args(["--suite", "latency", "--quick", "--out-dir", "/tmp/x"]), "latency")
    assert out_dir == str(pathlib.Path("/tmp/x") / "e2_latency_quick.json")
    explicit = module.resolve_out(
        parser.parse_args(["--suite", "latency", "--quick", "--out", "/tmp/mine.json"]), "latency")
    assert explicit == "/tmp/mine.json"


# --------------------------------------------------------------------------- helpers
class ClockedBudget(harness.TimeBudget):
    """A budget whose clock advances 10 s per check: deterministic, no sleeping in the gate."""

    def __init__(self, max_seconds: float | None = None) -> None:
        super().__init__(max_seconds)
        self.checks = 0

    def elapsed(self) -> float:
        self.checks += 1
        return 10.0 * self.checks


def dev_items():
    from ggufone.bench import devset as devset_module
    return devset_module.load()


def reproduce_tool():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("e2_reproduce_quick", root / "tools" /
                                                  "e2_reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
