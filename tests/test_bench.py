"""E2 gates: the bench harness, the committed dev set and the five suites (SPEC 5 / A-E2-1..8).

Offline and model-free: every suite runs against the same seam a real model implements
(`harness.ModelLike`), so the statistics, the report shape, the dev-set contract, the wave
accounting and the registry/network isolation are pinned without a GGUF file or a runtime
bundle. The live numbers live in `docs/BENCHMARKS.md` and `tests/test_bench_live.py`.
"""
from __future__ import annotations

import json
import pathlib
import socket

import pytest

from ggufone import cli, schema
from ggufone.bench import devset as devset_module
from ggufone.bench import harness, suites
from tests.fake_engine import BenchModel

#: the real resolver, captured before the autouse fixture replaces the module attribute
_REAL_BACKEND_RUNTIMES = harness.backend_runtimes


@pytest.fixture(autouse=True)
def _local_cpu_bundle(monkeypatch, tmp_path):
    """Pretend one local CPU bundle exists: no test here may need (or download) a real one."""
    monkeypatch.setattr(harness, "backend_runtimes",
                        lambda **kwargs: {"cpu": tmp_path / "b11026-linux-x64-cpu"})


def bench_factory(*, script: dict[str, str] | None = None, **kwargs):
    """`factory(spec) -> BenchModel`: the seam a live run fills with `suites.live_factory`."""
    kwargs.setdefault("prefill_ms_per_token", 0.001)

    def make(spec: harness.ModelSpec) -> BenchModel:
        return BenchModel(spec, script=script, **kwargs)

    return make


def devset_script(*, right: bool) -> dict[str, str]:
    """A per-item preference: the gold label, or the first candidate that is not the gold one."""
    script: dict[str, str] = {}
    for item in devset_module.load():
        labels = list(devset_module.labels_of(item))
        gold = devset_module.gold_label(item)
        script[item.state] = gold if right else next(label for label in labels if label != gold)
    return script


# --------------------------------------------------------------------------- statistics
def test_percentile_pins_the_documented_interpolated_definition():
    values = [4.0, 1.0, 3.0, 2.0]
    assert harness.percentile(values, 0) == 1.0
    assert harness.percentile(values, 50) == 2.5
    assert harness.percentile(values, 100) == 4.0
    assert harness.percentile(values, 95) == pytest.approx(3.85)
    assert harness.percentile([7.0], 50) == 7.0
    with pytest.raises(ValueError):
        harness.percentile([], 50)


def test_summarise_reports_n_min_p50_p95_max_mean():
    row = harness.summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert row["n"] == 5
    assert row["min"] == 1.0
    assert row["p50"] == 3.0
    assert row["p95"] == pytest.approx(4.8)
    assert row["max"] == 5.0
    assert row["mean"] == 3.0
    assert json.loads(json.dumps(row)) == row          # JSON-clean, no NaN/numpy types
    assert harness.summarise([]) == {"n": 0, "min": None, "p50": None, "p95": None, "max": None,
                                     "mean": None}


def test_ratio_summarise_never_divides_by_zero():
    row = harness.ratio_summarise([100, 200], [1.0, 0.0])
    assert row["n"] == 1 and row["p50"] == pytest.approx(100.0)


def test_wilson_interval_matches_the_textbook_example():
    low, high = harness.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=5e-4)
    assert high == pytest.approx(0.5962, abs=5e-4)
    low, high = harness.wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=5e-4)
    assert high == 1.0
    low, high = harness.wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=5e-4)
    assert harness.wilson_interval(0, 0) == (0.0, 1.0)   # no data: the honest full range


def test_reliability_bins_cover_every_sample_and_handle_empty_bins():
    confidences = [0.05, 0.15, 0.55, 0.95, 0.85]
    correct = [False, True, True, True, False]
    bins = harness.reliability_bins(confidences, correct, n_bins=5)
    assert len(bins) == 5
    assert sum(entry["n"] for entry in bins) == len(confidences)
    assert bins[0]["lo"] == 0.0 and bins[-1]["hi"] == 1.0
    for entry in bins:
        if entry["n"] == 0:
            assert entry["accuracy"] is None and entry["gap"] is None
        else:
            assert 0.0 <= entry["accuracy"] <= 1.0
            assert entry["gap"] == pytest.approx(entry["accuracy"] - entry["mean_confidence"])
    # confidence 1.0 belongs to the last bin rather than falling off the end
    assert harness.reliability_bins([1.0], [True], n_bins=4)[-1]["n"] == 1
    with pytest.raises(ValueError):
        harness.reliability_bins([0.1], [True], n_bins=0)
    with pytest.raises(ValueError):
        harness.reliability_bins([0.1, 0.2], [True], n_bins=4)


def test_ece_pins_a_hand_computed_example():
    # two bins, 4 samples: bin 1 (0.0-0.5) has 2 samples, 1 correct, mean conf 0.3
    #                       bin 2 (0.5-1.0) has 2 samples, 2 correct, mean conf 0.9
    # ECE = (2/4)*|0.5-0.3| + (2/4)*|1.0-0.9| = 0.10 + 0.05 = 0.15
    bins = harness.reliability_bins([0.2, 0.4, 0.8, 1.0], [True, False, True, True], n_bins=2)
    assert harness.ece(bins) == pytest.approx(0.15, abs=1e-9)
    assert harness.ece(harness.reliability_bins([0.0, 1.0], [False, True], n_bins=2)) == \
        pytest.approx(0.0, abs=1e-9)


def test_pearson_pins_the_degenerate_cases():
    assert harness.pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert harness.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert harness.pearson([1, 2, 3], [2, 2, 2]) is None       # zero variance
    assert harness.pearson([1], [1]) is None                   # n < 2
    assert harness.pearson([], []) is None


# --------------------------------------------------------------------------- determinism digest
def test_a_timings_only_delta_never_changes_the_digest():
    base = {"model": "m", "answers": {"a": {"probabilities": {"x": 0.6, "y": 0.4}}},
            "timings": {"model_load_ms": 1.0, "prefill_ms": 2.0, "total_ms": 3.0},
            "usage": {"questions": 1}}
    other = json.loads(json.dumps(base))
    other["timings"] = {"model_load_ms": 99.0, "prefill_ms": 88.0, "total_ms": 77.0}
    assert harness.digest(base) == harness.digest(other)
    assert "timings" not in harness.strip_timings(base)
    assert "timings" in base                                   # strip must not mutate its input
    moved = json.loads(json.dumps(base))
    moved["answers"]["a"]["probabilities"]["x"] = 0.6001
    assert harness.digest(base) != harness.digest(moved)
    assert harness.digest({"b": 1, "a": 2}) == harness.digest({"a": 2, "b": 1})   # key order free


# --------------------------------------------------------------------------- the dev set
def test_the_committed_dev_set_satisfies_the_e2_contract():
    items = devset_module.load()
    assert len(items) >= 50
    assert {"choice", "score", "noul"} <= {item.type for item in items}
    assert len({item.id for item in items}) == len(items)      # ids unique
    assert devset_module.validate(items) == []                 # every rule, not just a sample
    counts = devset_module.counts(items)
    assert set(counts) == {"choice", "score", "noul"}
    assert all(count >= 5 for count in counts.values())
    assert all(len(item.state.split()) <= devset_module.MAX_STATE_WORDS for item in items)


def test_validate_rejects_the_mistakes_it_can_see():
    good = devset_module.load()[0]
    assert devset_module.validate([good] * 1) != []            # too few items / duplicate ids
    broken = devset_module.DevItem(id="x", type="choice", state="s", instructions="i",
                                   criteria={"a": None, "b": None}, gold="c")
    problems = devset_module.validate([broken])
    assert any("gold" in problem for problem in problems)
    vendor = devset_module.DevItem(id="x", type="noul", state="s", instructions="i",
                                   criteria={"true": "y", "false": "n"}, gold=True,
                                   provenance="copied from a vendor eval")
    assert any("vendor" in problem for problem in devset_module.validate([vendor]))


def test_every_dev_item_builds_a_valid_request_with_its_gold_in_the_criteria():
    for item in devset_module.load():
        payload = devset_module.request_for(item, model="m.gguf")
        request = schema.parse_request(payload)                # raises on a malformed item
        assert len(request.questions) == 1
        question = request.questions[0]
        assert question.type == item.type
        assert item.id in payload["questions"]
        assert devset_module.gold_key(item) in question.options


def test_the_dev_set_is_committed_inside_the_package_and_has_no_duplicate_states():
    path = devset_module.devset_path()
    assert path.exists() and path.is_file()
    assert path.parent == pathlib.Path(devset_module.__file__).parent
    items = devset_module.load()
    states = [item.state for item in items]
    assert len(set(states)) == len(states)                     # no copy-pasted states
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(items)


# --------------------------------------------------------------------------- isolation (A-E2-7)
def test_no_suite_touches_the_registry_or_the_network(monkeypatch):
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a benchmark must never reach the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    from ggufone.registry import store

    def no_registry(*args: object, **kwargs: object) -> None:
        raise AssertionError("a benchmark must never read the model registry")

    for name in ("load_registry", "resolve", "registry_path", "data_home"):
        monkeypatch.setattr(store, name, no_registry)

    for suite in harness.SUITES:
        report = suites.run_suite(harness.BenchConfig(suite=suite, model_path="/tmp/fake.gguf",
                                                      runs=3, items=6),
                                  factory=bench_factory(script=devset_script(right=True)))
        assert report["schema"] == harness.SCHEMA
        assert report["suite"] == suite
        json.loads(json.dumps(report))                         # JSON-clean end to end


def test_the_bench_modules_never_import_the_registry_at_module_level():
    """A-E2-7 at the source level: the registry stack may only be imported lazily, in-function.

    (`ggufone.runtime.finder` — which bench uses to locate a *runtime bundle* — happens to import
    the registry package itself; what bench must never do is read `registry.json`, resolve an
    alias or pull a model. The lazy-import rule keeps that visible in the source, and
    `test_no_suite_touches_the_registry_or_the_network` pins the behaviour.)
    """
    import ast

    root = pathlib.Path(__file__).resolve().parents[1]
    checked = 0
    for module in ("harness", "suites", "devset"):
        path = root / "src" / "ggufone" / "bench" / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:                                  # module level only
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("ggufone.registry", "ggufone.runtime"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("ggufone.registry", "ggufone.runtime")), \
                        f"{module}.py imports {alias.name} at module level"
        checked += 1
    assert checked == 3


def test_the_model_must_be_a_file_never_an_alias():
    with pytest.raises(cli.UserError) as excinfo:
        harness.resolve_model_path("/nope/missing.gguf")
    assert excinfo.value.code == "E_BENCH_MODEL"
    assert "registry" in str(excinfo.value)
    with pytest.raises(cli.UserError):
        harness.resolve_model_path(None, env={})
    assert harness.resolve_model_path(None, env={"GGUFONE_BENCH_MODEL": __file__}) == __file__


# --------------------------------------------------------------------------- the suites
def test_latency_reports_model_load_prefill_sizes_and_candidate_counts():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=5, threads=2),
                              factory=bench_factory())
    assert report["model_load"]["n"] >= 5
    sizes = [row["tokens"] for row in report["prefill"]]
    assert sorted(sizes) == [256, 2048, 8192]
    for row in report["prefill"]:
        assert row["tok_per_s"]["n"] >= 5
        assert row["ms"]["n"] >= 5
        assert row["ms"]["p50"] > 0 and row["ms"]["p95"] >= row["ms"]["p50"]
        assert row["tok_per_s"]["p50"] > 0
    candidates = [row["candidates"] for row in report["per_question"]]
    assert sorted(candidates) == [2, 4, 10]
    for row in report["per_question"]:
        assert row["ms"]["n"] >= 5
        assert row["decode_steps"] >= row["candidates"]
        assert row["prefill_reused"] is True                   # warm cache, per the card
    waves = report["wave_scaling"]
    assert [row["questions"] for row in waves] == list(range(1, 17))
    for row in waves:
        assert row["ms"]["n"] >= 5
        assert row["waves"] >= 1
    warm = report["warm_cache"]
    assert warm["prefill_reused"] is True
    assert warm["prefill_ms"]["p50"] == 0.0
    assert report["host"]["cpu_count"] >= 1
    assert report["commands"]["reproduce"].startswith("uv run ggufone bench --suite latency")


def test_latency_also_reports_what_load_reuse_saves():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=4, threads=2),
                              factory=bench_factory())
    amortised = report["load_amortisation"]
    assert amortised["calls"] == 4
    assert amortised["model_load_ms"] > 0
    assert amortised["serve_ms_per_request"] > 0
    assert amortised["one_shot_ms_per_request"] == pytest.approx(
        amortised["model_load_ms"] + amortised["serve_ms_per_request"])
    assert amortised["saved_ms_per_request"] == amortised["model_load_ms"]


def test_the_prefill_sizes_are_configurable_for_smoke_runs():
    config = harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf", runs=2,
                                 prefill_sizes=(128, 512))
    report = suites.run_suite(config, factory=bench_factory())
    assert [row["tokens"] for row in report["prefill"]] == [128, 512]
    assert cli._bench_sizes("128, 512") == (128, 512)
    assert cli._bench_sizes(None) == harness.PREFILL_SIZES
    with pytest.raises(cli.UserError):
        cli._bench_sizes("0")
    with pytest.raises(cli.UserError):
        cli._bench_sizes("1024,abc")


def test_a_backend_bundle_also_answers_for_cpu(tmp_path, monkeypatch):
    """A Vulkan/Metal bundle carries the CPU backend too: `--backend auto` still finds `cpu`."""
    from ggufone.runtime import finder

    root = tmp_path / "runtime"
    bundle = root / "b11026-linux-x64-vulkan"
    bundle.mkdir(parents=True)
    (bundle / finder.library_names()["llama"]).write_bytes(b"")
    (bundle / "libggml-vulkan.so").write_bytes(b"")
    monkeypatch.setattr(harness, "backend_runtimes", _REAL_BACKEND_RUNTIMES)
    monkeypatch.setenv("GGUFONE_BENCH_RUNTIME_DIR", str(root))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    found = harness.backend_runtimes(home=tmp_path / "data-home")
    assert found["vulkan"] == bundle
    assert found["cpu"] == bundle
    assert "cuda" not in found
    report = suites.run_suite(harness.BenchConfig(suite="throughput", model_path="/tmp/fake.gguf",
                                                  runs=1, backend="all"),
                              factory=bench_factory())
    assert [row["backend"] for row in report["backends"]] == ["cpu", "vulkan", "cuda"]
    assert report["backends"][0]["runtime_dir"] == str(bundle)


def test_a_cpu_only_bundle_is_not_mistaken_for_an_accelerator(tmp_path, monkeypatch):
    from ggufone.runtime import finder

    bundle = tmp_path / "runtime" / "b11026-linux-x64-cpu"
    bundle.mkdir(parents=True)
    (bundle / finder.library_names()["llama"]).write_bytes(b"")
    monkeypatch.setattr(harness, "backend_runtimes", _REAL_BACKEND_RUNTIMES)
    monkeypatch.setenv("GGUFONE_BENCH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("GGUFONE_RUNTIME_DIR", raising=False)
    found = harness.backend_runtimes(home=tmp_path / "data-home")
    assert set(found) == {"cpu"}
    assert "libggml-vulkan.so" in harness.backend_unavailable_reason("vulkan")


def test_latency_records_the_threads_per_row_because_they_change_everything():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=3, threads=1),
                              factory=bench_factory())
    assert report["config"]["threads"] == 1
    assert report["per_question"][0]["threads"] == 1
    assert report["wave_scaling"][0]["threads"] == 1
    assert report["warm_cache"]["threads"] == 1


def test_throughput_measures_each_local_backend_and_reports_the_missing_ones():
    report = suites.run_suite(harness.BenchConfig(suite="throughput", model_path="/tmp/fake.gguf",
                                                  runs=3, backend="all"),
                              factory=bench_factory())
    measured = {row["backend"] for row in report["backends"] if row.get("measured")}
    unavailable = {row["backend"]: row["reason"] for row in report["backends"]
                   if not row.get("measured")}
    assert measured == {"cpu"}
    assert set(unavailable) == {"vulkan", "cuda"}
    assert all(reason for reason in unavailable.values())
    assert report["ok"] is True
    row = next(row for row in report["backends"] if row["measured"])
    assert row["prefill_tok_per_s"]["n"] >= 1
    assert row["decision_ms"]["p50"] > 0
    assert row["decision_tok_per_s"]["p50"] > 0
    assert row["placement"] == "n_gpu_layers=0"                # cpu placement is explicit


def test_a_forced_backend_without_a_bundle_is_an_error_not_a_silent_fallback():
    with pytest.raises(cli.UserError) as excinfo:
        suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                             runs=1, backend="cuda"),
                         factory=bench_factory())
    assert excinfo.value.code in ("E_BENCH_BACKEND",)


def test_quality_reports_exact_match_per_type_and_overall_with_wilson_cis():
    report = suites.run_suite(harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf",
                                                  runs=1),
                              factory=bench_factory(script=devset_script(right=True)))
    assert report["devset"]["items"] >= 50
    for qtype, row in report["per_type"].items():
        assert qtype in ("choice", "score", "noul")
        assert row["n"] >= 5
        assert 0.0 <= row["agreement"] <= 1.0
        low, high = row["ci"]
        assert low <= row["agreement"] <= high
    overall = report["overall"]
    assert overall["n"] == report["devset"]["items"]
    assert overall["agreement"] == pytest.approx(1.0)
    assert len(report["items"]) == overall["n"]
    assert all({"id", "type", "expected", "got", "correct"} <= set(row)
               for row in report["items"])


def test_quality_honestly_reports_disagreement():
    report = suites.run_suite(harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf",
                                                  runs=1),
                              factory=bench_factory(script=devset_script(right=False)))
    assert report["overall"]["agreement"] == pytest.approx(0.0)
    assert report["overall"]["ci"][1] < 0.10
    assert any(not row["correct"] for row in report["items"])


def test_quality_caps_the_dev_set_with_items_for_smoke_runs():
    report = suites.run_suite(harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf",
                                                  runs=1, items=6),
                              factory=bench_factory(script=devset_script(right=True)))
    assert report["devset"]["items"] == 6
    assert report["overall"]["n"] == 6


def test_calibration_reports_bins_ece_coverage_correlation_and_all_three_modes():
    report = suites.run_suite(harness.BenchConfig(suite="calibration", model_path="/tmp/fake.gguf",
                                                  runs=1),
                              factory=bench_factory(script=devset_script(right=True)))
    assert report["n_bins"] == 10
    bins = report["reliability"]
    assert len(bins) == 10
    assert sum(entry["n"] for entry in bins) == report["n"]
    assert 0.0 <= report["ece"] <= 1.0
    assert set(report["modes"]) == {"normalized_peak", "entropy", "margin"}
    for _mode, row in report["modes"].items():
        assert 0.0 <= row["ece"] <= 1.0
        assert row["n"] == report["n"]
        assert len(row["bins"]) == 10
    correlation = report["confidence_coverage_correlation"]
    assert correlation is None or -1.0 <= correlation <= 1.0
    assert report["coverage"]["mean"] > 0.0
    assert all("probabilities" in row for row in report["items"])
    assert report["per_type"]["choice"]["n"] >= 5              # the quality table comes along


def test_determinism_runs_three_repeats_per_backend_and_compares_timings_stripped_bytes():
    report = suites.run_suite(harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf",
                                                  backend="all"),
                              factory=bench_factory())
    assert report["repeats"] == 3
    assert [row["backend"] for row in report["backends"]] == ["cpu"]
    row = report["backends"][0]
    assert len(row["digests"]) == 3
    assert len(set(row["digests"])) == 1
    assert row["identical"] is True
    assert row["ok"] is True
    assert report["ok"] is True
    assert report["threads"] == 1


def test_determinism_flags_a_nondeterministic_model():
    # the determinism request's state, with the label the deterministic version would pick
    script = {"The billing dashboard is blank for every user after login since 09:12.": "billing"}
    report = suites.run_suite(harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf",
                                                  backend="cpu"),
                              factory=bench_factory(script=script, nondeterministic=True))
    row = report["backends"][0]
    assert len(set(row["digests"])) == 3
    assert row["identical"] is False
    assert row["ok"] is False
    assert report["ok"] is False
    assert any("differ" in note.lower() for note in report["notes"])


# --------------------------------------------------------------------------- the wave accounting
def test_the_wave_count_is_a_decode_batch_count_not_a_fork_bucket():
    """The headline of BENCHMARKS.md: `usage.waves` counts decode calls after the prefill.

    Five candidates in one question cost `1 (suffix) + (max candidate length - 1)` batches; the
    request-level `waves` is the sum over question groups — never `ceil(forks / n_seq_max)`.
    """
    request = schema.parse_request({
        "state": "The billing dashboard is blank after login.",
        "questions": {"area": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": None, "technical": None,
                                            "platform": None, "support": None,
                                            "sales": None}}},
    })
    make = bench_factory()
    model = make(harness.ModelSpec(path="/tmp/fake.gguf", backend="cpu", threads=1))
    result = model.decide(request, n_seq_max=6, threads=1)
    assert result.usage["forks"] == 5
    # five single-token labels: one decode batch (the suffix) and zero candidate steps
    assert result.usage["waves"] == 1
    assert suites.planned_wave_breakdown(model, request, n_seq_max=6) == {
        "groups": 1, "suffix_decodes": 1, "step_decodes": 0, "waves": 1}
    # a 10-candidate question with n_seq_max=4 leaves room for 3 candidates per wave:
    # 10 candidates -> 4 groups (3 + 3 + 3 + 1), so 4 suffix decodes and 4 waves
    wide = schema.parse_request({
        "state": "x",
        "questions": {"area": {"type": "choice", "instructions": "Which?",
                               "criteria": {f"option{index}": None for index in range(10)}}},
    })
    breakdown = suites.planned_wave_breakdown(model, wide, n_seq_max=4)
    assert breakdown["groups"] == 4
    assert breakdown["suffix_decodes"] == 4
    assert breakdown["waves"] == 4 + breakdown["step_decodes"]


def test_the_latency_report_carries_the_wave_accounting_headline():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=2),
                              factory=bench_factory())
    accounting = report["wave_accounting"]
    assert accounting["groups"] == 1
    assert accounting["waves"] >= 1
    assert any("waves" in note for note in report["notes"])


def test_wave_scaling_uses_the_sequence_cap_it_reports():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=3, n_seq_max=4),
                              factory=bench_factory())
    for row in report["wave_scaling"]:
        assert row["n_seq_max"] == 4
        assert row["waves"] >= 1
        assert row["ms_per_question"] >= 0


# --------------------------------------------------------------------------- the CLI surface
def test_cli_bench_requires_a_real_model_file_and_explains_why(capsys):
    code = cli.main(["bench", "--suite", "latency", "--model", "/nope/missing.gguf"])
    err = capsys.readouterr().err
    assert code == 2
    assert "E_BENCH_MODEL" in err
    assert "registry" in err.lower()                          # says why: never the registry


def test_cli_bench_rejects_an_unknown_suite(capsys, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\x00" * 16)
    code = cli.main(["bench", "--suite", "vibes", "--model", str(model)])
    assert code == 2
    assert "E_BENCH_SUITE" in capsys.readouterr().err


def test_cli_bench_json_and_out_file(tmp_path, capsys, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\x00" * 16)
    out = tmp_path / "report.json"
    script = devset_script(right=True)
    monkeypatch.setattr(suites, "live_factory", bench_factory(script=script))
    code = cli.main(["bench", "--suite", "quality", "--model", str(model), "--runs", "1",
                     "--items", "6", "--json", "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["suite"] == "quality"
    assert report["devset"]["items"] == 6
    assert "quality" in captured.out


def test_cli_bench_renders_a_human_table_without_json(tmp_path, capsys, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\x00" * 16)
    monkeypatch.setattr(suites, "live_factory", bench_factory())
    code = cli.main(["bench", "--suite", "latency", "--model", str(model), "--runs", "3",
                     "--items", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "latency" in out
    assert "p50" in out and "p95" in out


def test_cli_bench_exits_one_when_a_gate_fails(tmp_path, capsys, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\x00" * 16)
    script = {"The billing dashboard is blank for every user after login since 09:12.": "billing"}
    monkeypatch.setattr(suites, "live_factory",
                        bench_factory(script=script, nondeterministic=True))
    code = cli.main(["bench", "--suite", "determinism", "--model", str(model), "--backend",
                     "cpu"])
    assert code == 1


def test_cli_bench_help_lists_the_frozen_flags(capsys):
    code = cli.main(["bench", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--suite" in out and "--runs" in out and "--items" in out and "--backend" in out


# --------------------------------------------------------------------------- report rendering
def test_render_report_produces_a_markdown_table_per_section():
    report = suites.run_suite(harness.BenchConfig(suite="latency", model_path="/tmp/fake.gguf",
                                                  runs=3),
                              factory=bench_factory())
    markdown = harness.render_report(report)
    assert markdown.startswith("### ")
    assert "| p50 | p95 |" in markdown
    assert "| 256 |" in markdown and "| 8192 |" in markdown
    assert "latency" in markdown


def test_render_report_covers_the_other_four_suites():
    for config, factory in (
            (harness.BenchConfig(suite="throughput", model_path="/tmp/fake.gguf", runs=2,
                                 backend="all"), bench_factory()),
            (harness.BenchConfig(suite="quality", model_path="/tmp/fake.gguf", runs=1, items=6),
             bench_factory(script=devset_script(right=True))),
            (harness.BenchConfig(suite="calibration", model_path="/tmp/fake.gguf", runs=1,
                                 items=8), bench_factory(script=devset_script(right=True))),
            (harness.BenchConfig(suite="determinism", model_path="/tmp/fake.gguf"),
             bench_factory())):
        report = suites.run_suite(config, factory=factory)
        markdown = harness.render_report(report)
        assert markdown.startswith("### ")
        assert report["suite"] in markdown
