"""E3 evidence math: the comparison table is computed from stored quality reports.

The published E3 table compares two `ggufone bench --suite quality` reports (the 4B default and
Occamy 1.0) without re-reading any model: everything it prints comes from the rows the reports
already store (`correct`, `type`, `coverage`, `reliability`). That makes the table reproducible
from the committed JSON and testable without a GPU — which is what these gates pin.
"""
from __future__ import annotations

import pytest

from ggufone.bench import compare, devset, harness


def row(item_id: str, qtype: str, *, correct: bool, coverage: float,
        reliability: str | None = None) -> dict:
    payload = {"id": item_id, "type": qtype, "expected": "a", "got": "a" if correct else "b",
               "correct": correct, "confidence": 0.5, "coverage": coverage,
               "probabilities": {"a": 0.6, "b": 0.4}}
    if reliability is not None:
        payload["reliability"] = reliability
    return payload


def report(rows: list[dict], *, label: str = "model") -> dict:
    return {"schema": harness.SCHEMA, "suite": "quality", "model": {"name": label},
            "items": rows, "ok": True}


def test_low_mass_split_prefers_the_engines_own_verdict():
    rows = [row("a", "choice", correct=True, coverage=0.9, reliability="ok"),
            row("b", "choice", correct=False, coverage=0.02, reliability="low_mass")]
    split = compare.split_by_mass(rows)
    assert [item["id"] for item in split["low_mass"]] == ["b"]
    assert [item["id"] for item in split["measured"]] == ["a"]


def test_low_mass_split_falls_back_to_the_coverage_floor_when_the_row_has_no_verdict():
    """The E2 reports predate a stored `reliability`; the default floor is the engine's (0.10)."""
    rows = [row("a", "choice", correct=True, coverage=0.09),
            row("b", "choice", correct=True, coverage=0.10)]
    split = compare.split_by_mass(rows)
    assert [item["id"] for item in split["low_mass"]] == ["a"]
    assert [item["id"] for item in split["measured"]] == ["b"]


def test_the_split_partitions_every_row():
    rows = [row("a", "choice", correct=True, coverage=0.9, reliability="ok"),
            row("b", "score", correct=False, coverage=0.01),
            row("c", "noul", correct=True, coverage=0.5, reliability="low_confidence")]
    split = compare.split_by_mass(rows)
    assert len(split["low_mass"]) + len(split["measured"]) == len(rows)
    assert not ({item["id"] for item in split["low_mass"]}
                & {item["id"] for item in split["measured"]})


def test_agreement_is_the_harness_wilson_interval():
    rows = [row(str(index), "choice", correct=index < 3, coverage=0.5) for index in range(5)]
    block = compare.agreement_block(rows)
    assert block["n"] == 5
    assert block["correct"] == 3
    assert block["agreement"] == 0.6
    assert block["ci"] == list(harness.wilson_interval(3, 5))


def test_empty_blocks_report_zero_without_dividing_by_zero():
    block = compare.agreement_block([])
    assert block == {"n": 0, "correct": 0, "agreement": 0.0, "ci": [0.0, 1.0]}


def test_model_row_carries_overall_per_type_and_the_mass_split():
    rows = [row("c1", "choice", correct=True, coverage=0.9, reliability="ok"),
            row("c2", "choice", correct=False, coverage=0.02, reliability="low_mass"),
            row("s1", "score", correct=True, coverage=0.4, reliability="ok")]
    entry = compare.model_row(report(rows, label="Spark-X2.5-4B"), label="4B default")
    assert entry["label"] == "4B default"
    assert entry["overall"]["n"] == 3 and entry["overall"]["correct"] == 2
    assert set(entry["per_type"]) == {"choice", "score"}
    assert entry["per_type"]["choice"]["n"] == 2
    assert entry["low_mass"]["n"] == 1
    assert entry["measured"]["n"] == 2


def test_comparison_puts_the_baseline_first_and_reports_the_delta():
    baseline = model_rows(correct_first=1)
    challenger = model_rows(correct_first=3)
    table = compare.comparison(baseline, challenger, labels=("4B default", "Occamy 1.0"))
    assert [entry["label"] for entry in table["models"]] == ["4B default", "Occamy 1.0"]
    assert table["models"][0]["overall"]["agreement"] == pytest.approx(0.25)
    assert table["models"][1]["overall"]["agreement"] == pytest.approx(0.75)
    assert table["delta"] == pytest.approx(0.5)


def model_rows(*, correct_first: int) -> dict:
    rows = [row(f"i{index}", "choice", correct=index < correct_first, coverage=0.5,
                reliability="ok") for index in range(4)]
    return report(rows)


def test_render_names_every_model_question_type_and_the_split():
    baseline = model_rows(correct_first=1)
    challenger = model_rows(correct_first=3)
    rendered = compare.render_comparison(
        compare.comparison(baseline, challenger, labels=("4B default", "Occamy 1.0")))
    assert "4B default" in rendered and "Occamy 1.0" in rendered
    assert "overall" in rendered and "low_mass" in rendered and "choice" in rendered
    assert "0.25" in rendered and "0.75" in rendered       # 1/4 and 3/4 agreement
    # every table line is a markdown row with the same number of cells
    widths = {line.count("|") for line in rendered.splitlines() if line.startswith("| ")}
    assert len(widths) == 1


def test_reports_without_rows_are_rejected():
    with pytest.raises(compare.ComparisonError):
        compare.model_row({"suite": "quality"}, label="empty")


def test_non_quality_reports_are_rejected():
    with pytest.raises(compare.ComparisonError):
        compare.model_row({"suite": "latency", "items": []}, label="latency")


# --------------------------------------------------------------- the 20-question batch (A-E3-2)
def test_batch_payload_merges_the_first_items_into_one_request():
    items = devset.load()
    payload = devset.batch_payload(items, model="occamy", limit=20)
    assert len(payload["questions"]) == 20
    assert list(payload["questions"]) == [item.id for item in items[:20]]
    assert payload["state"] == items[0].state
    assert payload["model"] == "occamy"
    first = payload["questions"][items[0].id]
    assert first["type"] == items[0].type and first["criteria"] == items[0].criteria


def test_batch_payload_can_constrain_the_sequence_cap():
    payload = devset.batch_payload(devset.load(), model="occamy", limit=20, n_seq_max=4)
    assert payload["options"]["n_seq_max"] == 4


def test_batch_payload_rejects_an_empty_selection():
    with pytest.raises(ValueError):
        devset.batch_payload([], model="occamy")


# ------------------------------------------------- chunked campaigns: subsets and merges
def dev_items(per_type: int = 4) -> list[devset.DevItem]:
    items: list[devset.DevItem] = []
    for qtype in devset.QUESTION_TYPES:
        criteria = {"a": "x", "b": "y"} if qtype == "choice" else (
            ["low", "high"] if qtype == "score" else {"true": "t", "false": "f"})
        gold = "a" if qtype == "choice" else (0 if qtype == "score" else True)
        for index in range(per_type):
            items.append(devset.DevItem(id=f"{qtype}{index}", type=qtype, state="state",
                                        instructions="", criteria=criteria, gold=gold))
    return items


def test_stratified_chunks_interleave_the_question_types():
    chunks = devset.stratified_chunks(dev_items(), 6)
    assert len(chunks) == 2
    assert [item.type for item in chunks[0]] == ["choice", "score", "noul",
                                                 "choice", "score", "noul"]
    assert [item.type for item in chunks[1]] == ["choice", "score", "noul",
                                                 "choice", "score", "noul"]
    assert len({item.id for chunk in chunks for item in chunk}) == 12


def test_stratified_chunks_keep_a_short_tail_a_mixture():
    chunks = devset.stratified_chunks(dev_items(per_type=2), 5)
    assert [len(chunk) for chunk in chunks] == [5, 1]
    assert len({item.id for chunk in chunks for item in chunk}) == 6


def test_stratified_chunks_reject_a_zero_size():
    with pytest.raises(ValueError):
        devset.stratified_chunks([], 0)


def test_merge_reports_concatenates_rows_and_recomputes_agreement():
    first = report([row("a1", "choice", correct=True, coverage=0.9, reliability="ok")])
    second = report([row("b1", "choice", correct=False, coverage=0.9, reliability="ok"),
                     row("b2", "score", correct=True, coverage=0.9, reliability="ok")])
    merged = compare.merge_reports([first, second])
    assert merged["overall"]["n"] == 3 and merged["overall"]["correct"] == 2
    assert merged["per_type"]["score"]["n"] == 1
    assert [chunk["items"] for chunk in merged["chunks"]] == [1, 2]


def test_merge_reports_refuses_an_item_measured_twice():
    first = report([row("a1", "choice", correct=True, coverage=0.9, reliability="ok")])
    second = report([row("a1", "choice", correct=True, coverage=0.9, reliability="ok")])
    with pytest.raises(compare.ComparisonError):
        compare.merge_reports([first, second])


def test_merge_reports_needs_a_chunk():
    with pytest.raises(compare.ComparisonError):
        compare.merge_reports([])


def test_merge_reports_labels_the_merged_model_and_flags_a_mixed_merge():
    """`--suite merge --label` names the report; merging two different models is recorded."""
    occamy = report([row("a1", "choice", correct=True, coverage=0.9, reliability="ok")],
                    label="Occamy 1.0")
    other = report([row("b1", "choice", correct=False, coverage=0.9, reliability="ok")],
                   label="Tiel-Coder")
    merged = compare.merge_reports([occamy, other], label="Occamy 1.0 (chunks)")
    assert merged["model"]["name"] == "Occamy 1.0 (chunks)"
    assert merged["models_merged"] == ["Occamy 1.0", "Tiel-Coder"]


def test_render_calls_a_worse_and_an_equal_challenger_by_name():
    worse = compare.render_comparison(compare.comparison(
        model_rows(correct_first=3), model_rows(correct_first=1), labels=("base", "worse")))
    assert "is worse than" in worse
    equal = compare.render_comparison(compare.comparison(
        model_rows(correct_first=2), model_rows(correct_first=2), labels=("base", "twin")))
    assert "matches" in equal and "exactly" in equal


def test_align_keeps_only_the_items_both_models_measured():
    baseline = report([row("a1", "choice", correct=True, coverage=0.9, reliability="ok"),
                       row("a2", "choice", correct=False, coverage=0.9, reliability="ok")])
    challenger = report([row("a2", "choice", correct=True, coverage=0.9, reliability="ok"),
                         row("a3", "score", correct=True, coverage=0.9, reliability="ok")])
    paired = compare.align(baseline, challenger)
    assert paired["items"] == 1
    assert [item["id"] for item in paired["baseline"]["items"]] == ["a2"]
    assert [item["id"] for item in paired["challenger"]["items"]] == ["a2"]
    assert paired["dropped"] == {"baseline": 1, "challenger": 1}


def test_align_refuses_reports_without_a_shared_item():
    baseline = report([row("a1", "choice", correct=True, coverage=0.9, reliability="ok")])
    challenger = report([row("z9", "choice", correct=True, coverage=0.9, reliability="ok")])
    with pytest.raises(compare.ComparisonError):
        compare.align(baseline, challenger)


# ------------------------------------------------- the quality report proves its compute path
def test_the_quality_report_carries_the_device_attribution(monkeypatch, tmp_path):
    """A table on a box with several bundles must say what *computed* it, not what was requested.

    The coordinator's E2-provenance audit found bench rows labelled `cpu` that had measured a GPU
    path (and a `vulkan` row with zero Vulkan buffers). E3 publishes a quality table through a
    forced backend, so the suite that produces it must carry the device evidence the engine's own
    log proves (`harness.device_usage`: device set, compute buffers per device, effective backend —
    card t_603a35a0). Pinning it here keeps E3's table from silently losing its attribution.
    """
    from ggufone.bench import suites as suites_module
    from tests.fake_engine import BenchModel

    monkeypatch.setattr(harness, "backend_runtimes",
                        lambda **kwargs: {"cpu": tmp_path / "b11026-linux-x64-cpu"})
    config = harness.BenchConfig(suite="quality", model_path="/fake/model.gguf",
                                 backend="cpu", items=1)
    report = suites_module.run_suite(config, factory=lambda spec: BenchModel(spec))
    assert report["devices"] == []                  # the fake engine logs no device line
    assert report["device_buffers"] == {}
    assert report["effective_backend"] is None      # unverified, never a claim
    assert report["backend_selection"]["requested"] == "cpu"


def test_the_chunk_writer_round_trips_through_the_devset_parser(tmp_path):
    """`--write-chunks` must produce files the dev-set parser reads back — that is the campaign.

    The chunks are what a *published* run is fed (`--devset .e3/chunks/devset_00N.jsonl`), so a
    chunk file that does not parse (or silently drops an item) would invalidate the table without
    failing anything else.
    """
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("e3_reproduce_tool",
                                                 root / "tools" / "e3_reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    written = module.write_chunks(tmp_path, size=10)
    assert [path.name for path in written] == [f"devset_{number:03d}.jsonl"
                                               for number in range(1, 7)]
    items = [item for path in written for item in devset.load(path)]
    assert len(items) == len(devset.load()) == 60
    assert len({item.id for item in items}) == 60
    assert [item.type for item in devset.load(written[0])][:3] == ["choice", "score", "noul"]
    # a chunk is a valid dev set except for the set-size floor A-E2-3 puts on the full set
    assert [problem for problem in devset.validate(devset.load(written[0]))
            if not problem.startswith("only 10 items")] == []
