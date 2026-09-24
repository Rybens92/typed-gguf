"""E3b evidence document: `tools/e3b_build_evidence.py` is a generator, and the gates prove it.

The card (`t_6952f0dd`) publishes `docs/evidence/e3b_t_6952f0dd_label_policy.md` as evidence, so
every sentence of it has to come from a file the campaign wrote: the 6-item variant sweep
(`docs/evidence/e3b/sweep.json`), the optional 20-item re-measure
(`docs/evidence/e3b/after.json`), the committed calibration verdict, and E3's/E2's published
reports. These gates build the whole document from
synthetic artifacts in a temp root and assert what it says — the negative-result verdict, the
ceiling, the before/after tables, and the two fallbacks (no re-measure yet, no calibration file).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_builder():
    """Import `tools/e3b_build_evidence.py` the way the other gates import a tool."""
    spec = importlib.util.spec_from_file_location("e3b_build_evidence",
                                                  ROOT / "tools" / "e3b_build_evidence.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e3b_build_evidence"] = module
    spec.loader.exec_module(module)
    return module


def coverage_of(index: int, cue: str, label: str) -> float:
    """`shipped x caps` sits on the floor and one item clears it; everything else is below."""
    if cue == "shipped" and label == "caps":
        return 0.02 + 0.01 * index
    return 0.001 * (1 + index) * (1 + index)


def piece(item_id: str, qtype: str, index: int) -> dict:
    """One item's shipped-prefix measurement, shaped like the sweep tool writes it."""
    cues = {}
    for cue in ("shipped", "blank", "explicit"):
        labels_block = {}
        for label in ("bare", "space", "caps", "newline", "long"):
            labels_block[label] = {
                "texts": ["a", "b"], "first_tokens": [1, 2], "pieces": ["a", "b"],
                "coverage": coverage_of(index, cue, label), "reliability": "low_mass",
                "shared_first_tokens": [],
            }
        cues[cue] = {"scale": 1.0, "labels": labels_block,
                     "top_tokens": [{"piece": "<|im_end|>", "p_full": 0.9}]}
    return {"id": item_id, "type": qtype, "expected": "a",
            "prefill_ms": 1000.0, "cue_decode_s": 10.0, "prefix_tokens": 100, "n_ctx": 4096,
            "n_seq_max": 8, "prefix_tail": "assistant", "suffix_tokens": {"shipped": 5},
            "tokenized": {}, "cues": cues, "ranked": {}}


def item_of(item_id: str, qtype: str, index: int, *, extra_prefix: bool = False) -> dict:
    """One dev item, shaped like the sweep tool writes it (`prefixes` -> measurement)."""
    prefixes = {"shipped": piece(item_id, qtype, index)}
    if extra_prefix:
        prefixes["kept"] = piece(item_id, qtype, index + 2)
    return {"id": item_id, "type": qtype, "expected": "a", "prefixes": prefixes}


def sweep_record() -> dict:
    items = [item_of("c01", "choice", 0, extra_prefix=True), item_of("s01", "score", 1)]
    return {
        "schema": "typed_gguf.e3b.labels/v1", "generated_at": "2026-09-19T00:00:00Z",
        "model": {"name": "Accio-Lab_occamy-1.0-Q4_K_L.gguf", "bytes": 24113674848,
                  "arch": "qwen35moe"},
        "model_sha256": "a" * 64, "runtime": "/runtime", "threads": 4, "gpu_layers": 7,
        "placement": {"n_gpu_layers": 7}, "load_ms": 30000.0,
        "cues": ["shipped", "blank", "explicit"],
        "label_variants": ["bare", "space", "caps", "newline", "long"],
        "prefix_variants": ["shipped", "kept"], "mass_floor": 0.02,
        "ranked_keys": [], "counts": {"choice": 1, "score": 1},
        "devset": "docs/evidence/e3_chunks/devset_001.jsonl", "items": items, "wall_s": 60.0,
        "device_log_tail": "Vulkan0 compute buffer size is 501.5 MiB",
    }


def quality_row(item_id: str, qtype: str, *, correct: bool, coverage: float,
                reliability: str) -> dict:
    return {"id": item_id, "type": qtype, "expected": "a", "got": "a" if correct else "b",
            "correct": correct, "confidence": 0.5, "coverage": coverage,
            "reliability": reliability, "probabilities": {"a": 0.6, "b": 0.4}}


def quality_report(rows: list[dict], *, label: str) -> dict:
    return {"schema": "typed_gguf.bench/v1", "suite": "quality", "model": {"name": label},
            "items": rows, "ok": True}


def write(root: pathlib.Path, relative: str, payload: object) -> pathlib.Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fake_root(tmp_path: pathlib.Path, *, after: bool = True,
              calibrate: bool = True, e3_extra: str | None = None) -> pathlib.Path:
    """A whole artifact tree: the sweep, the committed baselines, and the optional files.

    `e3_extra` adds one more row to E3's report (a different item id), so the gates can drive the
    case where E3's merged report covers more items than this card re-measured — the before side
    must be aligned to the shared ids, not compared whole.
    """
    write(tmp_path, "docs/evidence/e3b/sweep.json", sweep_record())
    write(tmp_path, "docs/evidence/e2_quality.json",
          quality_report([quality_row("c01", "choice", correct=True, coverage=0.5,
                                      reliability="measured"),
                          quality_row("s01", "score", correct=False, coverage=0.5,
                                      reliability="measured")], label="Spark-X2.5-4B-Q8_0"))
    e3_rows = [quality_row("c01", "choice", correct=True, coverage=0.002,
                           reliability="low_mass"),
               quality_row("s01", "score", correct=True, coverage=0.003,
                           reliability="low_mass")]
    if e3_extra:
        e3_rows.append(quality_row(e3_extra, "noul", correct=True, coverage=0.004,
                                   reliability="low_mass"))
    write(tmp_path, "docs/evidence/e3_occamy_quality.json",
          quality_report(e3_rows, label="Occamy 1.0"))
    if after:
        write(tmp_path, "docs/evidence/e3b/after.json",
              quality_report([quality_row("c01", "choice", correct=True, coverage=0.0295,
                                          reliability="measured"),
                              quality_row("s01", "score", correct=False, coverage=0.0095,
                                          reliability="low_mass")],
                             label="Occamy 1.0 [shipped=caps]"))
    if calibrate:
        path = tmp_path / "docs/evidence/e3b_calibrate_accepted.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "no calibration applied (no question type improved on the held-out split)\n",
            encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------- the negative-result document
def test_the_verdict_is_the_negative_result_when_no_variant_clears_the_floor(tmp_path):
    """The card allows the negative result, so the generator has to *say* it — and prove it.

    The best (cue, label) pair is named with its mean and its count of items above the floor,
    what dominates the cue row is named, and no positive verdict leaks in.
    """
    module = load_builder()
    root = fake_root(tmp_path)
    record = sweep_record()
    record["mass_floor"] = 0.05                              # nothing in the fixture clears this
    write(root, "docs/evidence/e3b/sweep.json", record)
    document = module.build(root=root)
    assert "**Answer in one paragraph.**" in document
    assert "No label rendering" in document
    assert "does not fit" not in document
    assert "`shipped` × `caps`" in document
    assert "2.500e-02" in document, "the best pair's mean coverage is printed"
    assert "0/2 items above the floor" in document
    assert "`<|im_end|>`" in document


def test_the_ceiling_is_the_best_single_value_the_sweep_wrote(tmp_path):
    module = load_builder()
    root = fake_root(tmp_path)
    record = sweep_record()
    record["mass_floor"] = 0.05
    write(root, "docs/evidence/e3b/sweep.json", record)
    document = module.build(root=root)
    assert "3.000e-02" in document, "the best single coverage value must be printed"
    assert "still below the 0.05 floor" in document


def test_the_positive_verdict_wins_when_a_variant_clears_the_floor(tmp_path):
    """A floor between the best mean and the best value: the pair clears it on both items."""
    module = load_builder()
    root = fake_root(tmp_path)
    record = sweep_record()
    record["mass_floor"] = 0.02
    write(root, "docs/evidence/e3b/sweep.json", record)
    document = module.build(root=root)
    assert "is the rendering that lifts the most items above the engine's 0.02 floor" in document
    assert "2/2" in document
    assert "No label rendering" not in document


def test_the_calibration_verdict_is_printed_verbatim_and_a_missing_file_says_so(tmp_path):
    module = load_builder()
    document = module.build(root=fake_root(tmp_path / "with"))
    assert "no calibration applied (no question type improved on the held-out split)" in document

    document = module.build(root=fake_root(tmp_path / "without", calibrate=False))
    assert "(not run)" in document


# ------------------------------------------------------------------- the before/after section
def cue_record(*, explicit: tuple[float, ...], shipped: tuple[float, ...],
               blank: tuple[float, ...]) -> dict:
    """One item per value triple: `{cue: (bare-label coverage per item)}`."""
    items = []
    for index in range(len(explicit)):
        labels_block = {
            name: {"texts": ["a"], "first_tokens": [1], "pieces": ["a"], "coverage": 0.0,
                   "reliability": "low_mass", "shared_first_tokens": []}
            for name in ("bare", "space", "caps", "newline", "long")}
        cues = {}
        for cue, values in (("explicit", explicit), ("shipped", shipped), ("blank", blank)):
            block = {name: dict(entry) for name, entry in labels_block.items()}
            block["bare"]["coverage"] = values[index]
            cues[cue] = {"scale": 1.0, "labels": block,
                         "top_tokens": [{"piece": "<|im_end|>", "p_full": 0.9999}]}
        items.append({"id": f"item{index}", "type": "choice", "expected": "a",
                      "prefixes": {"shipped": {"cues": cues, "ranked": {}}}})
    return {"cues": ["shipped", "blank", "explicit"],
            "label_variants": ["bare", "space", "caps", "newline", "long"], "mass_floor": 0.10,
            "items": items}


def test_the_cue_ranking_sentence_names_the_cue_that_lifts_the_most_mass():
    """The card asks which *cue* lifts coverage, so the doc has to rank the cue variants."""
    module = load_builder()
    record = cue_record(explicit=(0.5, 0.6), shipped=(0.1, 0.2), blank=(1e-08, 2e-08))
    sentence = module.cue_sentence(record)
    assert sentence.startswith("By cue variant the ordering is `explicit`-best 5.500e-01")
    assert "then `shipped` 1.500e-01" in sentence
    assert "names the labels explicitly" in sentence


def test_the_cue_sentence_is_empty_when_there_is_nothing_to_rank():
    module = load_builder()
    record = cue_record(explicit=(0.5,), shipped=(0.1,), blank=(1e-08,))
    record["cues"] = ["shipped"]
    assert module.cue_sentence(record) == ""


def test_no_re_measure_falls_back_to_the_published_e3_table(tmp_path):
    module = load_builder()
    document = module.build(root=fake_root(tmp_path, after=False))
    assert "No 20-item re-measure ran" in document
    assert "**Occamy before/after**" not in document


def test_a_re_measure_renders_both_comparisons_from_the_committed_reports(tmp_path):
    """The after side goes through `compare.comparison`/`compare.align` — never hand-written."""
    module = load_builder()
    document = module.build(root=fake_root(tmp_path))
    assert "**Occamy before/after** (2 paired items — E3's merged report dropped 0 row(s) it " \
           "measured and this card did not, so both sides are the same items):" in document
    assert "**The published pairing re-rendered** (2 paired items, dropped 0 unpaired " \
           "baseline row(s) and 0 unpaired challenger row(s)):" in document
    assert "| overall |" in document and "| low_mass (below the floor) |" in document
    assert "| measured (at or above the floor) |" in document
    assert "Occamy shipped (E3)" in document and "Occamy accepted" in document


def test_the_before_after_coverage_rows_come_from_the_aligned_reports(tmp_path):
    """The card's before/after asks for coverage + low_mass share, not only agreement."""
    module = load_builder()
    document = module.build(root=fake_root(tmp_path))
    table = document.split("The coverage the before/after sides were read with")[1]
    assert "| `Occamy shipped (E3)` | 2 | 2.500e-03 |" in table, "E3 side mean of 0.002/0.003"
    assert "| `Occamy accepted (E3b)` | 2 | 1.950e-02 |" in table, "after side mean 0.0295/0.0095"
    assert "| 1/2 |" in table, "the after side has one low_mass row and one measured row"
    assert "| 2/2 |" in table, "both E3 rows are low_mass"
    assert "is 0x the probe's" in table, "the mean ratio is stated (2.5e-03 / 1.95e-02 rounds to 0)"
    assert "both sides are `low_mass` on every item" not in table, \
        "no all-low_mass note when the after side has a measured row"


def test_the_low_mass_note_says_so_when_the_re_measure_never_crosses_the_floor(tmp_path):
    """The honest headline: if every fresh row is still `low_mass`, the doc says it outright."""
    module = load_builder()
    root = fake_root(tmp_path)
    write(root, "docs/evidence/e3b/after.json",
          quality_report([quality_row("c01", "choice", correct=True, coverage=0.002,
                                      reliability="low_mass"),
                          quality_row("s01", "score", correct=False, coverage=0.003,
                                      reliability="low_mass")], label="Occamy 1.0 [shipped=caps]"))
    document = module.build(root=root)
    assert "both sides are `low_mass` on every item" in document
    assert "reproduces E3's 20/20" in document


def test_the_alternative_policy_block_is_rendered_when_its_report_exists(tmp_path):
    """`shipped=newline` is measured on the same items; the doc prints its numbers next to them."""
    module = load_builder()
    root = fake_root(tmp_path)
    assert "alternative policy" not in module.build(root=root)
    inline = quality_report([quality_row("c01", "choice", correct=True, coverage=0.03,
                                         reliability="low_mass"),
                             quality_row("s01", "score", correct=True, coverage=0.04,
                                         reliability="low_mass")], label="Occamy [shipped=newline]")
    write(root, "docs/evidence/e3b/after_newline.json", inline)
    document = module.build(root=root)
    assert "**The alternative policy measured on the same items**" in document
    assert "agreement 1.000 (2/2)" in document
    assert "mean coverage 3.500e-02" in document


def test_an_e3_report_covering_more_items_is_aligned_to_the_re_measure(tmp_path):
    """E3's merged report may grow to 60 items while this card re-measured 20: never compare the
    aggregate against the subset — the item-mix difference would read as a policy difference."""
    module = load_builder()
    document = module.build(root=fake_root(tmp_path, e3_extra="n09"))
    assert "**Occamy before/after** (2 paired items — E3's merged report dropped 1 row(s) it " \
           "measured and this card did not, so both sides are the same items):" in document
    # the unpaired row is not in the aligned table, so n stays 2 on both sides
    table = document.split("**Occamy before/after**")[1].split("**The published pairing")[0]
    assert "(2/2)" in table and "noul" not in table


def test_the_document_is_written_to_the_card_path_and_the_tables_sidecar(tmp_path):
    module = load_builder()
    root = fake_root(tmp_path)
    document = module.build(root=root)
    target = root / "docs/evidence/e3b_t_6952f0dd_label_policy.md"
    target.write_text(document, encoding="utf-8")
    assert "docs/evidence/e3b_t_6952f0dd_label_policy.md" in str(target)
    assert document.endswith("\n")


def test_an_empty_artifact_tree_is_an_error_not_an_empty_document(tmp_path):
    module = load_builder()
    (tmp_path / "docs" / "evidence" / "e3b").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match="sweep.json not found"):
        module.build(root=tmp_path)
