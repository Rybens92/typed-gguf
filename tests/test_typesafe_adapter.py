"""A-E1b-11 (+A-E1b-6): the typesafe adapter against the doc-capture fixtures.

The fixtures (`tests/fixtures/typesafe_doc_captures.json`) are the probability vectors and the
documented confidence values captured from the adapter target's documentation — the same table
`docs/verify_runtime_contract.py` section C executes. What this file pins:

  * the confidence statistic reproduces the documented values within ±0.02 (the documented
    outlier is reproduced verbatim and carries no parity claim);
  * the native and the typesafe rendering carry byte-comparable numbers;
  * the answer key sets are exactly the documented ones:
    `type` + value key + `probabilities` + `confidence` (+ `legend` for score);
  * `noul` answers carry no confidence at all.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ggufone import schema
from ggufone.engine import readout

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "typesafe_doc_captures.json"
CAPTURES = json.loads(FIXTURES.read_text(encoding="utf-8"))["captures"]
OUTLIER = json.loads(FIXTURES.read_text(encoding="utf-8"))["_provenance"]["outlier"]


def native_result(capture: dict) -> dict:
    keys = list(capture["criteria"])
    probabilities = {key: value for key, value in zip(keys, capture["probabilities"],
                                                      strict=True)}
    confidence = readout.confidence_normalized_peak(list(probabilities.values()))
    if capture["type"] == "choice":
        answer = {"type": "choice", "choice": keys[readout.argmax_first(
            list(probabilities.values()))], "probabilities": probabilities,
            "confidence": confidence, "legend": {key: None for key in keys},
            "coverage": 0.93, "reliability": "ok", "decode_steps": len(keys)}
    else:
        answer = {"type": "score", "score": readout.score_weighted_mean(
            list(probabilities.values())), "probabilities": probabilities,
            "confidence": confidence, "legend": {str(i): level for i, level in enumerate(keys)},
            "coverage": 0.93, "reliability": "ok", "decode_steps": len(keys)}
    return {
        "model": "spark-x2.5-4b-q8_0",
        "engine": {"runtime": "llama.cpp b11026", "backend": "cpu", "readout": "sequence",
                   "kv_unified": True, "n_ctx": 2048, "n_seq_max": 9, "prefix_tokens": 128,
                   "state_id": "sha256:fixture", "prefill_reused": False},
        "answers": {capture["id"]: answer},
        "usage": {"input_tokens": 128, "output_tokens": len(keys), "questions": 1, "forks": 1,
                  "prefill_tokens": 128, "decode_steps": len(keys), "waves": 1},
        "timings": {"model_load_ms": 0.0, "prefill_ms": 1.0, "questions_ms": 1.0,
                    "total_ms": 2.0},
        "warnings": [],
    }


@pytest.mark.parametrize("capture", CAPTURES, ids=[c["id"] for c in CAPTURES])
def test_confidence_matches_the_documented_value(capture: dict) -> None:
    ours = readout.confidence_normalized_peak(capture["probabilities"])
    assert abs(ours - capture["documented_confidence"]) <= 0.02, capture["id"]


def test_the_documented_outlier_is_reproduced_without_a_parity_claim() -> None:
    ours = readout.confidence_normalized_peak(OUTLIER["probabilities"])
    assert abs(ours - 0.760) < 1e-3            # our statistic, not the documented 0.596
    assert abs(ours - OUTLIER["documented_confidence"]) > 0.02
    assert "no parity claim" in OUTLIER["note"].lower()


@pytest.mark.parametrize("capture", CAPTURES, ids=[c["id"] for c in CAPTURES])
def test_answer_key_sets_are_byte_comparable_to_the_documentation(capture: dict) -> None:
    result = native_result(capture)
    native = schema.render_response(result, format="native")
    typesafe = schema.render_response(result, format="typesafe")
    answer = typesafe["answers"][capture["id"]]
    value_key = {"choice": "choice", "score": "score", "noul": "noul"}[capture["type"]]
    expected = {"type", value_key, "probabilities", "confidence"}
    if capture["type"] == "score":
        expected.add("legend")
    assert set(answer) == expected
    assert list(answer) == ["type", value_key, "probabilities", "confidence"] + \
        (["legend"] if capture["type"] == "score" else [])
    # native exposes strictly more, and every shared number is identical
    assert set(native["answers"][capture["id"]]) > set(answer)
    assert native["answers"][capture["id"]]["probabilities"] == answer["probabilities"]
    assert native["answers"][capture["id"]]["confidence"] == answer["confidence"]
    assert set(typesafe) == {"model", "answers", "usage"}
    assert set(typesafe["usage"]) == {"input_tokens", "output_tokens"}
    assert "engine" not in typesafe and "timings" not in typesafe and "warnings" not in typesafe


def test_noul_answers_carry_no_confidence_in_either_format() -> None:
    result = native_result(CAPTURES[0])
    result["answers"] = {"page": {"type": "noul", "noul": 0.83,
                                  "probabilities": {"yes": 0.83, "no": 0.17},
                                  "coverage": 0.9, "reliability": "ok", "decode_steps": 1}}
    for fmt in ("native", "typesafe"):
        answer = schema.render_response(result, format=fmt)["answers"]["page"]
        assert "confidence" not in answer
        assert answer["noul"] == 0.83
