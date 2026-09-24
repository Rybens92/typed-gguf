"""E3e doc gate (card `t_4c48f40a`): the two switches are documented where a reader looks.

The repo's convention for a card that adds a knob: the option is named in the template document
(`docs/TEMPLATES.md`), the measured table is in `docs/BENCHMARKS.md`, and the evidence document is
the one the card's name points at. A doc-only deletion is exactly the kind of change that passes
every unit test and still misleads the next reader — this file fails the build if it happens.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES = (ROOT / "docs" / "TEMPLATES.md").read_text(encoding="utf-8")
BENCHMARKS = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
EVIDENCE = ROOT / "docs" / "evidence" / "e3e_role_split_t_4c48f40a.md"


def test_the_template_document_names_both_switches_and_the_contract() -> None:
    for needle in ("role_split", "json_instructed", "json_contract"):
        assert needle in TEMPLATES, f"docs/TEMPLATES.md no longer names `{needle}`"
    # the role split's acceptance is a *measured* property: the section must say so, not imply it
    section = TEMPLATES.split("### The question's placement and the instructed JSON", 1)
    assert len(section) == 2, "docs/TEMPLATES.md lost the E3e section"
    body = section[1].split("## 5.", 1)[0]
    assert "E_ROLE_SPLIT_UNSUPPORTED" in body
    assert "chat_template" in body


def test_the_benchmark_document_has_the_e3e_section_with_its_table() -> None:
    section = BENCHMARKS.split("## 9. E3e", 1)
    assert len(section) == 2, "docs/BENCHMARKS.md lost the E3e section"
    body = section[1]
    assert "role_split" in body and "json_instructed" in body
    assert "e3e_roles_decision" in body, "the section must name the tool that decided the table"
    assert re.search(r"\|.*agreement.*\|", body), "the section must carry the 3x2 table"
    assert "Frozen" in body or "frozen" in body, "the section must state that the defaults froze"


def test_the_evidence_document_exists_and_points_at_its_instruments() -> None:
    text = EVIDENCE.read_text(encoding="utf-8")
    assert "t_4c48f40a" in text
    assert "tools/e3e_roles_decision.py" in text
    assert "tools/e3e_role_render.py" in text
    # the baseline cell's provenance: the receipt keeps the pre-move spelling of the path it was
    # measured with (`docs/evidence/**` is frozen history — card t_f2636df1 moved the dirs, and a
    # receipt is a record), so the pin is the artifact's name, not the retired directory prefix.
    assert "bench_templated_shipped.json" in text


def test_the_freeze_claim_counts_the_cue_refusal_verdict() -> None:
    """F1 (audit `t_57bd3db2` §8.1): the honest re-score line, in both documents.

    The freeze check compares `got` + `reliability` + `cue.refused`; on this data one item moved —
    the refusal verdict on `n16`, not the answer — so every "60/60" / "no decision moved" sentence
    the correction touches must read 59/60 with the item named.
    """
    text = EVIDENCE.read_text(encoding="utf-8")
    assert "59/60 decisions identical" in text
    assert "refusal verdict" in text
    section = BENCHMARKS.split("## 9. E3e", 1)[1]
    assert "59 of 60 decisions" in section
    assert "on all 60 decisions" not in section, "the corrected line is back to 60/60"
    assert "refusal verdict" in section


def test_the_instrument_line_names_the_command_that_actually_ran() -> None:
    """F4 (audit `t_57bd3db2` §8.3): the arms ran the harness default, not `--runs 1`.

    `docs/evidence/e3e/run_arms.sh` passes no `--runs`, so `harness.DEFAULT_RUNS = 5` applied and
    the reports' own reproduce lines say `--runs 5`. The quality suite decodes each item once, so no
    number moved
    — the sentence was the only thing wrong.
    """
    head = EVIDENCE.read_text(encoding="utf-8").split("## Why this card exists", 1)[0]
    assert "`--runs 1`" not in head
    assert "--runs 5" in head
    assert "harness default" in head
