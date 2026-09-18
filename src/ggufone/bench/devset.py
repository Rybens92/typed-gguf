"""The committed labeled dev set for the quality/calibration suites (SPEC 5 / A-E2-3, S-10).

One JSONL record per item — `type` is exactly one of `choice | score | noul`, `criteria` is the
wire-shaped criteria of that type, and `gold` is the expected answer in the item's own domain:

| type   | `criteria`                    | `gold`              | candidate the model must pick |
|--------|-------------------------------|---------------------|-------------------------------|
| choice | `{name: description|null}`    | an option **name**  | that name                     |
| score  | `[level, ...]` (2..10)        | a **level index**   | `str(index)`                  |
| noul   | `{"true": str, "false": str}` | `true` or `false`   | `"yes"` / `"no"`              |

The comparison is uniform (`gold_key`): the answer's highest-probability candidate must be the
gold one — `choice == argmax(probabilities)` for choice, the argmax level for score, and the
`yes`/`no` argmax for noul. That is the "exact-match agreement" of A-E2-3, measured per type and
overall with Wilson intervals; it never looks at the model's confidence, which is what the
calibration suite is for.

**Provenance (S-10).** Every item was authored for this repository (E2, 2026-09-18) with a
deliberately unambiguous state, so that a human reader and the model should agree. No vendor
evaluation set, no scraped benchmark and no model output is reused anywhere in this file.
"""
from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEV_SET_SCHEMA = "ggufone.bench.devset/v1"
PROVENANCE = ("authored for ggufone E2 (2026-09-18); no vendor evaluation set is reused")
FILE_NAME = "devset.jsonl"
QUESTION_TYPES = ("choice", "score", "noul")
MAX_STATE_WORDS = 200            # A-E2-3: <= 200 tokens; the word count is the offline proxy
MIN_ITEMS = 50


@dataclass(frozen=True)
class DevItem:
    """One labeled decision problem."""

    id: str
    type: str
    state: str
    instructions: str
    criteria: Any
    gold: Any
    note: str = ""
    provenance: str = PROVENANCE

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "state": self.state,
                "instructions": self.instructions, "criteria": self.criteria, "gold": self.gold,
                "note": self.note, "provenance": self.provenance}


def devset_path(path: str | pathlib.Path | None = None) -> pathlib.Path:
    """The committed dev set: shipped inside the package, so no run ever downloads one."""
    return pathlib.Path(path) if path else pathlib.Path(__file__).with_name(FILE_NAME)


def load(path: str | pathlib.Path | None = None) -> list[DevItem]:
    """Parse the JSONL dev set (a malformed line names itself instead of raising KeyError)."""
    target = devset_path(path)
    items: list[DevItem] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{number} is not JSON: {exc}") from exc
        items.append(DevItem(
            id=record["id"], type=record["type"], state=record["state"],
            instructions=record.get("instructions", ""), criteria=record["criteria"],
            gold=record["gold"], note=record.get("note", ""),
            provenance=record.get("provenance", PROVENANCE)))
    return items


def counts(items: list[DevItem]) -> dict[str, int]:
    """Items per question type (the A-E2-3 ">= 3 question types" witness)."""
    result: dict[str, int] = {}
    for item in items:
        result[item.type] = result.get(item.type, 0) + 1
    return result


def labels_of(item: DevItem) -> tuple[str, ...]:
    """The candidate texts in wire order — exactly the keys the response reports back."""
    if item.type == "choice":
        return tuple(item.criteria.keys())
    if item.type == "score":
        return tuple(str(index) for index in range(len(item.criteria)))
    return ("yes", "no")


def stratify(items: list[DevItem], *, per_type: int) -> list[DevItem]:
    """At most `per_type` items of every question type, in the committed `QUESTION_TYPES` order.

    The committed dev set is type-blocked (24 choice / 18 score / 18 noul), so a "first N items"
    cut measures one type and nothing else. `bench --quick` wants the opposite: a few items of
    *each* type, so every per-type row has samples and the three confidence modes have something
    to disagree about. A type the set does not carry — or one it is short of — contributes what it
    has: never an error, never padding, never a re-ordering of the item itself.
    """
    if per_type <= 0:
        return []
    picked: list[DevItem] = []
    for qtype in QUESTION_TYPES:
        picked.extend([item for item in items if item.type == qtype][:per_type])
    return picked


def gold_key(item: DevItem) -> str:
    """The gold answer in the candidate domain (the key that must win the restricted softmax)."""
    if item.type == "choice":
        return str(item.gold)
    if item.type == "score":
        return str(int(item.gold))
    return "yes" if item.gold else "no"


def gold_label(item: DevItem) -> str:
    """The gold answer as the model should *say* it (same domain: labels are the wire keys)."""
    return gold_key(item)


def request_for(item: DevItem, *, model: str, **options: Any) -> dict[str, Any]:
    """The native request payload for one item (one question, id `item.id`)."""
    question: dict[str, Any] = {"type": item.type, "criteria": item.criteria}
    if item.instructions:
        question["instructions"] = item.instructions
    payload: dict[str, Any] = {"state": item.state, "model": model,
                               "questions": {item.id: question}}
    if options:
        payload["options"] = dict(options)
    return payload


def batch_payload(items: Sequence[DevItem], *, model: str, limit: int = 20,
                  n_seq_max: int | None = None) -> dict[str, Any]:
    """One request that asks the first `limit` items as a *batch* on a single state (A-E3-2).

    The engine answers N questions against one prefilled state; the E3 gate needs 20 of them on a
    model that does not fit in VRAM, with `n_seq_max` small enough that the branches no longer fit
    in one decode batch (waves). The state of the first item is reused for the whole batch — the
    items are independent decisions, and this is a load/wave test, not an agreement measurement
    (that is `--suite quality`, which asks each item against its own state).
    """
    chosen = list(items)[:int(limit)]
    if not chosen:
        raise ValueError("a batch needs at least one dev item")
    questions: dict[str, Any] = {}
    for item in chosen:
        body: dict[str, Any] = {"type": item.type, "criteria": item.criteria}
        if item.instructions:
            body["instructions"] = item.instructions
        questions[item.id] = body
    payload: dict[str, Any] = {"state": chosen[0].state, "model": model, "questions": questions}
    if n_seq_max is not None:
        payload["options"] = {"n_seq_max": int(n_seq_max)}
    return payload


def stratified_chunks(items: Sequence[DevItem], size: int) -> list[list[DevItem]]:
    """Split the dev set into chunks of `size`, round-robin over the question types.

    A prefix of the committed file is type-skewed, so a shortened run would measure one question
    type and call it the suite. Round-robin keeps every chunk (and therefore every prefix of
    chunks) a mixture of `choice | score | noul`, which is what makes an interrupted campaign
    still comparable — the pairing with the baseline stays valid however many chunks landed.
    """
    if size < 1:
        raise ValueError("a chunk needs size >= 1")
    buckets: dict[str, list[DevItem]] = {qtype: [] for qtype in QUESTION_TYPES}
    for item in items:
        buckets.setdefault(item.type, []).append(item)
    ordered: list[DevItem] = []
    while any(buckets.values()):
        for qtype in QUESTION_TYPES:
            queue = buckets.get(qtype) or []
            if queue:
                ordered.append(queue.pop(0))
    for leftover in buckets.values():
        if leftover:
            ordered.extend(leftover)
    return [ordered[start:start + size] for start in range(0, len(ordered), size)]


def validate(items: list[DevItem], *, max_words: int = MAX_STATE_WORDS) -> list[str]:
    """Every A-E2-3/S-10 constraint as a list of problems (empty list = the set is valid)."""
    problems: list[str] = []
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate item ids")
    states = [item.state for item in items]
    if len(states) != len(set(states)):
        problems.append("duplicate states")
    if len(items) < MIN_ITEMS:
        problems.append(f"only {len(items)} items (A-E2-3 needs >= {MIN_ITEMS})")
    seen: set[str] = set()
    for item in items:
        seen.add(item.type)
        if item.type not in QUESTION_TYPES:
            problems.append(f"{item.id}: unknown type {item.type!r}")
            continue
        if not isinstance(item.state, str) or not item.state.strip():
            problems.append(f"{item.id}: empty state")
        elif len(item.state.split()) > max_words:
            problems.append(f"{item.id}: state is {len(item.state.split())} words "
                            f"(A-E2-3 caps an item at {max_words} tokens)")
        if item.type == "choice":
            if not isinstance(item.criteria, dict) or len(item.criteria) < 2:
                problems.append(f"{item.id}: choice needs >= 2 criteria")
            elif item.gold not in item.criteria:
                problems.append(f"{item.id}: gold {item.gold!r} is not one of the criteria")
        elif item.type == "score":
            if not isinstance(item.criteria, list) or not 2 <= len(item.criteria) <= 10:
                problems.append(f"{item.id}: score needs 2..10 levels")
            elif not isinstance(item.gold, int) or not 0 <= item.gold < len(item.criteria):
                problems.append(f"{item.id}: gold {item.gold!r} is not a level index")
        else:
            if not isinstance(item.criteria, dict) or set(item.criteria) != {"true", "false"}:
                problems.append(f"{item.id}: noul criteria must be {{true, false}}")
            if not isinstance(item.gold, bool):
                problems.append(f"{item.id}: noul gold must be true|false")
        heads = [label.split()[0].lower() for label in labels_of(item) if label.split()]
        if len(heads) != len(set(heads)):
            problems.append(f"{item.id}: two candidates start with the same word "
                            f"({heads}) — the readout cannot tell them apart")
        if not item.provenance or "ggufone" not in item.provenance.lower():
            problems.append(f"{item.id}: provenance must name this repository "
                            f"(got {item.provenance!r})")
    if {"choice", "score", "noul"} - seen:
        problems.append(f"missing question types: {sorted({'choice', 'score', 'noul'} - seen)}")
    return problems
