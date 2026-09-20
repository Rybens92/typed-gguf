"""The E3 model comparison: two stored quality reports -> the published table (SPEC 5 / A-E3-3).

E3 compares the box's default 4B (`Spark-X2.5-4B-Q8_0`) with the local 23 GB MoE
(Occamy 1.0, `qwen35moe`) on the same committed dev set. Re-running both suites is expensive
(hours on a 35B-A3B with a partial offload), so the table is computed from the reports the suites
already wrote: every row carries `correct`, `type`, `coverage`, `reliability` and `probabilities`,
and this module only aggregates them.

Three things the published table must show, and therefore three things this module computes:

* **overall and per question type**, each with a 95 % Wilson interval (`harness.wilson_interval`,
  the same definition `suites.agreement` uses — the table never invents its own statistics);
* the **`low_mass` split**: an answer whose full-vocabulary mass is below the engine's floor is
  reported as `reliability: "low_mass"` (`W_LOW_MASS`), and such an answer is a *guess among the
  candidates the model happened to see*, not a decision about the question. Splitting the
  agreement into `low_mass` / `measured` keeps the comparison honest: the row that a reader
  should trust is `measured`;
* the **delta** against the baseline, so "is the 23 GB model better on this box?" is one number.

A row written before E3 stored no `reliability`; for those the split falls back to the engine's
own default floor (`schema.DEFAULT_OPTIONS["coverage_floor"]` = 0.10), which is the same threshold
`readout.reliability` applies.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from typed_gguf.bench import harness, suites
from typed_gguf.schema import OPTION_DEFAULTS

#: the engine's own floor: `readout.reliability(coverage, coverage_floor=0.10)`
DEFAULT_COVERAGE_FLOOR = float(OPTION_DEFAULTS["coverage_floor"])
LOW_MASS = "low_mass"
MEASURED = "measured"


class ComparisonError(harness.BenchError):
    """A report that cannot be compared (wrong suite, no rows) — exit code 2, like `bench`."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="E_BENCH_COMPARE")


def rows_of(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The item rows of a `quality` report, or a `ComparisonError` explaining what is missing."""
    suite = report.get("suite")
    if suite != "quality":
        raise ComparisonError(
            f"only `--suite quality` reports can be compared (got suite={suite!r}); the E3 "
            f"comparison is the agreement of two runs of the committed dev set")
    rows = report.get("items")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ComparisonError(
            "the report carries no item rows (a `--suite quality` run that measured nothing "
            "cannot be compared)")
    return list(rows)


def is_low_mass(row: Mapping[str, Any], *, coverage_floor: float = DEFAULT_COVERAGE_FLOOR) -> bool:
    """The engine's verdict when the row stores one, else the coverage floor it would use."""
    verdict = row.get("reliability")
    if isinstance(verdict, str) and verdict:
        return verdict == LOW_MASS
    return float(row.get("coverage") or 0.0) < coverage_floor


def split_by_mass(rows: Sequence[Mapping[str, Any]], *,
                  coverage_floor: float = DEFAULT_COVERAGE_FLOOR) -> dict[str, list[Any]]:
    """`{low_mass, measured}` — a partition of `rows` by the answer's own mass."""
    low = [row for row in rows if is_low_mass(row, coverage_floor=coverage_floor)]
    rest = [row for row in rows if not is_low_mass(row, coverage_floor=coverage_floor)]
    return {LOW_MASS: low, MEASURED: rest}


def agreement_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`n / correct / agreement / ci` — `suites.agreement`, kept in one place for the table."""
    return suites.agreement(rows)


def merge_reports(reports: Sequence[Mapping[str, Any]], *,
                  label: str | None = None) -> dict[str, Any]:
    """Stitch chunk reports of the same model/dev set into one `quality` report.

    A 23 GB model on a small box costs minutes per item, so the campaign is run (and committed) in
    chunks; the published table must still be one report over all measured items. Every chunk must
    come from the same suite and the same model, and no item may be measured twice — a duplicate id
    means two chunks overlapped and the agreement would be double-counted.
    """
    if not reports:
        raise ComparisonError("nothing to merge: no chunk reports")
    rows: list[Any] = []
    seen: set[str] = set()
    models = set()
    for report in reports:
        for row in rows_of(report):
            item_id = str(row.get("id"))
            if item_id in seen:
                raise ComparisonError(
                    f"item {item_id!r} was measured twice: two chunks overlap, so the merged "
                    f"agreement would count it twice")
            seen.add(item_id)
            rows.append(row)
        model = report.get("model")
        models.add(model.get("name") if isinstance(model, Mapping) else None)
    first = dict(reports[0])
    first["items"] = rows
    first["per_type"] = {qtype: agreement_block([row for row in rows if row["type"] == qtype])
                         for qtype in sorted({str(row["type"]) for row in rows})}
    first["overall"] = agreement_block(rows)
    first["chunks"] = [{"items": len(rows_of(report)),
                        "model": (report.get("model") or {}).get("name")
                        if isinstance(report.get("model"), Mapping) else None,
                        "reproduce": (report.get("commands") or {}).get("reproduce")}
                       for report in reports]
    if label:
        first["model"] = dict(first.get("model") or {})
        if isinstance(first["model"], dict):
            first["model"]["name"] = label
    if len(models) > 1:
        first["models_merged"] = sorted(str(name) for name in models)
    return first


def align(baseline: Mapping[str, Any], challenger: Mapping[str, Any]) -> dict[str, Any]:
    """Restrict two reports to the items they *both* measured (the paired comparison).

    A 23 GB model on a small box is measured in chunks that may not all land, while the baseline
    may have been measured in one long pass (`e2_quality.json`: all 60 items). Comparing "whatever
    the small model finished" against "everything the baseline did" is not a comparison of the two
    models — it mixes in the models' different item mixes. So both sides are cut to the intersection
    of their ids, and the result says how many rows that dropped on each side.
    """
    baseline_rows = rows_of(baseline)
    challenger_rows = rows_of(challenger)
    baseline_ids = {str(row.get("id")) for row in baseline_rows}
    challenger_ids = {str(row.get("id")) for row in challenger_rows}
    shared = baseline_ids & challenger_ids
    if not shared:
        raise ComparisonError(
            "the two reports share no dev item: they were not measured on the same set, so their "
            "agreements are not comparable")
    first = dict(baseline)
    first["items"] = [row for row in baseline_rows if str(row.get("id")) in shared]
    second = dict(challenger)
    second["items"] = [row for row in challenger_rows if str(row.get("id")) in shared]
    return {"baseline": first, "challenger": second,
            "items": len(shared),
            "dropped": {"baseline": len(baseline_ids - shared),
                        "challenger": len(challenger_ids - shared)}}


def model_row(report: Mapping[str, Any], *, label: str | None = None, name: str | None = None,
              coverage_floor: float = DEFAULT_COVERAGE_FLOOR) -> dict[str, Any]:
    """One model's side of the table: overall, per type, and the mass split."""
    rows = rows_of(report)
    model = report.get("model")
    model_name = name or (model.get("name") if isinstance(model, Mapping) else None)
    types = sorted({str(row["type"]) for row in rows})
    split = split_by_mass(rows, coverage_floor=coverage_floor)
    return {
        "label": label or model_name or "model",
        "model": model_name,
        "coverage_floor": coverage_floor,
        "overall": agreement_block(rows),
        "per_type": {qtype: agreement_block([row for row in rows if row["type"] == qtype])
                     for qtype in types},
        LOW_MASS: agreement_block(split[LOW_MASS]),
        MEASURED: agreement_block(split[MEASURED]),
    }


def comparison(baseline: Mapping[str, Any], challenger: Mapping[str, Any], *,
               labels: tuple[str, str] = ("baseline", "challenger"),
               coverage_floor: float = DEFAULT_COVERAGE_FLOOR) -> dict[str, Any]:
    """The whole table: both sides, the question types they share, and the overall delta."""
    first = model_row(baseline, label=labels[0], coverage_floor=coverage_floor)
    second = model_row(challenger, label=labels[1], coverage_floor=coverage_floor)
    types = sorted(set(first["per_type"]) | set(second["per_type"]))
    return {
        "labels": [first["label"], second["label"]],
        "coverage_floor": coverage_floor,
        "types": types,
        "models": [first, second],
        "delta": second["overall"]["agreement"] - first["overall"]["agreement"],
    }


def _cell(block: Mapping[str, Any]) -> str:
    if not block["n"]:
        return "—"
    low, high = block["ci"]
    return f"{block['agreement']:.3f} ({block['correct']}/{block['n']}) [{low:.3f}–{high:.3f}]"


def _delta(block_a: Mapping[str, Any], block_b: Mapping[str, Any]) -> str:
    if not block_a["n"] or not block_b["n"]:
        return "—"
    return f"{block_b['agreement'] - block_a['agreement']:+.3f}"


def render_comparison(table: Mapping[str, Any]) -> str:
    """The markdown table `docs/BENCHMARKS.md` publishes (one row per metric, one column each)."""
    first, second = table["models"]
    baseline, challenger = table["labels"]
    lines = [
        f"Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the "
        f"engine's own verdict, or `coverage < {table['coverage_floor']:.2f}` where a report "
        f"predates it.",
        "",
        f"| metric | {baseline} | {challenger} | delta |",
        "|---|---|---|---|",
    ]
    rows: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = [
        ("overall", first["overall"], second["overall"])]
    for qtype in table["types"]:
        rows.append((qtype, first["per_type"].get(qtype) or suites.agreement([]),
                     second["per_type"].get(qtype) or suites.agreement([])))
    rows.append((f"{LOW_MASS} (below the floor)", first[LOW_MASS], second[LOW_MASS]))
    rows.append((f"{MEASURED} (at or above the floor)", first[MEASURED], second[MEASURED]))
    for label, left, right in rows:
        lines.append(f"| {label} | {_cell(left)} | {_cell(right)} | {_delta(left, right)} |")
    lines.append("")
    got = second["overall"]["agreement"]
    base = first["overall"]["agreement"]
    if table["delta"] > 0:
        verdict = f"is better than `{baseline}` by {table['delta']:+.3f}"
    elif table["delta"] < 0:
        verdict = f"is worse than `{baseline}` by {table['delta']:+.3f}"
    else:
        verdict = f"matches `{baseline}` exactly"
    lines.append(f"`{challenger}` {verdict} overall ({base:.3f} -> {got:.3f}); the `{MEASURED}` "
                 f"row is the one to read first.")
    return "\n".join(lines)
