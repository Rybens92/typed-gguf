#!/usr/bin/env python3
"""E3e (card t_4c48f40a): which prompt policy beats the shipped one — the 3x2 table, decided.

E3c measured seven cue shapes on six dev items; E3d took the question to the full 60-item dev set
and found the opener (`json_field`) the best shape, with the two-step readout close behind. Both
kept the question where the shipped instrument had it: **prefilled inside the assistant turn**,
after the generation prompt. E3e adds the two levers the E3d evidence pointed at and this tool
decides between them:

* `options.cue = json_instructed` — the instruction *instructs*: the ask line is a JSON contract and
  the assistant turn is prefilled with the opened field, so what the model answers and what the
  prompt asked for cannot disagree.
* `options.chat_format = role_split` — the question is rendered as its own **user** turn through the
  model's own chat template, so no template has to swallow a question inside an assistant turn.
* `options.json_contract = question|system` — where the contract is stated (the amendment's two
  variants): next to the candidates, or once in the system framing.

Every cell is measured on the **same 60 committed dev items**, with the same instrument
(temperature 0, fixed seed, `--backend vulkan`) — so the comparison is the **discordant pairs** and
the numbers are read *directly* from the reports: no resampling, no invented rows. The single-cell
interval is Wilson; the paired difference is the analytic (Wald) interval on the discordant counts
with its caveat named, and the test is the **exact** two-sided McNemar binomial — 60 items is small
and the chi-square approximation is not honest there. The decision rule is stated, not implied
(`decide()`).

    python3 tools/e3e_roles_decision.py \\
        --report .e3e/bench_shipped.json --report .e3e/bench_role_split_shipped.json ... \\
        --json docs/evidence/e3e_roles_decision.json \\
        --report-file docs/evidence/e3e_roles_decision.md

The tool never loads a model and never touches the network: it reads reports.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "ggufone.e3e.roles-decision/v1"
Z = 1.959963984540054          # 95 % normal quantile (Wilson / Wald)
UNIT = "discordant-pairs"      # the unit of comparison: the same items, not two samples


class DecisionError(Exception):
    """A report that cannot take part in the table (wrong suite, no items, missing config)."""


# ------------------------------------------------------------------ statistics (direct)
def wilson(successes: int, n: int) -> tuple[float | None, float | None]:
    """The Wilson score interval for one proportion — the interval every E-row already quotes."""
    if n == 0:
        return (None, None)
    phat = successes / n
    denominator = 1.0 + Z * Z / n
    centre = (phat + Z * Z / (2 * n)) / denominator
    half = Z * math.sqrt(phat * (1 - phat) / n + Z * Z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p on the discordant counts (binomial tail, no chi-square)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_difference(b: int, c: int, n: int) -> dict[str, Any]:
    """The paired risk difference `(b - c) / n` with the analytic (Wald) interval on it.

    The Wald interval is the closed-form one; it is named in the report because at 60 items and
    near-zero discordance it is optimistic (the exact test above is what decides a *difference*,
    this interval is the effect size the decision quotes).
    """
    if n == 0:
        return {"difference": None, "ci": [None, None], "discordant": 0, "caveat": "no items"}
    difference = (b - c) / n
    spread = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    return {"difference": difference,
            "ci": [max(-1.0, difference - Z * spread), min(1.0, difference + Z * spread)],
            "discordant": b + c,
            "caveat": "Wald interval on the discordant counts (optimistic near zero discordance); "
                      "the exact McNemar p is the test"}


# ------------------------------------------------------------------ reports -> cells
def load_report(path: str | pathlib.Path) -> dict[str, Any]:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    report = json.loads(text)
    if report.get("suite") != "quality":
        raise DecisionError(f"{path}: suite is {report.get('suite')!r}, not 'quality'")
    if not report.get("items"):
        raise DecisionError(f"{path}: no items")
    return report


def cell_label(report: Mapping[str, Any]) -> str:
    """`<cue>/<chat_format>` (+ `/system` when the contract is stated in the framing)."""
    config = report.get("config") or {}
    cue = str(config.get("cue") or "shipped")
    placement = str(config.get("chat_format") or "answer_sheet")
    label = f"{cue}/{placement}"
    if cue == "json_instructed" and str(config.get("json_contract") or "question") == "system":
        label += "/system"
    return label


def rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in report["items"]]


def _coverage_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(float(item.get("coverage") or 0.0) for item in items)
    if not values:
        return {"n": 0, "min": None, "p50": None, "max": None}
    return {"n": len(values), "min": values[0], "p50": values[len(values) // 2], "max": values[-1]}


def cell_stats(report: Mapping[str, Any]) -> dict[str, Any]:
    """One cell: what the report says, read straight out of its rows."""
    items = rows(report)
    correct = sum(1 for item in items if item.get("correct"))
    low, high = wilson(correct, len(items))
    per_type: dict[str, Any] = {}
    for item in items:
        bucket = per_type.setdefault(str(item.get("type")), {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += 1 if item.get("correct") else 0
    for bucket in per_type.values():
        bucket["agreement"] = bucket["correct"] / bucket["n"] if bucket["n"] else None
        bucket["ci"] = list(wilson(bucket["correct"], bucket["n"]))
    verdicts: dict[str, int] = {}
    refusals = 0
    for item in items:
        cue = item.get("cue") or {}
        if cue.get("refused"):
            refusals += 1
        if "verdict" in cue:
            verdicts[str(cue["verdict"])] = verdicts.get(str(cue["verdict"]), 0) + 1
    framing = report.get("framing") or {}
    item_framing = (items[0].get("framing") or {}) if items else {}
    prefix_tokens = [int(item.get("prefix_tokens") or 0) for item in items]
    return {
        "label": cell_label(report),
        "path": str(report.get("_path") or ""),
        "suite": report.get("suite"),
        "model": (report.get("model") or {}).get("name"),
        "cue": (report.get("config") or {}).get("cue"),
        "chat_format": (report.get("config") or {}).get("chat_format"),
        "json_contract": (report.get("config") or {}).get("json_contract"),
        "items": len(items),
        "correct": correct,
        "agreement": correct / len(items) if items else None,
        "ci": [low, high],
        "per_type": per_type,
        "low_mass": sum(1 for item in items if item.get("reliability") == "low_mass"),
        "refusals": refusals,
        "verdicts": verdicts,
        "coverage": _coverage_summary(items),
        "prefix_tokens": {"min": min(prefix_tokens, default=None),
                          "max": max(prefix_tokens, default=None)},
        "framing": {"kind": item_framing.get("kind"),
                    "renderer": item_framing.get("renderer"),
                    "family": item_framing.get("family"),
                    "thinking": item_framing.get("thinking"),
                    "labels": framing.get("labels") or [],
                    "mixed": framing.get("mixed"),
                    "warnings": item_framing.get("warnings") or []},
        "wall_ms": report.get("wall_ms"),
        "report_warnings": sorted({str(w).split(":")[0] for w in (report.get("warnings") or [])}),
        "ok": bool(report.get("ok")),
    }


def pair_stats(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """The paired comparison of two cells, item by item (same dev set, same order)."""
    left_rows = {str(item["id"]): item for item in rows(left)}
    right_rows = {str(item["id"]): item for item in rows(right)}
    shared = sorted(set(left_rows) & set(right_rows))
    if not shared:
        raise DecisionError("the two reports share no item ids — not the same dev set")
    b = sum(1 for key in shared if right_rows[key].get("correct") and not
            left_rows[key].get("correct"))
    c = sum(1 for key in shared if left_rows[key].get("correct") and not
            right_rows[key].get("correct"))
    both = sum(1 for key in shared
               if left_rows[key].get("correct") and right_rows[key].get("correct"))
    neither = len(shared) - b - c - both
    difference = paired_difference(b, c, len(shared))
    return {
        "baseline": cell_label(left),
        "challenger": cell_label(right),
        "n": len(shared),
        "both_correct": both,
        "neither_correct": neither,
        "baseline_only": c,       # the baseline won these, the challenger missed them
        "challenger_only": b,     # the challenger won these, the baseline missed them
        "difference": difference["difference"],
        "ci": difference["ci"],
        "caveat": difference["caveat"],
        "mcnemar_p": mcnemar_exact(b, c),
        "challenger_wins": bool(difference["difference"] is not None
                                and difference["ci"][0] is not None
                                and difference["ci"][0] > 0.0
                                and mcnemar_exact(b, c) < 0.05),
    }


# ------------------------------------------------------------------ the rule, stated
#: what a non-default cell costs in comparability — printed with every table, because a number is
#: only as good as the prompt it was measured under (card t_4c48f40a, acceptance 6).
COMPARABILITY = (
    "**Comparability cost.** Every cell in this table is measured under the *same* instrument (the "
    "same 60 committed dev items, temperature 0, fixed seed, `--backend vulkan`) — so the table "
    "compares itself and nothing else. What it does **not** compare against is any published row: "
    "a `role_split` placement, a `json_instructed` ask line, or a contract stated in the framing "
    "each "
    "change the bytes the model sees. A cell published as *the* quality row would carry its "
    "policy line (`- prompt policy: …`) and every other published quality row would have to be "
    "re-measured under the same policy before it could sit next to it."
)


def decide(cells: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], *,
           baseline: str | None = None) -> dict[str, Any]:
    """The decision rule, stated: a challenger displaces the default only by more than the noise.

    Ranked by agreement (fewest misses), ties broken by the refusal/low-mass count that produced
    them. A challenger is called a **win** only when the paired difference's 95 % interval excludes
    zero *and* the exact McNemar p is below 0.05; equal rows are called equal, everything else is
    "not by more than the CI noise" — the same rule E3d's tool applied, on the same unit.
    """
    if not cells:
        raise DecisionError("no cells")
    labels = [str(cell["label"]) for cell in cells]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise DecisionError("two cells share a label (" + ", ".join(duplicates)
                            + "): the label *is* the policy, so a table with a repeated one cannot "
                              "say which row a verdict belongs to")
    baseline = baseline or labels[0]
    if baseline not in labels:
        raise DecisionError(f"baseline {baseline!r} is not in the table ({', '.join(labels)})")
    ranked = sorted(cells, key=lambda cell: (-int(cell["correct"]), int(cell["refusals"]),
                                             int(cell["low_mass"]), str(cell["label"])))
    best = ranked[0]
    against: dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        base = str(pair["baseline"])
        challenger = str(pair["challenger"])
        if base == baseline:
            against[challenger] = pair
        elif challenger == baseline:
            flipped = dict(pair)
            flipped["baseline"], flipped["challenger"] = challenger, base
            flipped["baseline_only"], flipped["challenger_only"] = \
                pair["challenger_only"], pair["baseline_only"]
            flipped["difference"] = -(pair["difference"] or 0.0)
            flipped["ci"] = [-pair["ci"][1], -pair["ci"][0]]
            flipped["challenger_wins"] = bool(pair["ci"][1] < 0.0 and pair["mcnemar_p"] < 0.05)
            against[base] = flipped
    verdicts = []
    for cell in ranked:
        label = str(cell["label"])
        if label == baseline:
            continue
        pair = against.get(label)
        if pair is None:
            verdicts.append({"label": label, "verdict": "no paired comparison",
                             "why": "the two reports share no items"})
            continue
        why = (f"{pair['challenger_only']} items only it got right, "
               f"{pair['baseline_only']} only {baseline} did; difference "
               f"{(pair['difference'] or 0.0):+.3f} (95 % CI "
               f"{pair['ci'][0]:+.3f}..{pair['ci'][1]:+.3f}), "
               f"exact McNemar p={pair['mcnemar_p']:.3f}")
        if pair["challenger_wins"]:
            verdicts.append({"label": label, "verdict": "wins", "why": why})
        elif pair["challenger_only"] == 0 and pair["baseline_only"] == 0:
            verdicts.append({"label": label, "verdict": "identical on these 60 items", "why": why})
        else:
            verdicts.append({"label": label, "verdict": "not by more than the CI noise",
                             "why": why})
        verdicts[-1].update({
            "challenger_only": pair["challenger_only"],
            "baseline_only": pair["baseline_only"],
            "difference": pair["difference"],
            "ci": list(pair["ci"]),
            "mcnemar_p": pair["mcnemar_p"],
        })
    return {
        "baseline": baseline,
        "rule": ("agreement on the 60 committed dev items, paired by item: a challenger wins only "
                 "when the paired difference's 95 % interval excludes zero and the exact two-sided "
                 "McNemar p < 0.05; ties are called identical, everything else is not by more than "
                 "the CI noise"),
        "best": best["label"],
        "best_correct": best["correct"],
        "best_refusals": best["refusals"],
        "verdicts": verdicts,
        "ranking": [{"label": cell["label"], "correct": cell["correct"],
                     "agreement": cell["agreement"], "refusals": cell["refusals"],
                     "low_mass": cell["low_mass"]} for cell in ranked],
    }


def analyses(reports: Sequence[Mapping[str, Any]], *,
             baseline: str | None = None) -> dict[str, Any]:
    cells = [cell_stats(report) for report in reports]
    pairs = []
    for index, left in enumerate(reports):
        for right in reports[index + 1:]:
            pairs.append(pair_stats(left, right))
    return {"cells": cells, "pairs": pairs, "decision": decide(cells, pairs, baseline=baseline)}


# ------------------------------------------------------------------ rendering
def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f} %"


def _interval(ci: Any) -> str:
    if not ci or ci[0] is None:
        return "—"
    return f"{100 * float(ci[0]):.1f}–{100 * float(ci[1]):.1f} %"


def _type_cell(types: Mapping[str, Any], name: str) -> str:
    bucket = types.get(name)
    return "—" if not bucket else f"{bucket['correct']}/{bucket['n']}"


def cell_table(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| cell | correct | agreement (Wilson 95 %) | choice | noul | score | low_mass | "
             "refusals | cue verdicts | coverage p50 | prefix tokens |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cell in cells:
        types = cell["per_type"]
        verdicts = ", ".join(f"{key} {value}" for key, value in sorted(cell["verdicts"].items()))
        lines.append(
            f"| `{cell['label']}` | {cell['correct']}/{cell['items']} | "
            f"{_pct(cell['agreement'])} ({_interval(cell['ci'])}) | "
            f"{_type_cell(types, 'choice')} | "
            f"{_type_cell(types, 'noul')} | {_type_cell(types, 'score')} | {cell['low_mass']} | "
            f"{cell['refusals']} | "
            f"{verdicts or '—'} | {_pct(cell['coverage']['p50'])} | "
            f"{cell['prefix_tokens']['min']}–{cell['prefix_tokens']['max']} |")
    return lines


def policy_table(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| cell | cue | chat_format | json_contract | framing | renderer (thinking) | family | "
             "labels | warnings |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cell in cells:
        framing = cell["framing"]
        lines.append(
            f"| `{cell['label']}` | {cell['cue']} | {cell['chat_format']} | "
            f"{cell['json_contract']} | {framing['kind']} | {framing['renderer']} "
            f"({framing['thinking']}) | {framing['family']} | "
            f"{', '.join(str(label) for label in framing['labels']) or '—'} | "
            f"{', '.join(cell['report_warnings']) or '—'} |")
    return lines


def pair_table(pairs: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| baseline | challenger | only challenger | only baseline | both | neither | "
             "difference (95 % CI) | exact McNemar p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for pair in pairs:
        lines.append(
            f"| `{pair['baseline']}` | `{pair['challenger']}` | {pair['challenger_only']} | "
            f"{pair['baseline_only']} | {pair['both_correct']} | {pair['neither_correct']} | "
            f"{pair['difference']:+.3f} ({pair['ci'][0]:+.3f}..{pair['ci'][1]:+.3f}) | "
            f"{pair['mcnemar_p']:.3f} |")
    return lines


def _devset_line(devset: Any) -> str:
    if isinstance(devset, Mapping):
        path = devset.get("path") or "?"
        counts = devset.get("counts") or {}
        spelled = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
        return f"`{pathlib.Path(str(path)).name}` ({spelled})"
    return str(devset)


def render(record: Mapping[str, Any]) -> str:
    decision = record["decision"]
    lines = [f"# E3e — which prompt policy beats the shipped one ({record['generated_at']})", "",
             f"- dev set: {_devset_line(record['devset'])} · items per cell: {record['items']} · "
             f"baseline: `{decision['baseline']}`",
             f"- model: {record['model']} · backend: {record['backend']} · "
             f"gpu-layers {record['gpu_layers']} · threads {record['threads']}",
             f"- reports: {len(record['cells'])} cells, {len(record['pairs'])} paired comparisons "
             f"({UNIT})", "",
             "## The cells", ""]
    lines += cell_table(record["cells"])
    lines += ["", "## The policy each cell ran under", ""]
    lines += policy_table(record["cells"])
    lines += ["", "## Paired comparisons (same items, item by item)", ""]
    lines += pair_table(record["pairs"])
    lines += ["", "## Decision", "",
              f"- rule: {decision['rule']}",
              f"- best cell: `{decision['best']}` — {decision['best_correct']}/{record['items']} "
              f"correct, {decision['best_refusals']} refusals", ""]
    for verdict in decision["verdicts"]:
        lines.append(f"- `{verdict['label']}` vs `{decision['baseline']}`: "
                     f"**{verdict['verdict']}** — {verdict['why']}")
    lines += ["", "## Recommendation", "", record["recommendation"], "",
              "## Caveats the numbers carry", ""]
    lines += [f"- {note}" for note in record["caveats"]]
    lines += ["", COMPARABILITY]
    return "\n".join(lines) + "\n"


def build_caveats(cells: Sequence[Mapping[str, Any]],
                  pairs: Sequence[Mapping[str, Any]]) -> list[str]:
    """The limits the table is published with — computed from it, not boilerplate."""
    notes = []
    items = cells[0]["items"] if cells else 0
    width = 0.0
    for cell in cells:
        if cell["ci"][0] is not None:
            width = max(width, float(cell["ci"][1]) - float(cell["ci"][0]))
    notes.append(f"{items} committed dev items: a single cell's 95 % interval is up to "
                 f"{100 * width:.1f} points wide, so single-cell differences below that are noise "
                 f"by construction — which is what the paired columns are for.")
    thin = [cell["label"] for cell in cells
            if min((bucket["n"] for bucket in cell["per_type"].values()), default=0) < 10]
    if thin:
        notes.append("per-type cells are thin (n < 10) in: "
                     + ", ".join(f"`{label}`" for label in thin)
                     + " — a per-type swing there is a lead, not a result.")
    if pairs and all(pair["mcnemar_p"] >= 0.05 for pair in pairs):
        notes.append("no paired comparison reaches p < 0.05: on this dev set the levers move the "
                     "verdict distribution more clearly than the agreement count.")
    notes.append("`score` is the type E3c/E3d found hardest: a policy that helps choice/noul and "
                 "does not help score is still a policy decision, not a quality result.")
    return notes


def report(record: Mapping[str, Any], *, json_path: str | None = None,
           report_path: str | None = None) -> str:
    text = render(record)
    if json_path:
        pathlib.Path(json_path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    if report_path:
        pathlib.Path(report_path).write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", default=[], dest="reports",
                        help="a quality report (repeat; the first is the baseline)")
    parser.add_argument("--baseline", default=None, help="the cell label to compare against")
    parser.add_argument("--json", default=None, help="write the record here")
    parser.add_argument("--report-file", default=None, help="write the markdown report here")
    parser.add_argument("--recommendation", default=None,
                        help="the recommendation paragraph (the card's own words)")
    args = parser.parse_args(argv)
    if not args.reports:
        parser.error("at least one --report is required")
    loaded = []
    for path in args.reports:
        report_data = load_report(path)
        report_data["_path"] = str(path)
        loaded.append(report_data)
    record = analyses(loaded, baseline=args.baseline)
    first = loaded[0]
    config = first.get("config") or {}
    record.update({
        "schema": SCHEMA,
        "generated_at": first.get("generated_at"),
        "devset": first.get("devset"),
        "items": len(first["items"]),
        "model": (first.get("model") or {}).get("name"),
        "backend": config.get("backend"),
        "gpu_layers": config.get("gpu_layers"),
        "threads": config.get("threads"),
        "recommendation": args.recommendation or (
            "Fill this in from the table above (the card asks for an explicit recommendation): "
            "the best cell, the paired evidence for it, and what it costs in comparability."),
        "caveats": build_caveats(record["cells"], record["pairs"]),
    })
    text = report(record, json_path=args.json, report_path=args.report_file)
    if not args.report_file:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

