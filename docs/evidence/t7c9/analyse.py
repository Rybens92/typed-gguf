#!/usr/bin/env python3
"""t_7c926398 — analyse the corrected-instrument Tiel row and pair it with the pre-fix row.

Reads the merged corrected report (and its per-chunk reports + placement sinks), the pre-fix
merged report (`docs/evidence/tiel_quality.json`) and the pre-fix placement sinks, and emits:

  * `.t7c9/tiel_corrected_stats.json` — every number the evidence doc and BENCHMARKS §7.4 quote;
  * stdout — the rendered tables (the doc is assembled from these, never retyped).

Statistics come from the committed helpers (`ggufone.bench.harness.wilson_interval`,
`ggufone.bench.compare.agreement_block`) and the E3d probe's paired arithmetic
(`tools/e3d_cue_decision.py`: exact McNemar + seeded percentile bootstrap), never a fresh
implementation — a second implementation could disagree with the tables this row joins.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import e3d_cue_decision as e3d  # noqa: E402

from ggufone.bench import compare, harness, suites  # noqa: E402

FLOOR = 0.10


def load(path: str | pathlib.Path) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row.get("coverage") or 0.0) for row in rows)
    return {
        "n": len(values),
        "min": values[0] if values else None,
        "p25": e3d._quantile(values, 0.25) if values else None,
        "median": e3d._quantile(values, 0.5) if values else None,
        "p75": e3d._quantile(values, 0.75) if values else None,
        "max": values[-1] if values else None,
        "above_floor": sum(1 for value in values if value >= FLOOR),
        "at_or_below_floor": sum(1 for value in values if value < FLOOR),
    }


def mass_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split = compare.split_by_mass(list(rows), coverage_floor=FLOOR)
    out: dict[str, Any] = {}
    for name, subset in split.items():
        block = compare.agreement_block(subset) if subset else {"n": 0, "correct": 0,
                                                                "agreement": None, "ci": None}
        out[name] = {"items": len(subset), "correct": block["correct"],
                     "agreement": block["agreement"], "ci": block["ci"],
                     "ids": [row["id"] for row in subset]}
    return out


def cue_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cues = [row.get("cue") or {} for row in rows]
    refused = [row for row, cue in zip(rows, cues, strict=True) if cue.get("refused")]
    closers = Counter(str(cue.get("closer")) for cue in cues if cue.get("refused"))
    tokens = Counter(int(cue.get("token")) for cue in cues if cue.get("token") is not None)
    return {
        "refused": len(refused),
        "refused_ids": [row["id"] for row in refused],
        "refused_closers": dict(closers),
        "cue_tokens": {str(key): value for key, value in tokens.most_common()},
        "cue_mass_median": e3d._quantile(sorted(float(cue.get("mass") or 0.0) for cue in cues), 0.5)
        if cues else None,
    }


def framing_facts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The framing surface **per row of the merged report** — the report-level block is the first
    chunk's only (`compare.merge_reports` keeps `reports[0]`'s), so a 60-row table has to verify the
    surface across all rows instead of quoting the first chunk's block."""
    labels = sorted({harness.framing_label(row.get("framing") or {}) for row in rows})
    prefixes = sorted(int(row["prefix_tokens"]) for row in rows
                      if row.get("prefix_tokens") is not None)
    return {"labels": labels, "mixed": len(labels) > 1, "rows": len(rows),
            "prefix_tokens": {"count": len(prefixes), "min": prefixes[0] if prefixes else None,
                              "max": prefixes[-1] if prefixes else None}}


def framing_surface(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`kind` / `renderer` / `source` / `thinking` / warnings of the rows' own framing blocks.

    This is the part a table must not hide: for `qwen35moe` the *internal* renderer rejected the
    template, so the runtime's built-in family table rendered it (`W_TEMPLATE_FALLBACK`) — the rows
    still say which surface produced the bytes, and the live parity gate proves that surface is the
    one the product's `ask`/`run` path sends.
    """
    surfaces: dict[str, Counter[str]] = {key: Counter() for key in
                                         ("kind", "renderer", "source", "thinking")}
    warnings: Counter[str] = Counter()
    notes: set[str] = set()
    for row in rows:
        framing = row.get("framing") or {}
        for key in surfaces:
            surfaces[key][str(framing.get(key))] += 1
        for warning in framing.get("warnings") or []:
            warnings[str(warning)] += 1
        for note in framing.get("notes") or []:
            notes.add(str(note))
    return {**{key: dict(counter) for key, counter in surfaces.items()},
            "warnings": dict(warnings), "notes": sorted(notes)}


def type_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return suites.agreement_by_type(list(rows))


def chunk_ledger(report_paths: Sequence[str], placement_paths: Sequence[str],
                 prefix_report_paths: Sequence[str],
                 prefix_placement_paths: Sequence[str]) -> list[dict[str, Any]]:
    ledger = []
    for report_path, sink_path, pre_report_path, pre_path in zip(
            report_paths, placement_paths, prefix_report_paths, prefix_placement_paths,
            strict=True):
        report = load(report_path)
        sink = load(sink_path)
        pre = load(pre_path)
        pre_rows = compare.rows_of(load(pre_report_path))
        rows = compare.rows_of(report)
        decisions = sorted(float(row["questions_ms"]) / 1000.0 for row in rows)
        pre_decisions = sorted(float(row["questions_ms"]) / 1000.0 for row in pre_rows)
        ledger.append({
            "report": pathlib.Path(report_path).name,
            "items": len(rows),
            "correct": sum(1 for row in rows if row["correct"]),
            "types": dict(Counter(str(row["type"]) for row in rows)),
            "ngl_requested": sink.get("gpu_layers_requested"),
            "ngl_used": (sink.get("placement") or {}).get("n_gpu_layers"),
            "degraded": (sink.get("placement") or {}).get("degraded"),
            "attempts": (sink.get("placement") or {}).get("attempts"),
            "warnings": (sink.get("placement") or {}).get("warnings"),
            "kv_type_used": sink.get("kv_type_used"),
            "n_ctx": sink.get("n_ctx"),
            "n_prefix": sink.get("n_prefix"),
            "n_seq_max": sink.get("n_seq_max"),
            "load_wall_s": sink.get("load_wall_s"),
            "chunk_wall_s": sink.get("wall_s"),
            "decision_median_s": e3d._quantile(decisions, 0.5),
            "decision_min_s": decisions[0] if decisions else None,
            "decision_max_s": decisions[-1] if decisions else None,
            "low_mass": sum(1 for row in rows if compare.is_low_mass(row)),
            "refused": sum(1 for row in rows if (row.get("cue") or {}).get("refused")),
            "framing": report.get("framing"),
            "pre_ngl_used": (pre.get("placement") or {}).get("n_gpu_layers"),
            "pre_kv_type_used": pre.get("kv_type_used"),
            "pre_chunk_wall_s": pre.get("wall_s"),
            "pre_decision_median_s": e3d._quantile(pre_decisions, 0.5),
            "pre_correct": sum(1 for row in pre_rows if row["correct"]),
        })
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corrected", required=True, help="merged corrected-instrument report")
    parser.add_argument("--baseline", required=True, help="merged pre-fix report")
    parser.add_argument("--chunk-reports", nargs="+", required=True)
    parser.add_argument("--chunk-placements", nargs="+", required=True)
    parser.add_argument("--prefix-reports", nargs="+", required=True)
    parser.add_argument("--prefix-placements", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=e3d.BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    corrected, baseline = load(args.corrected), load(args.baseline)
    rows_c, rows_p = compare.rows_of(corrected), compare.rows_of(baseline)

    by_id_c = {str(row["id"]): row for row in rows_c}
    by_id_p = {str(row["id"]): row for row in rows_p}
    shared = sorted(set(by_id_c) & set(by_id_p))
    only_c = sorted(set(by_id_c) - set(by_id_p))
    only_p = sorted(set(by_id_p) - set(by_id_c))
    paired_c = [by_id_c[item] for item in shared]
    paired_p = [by_id_p[item] for item in shared]

    paired = e3d.paired_diff([bool(row["correct"]) for row in paired_p],
                             [bool(row["correct"]) for row in paired_c],
                             iters=args.iters, seed=args.seed)

    stats: dict[str, Any] = {
        "schema": "ggufone.t7c926398.tiel_corrected/v1",
        "card": "t_7c926398",
        "instrument": {
            "corrected": {"path": str(args.corrected), "framing": corrected.get("framing"),
                          "generated_at": corrected.get("generated_at"),
                          "model": corrected.get("model"), "chunks": corrected.get("chunks")},
            "pre_fix": {"path": str(args.baseline), "framing": "plain (prompt.py E1b framing)",
                        "generated_at": baseline.get("generated_at"),
                        "model": baseline.get("model")},
        },
        "n": {"corrected": len(rows_c), "pre_fix": len(rows_p), "paired": len(shared),
              "only_corrected": only_c, "only_pre_fix": only_p},
        "corrected": {
            "overall": compare.agreement_block(list(rows_c)),
            "per_type": type_table(rows_c),
            "mass_split": mass_split(rows_c),
            "coverage": coverage_summary(rows_c),
            "cue": cue_block(rows_c),
            "framing_label": (corrected.get("framing") or {}).get("labels"),
            "framing_mixed": (corrected.get("framing") or {}).get("mixed"),
            "prefix_tokens": (corrected.get("framing") or {}).get("prefix_tokens"),
            "framing_per_row": framing_facts(rows_c),
            "framing_surface": framing_surface(rows_c),
            "warnings": corrected.get("warnings"),
            "misses": [{"id": row["id"], "type": row["type"], "expected": row["expected"],
                        "got": row["got"], "coverage": row["coverage"],
                        "reliability": row["reliability"]}
                       for row in rows_c if not row["correct"]],
        },
        "pre_fix": {
            "overall": compare.agreement_block(list(rows_p)),
            "per_type": type_table(rows_p),
            "mass_split": mass_split(rows_p),
            "coverage": coverage_summary(rows_p),
            "cue": cue_block(rows_p),
            "misses": [{"id": row["id"], "type": row["type"], "expected": row["expected"],
                        "got": row["got"], "coverage": row["coverage"],
                        "reliability": row["reliability"]}
                       for row in rows_p if not row["correct"]],
        },
        "paired": paired,
        "item_flips": {
            "pre_fix_only": [{"id": item, "type": by_id_p[item]["type"],
                              "expected": by_id_p[item]["expected"],
                              "pre_got": by_id_p[item]["got"],
                              "now_got": by_id_c[item]["got"]}
                             for item in shared
                             if by_id_p[item]["correct"] and not by_id_c[item]["correct"]],
            "corrected_only": [{"id": item, "type": by_id_c[item]["type"],
                                "expected": by_id_c[item]["expected"],
                                "pre_got": by_id_p[item]["got"],
                                "now_got": by_id_c[item]["got"]}
                               for item in shared
                               if by_id_c[item]["correct"] and not by_id_p[item]["correct"]],
        },
        "chunks": chunk_ledger(args.chunk_reports, args.chunk_placements,
                               args.prefix_reports, args.prefix_placements),
    }
    pathlib.Path(args.out).write_text(json.dumps(stats, indent=1, sort_keys=True), encoding="utf-8")
    print(f"stats: {args.out}")

    # ---- rendered summary (the doc's tables come from here, not from a retyped copy)
    def line(block: Mapping[str, Any]) -> str:
        if not block["n"]:
            return "—"
        low, high = block["ci"]
        return (f"{block['correct']}/{block['n']} = {block['agreement']:.3f} "
                f"[{low:.3f}–{high:.3f}]")

    print("\n## corrected instrument (chat-template framing)")
    print(f"framing: {stats['corrected']['framing_label']} · mixed: "
          f"{stats['corrected']['framing_mixed']}")
    print(f"overall: {line(stats['corrected']['overall'])}")
    for qtype, block in stats["corrected"]["per_type"].items():
        print(f"  {qtype}: {line(block)}")
    coverage = stats["corrected"]["coverage"]
    total_c = stats["n"]["corrected"]
    print(f"low_mass: {stats['corrected']['mass_split']['low_mass']['items']}/{total_c} · "
          f"measured: {stats['corrected']['mass_split']['measured']['items']}/{total_c}")
    print(f"coverage median {coverage['median']:.4g} · min {coverage['min']:.4g} · "
          f"p25 {coverage['p25']:.4g} · p75 {coverage['p75']:.4g} · max {coverage['max']:.4g} · "
          f"above floor {coverage['above_floor']}/{total_c}")
    print(f"refused at the cue: {stats['corrected']['cue']['refused']}/{total_c} "
          f"{stats['corrected']['cue']['refused_closers']}")
    print("\n## pre-fix instrument (plain framing, published row)")
    print(f"overall: {line(stats['pre_fix']['overall'])}")
    for qtype, block in stats["pre_fix"]["per_type"].items():
        print(f"  {qtype}: {line(block)}")
    print(f"low_mass: {stats['pre_fix']['mass_split']['low_mass']['items']}/60 · "
          f"refused at the cue: {stats['pre_fix']['cue']['refused']}/60")
    print("\n## paired (same 60 items)")
    print(f"risk difference {paired['difference']:+.3f} "
          f"[{paired['low']:+.3f}…{paired['high']:+.3f}] · exact McNemar p = "
          f"{paired['mcnemar_p']:.4g} · discordant: pre-fix-only "
          f"{paired['discordant']['a_win']}, corrected-only {paired['discordant']['b_win']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
