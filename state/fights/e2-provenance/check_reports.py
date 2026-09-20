#!/usr/bin/env python3
"""Audit helper (t_78f5ea7a): internal-consistency checks of the published E2 reports.

Checks aggregates that a *real* measurement run produces and that fabrication tends to
break: percentile ordering, sample counts, arithmetic identities between reported fields.
Read-only; prints one line per check.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path("/home/rybens/workspace/ggufone")
EV = REPO / "docs" / "evidence"

problems: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        problems.append(f"{label}: {detail}")
    print(("  ok  " if ok else " FAIL ") + label + (f"  [{detail}]" if detail else ""))


def summaries(node, path="root"):
    """Yield every dict that looks like a harness.summarise() summary."""
    if isinstance(node, dict):
        if {"min", "p50", "p95", "max", "n"} <= set(node):
            yield path, node
        for key, value in node.items():
            yield from summaries(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from summaries(value, f"{path}[{index}]")


def check_percentiles(name: str) -> dict:
    data = json.loads((EV / name).read_text())
    for path, summary in summaries(data, name):
        n = summary["n"]
        lo, p50, p95, hi = summary["min"], summary["p50"], summary["p95"], summary["max"]
        ordered = lo <= p50 <= hi and p50 <= p95 <= hi and lo <= p95
        check(ordered, f"{path} ordered min<=p50<=p95<=max",
              f"{lo:.3f} {p50:.3f} {p95:.3f} {hi:.3f} n={n}")
        if n >= 2:
            check(p50 >= lo and p95 <= hi, f"{path} spread", f"n={n}")
    return data


print("== e2_latency.json ==")
lat = check_percentiles("e2_latency.json")
# load amortisation identity: one_shot == serve + model_load
am = lat["load_amortisation"]
check(abs(am["one_shot_ms_per_request"] - (am["serve_ms_per_request"] + am["model_load_ms"])) < 1e-6,
      "latency load_amortisation: one_shot == serve + model_load",
      f'{am["one_shot_ms_per_request"]:.3f} vs {am["serve_ms_per_request"] + am["model_load_ms"]:.3f}')
check(am["saved_ms_per_request"] == am["model_load_ms"], "latency saved == model_load")
check(lat["wave_accounting"] == {"groups": 1, "suffix_decodes": 1, "step_decodes": 1, "waves": 2},
      "latency wave_accounting")
# wave scaling monotone in questions at p50 up to noise: N=1..6 must not decrease wildly
ws = {row["questions"]: row["ms"]["p50"] for row in lat["wave_scaling"]}
check(len(ws) == 16, "latency wave scaling has 16 rows", f"n={len(ws)}")
check(ws[16] > ws[1], "latency ws[16] > ws[1]", f'{ws[16]:.0f} > {ws[1]:.0f}')
check(ws[2] > ws[1], "latency ws[2] > ws[1]", f'{ws[2]:.0f} > {ws[1]:.0f}')

print("== e2_quality.json ==")
qual = check_percentiles("e2_quality.json")
per_type: dict[str, list[int]] = {}
for item in qual["items"]:
    per_type.setdefault(item["type"], [0, 0])
    per_type[item["type"]][0] += 1
    per_type[item["type"]][1] += int(bool(item["correct"]))
check(sum(v[0] for v in per_type.values()) == 60, "quality 60 items", str(per_type))
for t, (n, correct) in per_type.items():
    rep = qual["per_type"][t]
    check(rep["n"] == n and rep["correct"] == correct,
          f"quality per_type[{t}] counts match items", f'items {correct}/{n} vs report {rep["correct"]}/{rep["n"]}')
check(qual["overall"]["correct"] == sum(v[1] for v in per_type.values()),
      "quality overall correct == sum of items")
check(abs(qual["overall"]["agreement"] - qual["overall"]["correct"] / 60) < 1e-9, "quality agreement == correct/n")
# probabilities are a distribution
bad = [i["id"] for i in qual["items"] if abs(sum(i["probabilities"].values()) - 1.0) > 1e-9]
check(not bad, "quality per-item probabilities sum to 1", ",".join(bad))

print("== e2_calibration.json ==")
cal = check_percentiles("e2_calibration.json")
# ECE recomputed from the published bins must equal the published ECE
for mode, payload in cal["modes"].items():
    bins = [b for b in payload["bins"] if b["n"] and b["accuracy"] is not None]
    n_total = sum(b["n"] for b in bins)
    ece = sum(b["n"] / n_total * abs(b["accuracy"] - b["mean_confidence"]) for b in bins)
    check(abs(ece - payload["ece"]) < 1e-9, f"calibration {mode}: ECE == sum over bins",
          f'{ece:.15f} vs {payload["ece"]:.15f}')
    check(n_total == cal["n"], f"calibration {mode}: bins n sum == items", f"{n_total} vs {cal['n']}")
check(abs(cal["overall"]["agreement"] - cal["overall"]["correct"] / cal["n"]) < 1e-9,
      "calibration agreement == correct/n")

print("== e2_determinism.json / e2_throughput.json / e2_qwen_*.json ==")
det = check_percentiles("e2_determinism.json")
for row in det["backends"]:
    if row.get("measured", True):
        check(len(set(row["digests"])) == 1 and row["identical"], f'determinism {row["backend"]} identical')
thr = check_percentiles("e2_throughput.json")
for row in thr["backends"]:
    if not row.get("measured", True):
        check("reason" in row, f'throughput {row["backend"]} has a reason')
        continue
    check(abs(row["prefill_tok_per_s"]["p50"] - row["prefill"][0]["tok_per_s"]["p50"]) < 1e-9,
          f'throughput {row["backend"]}: prefill_tok_per_s == prefill[0]')
ql = check_percentiles("e2_qwen_latency.json")
qq = check_percentiles("e2_qwen_quality.json")
check(qq["overall"]["correct"] == 28, "qwen quality 28/60", str(qq["overall"]["correct"]))

print()
print(f"{checks} checks, {len(problems)} problems")
for problem in problems:
    print("  PROBLEM:", problem)
sys.exit(1 if problems else 0)
