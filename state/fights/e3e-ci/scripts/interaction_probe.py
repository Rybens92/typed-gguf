#!/usr/bin/env python3
"""The two_step x role_split collapse, read from the raw rows (audit t_57bd3db2).

Prints, for the four arms of the interaction, the per-item record of the items the doc names
(c01), plus aggregate counts: refusals, low_mass, coverage p50, and the cue block's token/mass.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
ARMS = {
    "shipped/answer_sheet": ".e3e/bench_shipped_answer_sheet.json",
    "shipped/role_split": ".e3e/bench_shipped_role_split.json",
    "two_step/answer_sheet": ".e3e/bench_two_step_answer_sheet.json",
    "two_step/role_split": ".e3e/bench_two_step_role_split.json",
}


def load(path):
    return json.loads((ROOT / path).read_text())


def rows(report):
    return {str(it["id"]): it for it in report["items"]}


reports = {label: load(path) for label, path in ARMS.items()}
by_id = {label: rows(rep) for label, rep in reports.items()}

print("=== c01 across the four arms ===")
for label, table in by_id.items():
    it = table["c01"]
    cov = sorted(float(x["coverage"]) for x in reports[label]["items"])
    print(f"\n-- {label}")
    print(f"   correct={it['correct']} got={it['got']!r} expected={it['expected']!r} "
          f"reliability={it['reliability']!r} coverage={it['coverage']:.3e} "
          f"prefix_tokens={it['prefix_tokens']}")
    print(f"   cue: token={it['cue']['token']} mass={it['cue']['mass']:.6f} "
          f"refused={it['cue']['refused']} closer={it['cue']['closer']!r} "
          f"verdict={it['cue'].get('verdict')!r}")
    print(f"   probabilities={ {k: round(v, 6) for k, v in it['probabilities'].items()} }")
    print(f"   arm coverage p50 = {cov[len(cov) // 2]:.3e}")

print("\n=== aggregate, per arm ===")
print(f"{'arm':<24} {'correct':>7} {'low_mass':>8} {'refused':>7} {'cue.mass<0.1':>12} "
      f"{'cov p50':>10} {'cov==0 cnt':>10}")
for label, rep in reports.items():
    items = rep["items"]
    lm = sum(1 for it in items if it["reliability"] == "low_mass")
    rf = sum(1 for it in items if it["cue"]["refused"])
    low_mass_cue = sum(1 for it in items if float(it["cue"]["mass"]) < 0.1)
    cov = sorted(float(it["coverage"]) for it in items)
    zero = sum(1 for it in items if float(it["coverage"]) < 1e-6)
    print(f"{label:<24} {sum(1 for it in items if it['correct']):>7} {lm:>8} {rf:>7} "
          f"{low_mass_cue:>12} {cov[len(cov)//2]:>10.3e} {zero:>10}")

print("\n=== two_step/role_split: the 20 items whose cue did NOT close ===")
tsr = reports["two_step/role_split"]
for it in tsr["items"]:
    if not it["cue"]["refused"]:
        print(f"  {it['id']:>4} type={it['type']:<6} correct={str(it['correct']):<5} "
              f"got={it['got']!r:<10} cue.token={it['cue']['token']:<7} "
              f"cue.mass={float(it['cue']['mass']):.4f} coverage={float(it['coverage']):.3e} "
              f"reliability={it['reliability']}")

print("\n=== the refusals: what token closed the cue (two_step/role_split) ===")
from collections import Counter
tokens = Counter((it["cue"]["token"], it["cue"]["closer"]) for it in tsr["items"]
                 if it["cue"]["refused"])
for (token, closer), count in tokens.most_common():
    print(f"  token={token} closer={closer!r}  x{count}")

print("\n=== the same tokens in shipped/role_split and shipped/answer_sheet (top cue tokens) ===")
for label in ("shipped/answer_sheet", "shipped/role_split"):
    counter = Counter(it["cue"]["token"] for it in reports[label]["items"])
    print(f"  {label}: {counter.most_common(6)}")

print("\n=== mass-at-cue: distribution comparison (median / max) ===")
for label, rep in reports.items():
    masses = sorted(float(it["cue"]["mass"]) for it in rep["items"])
    print(f"  {label:<24} min={masses[0]:.3e} p50={masses[len(masses)//2]:.4f} max={masses[-1]:.4f}")
