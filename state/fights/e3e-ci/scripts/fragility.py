#!/usr/bin/env python3
"""How wide is the borderline? One-item fragility of the E3e winner and its runner-up.

Audit t_57bd3db2. Uses the record's own discordant splits and the same exact McNemar the tool
prints; the point is to show how much of the verdict rests on one item in sixty.
"""
import math
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parents[4]
Z = 1.959963984540054
record = json.loads((ROOT / "docs/evidence/e3e_roles_decision.json").read_text())


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def wald(b, c, n):
    diff = (b - c) / n
    spread = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    return diff, [diff - Z * spread, diff + Z * spread]


pairs = {(p["baseline"], p["challenger"]): p for p in record["pairs"]}
for challenger in ("json_instructed/answer_sheet", "json_instructed/role_split",
                   "shipped/role_split"):
    p = pairs[("shipped/answer_sheet", challenger)]
    b, c = p["challenger_only"], p["baseline_only"]
    print(f"\n{challenger}: recorded split {b} vs {c}, p={p['mcnemar_p']:.4f}, "
          f"ci={p['ci'][0]:+.3f}..{p['ci'][1]:+.3f}, wins={p['challenger_wins']}")
    for delta in (0, 1, 2):
        bb, cc = b - delta, c + delta
        diff, ci = wald(bb, cc, p["n"])
        verdict = ("wins" if ci[0] > 0 and mcnemar(bb, cc) < 0.05 else
                   "not more than the noise" if not (bb == 0 and cc == 0) else "identical")
        print(f"   move {delta} item(s) from the challenger to the baseline: "
              f"{bb} vs {cc} -> p={mcnemar(bb, cc):.4f}, ci={ci[0]:+.3f}..{ci[1]:+.3f} [{verdict}]")
    for delta in (1, 2):
        bb, cc = b + delta, c - delta
        if cc < 0:
            break
        diff, ci = wald(bb, cc, p["n"])
        print(f"   one more item to the challenger (+{delta}): {bb} vs {cc} -> "
              f"p={mcnemar(bb, cc):.4f}, ci={ci[0]:+.3f}..{ci[1]:+.3f}")

print("\nThe winner's margin over the 0.05 gate: "
      f"{0.05 - pairs[('shipped/answer_sheet', 'json_instructed/answer_sheet')]['mcnemar_p']:.4f} "
      "of p — one item is "
      f"{abs(mcnemar(12, 5) - pairs[('shipped/answer_sheet', 'json_instructed/answer_sheet')]['mcnemar_p']):.4f}")
