"""Recompute the E3d §6 bench-arm table from the two arm JSONs."""
import json
import math

W = "/var/home/rybens/workspace/ggufone/docs/evidence/e3d/"


def wilson(s, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = s / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (max(0.0, c - m), min(1.0, c + m))


for name in ("bench_shipped", "bench_two_step"):
    d = json.load(open(W + name + ".json"))
    rows = d["items"]
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    low = sum(1 for r in rows if r["reliability"] == "low_mass")
    refused = sum(1 for r in rows if (r.get("cue") or {}).get("refused"))
    covs = sorted(float(r["coverage"]) for r in rows)
    above = sum(1 for c in covs if c >= 0.10)
    median = covs[len(covs) // 2]
    lo, hi = wilson(correct, n)
    ov = d["overall"]
    per = d["per_type"]
    print(f"=== {name} (cue={d['config']['cue']}, backend={d['effective_backend']})")
    print(f"  recomputed: {correct}/{n} = {correct/n:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"low_mass {low}/{n}  refused {refused}/{n}  median cov {median:.4f}  above {above}/{n}")
    print(f"  recorded overall: {ov['correct']}/{ov['n']} = {ov['agreement']:.3f}  ci={ov['ci']}")
    print(f"  per_type: " + "  ".join(
        f"{k}: {v['correct']}/{v['n']}" for k, v in per.items()))
    print(f"  warnings: {d.get('warnings')}")
    print()

# cross-arm: items where the two arms disagree
a = json.load(open(W + "bench_shipped.json"))
b = json.load(open(W + "bench_two_step.json"))
by_a = {r["id"]: r for r in a["items"]}
by_b = {r["id"]: r for r in b["items"]}
diff = [(i, by_a[i]["correct"], by_b[i]["correct"]) for i in by_a if by_a[i]["correct"] != by_b[i]["correct"]]
print(f"items with different correctness between arms: {len(diff)}")
print(diff[:20])

# c01 numbers vs the doc's claims
c1a, c1b = by_a["c01"], by_b["c01"]
print("\nc01 shipped arm: cov", c1a["coverage"], "cue.mass", c1a["cue"]["mass"], "got", c1a["got"])
print("c01 two_step arm: cov", c1b["coverage"], "cue.mass", c1b["cue"]["mass"], "got", c1b["got"])
