#!/usr/bin/env python3
"""INDEPENDENT recomputation of the E3d cue-shape statistics from docs/evidence/e3d/full.json.

Everything here is implemented from scratch (no import of ggufone / no reuse of
tools/e3d_cue_decision.py) so the numbers can disagree. Implements:
  - agreement counts + Wilson score intervals (z=1.96, standard formula)
  - exact two-sided McNemar (binomial tail)
  - seeded percentile bootstrap (replicating the tool's exact convention) AND
  - the EXACT bootstrap distribution via DP (no Monte Carlo dependence)
  - coverage distribution summaries
  - per-type counts
  - an extrapolation: minimal item count at which the two_step paired 2.5%
    quantile would exceed zero at the observed per-item rates
"""
import json
import math
import random

REC = "/var/home/rybens/workspace/ggufone/docs/evidence/e3d/full.json"

d = json.load(open(REC))
items = d["items"]
policies = list(d["ranked_keys"])
floor = float(d["mass_floor"])
print(f"record: {len(items)} items · policies {policies} · floor {floor}")
print(f"counts: {d['counts']}")


def get_rows():
    rows = {p: [] for p in policies}
    for it in items:
        for p in policies:
            r = it["ranked"][p]
            rows[p].append({
                "id": it["id"], "type": it["type"], "expected": it["expected"],
                "got": r["got"], "correct": bool(r["correct"]),
                "coverage": float(r["coverage"]), "reliability": r["reliability"],
                "confidence": r.get("confidence"),
            })
    return rows


rows = get_rows()


# ---------------------------------------------------------------- statistics
def wilson(s, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = s / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (max(0.0, c - m), min(1.0, c + m))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def quantile(ordered, frac):
    """Linear-interpolated (numpy-default) quantile of a sorted list."""
    m = len(ordered)
    if m == 1:
        return float(ordered[0])
    pos = frac * (m - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, m - 1)
    w = pos - lo
    return float(ordered[lo]) * (1 - w) + float(ordered[hi]) * w


def bootstrap_mc(a, b, iters, seed):
    """Replicates tools/e3d_cue_decision.py::paired_diff exactly (order-statistic percentiles)."""
    n = len(a)
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        tot = 0
        for _ in range(n):
            i = rng.randrange(n)
            tot += int(b[i]) - int(a[i])
        draws.append(tot / n)
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))], draws


def exact_delta_dist(n, p_a, p_b):
    """Exact distribution of the resampled total delta: each of n items contributes
    -1 with prob p_a, +1 with prob p_b, 0 else. Returns {delta: prob}."""
    dist = {0: 1.0}
    for _ in range(n):
        nd = {}
        for v, pr in dist.items():
            nd[v - 1] = nd.get(v - 1, 0.0) + pr * p_a
            nd[v + 1] = nd.get(v + 1, 0.0) + pr * p_b
            nd[v] = nd.get(v, 0.0) + pr * (1 - p_a - p_b)
        dist = nd
    return dist


def bootstrap_exact(a, b):
    """Exact bootstrap distribution of the paired difference by DP (iid item resampling)."""
    n = len(a)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    dist = exact_delta_dist(n, a_only / n, b_only / n)
    vs = sorted(dist)
    cum = 0.0
    q_lo = q_hi = None
    for v in vs:
        cum += dist[v]
        if q_lo is None and cum >= 0.025:
            q_lo = v
        if q_hi is None and cum >= 0.975:
            q_hi = v
    p_le0 = sum(pr for v, pr in dist.items() if v <= 0)
    p_lt0 = sum(pr for v, pr in dist.items() if v < 0)
    return q_lo / n, q_hi / n, (a_only, b_only), (p_le0, p_lt0)


def summary(values):
    o = sorted(values)
    return {
        "min": o[0], "q1": quantile(o, 0.25), "median": quantile(o, 0.5),
        "q3": quantile(o, 0.75), "max": o[-1], "mean": sum(o) / len(o),
    }


# ---------------------------------------------------------------- per-policy
print("\n=== per-policy recomputation ===")
out = {}
for p in policies:
    rs = rows[p]
    n = len(rs)
    correct = sum(r["correct"] for r in rs)
    agg = correct / n
    lo, hi = wilson(correct, n)
    covs = summary([r["coverage"] for r in rs])
    above = sum(1 for r in rs if r["coverage"] >= floor)
    low = sum(1 for r in rs if r["reliability"] == "low_mass")
    out[p] = {
        "n": n, "correct": correct, "agreement": agg, "wilson": [lo, hi],
        "above_floor": above, "low_mass": low,
        "coverage": covs,
    }
    print(f"{p:22s} {correct}/{n} = {agg:.4f}  Wilson [{lo:.4f}, {hi:.4f}]  "
          f"above {above}/{n}  low_mass {low}/{n}")
    print(f"   cov min={covs['min']:.4g} q1={covs['q1']:.4g} med={covs['median']:.4g} "
          f"q3={covs['q3']:.4g} max={covs['max']:.4g} mean={covs['mean']:.4g}")

# ---------------------------------------------------------------- per-type
print("\n=== per-type agreement (correct/n) ===")
types = sorted({r["type"] for r in rows[policies[0]]})
for p in policies:
    cells = []
    for t in types:
        sub = [r for r in rows[p] if r["type"] == t]
        cells.append(f"{t}: {sum(r['correct'] for r in sub)}/{len(sub)}")
    print(f"{p:22s} " + "  ".join(cells))

# ---------------------------------------------------------------- paired
print("\n=== paired readings (baseline = shipped=bare) ===")
base = rows[policies[0]]
base_by_id = {r["id"]: r for r in base}
for chal in policies[1:]:
    new = rows[chal]
    new_by_id = {r["id"]: r for r in new}
    ids = [r["id"] for r in base]
    a = [base_by_id[i]["correct"] for i in ids]
    b = [new_by_id[i]["correct"] for i in ids]
    a_win = sum(1 for x, y in zip(a, b) if x and not y)
    b_win = sum(1 for x, y in zip(a, b) if y and not x)
    diff = (sum(b) - sum(a)) / len(a)
    p_val = mcnemar_exact(a_win, b_win)
    lo_rep, hi_rep, _ = bootstrap_mc(a, b, 10000, 20260919)
    lo_ex, hi_ex, counts, pmass = bootstrap_exact(a, b)
    print(f"\n{chal} vs {policies[0]}")
    print(f"  discordant: a_win={a_win} b_win={b_win}  diff={diff:+.6f}  mcnemar p={p_val:.8f}")
    print(f"  bootstrap (tool's exact MC convention, seed 20260919, 10k): [{lo_rep:+.6f}, {hi_rep:+.6f}]")
    print(f"  bootstrap EXACT distribution (DP):                          [{lo_ex:+.6f}, {hi_ex:+.6f}]")
    print(f"  exact P(resampled diff <= 0) = {pmass[0]:.4f}   P(< 0) = {pmass[1]:.4f}")
    # my own independent MC runs
    for seed in (1, 42, 777, 123456):
        lo2, hi2, _ = bootstrap_mc(a, b, 100000, seed)
        print(f"  bootstrap MC own (seed {seed}, 100k): [{lo2:+.6f}, {hi2:+.6f}]")

print("\n=== fragility: one-item flips (which numbers would change the sign of the readings) ===")
for chal, (aw, bw) in ((policies[1], (3, 7)), (policies[2], (3, 12))):
    for d_aw, d_bw, note in ((-1, +1, "one challenger win becomes a baseline win"),
                             (+1, -1, "one baseline win becomes a challenger win"),
                             (0, +1, "one more challenger win (60 -> 61 correct)"),
                             (-1, 0, "one challenger win lost outright")):
        a2, b2 = aw + d_aw, bw + d_bw
        if a2 < 0 or b2 < 0 or (a2 + b2) > 60:
            continue
        n2 = a2 + b2
        p2 = mcnemar_exact(a2, b2)
        dist = exact_delta_dist(60, a2 / 60, b2 / 60)
        vs = sorted(dist)
        cum = 0.0
        q_lo = None
        for v in vs:
            cum += dist[v]
            if cum >= 0.025:
                q_lo = v
                break
        verdict = "excludes 0" if q_lo > 0 else "includes 0"
        print(f"  {chal} [{note}]: a={a2} b={b2} -> p={p2:.5f}, 2.5% q={q_lo/60:+.4f} ({verdict})")

# ---------------------------------------------------------------- extrapolation
print("\n=== how many items would two_step need for the paired 2.5% quantile > 0? ===")
p_a, p_b = 3 / 60, 7 / 60


def exact_quantile(n, p_a, p_b, q=0.025):
    dist = {0: 1.0}
    for _ in range(n):
        nd = {}
        for v, pr in dist.items():
            nd[v - 1] = nd.get(v - 1, 0.0) + pr * p_a
            nd[v + 1] = nd.get(v + 1, 0.0) + pr * p_b
            nd[v] = nd.get(v, 0.0) + pr * (1 - p_a - p_b)
        dist = nd
    cum = 0.0
    for v in sorted(dist):
        cum += dist[v]
        if cum >= q:
            return v / n
    return None


for nn in (60, 80, 100, 120, 140, 160, 200, 300):
    print(f"  n={nn:4d}: 2.5% quantile = {exact_quantile(nn, p_a, p_b):+.4f}"
          + ("  <- excludes 0" if exact_quantile(nn, p_a, p_b) > 0 else ""))

# ---------------------------------------------------------------- compare vs committed
print("\n=== compare against committed docs/evidence/e3d_cue_decision_4b.json ===")
committed = json.load(open("/var/home/rybens/workspace/ggufone/docs/evidence/e3d_cue_decision_4b.json"))
ok = True
for p in policies:
    c = committed["shapes"][p]
    mine = out[p]
    checks = [
        ("n", c["n"], mine["n"]),
        ("correct", c["correct"], mine["correct"]),
        ("agreement", round(c["agreement"], 12), round(mine["agreement"], 12)),
        ("wilson_lo", round(c["wilson"][0], 12), round(mine["wilson"][0], 12)),
        ("wilson_hi", round(c["wilson"][1], 12), round(mine["wilson"][1], 12)),
        ("low_mass", c["low_mass"], mine["low_mass"]),
        ("above_floor", c["coverage"]["above_floor"], mine["above_floor"]),
        ("cov_min", c["coverage"]["min"], mine["coverage"]["min"]),
        ("cov_median", c["coverage"]["median"], mine["coverage"]["median"]),
        ("cov_max", c["coverage"]["max"], mine["coverage"]["max"]),
        ("cov_mean", round(c["coverage"]["mean"], 12), round(mine["coverage"]["mean"], 12)),
    ]
    for name, want, got in checks:
        same = (abs(want - got) <= 1e-9) if isinstance(want, float) else (want == got)
        if not same:
            ok = False
            print(f"  MISMATCH {p} {name}: committed={want} recomputed={got}")
print("all per-shape agreement/wilson/coverage checks:", "PASS" if ok else "FAIL")

for dec in committed["decisions"]:
    chal = dec["challenger"]
    new = rows[chal]
    new_by_id = {r["id"]: r for r in new}
    ids = [r["id"] for r in base]
    a = [base_by_id[i]["correct"] for i in ids]
    b = [new_by_id[i]["correct"] for i in ids]
    a_win = sum(1 for x, y in zip(a, b) if x and not y)
    b_win = sum(1 for x, y in zip(a, b) if y and not x)
    p_val = mcnemar_exact(a_win, b_win)
    lo_rep, hi_rep, _ = bootstrap_mc(a, b, 10000, 20260919)
    P = dec["paired"]
    print(f"  {chal}: committed diff={P['difference']:.6f} lo={P['low']:.6f} hi={P['high']:.6f} "
          f"a={P['discordant']['a_win']} b={P['discordant']['b_win']} p={P['mcnemar_p']}")
    print(f"      recomputed diff={(sum(b)-sum(a))/len(a):.6f} lo={lo_rep:.6f} hi={hi_rep:.6f} "
          f"a={a_win} b={b_win} p={p_val}")
print("\ndone")
