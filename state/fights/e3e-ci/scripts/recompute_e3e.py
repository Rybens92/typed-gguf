#!/usr/bin/env python3
"""Independent recomputation of the E3e eight-arm table (audit card t_57bd3db2).

Written by the auditor from scratch: it reads ONLY the raw arm reports
(`docs/evidence/e3e/bench_*.json`, `docs/evidence/e3e/probe_default.json`), the committed baseline
(`docs/evidence/e3d/bench_templated_shipped.json`) and the committed dev set, and re-derives
every number the record `docs/evidence/e3e_roles_decision.json` publishes —
cells, per-type cells, `low_mass`, refusals, cue verdicts, coverage percentiles,
prefix-token ranges, Wilson intervals, all 28 paired readings, the exact McNemar
p-values, the Wald risk-difference intervals and an exact-DP (seed-free) bootstrap
of the paired difference — then diffs my reading against the record.

Nothing here imports the repo's tools: the semantics are re-implemented.

Run:  python3 state/fights/e3e-ci/scripts/recompute_e3e.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
Z = 1.959963984540054

# The record's own report order (from `docs/evidence/e3e/report.sh`), so pairs can be matched by label.
ARM_FILES = [
    "docs/evidence/e3e/bench_shipped_answer_sheet.json",
    "docs/evidence/e3e/bench_shipped_role_split.json",
    "docs/evidence/e3e/bench_two_step_answer_sheet.json",
    "docs/evidence/e3e/bench_two_step_role_split.json",
    "docs/evidence/e3e/bench_json_instructed_answer_sheet.json",
    "docs/evidence/e3e/bench_json_instructed_role_split.json",
    "docs/evidence/e3e/bench_json_instructed_answer_sheet_system.json",
    "docs/evidence/e3e/bench_json_instructed_role_split_system.json",
]
RECORD = "docs/evidence/e3e_roles_decision.json"
BASELINE = "docs/evidence/e3d/bench_templated_shipped.json"
PROBE = "docs/evidence/e3e/probe_default.json"
DEVSET = "src/ggufone/bench/devset.jsonl"

DISAGREEMENTS: list[str] = []
CHECKS = 0


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def label_of(report: dict) -> str:
    config = report.get("config") or {}
    cue = str(config.get("cue") or "shipped")
    fmt = str(config.get("chat_format") or "answer_sheet")
    text = f"{cue}/{fmt}"
    if cue == "json_instructed" and str(config.get("json_contract") or "question") == "system":
        text += "/system"
    return text


def check(name: str, record_value, mine, tol=0.0) -> None:
    """Diff one number against the record; collect, never raise."""
    global CHECKS
    CHECKS += 1
    ok = True
    if record_value is None and mine is None:
        ok = True
    elif isinstance(record_value, (int, float)) and isinstance(mine, (int, float)):
        ok = abs(float(record_value) - float(mine)) <= tol
    else:
        ok = record_value == mine
    if not ok:
        DISAGREEMENTS.append(f"{name}: record={record_value!r} mine={mine!r}")


# ---------------------------------------------------------------- statistics
def wilson96(k: int, n: int) -> tuple[float, float]:
    """The arm harness's own convention (`harness.wilson_interval`, z=1.96 exactly)."""
    return wilson_z(k, n, 1.96)


def wilson_z(k: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, mid - half), min(1.0, mid + half))


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for k/n (z = 1.96 to 1e-12) — the decision tool's convention."""
    return wilson_z(k, n, Z)


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by continued fraction (Lentz) — a second,
    algorithmically different route to the binomial tail than the comb sum."""
    def cf(a: float, b: float, x: float) -> float:
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 500):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-16:
                break
        return h

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * cf(a, b, x) / a
    return 1.0 - front * cf(b, a, 1.0 - x) / b


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p = min(1, 2 * P(X <= min(b,c))), X ~ Binom(b+c, 1/2).

    Computed twice: the comb sum (exact rationals) and the incomplete-beta identity;
    a mismatch beyond 1e-12 is a bug, not a finding.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    comb_tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    p_comb = min(1.0, 2.0 * comb_tail)
    p_beta = min(1.0, 2.0 * betainc(n - k, k + 1.0, 0.5))
    if abs(p_comb - p_beta) > 1e-12:
        DISAGREEMENTS.append(f"mcnemar self-check b={b} c={c}: comb={p_comb} beta={p_beta}")
    return p_comb


def wald(b: int, c: int, n: int) -> tuple[float, list[float]]:
    diff = (b - c) / n
    spread = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    return diff, [max(-1.0, diff - Z * spread), min(1.0, diff + Z * spread)]


def dp_bootstrap(b: int, c: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Exact bootstrap distribution of the paired difference, no seed.

    The bootstrap resamples the n items with replacement; each draw is +1 (challenger-only),
    -1 (baseline-only) or 0 (concordant) with the observed frequencies. The distribution of
    (B* - C*)/n is therefore the n-fold convolution of a 3-point distribution, computed here
    by dynamic programming over the count of +1 and -1 draws — the exact object a Monte-Carlo
    bootstrap estimates. Returns (q_lo, q_hi, P(diff <= 0)) with the empirical-CDF quantiles.
    """
    dist: dict[int, float] = {0: 1.0}
    pb, pc = b / n, c / n
    for _ in range(n):
        nxt: dict[int, float] = {}
        for v, p in dist.items():
            nxt[v] = nxt.get(v, 0.0) + p * (1.0 - pb - pc)
            nxt[v + 1] = nxt.get(v + 1, 0.0) + p * pb
            nxt[v - 1] = nxt.get(v - 1, 0.0) + p * pc
        dist = nxt
    values = sorted(dist)
    cdf = 0.0
    q_lo = q_hi = None
    tail = 0.0
    for v in values:
        cdf += dist[v]
        if q_lo is None and cdf >= alpha / 2:
            q_lo = v / n
        if q_hi is None and cdf >= 1 - alpha / 2:
            q_hi = v / n
        if v <= 0:
            tail += dist[v]
        if q_lo is not None and q_hi is not None:
            pass
    return q_lo, q_hi, tail


# ---------------------------------------------------------------- cells
def my_cell(report: dict) -> dict:
    items = report["items"]
    n = len(items)
    correct = sum(1 for it in items if it.get("correct"))
    per_type: dict[str, dict] = {}
    for it in items:
        bucket = per_type.setdefault(str(it.get("type")), {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += 1 if it.get("correct") else 0
    for bucket in per_type.values():
        bucket["agreement"] = bucket["correct"] / bucket["n"]
        bucket["ci"] = list(wilson(bucket["correct"], bucket["n"]))
    verdicts: dict[str, int] = {}
    refusals = 0
    for it in items:
        cue = it.get("cue") or {}
        refusals += 1 if cue.get("refused") else 0
        if "verdict" in cue:
            verdicts[str(cue["verdict"])] = verdicts.get(str(cue["verdict"]), 0) + 1
    cov = sorted(float(it.get("coverage") or 0.0) for it in items)
    prefix = [int(it.get("prefix_tokens") or 0) for it in items]
    return {
        "label": label_of(report),
        "items": n,
        "correct": correct,
        "agreement": correct / n,
        "ci": list(wilson(correct, n)),
        "per_type": per_type,
        "low_mass": sum(1 for it in items if it.get("reliability") == "low_mass"),
        "refusals": refusals,
        "verdicts": verdicts,
        "coverage": {"n": len(cov), "min": cov[0], "p50": cov[len(cov) // 2], "max": cov[-1]},
        "prefix_tokens": {"min": min(prefix), "max": max(prefix)},
    }


def compare_cells(record: dict) -> dict[str, dict]:
    mine_by_label: dict[str, dict] = {}
    record_by_label = {cell["label"]: cell for cell in record["cells"]}
    for path in ARM_FILES:
        report = load(path)
        cell = my_cell(report)
        mine_by_label[cell["label"]] = cell
        # the arm's own stored blocks must match the rows (raw-file internal consistency).
        # The harness stores its CIs with z=1.96 exactly (`harness.wilson_interval` default),
        # which is *not* the decision tool's z — the tool recomputes from the rows, so this
        # block checks the raw file against its own convention only.
        overall = report.get("overall") or {}
        check(f"{path} overall.correct", overall.get("correct"), cell["correct"])
        check(f"{path} overall.agreement", overall.get("agreement"), cell["agreement"], 1e-12)
        lo96, hi96 = wilson96(cell["correct"], cell["items"])
        check(f"{path} overall.ci[0] (z=1.96)", (overall.get("ci") or [None])[0], lo96, 1e-12)
        check(f"{path} overall.ci[1] (z=1.96)", (overall.get("ci") or [None])[1], hi96, 1e-12)
        for name, bucket in (report.get("per_type") or {}).items():
            check(f"{path} per_type.{name}.correct", bucket.get("correct"),
                  cell["per_type"][name]["correct"])
            lo_t, hi_t = wilson96(bucket["correct"], bucket["n"])
            check(f"{path} per_type.{name}.ci[0] (z=1.96)", (bucket.get("ci") or [None])[0],
                  lo_t, 1e-12)
            check(f"{path} per_type.{name}.ci[1] (z=1.96)", (bucket.get("ci") or [None])[1],
                  hi_t, 1e-12)
        # then the record's cell block against my reading of the same rows
        rec = record_by_label.get(cell["label"])
        if rec is None:
            DISAGREEMENTS.append(f"record has no cell {cell['label']}")
            continue
        check(f"{cell['label']} correct", rec["correct"], cell["correct"])
        check(f"{cell['label']} items", rec["items"], cell["items"])
        check(f"{cell['label']} agreement", rec["agreement"], cell["agreement"], 1e-12)
        check(f"{cell['label']} ci[0]", rec["ci"][0], cell["ci"][0], 1e-12)
        check(f"{cell['label']} ci[1]", rec["ci"][1], cell["ci"][1], 1e-12)
        check(f"{cell['label']} low_mass", rec["low_mass"], cell["low_mass"])
        check(f"{cell['label']} refusals", rec["refusals"], cell["refusals"])
        check(f"{cell['label']} verdicts", rec["verdicts"], cell["verdicts"])
        check(f"{cell['label']} coverage.min", rec["coverage"]["min"], cell["coverage"]["min"], 1e-18)
        check(f"{cell['label']} coverage.p50", rec["coverage"]["p50"], cell["coverage"]["p50"], 1e-18)
        check(f"{cell['label']} coverage.max", rec["coverage"]["max"], cell["coverage"]["max"], 1e-18)
        check(f"{cell['label']} prefix.min", rec["prefix_tokens"]["min"], cell["prefix_tokens"]["min"])
        check(f"{cell['label']} prefix.max", rec["prefix_tokens"]["max"], cell["prefix_tokens"]["max"])
        for name, bucket in cell["per_type"].items():
            rb = (rec.get("per_type") or {}).get(name)
            if rb is None:
                DISAGREEMENTS.append(f"{cell['label']} per_type {name} missing in record")
                continue
            check(f"{cell['label']}.{name}.n", rb["n"], bucket["n"])
            check(f"{cell['label']}.{name}.correct", rb["correct"], bucket["correct"])
            check(f"{cell['label']}.{name}.ci[0]", rb["ci"][0], bucket["ci"][0], 1e-12)
            check(f"{cell['label']}.{name}.ci[1]", rb["ci"][1], bucket["ci"][1], 1e-12)
            check(f"{cell['label']}.{name}.agreement", rb["agreement"], bucket["agreement"], 1e-12)
        # policy identity: the record's framing/config fields vs the raw file's
        config = report.get("config") or {}
        check(f"{cell['label']} cue", rec["cue"], config.get("cue"))
        check(f"{cell['label']} chat_format", rec["chat_format"], config.get("chat_format"))
        check(f"{cell['label']} json_contract", rec["json_contract"], config.get("json_contract"))
    return mine_by_label


# ---------------------------------------------------------------- pairs
def my_pairs(reports: list[dict], baseline_label: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for i, left in enumerate(reports):
        for right in reports[i + 1:]:
            lrows = {str(it["id"]): it for it in left["items"]}
            rrows = {str(it["id"]): it for it in right["items"]}
            shared = sorted(set(lrows) & set(rrows))
            b = sum(1 for k in shared if rrows[k].get("correct") and not lrows[k].get("correct"))
            c = sum(1 for k in shared if lrows[k].get("correct") and not rrows[k].get("correct"))
            both = sum(1 for k in shared if lrows[k].get("correct") and rrows[k].get("correct"))
            neither = len(shared) - b - c - both
            diff, ci = wald(b, c, len(shared))
            p = mcnemar_exact(b, c)
            out[(label_of(left), label_of(right))] = {
                "n": len(shared), "challenger_only": b, "baseline_only": c,
                "both_correct": both, "neither_correct": neither,
                "difference": diff, "ci": ci, "mcnemar_p": p,
                "challenger_wins": bool(ci[0] > 0.0 and p < 0.05),
            }
    return out


def compare_pairs(record: dict, reports: list[dict]) -> None:
    mine = my_pairs(reports, record["decision"]["baseline"])
    for pair in record["pairs"]:
        key = (pair["baseline"], pair["challenger"])
        got = mine.get(key)
        if got is None:
            DISAGREEMENTS.append(f"pair {key} missing from my recomputation")
            continue
        tag = f"{key[0]} vs {key[1]}"
        check(f"{tag} n", pair["n"], got["n"])
        check(f"{tag} challenger_only", pair["challenger_only"], got["challenger_only"])
        check(f"{tag} baseline_only", pair["baseline_only"], got["baseline_only"])
        check(f"{tag} both_correct", pair["both_correct"], got["both_correct"])
        check(f"{tag} neither_correct", pair["neither_correct"], got["neither_correct"])
        check(f"{tag} difference", pair["difference"], got["difference"], 1e-15)
        check(f"{tag} ci[0]", pair["ci"][0], got["ci"][0], 1e-15)
        check(f"{tag} ci[1]", pair["ci"][1], got["ci"][1], 1e-15)
        check(f"{tag} mcnemar_p", pair["mcnemar_p"], got["mcnemar_p"], 1e-15)
        check(f"{tag} challenger_wins", pair["challenger_wins"], got["challenger_wins"])
        if pair["baseline"] == "shipped/answer_sheet":
            q_lo, q_hi, tail = dp_bootstrap(got["challenger_only"], got["baseline_only"], got["n"])
            print(f"  bootstrap  {pair['challenger']:<38} exact-DP CI [{q_lo:+.3f}, {q_hi:+.3f}] "
                  f"P(diff<=0)={tail:.4f}  wald [{got['ci'][0]:+.3f}, {got['ci'][1]:+.3f}]")
    # verdicts against the record's baseline
    decision = record["decision"]
    baseline = decision["baseline"]
    record_verdicts = {v["label"]: v for v in decision["verdicts"]}
    for label, got in mine.items():
        if label[0] != baseline:
            continue
        challenger = label[1]
        rec = record_verdicts.get(challenger)
        if rec is None:
            DISAGREEMENTS.append(f"record has no verdict for {challenger}")
            continue
        if got["challenger_wins"]:
            v = "wins"
        elif got["challenger_only"] == 0 and got["baseline_only"] == 0:
            v = "identical on these 60 items"
        else:
            v = "not by more than the CI noise"
        check(f"verdict {challenger}", rec["verdict"], v)
    # ranking
    ranking = sorted(record["cells"], key=lambda c: (-c["correct"], c["refusals"], c["low_mass"], c["label"]))
    if [c["label"] for c in ranking] != [r["label"] for r in decision["ranking"]]:
        DISAGREEMENTS.append("ranking order differs")


# ---------------------------------------------------------------- freeze
def freeze_mine(probe_path: str, base_path: str, tolerance: float) -> dict:
    probe = load(probe_path)
    base = load(base_path)
    prows = {str(it["id"]): it for it in probe["items"]}
    brows = {str(it["id"]): it for it in base["items"]}
    shared = sorted(set(prows) & set(brows))
    hard: list[dict] = []
    numeric_only: list[str] = []
    decisions_agree = 0
    prefix_differ: list[str] = []
    max_prob = max_cov = 0.0
    for key in shared:
        left, right = prows[key], brows[key]
        same = (left.get("got") == right.get("got")
                and left.get("reliability") == right.get("reliability"))
        decisions_agree += int(same)
        problems = [] if same else [f"got {left.get('got')!r} != {right.get('got')!r}"]
        if left.get("prefix_tokens") != right.get("prefix_tokens"):
            prefix_differ.append(key)
            problems.append(f"prefix_tokens {left.get('prefix_tokens')} != {right.get('prefix_tokens')}")
        cov_delta = abs(float(left.get("coverage") or 0.0) - float(right.get("coverage") or 0.0))
        max_cov = max(max_cov, cov_delta)
        soft = [] if cov_delta <= tolerance else [f"coverage delta {cov_delta:.3e}"]
        for name, value in (right.get("probabilities") or {}).items():
            other = (left.get("probabilities") or {}).get(name)
            delta = abs(float(other) - float(value)) if other is not None else float("inf")
            max_prob = max(max_prob, delta) if math.isfinite(delta) else max_prob
            if delta > tolerance:
                soft.append(f"p({name}) delta {delta:.3e}")
        if problems:
            hard.append({"id": key, "problems": problems + soft})
        elif soft:
            numeric_only.append(key)
    return {"n": len(shared), "frozen": not hard, "bit_frozen": not hard and not numeric_only,
            "decisions": len(shared), "decisions_agree": decisions_agree,
            "differences": hard, "numeric_only": numeric_only,
            "prefix_tokens_differ": prefix_differ,
            "max_probability_delta": max_prob, "max_coverage_delta": max_cov,
            "tolerance": tolerance}


def compare_freeze(record: dict) -> None:
    rec_probe = record["freeze"]
    mine = freeze_mine(PROBE, BASELINE, 1e-9)
    check("probe frozen", rec_probe["frozen"], mine["frozen"])
    check("probe bit_frozen", rec_probe["bit_frozen"], mine["bit_frozen"])
    check("probe decisions_agree", rec_probe["decisions_agree"], mine["decisions_agree"])
    check("probe n", rec_probe["n"], mine["n"])
    check("probe prefix_tokens_differ", rec_probe["prefix_tokens_differ"], mine["prefix_tokens_differ"])
    check("probe numeric_only", rec_probe["numeric_only"], mine["numeric_only"])
    check("probe max_probability_delta", rec_probe["max_probability_delta"],
          mine["max_probability_delta"], 1e-15)
    check("probe max_coverage_delta", rec_probe["max_coverage_delta"],
          mine["max_coverage_delta"], 1e-15)
    rec_place = record["placement"]
    mine_place = freeze_mine("docs/evidence/e3e/bench_shipped_answer_sheet.json", BASELINE, 5e-3)
    check("placement frozen", rec_place["frozen"], mine_place["frozen"])
    check("placement decisions_agree", rec_place["decisions_agree"], mine_place["decisions_agree"])
    check("placement n", rec_place["n"], mine_place["n"])
    check("placement numeric_only", rec_place["numeric_only"], mine_place["numeric_only"])
    check("placement prefix_tokens_differ", rec_place["prefix_tokens_differ"],
          mine_place["prefix_tokens_differ"])
    check("placement max_probability_delta", rec_place["max_probability_delta"],
          mine_place["max_probability_delta"], 1e-15)
    check("placement max_coverage_delta", rec_place["max_coverage_delta"],
          mine_place["max_coverage_delta"], 1e-15)
    check("placement tolerance", rec_place["tolerance"], mine_place["tolerance"])


# ---------------------------------------------------------------- devset identity
def wire_gold(value):
    """The dev set's gold in the wire mapping the arms use (bool -> yes/no, int -> str)."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def check_devset(reports: list[dict]) -> None:
    rows = [json.loads(line) for line in (ROOT / DEVSET).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    gold = {str(r.get("id")): r for r in rows}
    print(f"devset: {len(rows)} rows, types "
          f"{ {t: sum(1 for r in rows if r.get('type') == t) for t in ('choice', 'noul', 'score')} }")
    for report in reports:
        ids = [str(it["id"]) for it in report["items"]]
        if ids != [str(r.get("id")) for r in rows]:
            DISAGREEMENTS.append(f"{label_of(report)}: item ids differ from the committed dev set")
        for it in report["items"]:
            row = gold.get(str(it["id"]))
            if row is None:
                continue
            if str(it.get("type")) != str(row.get("type")):
                DISAGREEMENTS.append(f"{it['id']}: type {it.get('type')} != devset {row.get('type')}")
            if str(it.get("expected")) != wire_gold(row.get("gold")):
                DISAGREEMENTS.append(
                    f"{it['id']}: expected {it.get('expected')!r} != devset gold "
                    f"{wire_gold(row.get('gold'))!r}")
            if bool(it.get("correct")) != (str(it.get("got")) == wire_gold(row.get("gold"))):
                DISAGREEMENTS.append(f"{it['id']}: correct flag disagrees with got vs gold")


def main() -> int:
    record = load(RECORD)
    reports = [load(path) for path in ARM_FILES]
    print(f"record: {len(record['cells'])} cells, {len(record['pairs'])} pairs, "
          f"generated_at {record['generated_at']}")
    compare_cells(record)
    compare_pairs(record, reports)
    compare_freeze(record)
    check_devset(reports)
    print(f"\nchecks run: {CHECKS}")
    if DISAGREEMENTS:
        print(f"DISAGREEMENTS ({len(DISAGREEMENTS)}):")
        for line in DISAGREEMENTS:
            print("  -", line)
        return 1
    print("no disagreement: every checked number matches the record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
