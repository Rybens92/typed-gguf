"""exp1 analysis: accuracy vs baseline + Wilson CI, reliability flags, confidence, timing.

BASE resolution (same rule as run.sh): $REAL_PURPOSE_BASE, else the directory this script lives in.
"""
import json
import math
import os
import pathlib
import statistics

_HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path(os.environ.get("REAL_PURPOSE_BASE") or _HERE)
FLOOR_CONF = 0.5  # "the model is confident enough to auto-act" threshold used in the report


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x):
    return f"{100 * x:.1f}%"


items = {json.loads(l)["id"]: json.loads(l) for l in (BASE / "items.jsonl").read_text().splitlines() if l.strip()}
rows = []
for iid, item in items.items():
    payload = json.loads((BASE / "out" / f"{iid}.json").read_text())
    ans = payload["answers"]
    rows.append({"id": iid, "item": item, "payload": payload, "answers": ans})

n = len(rows)
print(f"items analysed: {n}")

# ---- choice: queue routing -------------------------------------------------
q_ok = [r for r in rows if r["answers"]["queue"]["choice"] == r["item"]["gold_queue"]]
q_acc = len(q_ok) / n
gold_counts = {}
for r in rows:
    gold_counts[r["item"]["gold_queue"]] = gold_counts.get(r["item"]["gold_queue"], 0) + 1
majority = max(gold_counts.values()) / n
lo, hi = wilson(len(q_ok), n)
print("\n== choice / queue ==")
print(f"accuracy {len(q_ok)}/{n} = {pct(q_acc)}  Wilson95 [{pct(lo)} , {pct(hi)}]")
print(f"majority-class baseline {pct(majority)} (class sizes {gold_counts})")
confusion = {}
for r in rows:
    key = (r["item"]["gold_queue"], r["answers"]["queue"]["choice"])
    confusion[key] = confusion.get(key, 0) + 1
print("gold -> predicted (off-diagonal only):")
for (g, p), c in sorted(confusion.items()):
    if g != p:
        print(f"  {g:>9} -> {p:<9} {c}")

# ---- noul: escalation gate -------------------------------------------------
e_ok = [r for r in rows if bool(r["answers"]["escalate"]["probabilities"]["yes"] > 0.5) == r["item"]["gold_esc"]]
n_true = sum(1 for r in rows if r["item"]["gold_esc"])
base_esc = max(n_true, n - n_true) / n
lo, hi = wilson(len(e_ok), n)
print("\n== noul / escalate ==")
print(f"accuracy {len(e_ok)}/{n} = {pct(len(e_ok) / n)}  Wilson95 [{pct(lo)} , {pct(hi)}]")
print(f"majority-class baseline {pct(base_esc)} (true {n_true} / false {n - n_true})")
for r in rows:
    if bool(r["answers"]["escalate"]["probabilities"]["yes"] > 0.5) != r["item"]["gold_esc"]:
        a = r["answers"]["escalate"]
        print(f"  miss {r['id']}: gold esc={r['item']['gold_esc']} p_yes={a['noul']:.4f} "
              f"trigger={r['item']['esc_trigger'] or '-'} :: {r['item']['state'][:70]}")
by_trigger = {}
for r in rows:
    t = r["item"]["esc_trigger"] or "none"
    by_trigger.setdefault(t, [0, 0])
    by_trigger[t][1] += 1
    if r["id"] in {x["id"] for x in e_ok}:
        by_trigger[t][0] += 1
print("escalate accuracy by truth-trigger:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_trigger.items())})

# ---- score: severity -------------------------------------------------------
exact = [r for r in rows if int(max(r["answers"]["severity"]["probabilities"], key=r["answers"]["severity"]["probabilities"].get)) == r["item"]["gold_sev"]]
err = [abs(int(max(r["answers"]["severity"]["probabilities"], key=r["answers"]["severity"]["probabilities"].get)) - r["item"]["gold_sev"]) for r in rows]
lo, hi = wilson(len(exact), n)
print("\n== score / severity ==")
print(f"argmax-level accuracy {len(exact)}/{n} = {pct(len(exact) / n)}  Wilson95 [{pct(lo)} , {pct(hi)}]")
print(f"MAE {statistics.mean(err):.3f} level(s); within one level {sum(1 for e in err if e <= 1)}/{n}")
print("score-value MAE vs gold level: "
      f"{statistics.mean(abs(r['answers']['severity']['score'] - r['item']['gold_sev']) for r in rows):.3f}")

# ---- reliability flags + coverage ------------------------------------------
print("\n== reliability flags (coverage floor 0.10) ==")
for qid in ("queue", "severity", "escalate"):
    rel = [r["answers"][qid]["reliability"] for r in rows if "reliability" in r["answers"][qid]]
    rel = rel or ["n/a"]
    cov = [r["answers"][qid]["coverage"] for r in rows if "coverage" in r["answers"][qid]]
    refused = sum(1 for r in rows if r["answers"][qid].get("cue", {}).get("refused"))
    print(f"  {qid:>8}: low_mass {rel.count('low_mass')}/{len(rel)}  low_confidence {rel.count('low_confidence')}/{len(rel)}"
          f"  median coverage {statistics.median(cov):.4f}  min {min(cov):.4f}  cue.refused {refused}")

# ---- confidence vs correctness (choice) ------------------------------------
print("\n== raw confidence vs correctness (choice queue; UNcalibrated) ==")
conf_ok = [r["answers"]["queue"]["confidence"] for r in q_ok]
conf_bad = [r["answers"]["queue"]["confidence"] for r in rows if r["id"] not in {x["id"] for x in q_ok}]
print(f"mean confidence correct {statistics.mean(conf_ok):.3f} (n={len(conf_ok)}) vs wrong "
      f"{statistics.mean(conf_bad) if conf_bad else float('nan'):.3f} (n={len(conf_bad)})")
for thr in (0.3, 0.4, 0.5, 0.6):
    sub = [r for r in rows if r["answers"]["queue"]["confidence"] >= thr]
    if sub:
        acc = sum(1 for r in sub if r["answers"]["queue"]["choice"] == r["item"]["gold_queue"]) / len(sub)
        print(f"  confidence >= {thr:.1f}: n={len(sub):>2}  accuracy {pct(acc)}")
    else:
        print(f"  confidence >= {thr:.1f}: n=0")

# ---- noul decisiveness vs correctness --------------------------------------
print("\n== noul decisiveness vs correctness (escalate) ==")
dec = [(max(a["probabilities"]["yes"], 1 - a["probabilities"]["yes"]), r) for r in rows for a in [r["answers"]["escalate"]]]
ok_ids = {r["id"] for r in e_ok}
for thr in (0.6, 0.9, 0.99):
    sub = [r for d, r in dec if d >= thr]
    if sub:
        acc = sum(1 for r in sub if r["id"] in ok_ids) / len(sub)
        print(f"  decisiveness >= {thr:.2f}: n={len(sub):>2}  accuracy {pct(acc)}")
    else:
        print(f"  decisiveness >= {thr:.2f}: n=0")
gt = [r["answers"]["escalate"]["noul"] for r in rows if r["item"]["gold_esc"]]
gf = [r["answers"]["escalate"]["noul"] for r in rows if not r["item"]["gold_esc"]]
print(f"mean p_yes where gold=true  {statistics.mean(gt):.3f} (n={len(gt)})")
print(f"mean p_yes where gold=false {statistics.mean(gf):.3f} (n={len(gf)})")
print("p_yes of gold=true items:", " ".join(f"{r['id']}={r['answers']['escalate']['noul']:.3f}" for r in rows if r["item"]["gold_esc"]))

# ---- timing ----------------------------------------------------------------
print("\n== timing (warm host) ==")
def med(field, sub="timings"):
    vals = [r["payload"][sub][field] for r in rows if field in r["payload"].get(sub, {})]
    return statistics.median(vals), min(vals), max(vals)
for field in ("model_load_ms", "prefill_ms", "questions_ms", "total_ms"):
    m, lo_, hi_ = med(field)
    print(f"  {field:<14} median {m:8.2f} ms   min {lo_:.2f}  max {hi_:.2f}")
wall = [int(l.split("wall_ms=")[1].split()[0]) for l in (BASE / "run.log").read_text().splitlines() if "wall_ms=" in l]
print(f"  wall clock per CLI call: median {statistics.median(wall):.0f} ms  min {min(wall)}  max {max(wall)}  sum {sum(wall) / 1000:.1f} s")
loads = sum(1 for r in rows if r["payload"]["timings"]["model_load_ms"] > 0)
print(f"  calls that paid a model load: {loads}/{n}; served_by: "
      f"{sorted({r['payload']['engine'].get('keep', {}).get('served_by') for r in rows})}")
toks = [r["payload"]["usage"]["input_tokens"] for r in rows]
print(f"  input tokens per request: median {statistics.median(toks)}  min {min(toks)}  max {max(toks)}")
nctx = sorted({r["payload"]["engine"]["n_ctx"] for r in rows})
print(f"  engine n_ctx per request: {nctx}")
print(f"  calibrated: {sorted({str(r['payload'].get('calibrated')) for r in rows})}")
print(f"  warnings: {sorted({w for r in rows for w in r['payload']['warnings']}) or 'none'}")
