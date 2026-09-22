#!/usr/bin/env python3
"""Build the committed machine JSONs for the Tiel arm (card t_0d5db2ef).

Writes <BASE>/report_tiel.json (this model's run) and <BASE>/compare_vs_4b.json (per-item
comparison against the committed 4B report), and prints the numbers the evidence doc quotes.

Usage: REAL_PURPOSE_BASE=/work/t0d5-tiel python3 report_tiel.py [--partial]
BASE resolution: $REAL_PURPOSE_BASE, else /work/t0d5-tiel when that run directory exists, else
the directory this script lives in.
"""
import json
import math
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT = pathlib.Path("/work/t0d5-tiel")
BASE = pathlib.Path(os.environ.get("REAL_PURPOSE_BASE") or (DEFAULT if (DEFAULT / "items.jsonl").exists() else HERE))
REPO = HERE.parents[2]  # docs/evidence/<dir>/.. = docs/evidence -> docs -> repo
FOURB_DIR = REPO / "docs/evidence/real-purpose-4b-2026-09-22"
FOURB_REPORT = FOURB_DIR / "report.json"
MODEL_PATH = pathlib.Path("/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf")
FLOOR = 0.10  # OPTION_DEFAULTS["coverage_floor"]
CONF_GATE = 0.5
DEC_GATE = 0.9
PARTIAL = "--partial" in sys.argv


def wilson(k, n, z=1.959963985):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def stats(rows, pred):
    k = sum(1 for r in rows if pred(r))
    return {"correct": k, "n": len(rows), "accuracy": round(k / len(rows), 4), "wilson95": wilson(k, len(rows))}


def parse_runlog(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "exit=" not in line or "wall_ms=" not in line:
            continue
        head, _, cmd = line.partition(" :: ")
        parts = head.split()
        iid = parts[0]
        out[iid] = {
            "exit": int(parts[1].split("=")[1]),
            "wall_ms": int(parts[2].split("=")[1]),
            "command": cmd,
        }
    return out


def argmax_int(d):
    return int(max(d, key=d.get))


items = [json.loads(line) for line in (BASE / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
runlog = parse_runlog(BASE / "run.log")

rows = []
missing = []
for item in items:
    iid = item["id"]
    p = BASE / "out" / f"{iid}.json"
    meta = runlog.get(iid, {})
    if not p.exists() or meta.get("exit") != 0:
        missing.append({"id": iid, "exit": meta.get("exit"), "file": p.exists()})
        continue
    payload = json.loads(p.read_text(encoding="utf-8"))
    ans = payload["answers"]
    queue = ans["queue"]["choice"]
    sev_probs = ans["severity"]["probabilities"]
    sev_level = argmax_int(sev_probs)
    esc = bool(ans["escalate"]["probabilities"]["yes"] > 0.5)
    row = {
        "id": iid,
        "state": item["state"],
        "cue_note": item.get("cue", ""),
        "gold": {"queue": item["gold_queue"], "severity": item["gold_sev"], "escalate": item["gold_esc"],
                 "escalate_trigger": item.get("esc_trigger") or ""},
        "got": {"queue": queue, "severity_level": sev_level, "severity_score": round(ans["severity"]["score"], 5),
                "escalate": esc},
        "correct": {"queue": queue == item["gold_queue"], "severity": sev_level == item["gold_sev"],
                    "escalate": esc == item["gold_esc"]},
        "confidence": {"queue": ans["queue"]["confidence"], "severity": ans["severity"]["confidence"],
                       "escalate_decisiveness": max(ans["escalate"]["probabilities"]["yes"],
                                                    1 - ans["escalate"]["probabilities"]["yes"])},
        "probabilities": {"queue": ans["queue"]["probabilities"], "severity": sev_probs,
                          "escalate": ans["escalate"]["probabilities"]},
        "coverage": {q: ans[q]["coverage"] for q in ("queue", "severity", "escalate")},
        "reliability": {q: ans[q]["reliability"] for q in ("queue", "severity", "escalate")},
        "cue_verdict": {q: ans[q]["cue"]["verdict"] for q in ("queue", "severity", "escalate")},
        "timings_ms": {k: payload["timings"][k] for k in ("model_load_ms", "prefill_ms", "questions_ms", "total_ms")},
        "n_ctx": payload["engine"]["n_ctx"],
        "prefix_tokens": payload["engine"].get("prefix_tokens") or payload["usage"]["prefill_tokens"],
        "input_tokens": payload["usage"]["input_tokens"],
        "state_id": payload["engine"].get("state_id"),
        "template": payload["engine"].get("template"),
        "calibrated": payload.get("calibrated"),
        "warnings": payload.get("warnings", []),
        "exit": meta.get("exit"),
        "wall_ms": meta.get("wall_ms"),
        "command": meta.get("command", ""),
    }
    rows.append(row)

n = len(rows)
print(f"items: {n} of {len(items)} have exit=0 responses in {BASE / 'out'}")
if missing:
    print(f"  not analysed ({len(missing)}): " + ", ".join(f"{m['id']}(exit={m['exit']},file={m['file']})" for m in missing))

# ---------------------------------------------------------------- accuracy
q = stats(rows, lambda r: r["correct"]["queue"])
e = stats(rows, lambda r: r["correct"]["escalate"])
s = stats(rows, lambda r: r["correct"]["severity"])
gold_counts = {}
for r in rows:
    gold_counts[r["gold"]["queue"]] = gold_counts.get(r["gold"]["queue"], 0) + 1
q["majority_baseline"] = round(max(gold_counts.values()) / n, 4)
n_true = sum(1 for r in rows if r["gold"]["escalate"])
e["majority_baseline"] = round(max(n_true, n - n_true) / n, 4)
sev_err = [abs(r["got"]["severity_level"] - r["gold"]["severity"]) for r in rows]
s["mae_level"] = round(statistics.mean(sev_err), 4)
s["within_one_level"] = sum(1 for d in sev_err if d <= 1)
s["score_value_mae"] = round(statistics.mean(abs(r["got"]["severity_score"] - r["gold"]["severity"]) for r in rows), 4)

print(f"\nqueue    {q['correct']}/{q['n']} = {q['accuracy']:.4f}  Wilson {q['wilson95']}  (majority {q['majority_baseline']})")
print(f"escalate {e['correct']}/{e['n']} = {e['accuracy']:.4f}  Wilson {e['wilson95']}  (majority {e['majority_baseline']}, true {n_true})")
print(f"severity {s['correct']}/{s['n']} = {s['accuracy']:.4f}  Wilson {s['wilson95']}  MAE {s['mae_level']} level, within-1 {s['within_one_level']}/{n}")

print("\nqueue errors (gold -> got):")
for r in rows:
    if not r["correct"]["queue"]:
        print(f"  {r['id']}: {r['gold']['queue']} -> {r['got']['queue']}  conf {r['confidence']['queue']:.3f}")
print("escalate errors:")
for r in rows:
    if not r["correct"]["escalate"]:
        print(f"  {r['id']}: gold {r['gold']['escalate']} (T{r['gold']['escalate_trigger'] or '-'}) -> got {r['got']['escalate']} p_yes {r['probabilities']['escalate']['yes']:.3f}")
print("severity errors:")
for r in rows:
    if not r["correct"]["severity"]:
        print(f"  {r['id']}: gold {r['gold']['severity']} -> {r['got']['severity_level']} (score {r['got']['severity_score']:.3f})")

# ---------------------------------------------------------------- flags
flags = {}
for qid in ("queue", "severity", "escalate"):
    rel = [r["reliability"][qid] for r in rows]
    cov = [r["coverage"][qid] for r in rows]
    flags[qid] = {
        "low_mass": rel.count("low_mass"),
        "low_confidence": rel.count("low_confidence"),
        "other_reliability": {v: rel.count(v) for v in set(rel) if v not in ("ok", "low_mass", "low_confidence")},
        "refused": sum(1 for r in rows if r["cue_verdict"][qid] == "refused"),
        "coverage_median": round(statistics.median(cov), 6),
        "coverage_min": round(min(cov), 6),
    }
print("\nflags:", json.dumps(flags, sort_keys=True))

# ---------------------------------------------------------------- confidence ordering
conf_ok = [r["confidence"]["queue"] for r in rows if r["correct"]["queue"]]
conf_bad = [r["confidence"]["queue"] for r in rows if not r["correct"]["queue"]]
conf_thr = {}
for thr in (0.3, 0.4, 0.5, 0.6):
    sub = [r for r in rows if r["confidence"]["queue"] >= thr]
    conf_thr[str(thr)] = {"n": len(sub),
                          "accuracy": round(sum(1 for r in sub if r["correct"]["queue"]) / len(sub), 4) if sub else None}
dec_thr = {}
for thr in (0.6, 0.9, 0.99):
    sub = [r for r in rows if r["confidence"]["escalate_decisiveness"] >= thr]
    dec_thr[str(thr)] = {"n": len(sub),
                         "accuracy": round(sum(1 for r in sub if r["correct"]["escalate"]) / len(sub), 4) if sub else None}
st = [r for r in rows if r["confidence"]["queue"] >= CONF_GATE and r["confidence"]["escalate_decisiveness"] >= DEC_GATE]
straight = {"band": f"queue confidence >= {CONF_GATE} and escalate decisiveness >= {DEC_GATE}",
            "n": len(st), "both_correct": sum(1 for r in st if r["correct"]["queue"] and r["correct"]["escalate"])}
print(f"\nconfidence: mean correct {statistics.mean(conf_ok) if conf_ok else float('nan'):.3f} "
      f"vs wrong {statistics.mean(conf_bad) if conf_bad else float('nan'):.3f}")
print("  thresholds:", json.dumps(conf_thr), " decisiveness:", json.dumps(dec_thr))
print("  straight-through:", json.dumps(straight))

# ---------------------------------------------------------------- timing
def tmed(field):
    vals = [r["timings_ms"][field] for r in rows]
    return {"median": round(statistics.median(vals), 2), "min": round(min(vals), 2), "max": round(max(vals), 2)}


walls = [r["wall_ms"] for r in rows]
loads = [r["timings_ms"]["model_load_ms"] for r in rows]
timing = {
    "model_load_ms_once": round(max(loads), 2),
    "calls_paying_a_model_load": sum(1 for x in loads if x > 0),
    "prefill_ms": tmed("prefill_ms"),
    "questions_ms": tmed("questions_ms"),
    "total_ms_engine": tmed("total_ms"),
    "wall_per_cli_call_ms": {"median": round(statistics.median(walls), 1), "min": min(walls), "max": max(walls),
                             "sum_s": round(sum(walls) / 1000, 1)},
    "per_item_note": "each item is its own request (own state): one prefix prefill + one candidate batch + the "
                     "label decodes; nothing is generated",
}
print("\ntiming:", json.dumps(timing, indent=1))

# ---------------------------------------------------------------- engine block (verbatim from the responses)
engine = {}
if rows:
    sample = json.loads((BASE / "out" / f"{rows[0]['id']}.json").read_text(encoding="utf-8"))
    eng = sample["engine"]
    engine = {
        "version": "typed-gguf 0.1.0",
        "model_alias": sample.get("model") if isinstance(sample.get("model"), str) else None,
        "runtime": "llama.cpp b11026 (bundle linux-x64-vulkan)",
        "backend": eng.get("backend"),
        "effective_backend": eng.get("effective_backend"),
        "fit_n_gpu_layers": eng.get("fit", {}).get("n_gpu_layers"),
        "fit": {k: eng.get("fit", {}).get(k) for k in ("arch", "n_ctx", "n_seq_max", "kv_type", "est_weights_bytes",
                                                       "budget_bytes", "host_fingerprint", "model_sha256", "notes",
                                                       "warnings", "created_at")},
        "readout": eng.get("readout"),
        "cue": eng.get("cue"),
        "template": eng.get("template"),
        "chat_format": eng.get("chat_format"),
        "n_ctx_per_request": sorted({r["n_ctx"] for r in rows}),
        "calibrated": sorted({str(r["calibrated"]) for r in rows}),
        "coverage_floor": FLOOR,
        "warnings": sorted({w for r in rows for w in r["warnings"]}),
        "served_by": sorted({json.loads((BASE / "out" / f"{r['id']}.json").read_text(encoding="utf-8"))["engine"]
                             .get("keep", {}).get("served_by") for r in rows}),
        "threads": sorted({json.loads((BASE / "out" / f"{r['id']}.json").read_text(encoding="utf-8"))["engine"]
                           .get("keep", {}).get("key", {}).get("threads") for r in rows}),
    }
    print("\nengine:", json.dumps(engine, indent=1)[:1500])

# ---------------------------------------------------------------- model block
model = {
    "path": str(MODEL_PATH),
    "size_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None,
    "quant": "Q4_K_XL (UD)",
    "arch": "qwen35moe",
    "sha256": engine.get("fit", {}).get("model_sha256"),
}

report = {
    "schema": "typed_gguf.evidence.real_purpose/v1",
    "generated": "2026-09-22",
    "card": "t_0d5db2ef",
    "purpose": "support-inbox triage: queue routing (choice) + severity (score) + human-escalation gate (noul)",
    "arm": "tiel-coder-35b-a3b",
    "protocol_note": "same frozen items.jsonl/questions.json as the 4B arm (docs/evidence/real-purpose-4b-2026-09-22), "
                     "one `typed-gguf run --questions` per item, defaults except --threads (performance knob; see "
                     "engine.threads)",
    "model": model,
    "engine": engine,
    "dataset": {
        "path": "docs/evidence/real-purpose-4b-2026-09-22/items.jsonl (byte-identical copy next to this report)",
        "items": len(items),
        "analysed": n,
        "basis": "30 synthetic support-inbox messages authored 2026-09-22; labels frozen before any model call",
        "queue_class_sizes": gold_counts,
        "escalate_true": n_true,
        "severity_levels": {str(v): sum(1 for r in rows if r["gold"]["severity"] == v) for v in (0, 1, 2)},
    },
    "results": {
        "choice_queue": q,
        "noul_escalate": e,
        "score_severity": s,
        "reliability_flags": flags,
        "confidence_vs_correctness": {
            "calibrated": False,
            "null_note": "raw probability-scale values, no calibration fitted (calibrated: false)",
            "queue_mean_confidence_correct": round(statistics.mean(conf_ok), 4) if conf_ok else None,
            "queue_mean_confidence_wrong": round(statistics.mean(conf_bad), 4) if conf_bad else None,
            "queue_accuracy_at_threshold": conf_thr,
            "escalate_accuracy_at_decisiveness": dec_thr,
            "escalate_mean_p_yes_gold_true": round(statistics.mean([r["probabilities"]["escalate"]["yes"] for r in rows if r["gold"]["escalate"]]), 4) if n_true else None,
            "escalate_mean_p_yes_gold_false": round(statistics.mean([r["probabilities"]["escalate"]["yes"] for r in rows if not r["gold"]["escalate"]]), 4) if (n - n_true) else None,
        },
        "straight_through": straight,
        "timing": timing,
    },
    "run": {"run_dir": str(BASE), "requests_exit0": n, "requests_total": len(items), "missing": missing,
            "run_log": "run.log next to this report (exit code + wall ms + exact command per item)"},
    "rows": rows,
}
(BASE / "report_tiel.json").write_text(json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")
print(f"\nwrote {BASE / 'report_tiel.json'}")

# ---------------------------------------------------------------- comparison vs the 4B
if not FOURB_REPORT.exists():
    print("no 4B report; skipping comparison")
    sys.exit(0)
b4 = json.loads(FOURB_REPORT.read_text(encoding="utf-8"))
b4_rows = {r["id"]: r for r in b4["rows"]}
common = [r for r in rows if r["id"] in b4_rows]
agree = {"queue": 0, "severity": 0, "escalate": 0}
dis = {"queue": {"n": 0, "tiel_only_right": 0, "b4_only_right": 0, "neither": 0},
       "severity": {"n": 0, "tiel_only_right": 0, "b4_only_right": 0, "neither": 0},
       "escalate": {"n": 0, "tiel_only_right": 0, "b4_only_right": 0, "neither": 0}}
per_item = []
for r in common:
    t = r
    b = b4_rows[r["id"]]
    same = {
        "queue": t["got"]["queue"] == b["got"]["queue"],
        "severity": t["got"]["severity_level"] == int(b["got"]["severity_level"]),
        "escalate": bool(t["got"]["escalate"]) == bool(b["got"]["escalate"]),
    }
    for k, v in same.items():
        if v:
            agree[k] += 1
        else:
            dis[k]["n"] += 1
            t_ok = t["correct"][k]
            b_ok = bool(b["correct"][k])
            if t_ok and not b_ok:
                dis[k]["tiel_only_right"] += 1
            elif b_ok and not t_ok:
                dis[k]["b4_only_right"] += 1
            elif not t_ok and not b_ok:
                dis[k]["neither"] += 1
    per_item.append({
        "id": r["id"], "gold": t["gold"],
        "tiel": t["got"], "b4": {"queue": b["got"]["queue"], "severity_level": int(b["got"]["severity_level"]),
                                 "escalate": bool(b["got"]["escalate"])},
        "tiel_correct": t["correct"], "b4_correct": {k: bool(b["correct"][k]) for k in ("queue", "severity", "escalate")},
        "identical": same,
    })

nc = len(common)
tiel_acc = {k: round(sum(1 for r in common if r["correct"][k]) / nc, 4) for k in ("queue", "severity", "escalate")}
b4_acc = {k: round(sum(1 for r in common if bool(b4_rows[r["id"]]["correct"][k])) / nc, 4) for k in ("queue", "severity", "escalate")}
comparison = {
    "schema": "typed_gguf.evidence.real_purpose_compare/v1",
    "card": "t_0d5db2ef",
    "baseline": {"report": "docs/evidence/real-purpose-4b-2026-09-22/report.json", "model": b4["model"]["path"]},
    "compared_items": nc,
    "agreement": {k: {"identical": agree[k], "n": nc, "rate": round(agree[k] / nc, 4)} for k in agree},
    "disagreements": dis,
    "accuracy_on_common": {"tiel": tiel_acc, "b4": b4_acc,
                           "net_delta_tiel_minus_b4": {k: round(tiel_acc[k] - b4_acc[k], 4) for k in tiel_acc}},
    "per_item": per_item,
}
(BASE / "compare_vs_4b.json").write_text(json.dumps(comparison, indent=1, sort_keys=False) + "\n", encoding="utf-8")
print(f"wrote {BASE / 'compare_vs_4b.json'}  (n={nc})")
print("agreement:", json.dumps(comparison["agreement"]))
print("disagreements:", json.dumps(comparison["disagreements"]))
print("accuracy on common:", json.dumps(comparison["accuracy_on_common"]))
if PARTIAL and nc < len(items):
    print(f"NOTE: partial run — {nc} of {len(items)} items compared")
