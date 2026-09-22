#!/usr/bin/env python3
"""Build the committed machine JSONs for the MiMo arm (card t_d199e09c).

Writes <BASE>/report_mimo.json (this model's run) and <BASE>/compare_vs_4b.json (per-item
comparison against the committed 4B report, incl. the money/legal escalation-false-negative
cohort that has been the headline safety metric since the Tiel arm), and prints the numbers the
evidence doc quotes.

Usage: REAL_PURPOSE_BASE=/work/t_d199e09c/mimo python3 report_mimo.py
BASE resolution: $REAL_PURPOSE_BASE, else the directory this script lives in.
"""
import json
import math
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path(os.environ.get("REAL_PURPOSE_BASE") or HERE)
# docs/evidence/<dir>/.. = docs/evidence -> docs -> repo. $REPO overrides that when the script is
# run from a staging directory that is not inside the checkout (the worker-sandbox run does this).
REPO = pathlib.Path(os.environ["REPO"]) if os.environ.get("REPO") else HERE.parents[2]
FOURB_DIR = REPO / "docs/evidence/real-purpose-4b-2026-09-22"
FOURB_REPORT = FOURB_DIR / "report.json"
TIEL_REPORT = REPO / "docs/evidence/real-purpose-tiel-35b-2026-09-22/report_tiel.json"
FLOOR = 0.10                               # OPTION_DEFAULTS["coverage_floor"]
CONF_GATE = 0.5
DEC_GATE = 0.9
MONEY_LEGAL = ("T1", "T2")                 # the two escalation triggers the frozen set defines
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
    """run.log -> {id: {exit, wall_ms, command}} plus the HOST/PLACEMENT header lines."""
    out, header = {}, {}
    if not path.exists():
        return out, header
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("HOST "):
            header["host"] = line
        elif line.startswith("PLACEMENT "):
            header["placement"] = line
        elif "exit=" in line and "wall_ms=" in line:
            head, _, cmd = line.partition(" :: ")
            parts = head.split()
            out[parts[0]] = {"exit": int(parts[1].split("=")[1]),
                             "wall_ms": int(parts[2].split("=")[1]), "command": cmd}
    return out, header


def argmax_int(d):
    return int(max(d, key=d.get))


def fn_cohort(rows):
    """Escalation false negatives (gold True -> got False), split by escalation trigger."""
    fn = [r for r in rows if r["gold"]["escalate"] and not r["got"]["escalate"]]
    per_trigger = {}
    for r in fn:
        per_trigger[r["gold"]["escalate_trigger"] or "-"] = per_trigger.get(r["gold"]["escalate_trigger"] or "-", 0) + 1
    money = [r for r in fn if (r["gold"]["escalate_trigger"] or "") in MONEY_LEGAL]
    gold_money = [r for r in rows if (r["gold"]["escalate_trigger"] or "") in MONEY_LEGAL]
    over = [r for r in rows if not r["gold"]["escalate"] and r["got"]["escalate"]]
    return {
        "false_negatives_total": len(fn),
        "false_negatives_ids": [r["id"] for r in fn],
        "false_negatives_by_trigger": per_trigger,
        "money_legal_gold_n": len(gold_money),
        "money_legal_false_negatives": len(money),
        "money_legal_false_negative_ids": [r["id"] for r in money],
        "money_legal_recall": round(sum(1 for r in gold_money if r["got"]["escalate"]) / len(gold_money), 4) if gold_money else None,
        "over_escalations": len(over),
        "over_escalation_ids": [r["id"] for r in over],
    }


items = [json.loads(line) for line in (BASE / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
runlog, run_header = parse_runlog(BASE / "run.log")
model_path_from_log = None
if run_header.get("host"):
    for token in run_header["host"].split():
        if token.startswith("model="):
            model_path_from_log = token.split("=", 1)[1]

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
    rows.append({
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
    })

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
escalation_safety = fn_cohort(rows)

print(f"\nqueue    {q['correct']}/{q['n']} = {q['accuracy']:.4f}  Wilson {q['wilson95']}  (majority {q['majority_baseline']})")
print(f"escalate {e['correct']}/{e['n']} = {e['accuracy']:.4f}  Wilson {e['wilson95']}  (majority {e['majority_baseline']}, true {n_true})")
print(f"severity {s['correct']}/{s['n']} = {s['accuracy']:.4f}  Wilson {s['wilson95']}  MAE {s['mae_level']} level, within-1 {s['within_one_level']}/{n}")
print("\nescalation safety:", json.dumps(escalation_safety))

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

# ---------------------------------------------------------------- protocol assertions (AC3)
def payload_of(iid):
    return json.loads((BASE / "out" / f"{iid}.json").read_text(encoding="utf-8"))


protocol_checks = {}
if rows:
    engines = [payload_of(r["id"])["engine"] for r in rows]
    diffs = [payload_of(r["id"]).get("calibrated") for r in rows]
    floors = [payload_of(r["id"]).get("coverage_floor", FLOOR) for r in rows]
    protocol_checks = {
        "readout": sorted({e.get("readout") for e in engines}),
        "cue": sorted({e.get("cue") for e in engines}),
        "chat_format": sorted({json.dumps({k: v for k, v in (e.get("chat_format") or {}).items()
                                           if k in ("kind", "contract", "question_turn")}, sort_keys=True)
                               for e in engines}),
        "prefix_chars_range": [min((e.get("chat_format") or {}).get("prefix_chars", 0) for e in engines),
                               max((e.get("chat_format") or {}).get("prefix_chars", 0) for e in engines)],
        "template": sorted({json.dumps({k: v for k, v in (e.get("template") or {}).items()
                                        if k in ("kind", "renderer", "source", "family", "thinking")},
                                       sort_keys=True) for e in engines}),
        "calibrated": sorted({str(d) for d in diffs}),
        "coverage_floor": sorted({f for f in floors}),
        "n_ctx_per_request": sorted({r["n_ctx"] for r in rows}),
        "fit_n_gpu_layers": sorted({e.get("fit", {}).get("n_gpu_layers") for e in engines}),
        "fit_n_ctx": sorted({e.get("fit", {}).get("n_ctx") for e in engines}),
        "fit_kv_type": sorted({e.get("fit", {}).get("kv_type") for e in engines}),
        "fit_n_seq_max": sorted({e.get("fit", {}).get("n_seq_max") for e in engines}),
        "fit_source": sorted({e.get("fit", {}).get("source") for e in engines}),
        "fit_warnings": sorted({w for e in engines for w in (e.get("fit", {}).get("warnings") or [])}),
        "placement_n_gpu_layers": sorted({(e.get("placement") or {}).get("n_gpu_layers") for e in engines}),
        "placement_kv_type": sorted({(e.get("placement") or {}).get("kv_type") for e in engines}),
        "placement_degraded": sorted({str((e.get("placement") or {}).get("degraded")) for e in engines}),
        "placement_cpu_only": sorted({str((e.get("placement") or {}).get("cpu_only")) for e in engines}),
        "placement_attempts": sorted({json.dumps((e.get("placement") or {}).get("attempts")) for e in engines}),
        "effective_backend": sorted({e.get("effective_backend") for e in engines}),
        "served_by": sorted({e.get("keep", {}).get("served_by") for e in engines}),
        "threads": sorted({e.get("keep", {}).get("key", {}).get("threads") for e in engines}),
        "fit_target_mb": sorted({e.get("keep", {}).get("key", {}).get("fit_target_mb") for e in engines}),
        "keep_pids": sorted({e.get("keep", {}).get("pid") for e in engines}),
    }
    protocol_checks["pass"] = {
        "readout_sequence": protocol_checks["readout"] == ["sequence"],
        "cue_json_instructed": protocol_checks["cue"] == ["json_instructed"],
        "chat_format_role_split_contract_question": all(
            json.loads(c)["kind"] == "role_split" and json.loads(c)["contract"] == "question"
            for c in protocol_checks["chat_format"]),
        "calibrated_false": protocol_checks["calibrated"] == ["False"],
        "coverage_floor_0_10": protocol_checks["coverage_floor"] == [FLOOR],
        "all_layers_offloaded": protocol_checks["fit_n_gpu_layers"] == protocol_checks["placement_n_gpu_layers"]
                                 and protocol_checks["fit_n_gpu_layers"] == [32],
        "fit_ctx_ge_4096": all(c >= 4096 for c in protocol_checks["fit_n_ctx"]),
        "not_degraded": protocol_checks["placement_degraded"] == ["False"],
        "backend_vulkan": protocol_checks["effective_backend"] == ["vulkan"],
        "one_keep_host": len(protocol_checks["keep_pids"]) == 1,
    }
    print("\nprotocol:", json.dumps(protocol_checks, sort_keys=True))

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

# ---------------------------------------------------------------- speed sample (AC5)
speed = None
if (BASE / "speed_sample.json").exists():
    speed = json.loads((BASE / "speed_sample.json").read_text(encoding="utf-8"))
    print("\nspeed sample:", json.dumps(speed, indent=1))

# ---------------------------------------------------------------- engine block (verbatim)
engine = {}
if rows:
    sample = payload_of(rows[0]["id"])
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
        "served_by": sorted({payload_of(r["id"])["engine"].get("keep", {}).get("served_by") for r in rows}),
        "threads": sorted({payload_of(r["id"])["engine"].get("keep", {}).get("key", {}).get("threads") for r in rows}),
        "fit_target_mb": sorted({payload_of(r["id"])["engine"].get("keep", {}).get("key", {}).get("fit_target_mb")
                                 for r in rows}),
        "keep_pids": sorted({payload_of(r["id"])["engine"].get("keep", {}).get("pid") for r in rows}),
    }
    print("\nengine:", json.dumps(engine, indent=1)[:1500])

model = {
    "path": model_path_from_log or "/work/t_d199e09c/models/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf",
    "size_bytes": 5841049120,
    "quant": "Q4_K_M",
    "arch": engine.get("fit", {}).get("arch"),
    "sha256": engine.get("fit", {}).get("model_sha256"),
    "hf_repo": "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF",
}

report = {
    "schema": "typed_gguf.evidence.real_purpose/v1",
    "generated": "2026-09-22",
    "card": "t_d199e09c",
    "purpose": "support-inbox triage: queue routing (choice) + severity (score) + human-escalation gate (noul)",
    "arm": "mimo-v2.6-distill-qwen-9b-q4_k_m",
    "protocol_note": "same frozen items.jsonl/questions.json as the 4B arm (docs/evidence/real-purpose-4b-2026-09-22), "
                     "one `typed-gguf run --questions` per item, defaults except --threads and --fit-target "
                     "(placement/performance knobs; see engine.threads / engine.fit_target_mb)",
    "model": model,
    "engine": engine,
    "protocol_checks": protocol_checks,
    "run_header": run_header,
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
        "escalation_safety": escalation_safety,
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
        "speed_sample": speed,
    },
    "run": {"run_dir": str(BASE), "requests_exit0": n, "requests_total": len(items), "missing": missing,
            "run_log": "run.log next to this report (exit code + wall ms + exact command per item)"},
    "rows": rows,
}
(BASE / "report_mimo.json").write_text(json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")
print(f"\nwrote {BASE / 'report_mimo.json'}")

# ---------------------------------------------------------------- comparison vs the 4B
if not FOURB_REPORT.exists():
    print("no 4B report; skipping comparison")
    sys.exit(0)
b4 = json.loads(FOURB_REPORT.read_text(encoding="utf-8"))
b4_rows = {r["id"]: r for r in b4["rows"]}
tiel_rows = {}
if TIEL_REPORT.exists():
    tiel_rows = {r["id"]: r for r in json.loads(TIEL_REPORT.read_text(encoding="utf-8"))["rows"]}
common = [r for r in rows if r["id"] in b4_rows]
agree = {"queue": 0, "severity": 0, "escalate": 0}
dis = {"queue": {"n": 0, "mimo_only_right": 0, "b4_only_right": 0, "neither": 0},
       "severity": {"n": 0, "mimo_only_right": 0, "b4_only_right": 0, "neither": 0},
       "escalate": {"n": 0, "mimo_only_right": 0, "b4_only_right": 0, "neither": 0}}
per_item = []
for r in common:
    m = r
    b = b4_rows[r["id"]]
    t = tiel_rows.get(r["id"])
    same = {
        "queue": m["got"]["queue"] == b["got"]["queue"],
        "severity": m["got"]["severity_level"] == int(b["got"]["severity_level"]),
        "escalate": bool(m["got"]["escalate"]) == bool(b["got"]["escalate"]),
    }
    for k, v in same.items():
        if v:
            agree[k] += 1
        else:
            dis[k]["n"] += 1
            m_ok = m["correct"][k]
            b_ok = bool(b["correct"][k])
            if m_ok and not b_ok:
                dis[k]["mimo_only_right"] += 1
            elif b_ok and not m_ok:
                dis[k]["b4_only_right"] += 1
            elif not m_ok and not b_ok:
                dis[k]["neither"] += 1
    row = {
        "id": r["id"], "gold": m["gold"],
        "mimo": m["got"], "b4": {"queue": b["got"]["queue"], "severity_level": int(b["got"]["severity_level"]),
                                 "escalate": bool(b["got"]["escalate"])},
        "mimo_correct": m["correct"], "b4_correct": {k: bool(b["correct"][k]) for k in ("queue", "severity", "escalate")},
        "identical": same,
    }
    if t:
        row["tiel"] = t["got"]
        row["tiel_correct"] = t["correct"]
    per_item.append(row)

nc = len(common)
acc = {k: round(sum(1 for r in common if r["correct"][k]) / nc, 4) for k in ("queue", "severity", "escalate")}
b4_acc = {k: round(sum(1 for r in common if bool(b4_rows[r["id"]]["correct"][k])) / nc, 4) for k in ("queue", "severity", "escalate")}
tiel_acc = None
if tiel_rows:
    tiel_acc = {k: round(sum(1 for r in common if bool(tiel_rows[r["id"]]["correct"][k])) / nc, 4) for k in ("queue", "severity", "escalate")}


def fn_of(rowset, truth_getter):
    fn = [iid for iid, rr in rowset.items() if truth_getter(rr)["gold_esc"] and not truth_getter(rr)["got"]["escalate"]]
    return fn


def contingency(rows, tag):
    """Escalation contingency for one arm's rows: FN / over-escalation, with the money-legal split."""
    gt = [r for r in rows if r["gold"]["escalate"]]
    gf = [r for r in rows if not r["gold"]["escalate"]]
    fn = [r for r in gt if not r["got"]["escalate"]]
    over = [r for r in gf if r["got"]["escalate"]]
    money = [r for r in fn if (r["gold"].get("escalate_trigger") or "") in MONEY_LEGAL]
    gold_money = [r for r in rows if (r["gold"].get("escalate_trigger") or "") in MONEY_LEGAL]
    return {
        "arm": tag,
        "gold_escalate_true": len(gt),
        "gold_escalate_false": len(gf),
        "false_negatives_total": len(fn),
        "false_negative_ids": [r["id"] for r in fn],
        "money_legal_gold": len(gold_money),
        "money_legal_false_negatives": len(money),
        "money_legal_false_negative_ids": [r["id"] for r in money],
        "money_legal_recall": round(sum(1 for r in gold_money if r["got"]["escalate"]) / len(gold_money), 4) if gold_money else None,
        "over_escalations": len(over),
        "over_escalation_ids": [r["id"] for r in over],
    }


b4_safety = {
    "false_negatives_total": sum(1 for r in common if b4_rows[r["id"]]["gold"]["escalate"] and not b4_rows[r["id"]]["got"]["escalate"]),
    "money_legal_false_negatives": sum(1 for r in common
                                        if b4_rows[r["id"]]["gold"]["escalate"]
                                        and not b4_rows[r["id"]]["got"]["escalate"]
                                        and (b4_rows[r["id"]]["gold"].get("escalate_trigger") or "") in MONEY_LEGAL),
    "false_negative_ids": [r["id"] for r in common if b4_rows[r["id"]]["gold"]["escalate"] and not b4_rows[r["id"]]["got"]["escalate"]],
}
contingency_table = {"mimo": contingency(common, "mimo-9b-q4_k_m"),
                     "b4": contingency([b4_rows[r["id"]] for r in common], "spark-4b-q8_0")}
if tiel_rows:
    contingency_table["tiel"] = contingency([tiel_rows[r["id"]] for r in common if r["id"] in tiel_rows],
                                            "tiel-coder-35b")
comparison = {
    "schema": "typed_gguf.evidence.real_purpose_compare/v1",
    "card": "t_d199e09c",
    "baseline": {"report": "docs/evidence/real-purpose-4b-2026-09-22/report.json", "model": b4["model"]["path"]},
    "compared_items": nc,
    "agreement": {k: {"identical": agree[k], "n": nc, "rate": round(agree[k] / nc, 4)} for k in agree},
    "disagreements": dis,
    "accuracy_on_common": {"mimo": acc, "b4": b4_acc,
                           "tiel": tiel_acc,
                           "net_delta_mimo_minus_b4": {k: round(acc[k] - b4_acc[k], 4) for k in acc}},
    "escalation_safety": {"mimo": escalation_safety, "b4": b4_safety},
    "escalation_contingency": contingency_table,
    "per_item": per_item,
}
(BASE / "compare_vs_4b.json").write_text(json.dumps(comparison, indent=1, sort_keys=False) + "\n", encoding="utf-8")
print(f"wrote {BASE / 'compare_vs_4b.json'}  (n={nc})")
print("agreement:", json.dumps(comparison["agreement"]))
print("disagreements:", json.dumps(comparison["disagreements"]))
print("accuracy on common:", json.dumps(comparison["accuracy_on_common"]))
print("escalation safety:", json.dumps(comparison["escalation_safety"]))
if PARTIAL and nc < len(items):
    print(f"NOTE: partial run — {nc} of {len(items)} items compared")
