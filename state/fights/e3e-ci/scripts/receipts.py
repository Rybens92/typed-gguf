#!/usr/bin/env python3
"""Receipts for the E3e audit (card t_57bd3db2): baselines, drift notes, byte-identity support.

1. the committed baseline `.e3d/bench_templated_shipped.json` — its own numbers, so the freeze
   claim has something to be checked against;
2. the E3d rows the recommendation quotes: `json_field` 51/60 and the `two_step` drift note
   (47/60 published vs 46/60 in the E3e instrument);
3. per-item `prefix_tokens` identity between the arms whose bytes the table claims are the same
   (`shipped`/`two_step` under the same chat_format), item by item — a count-level check of
   "the cue shape adds a decoded token, not a prompt byte".
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]


def load(path):
    return json.loads((ROOT / path).read_text())


def summary(path):
    report = load(path)
    items = report["items"]
    cov = sorted(float(it["coverage"]) for it in items)
    return {
        "correct": sum(1 for it in items if it["correct"]),
        "n": len(items),
        "low_mass": sum(1 for it in items if it["reliability"] == "low_mass"),
        "refused": sum(1 for it in items if it["cue"]["refused"]),
        "cov_p50": cov[len(cov) // 2],
        "prefix_tokens": sorted({it["prefix_tokens"] for it in items}),
        "backend": (report.get("config") or {}).get("backend"),
        "cue": (report.get("config") or {}).get("cue"),
        "chat_format": (report.get("config") or {}).get("chat_format"),
        "framing": (report.get("framing") or {}).get("labels"),
        "generated_at": report.get("generated_at"),
    }


print("=== baselines and the E3d rows the recommendation quotes ===")
for path in (".e3d/bench_templated_shipped.json", ".e3d/bench_templated_json_field.json",
             ".e3d/bench_templated_two_step.json", ".e3e/probe_default.json",
             ".e3e/bench_shipped_answer_sheet.json"):
    s = summary(path)
    print(f"{path:<42} {s['correct']}/{s['n']} low_mass={s['low_mass']} refused={s['refused']} "
          f"cov_p50={s['cov_p50']:.4e} backend={s['backend']} cue={s['cue']} "
          f"fmt={s['chat_format']} @{s['generated_at']}")
    print(f"    framing={s['framing']} prefix_token_counts={s['prefix_tokens']}")

print("\n=== per-item prefix_tokens: same bytes under one chat_format? ===")
pairs = [(".e3e/bench_shipped_answer_sheet.json", ".e3e/bench_two_step_answer_sheet.json",
          "answer_sheet: shipped vs two_step"),
         (".e3e/bench_shipped_role_split.json", ".e3e/bench_two_step_role_split.json",
          "role_split: shipped vs two_step"),
         (".e3e/bench_json_instructed_answer_sheet.json",
          ".e3e/bench_json_instructed_answer_sheet_system.json",
          "answer_sheet: json_instructed question vs system (expected to DIFFER)")]
for left_path, right_path, name in pairs:
    left = {it["id"]: it["prefix_tokens"] for it in load(left_path)["items"]}
    right = {it["id"]: it["prefix_tokens"] for it in load(right_path)["items"]}
    shared = sorted(set(left) & set(right))
    differ = [key for key in shared if left[key] != right[key]]
    print(f"{name}: {len(shared)} items, prefix_tokens differing on {len(differ)}"
          + (f" -> {differ[:6]}" if differ else " (identical item by item)"))

print("\n=== the dev set ids the probes measured ===")
probe = load(".e3e/probe_default.json")
base = load(".e3d/bench_templated_shipped.json")
print("probe ids:", [it["id"] for it in probe["items"]])
print("baseline ids (first 6):", [it["id"] for it in base["items"]][:6],
      "… total", len(base["items"]))
