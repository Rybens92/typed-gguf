#!/usr/bin/env python3
"""Item-level diff of the freeze comparison, including the fields the tool's freeze check does NOT
compare (audit t_57bd3db2): `cue.refused`, `cue.token`, `cue.mass`.

The record's freeze check compares `got` + `reliability` + `prefix_tokens`. The cue verdict
(`refused`) is a published classification (E3c's protocol) and is not part of that identity, so
this prints every item where it differs, with the numbers behind it.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]


def load(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return {str(it["id"]): it for it in load(path)["items"]}


def diff(name, probe_path, base_path):
    probe, base = rows(probe_path), rows(base_path)
    shared = sorted(set(probe) & set(base))
    print(f"\n=== {name}: {probe_path} vs {base_path} ({len(shared)} shared items)")
    counts = {"got": 0, "reliability": 0, "prefix_tokens": 0, "refused": 0, "cue.token": 0,
              "coverage>tol": 0}
    for key in shared:
        left, right = probe[key], base[key]
        lb, rb = left.get("cue") or {}, right.get("cue") or {}
        notes = []
        if left.get("got") != right.get("got"):
            counts["got"] += 1
            notes.append(f"got {left.get('got')!r} != {right.get('got')!r}")
        if left.get("reliability") != right.get("reliability"):
            counts["reliability"] += 1
            notes.append(f"reliability {left.get('reliability')!r} != {right.get('reliability')!r}")
        if left.get("prefix_tokens") != right.get("prefix_tokens"):
            counts["prefix_tokens"] += 1
            notes.append("prefix_tokens differ")
        if bool(lb.get("refused")) != bool(rb.get("refused")):
            counts["refused"] += 1
            notes.append(f"refused {bool(lb.get('refused'))} != {bool(rb.get('refused'))} "
                         f"(token {lb.get('token')}/{rb.get('token')}, "
                         f"mass {float(lb.get('mass') or 0):.6f}/{float(rb.get('mass') or 0):.6f}, "
                         f"closer {lb.get('closer')!r}/{rb.get('closer')!r})")
        if lb.get("token") != rb.get("token"):
            counts["cue.token"] += 1
        cov_left = float(left.get("coverage") or 0.0)
        cov_right = float(right.get("coverage") or 0.0)
        if abs(cov_left - cov_right) > 5e-3:
            counts["coverage>tol"] += 1
        if notes:
            print(f"  {key}: " + "; ".join(notes))
            print(f"      probe: got={left.get('got')!r} rel={left.get('reliability')!r} "
                  f"cov={cov_left:.6e} cue={lb}")
            print(f"      base : got={right.get('got')!r} rel={right.get('reliability')!r} "
                  f"cov={cov_right:.6e} cue={rb}")
    print(f"  counts: {counts}")
    return counts


diff("the six-item freeze probe", "docs/evidence/e3e/probe_default.json", "docs/evidence/e3d/bench_templated_shipped.json")
diff("the 60-item placement probe", "docs/evidence/e3e/bench_shipped_answer_sheet.json",
     "docs/evidence/e3d/bench_templated_shipped.json")
