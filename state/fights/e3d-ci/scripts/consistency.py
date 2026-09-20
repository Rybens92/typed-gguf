#!/usr/bin/env python3
"""Record-internal consistency + derivation checks for .e3d/full.json.

Verifies that every number the analysis reads out of the record is itself consistent
with the raw per-item payloads, and that the shape-specific claims (prompt bytes,
readout rows, advance semantics) hold across all 60 items.
"""
import json
from collections import Counter

REPO = "/var/home/rybens/workspace/ggufone"
d = json.load(open(f"{REPO}/.e3d/full.json"))
items = d["items"]
shapes = ["shipped", "two_step_shipped", "json_field"]
variants = d["label_variants"]
floor = d["mass_floor"]
fails = Counter()
notes = []


def check(tag, ok, detail=""):
    if not ok:
        fails[tag] += 1
        if fails[tag] <= 3:
            print(f"  FAIL[{tag}] {detail}")


# ---------- 1) ranked block == shape readout derivations
for it in items:
    for pol, r in it["ranked"].items():
        shape, variant = pol.split("=", 1)
        rd = it["shapes"][shape]["readout"]
        lab = rd["labels"][variant]
        # coverage equality
        check("ranked_cov_vs_labels", abs(r["coverage"] - lab["coverage"]) < 1e-12,
              f"{it['id']}/{pol}: {r['coverage']} vs {lab['coverage']}")
        # got == argmax(probabilities)
        pmax = max(r["probabilities"].items(), key=lambda kv: kv[1])
        check("got_is_argmax", pmax[0] == r["got"],
              f"{it['id']}/{pol}: argmax {pmax[0]} != got {r['got']}")
        # correct == got == expected
        check("correct_eq", r["correct"] == (r["got"] == it["expected"]),
              f"{it['id']}/{pol}: correct={r['correct']} got={r['got']} exp={it['expected']}")
        # top_coverage == argmax of first_mass -> texts
        fm = lab["first_mass"]
        amax = fm.index(max(fm))
        check("top_cov_eq", lab["texts"][amax] == r["top_coverage"],
              f"{it['id']}/{pol}: top_coverage={r['top_coverage']} argmax={lab['texts'][amax]}")
        # agrees == (top_coverage == got)?
        check("agrees_eq", r["agrees"] == (r["top_coverage"] == r["got"]),
              f"{it['id']}/{pol}: agrees={r['agrees']}")
        # reliability == ok iff coverage >= floor  (labels.py: 'from coverage alone')
        want = "ok" if lab["coverage"] >= floor else "low_mass"
        check("rel_eq", r["reliability"] == want,
              f"{it['id']}/{pol}: reliability={r['reliability']} cov={lab['coverage']}")
        # first_mass: sum == coverage for every variant
for it in items:
    for shape in shapes:
        rd = it["shapes"][shape]["readout"]
        for var, lab in rd["labels"].items():
            check("cov_eq_sum_first_mass",
                  abs(sum(lab["first_mass"]) - lab["coverage"]) < max(1e-9, 1e-9 * lab["coverage"]),
                  f"{it['id']}/{shape}/{var}: sum={sum(lab['first_mass'])} cov={lab['coverage']}")
            # first_mass for a label whose first token appears in top_tokens should equal p_full
            tops = {t["piece"]: t["p_full"] for t in rd["top_tokens"]}
            for i, tok in enumerate(lab["first_tokens"]):
                pass
        # cue block vs readout row
for it in items:
    for shape in shapes:
        rd = it["shapes"][shape]["readout"]
        cue = rd.get("cue") or {}
        check("cue_present", bool(cue), f"{it['id']}/{shape}")
        # closer reported as piece text
        # top_tokens contains the cue token at rank 1? check cue.mass == top_tokens[0].p_full when refused False
        if rd["top_tokens"]:
            t0 = rd["top_tokens"][0]
            if cue.get("token") == t0.get("token"):
                check("cue_mass_eq_top1", abs(cue.get("mass", -1) - t0["p_full"]) < 1e-12,
                      f"{it['id']}/{shape}: cue.mass={cue.get('mass')} top1={t0['p_full']}")

print("fail tallies so far:", dict(fails) or "none")

# ---------- 2) shape claims: prompt bytes / readout positions / advance
pos_delta_two = Counter()
pos_delta_json = Counter()
suffix_delta = Counter()
prefix_same = Counter()
row_pairs = Counter()
closers = []
refused_counts = Counter()
for it in items:
    sh = it["shapes"]["shipped"]["readout"]
    tw = it["shapes"]["two_step_shipped"]["readout"]
    js = it["shapes"]["json_field"]["readout"]
    row_pairs[(sh["row"], tw["row"], js["row"])] += 1
    pos_delta_two[tw["position"] - sh["position"]] += 1
    pos_delta_json[js["position"] - sh["position"]] += 1
    st = it["shapes"]["shipped"]["suffix_tokens"]
    tt = it["shapes"]["two_step_shipped"]["suffix_tokens"]
    jt = it["shapes"]["json_field"]["suffix_tokens"]
    suffix_delta[("shipped->two_step", tt - st)] += 1
    suffix_delta[("shipped->json", jt - st)] += 1
    for shape in shapes:
        cue = it["shapes"][shape]["readout"]["cue"]
        if cue.get("refused"):
            refused_counts[shape] += 1
        if cue.get("closer"):
            closers.append((it["id"], shape, cue["closer"]))
print("\nrow triples (shipped,two_step,json):", dict(row_pairs))
print("two_step pos delta vs shipped:", dict(pos_delta_two))
print("json pos delta vs shipped:", dict(pos_delta_json))
print("suffix token deltas:", dict(suffix_delta))
print("refused counts per shape:", dict(refused_counts))
print("closers found:", closers[:10], "..." if len(closers) > 10 else "", f"(total {len(closers)})")

# prefix_tokens per shape present? compare per-item block keys
it0 = items[0]
print("\nshape block keys (c01):")
for shape in shapes:
    print(" ", shape, sorted(it0["shapes"][shape].keys()))
print("item-level keys:", sorted(it0.keys()))
print("opener:", it0["opener"])

# ---------- 3) json_field opener tokens vs position delta
# count how many tokens the opener is, from raw pieces if present
print()
for it in items[:5]:
    op = it["opener"]["json_field"]
    print(it["id"], "json opener:", repr(op), "pos delta:", 
          it["shapes"]["json_field"]["readout"]["position"] - it["shapes"]["shipped"]["readout"]["position"])

# ---------- 4) devset correspondence
dev = [json.loads(line) for line in open(f"{REPO}/src/ggufone/bench/devset.jsonl")]
print("\ndevset lines:", len(dev), "record items:", len(items))
ids_dev = [x.get("id") for x in dev]
ids_rec = [it["id"] for it in items]
print("ids equal:", ids_dev == ids_rec)
mism = [(a, b) for a, b in zip(dev, items) if a.get("type") != b["type"] or a.get("expected") != b["expected"]]
print("type/expected mismatches:", len(mism), mism[:5])
print("devset sample keys:", sorted(dev[0].keys()))

# ---------- 5) top-level claims
print("\ntop-level: threads", d.get("threads"), "gpu_layers", d.get("gpu_layers"),
      "backend", d.get("backend_claim"), "| model", d["model"]["name"], d["model"]["bytes"],
      "| wall_s", d.get("wall_s"))
print("placement:", d.get("placement"))
print("label variants:", variants)

# ---------- 6) cue token distribution across items (what closes the turn at the cue)
cue_tok = Counter()
for it in items:
    for shape in shapes:
        rd = it["shapes"][shape]["readout"]
        t0 = rd["top_tokens"][0]
        cue_tok[(shape, t0["piece"][:12])] += 1
for k in sorted(cue_tok):
    pass
print("\ntop-1 token at readout per shape (piece -> count):")
for shape in shapes:
    c = Counter()
    for it in items:
        c[it["shapes"][shape]["readout"]["top_tokens"][0]["piece"][:14]] += 1
    print(" ", shape, dict(c))

print("\nTOTAL FAILS:", dict(fails) or "none")
