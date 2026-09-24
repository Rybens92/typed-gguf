#!/usr/bin/env python3
"""Record-internal consistency v2 (fixed: coverage cap, softmax derivation)."""
import json
import math
from collections import Counter

REPO = "/var/home/rybens/workspace/ggufone"
d = json.load(open(f"{REPO}/docs/evidence/e3d/full.json"))
items = d["items"]
shapes = ["shipped", "two_step_shipped", "json_field"]
variants = d["label_variants"]
floor = d["mass_floor"]
fails = Counter()
counts = Counter()


def check(tag, ok, detail=""):
    counts[tag] += 1
    if not ok:
        fails[tag] += 1
        if fails[tag] <= 2:
            print(f"  FAIL[{tag}] {detail}")


def softmax(vals):
    m = max(vals)
    ex = [math.exp(v - m) for v in vals]
    s = sum(ex)
    return [e / s for e in ex]


for it in items:
    for pol, r in it["ranked"].items():
        shape, variant = pol.split("=", 1)
        rd = it["shapes"][shape]["readout"]
        lab = rd["labels"][variant]
        check("ranked_cov_vs_labels", abs(r["coverage"] - lab["coverage"]) < 1e-12,
              f"{it['id']}/{pol}")
        # probabilities == softmax(sequence_scores) in wire order of the score dict
        scores = list(r["sequence_scores"].values())
        probs = list(r["probabilities"].values())
        sm = softmax(scores)
        check("probs_eq_softmax_scores",
              all(abs(a - b) < 1e-9 for a, b in zip(probs, sm)),
              f"{it['id']}/{pol}: {[round(a-b,12) for a,b in zip(probs,sm)]}")
        # got == argmax with frozen lowest-index tie-break
        vals = list(r["probabilities"].values())
        best = 0
        for i in range(1, len(vals)):
            if vals[i] > vals[best]:
                best = i
        check("got_is_argmax", list(r["probabilities"].keys())[best] == r["got"], f"{it['id']}/{pol}")
        check("correct_eq", r["correct"] == (r["got"] == it["expected"]), f"{it['id']}/{pol}")
        fm = lab["first_mass"]
        amax = 0
        for i in range(1, len(fm)):
            if fm[i] > fm[amax]:
                amax = i
        check("top_cov_eq", lab["texts"][amax] == r["top_coverage"], f"{it['id']}/{pol}")
        check("agrees_eq", r["agrees"] == (r["top_coverage"] == r["got"]), f"{it['id']}/{pol}")
        want = "ok" if lab["coverage"] >= floor else "low_mass"
        check("rel_eq", r["reliability"] == want, f"{it['id']}/{pol}")

for it in items:
    for shape in shapes:
        rd = it["shapes"][shape]["readout"]
        for var, lab in rd["labels"].items():
            check("cov_eq_capped_sum",
                  abs(min(1.0, sum(lab["first_mass"])) - lab["coverage"]) < max(1e-9, 1e-9 * lab["coverage"]),
                  f"{it['id']}/{shape}/{var}: sum={sum(lab['first_mass'])} cov={lab['coverage']}")

# positions / advance / cue_row
for it in items:
    sh, tw, js = (it["shapes"][s] for s in shapes)
    check("advance_pos", tw["readout"]["position"] == tw["cue_row"]["position"] + 1, it["id"])
    check("advance_rule", tw["advance"]["rule"] == "content", it["id"])
    check("img_advance_after_cue",
          tw["readout"]["position"] == sh["readout"]["position"] + 1, it["id"])
    check("json_pos_shift",
          js["readout"]["position"] - sh["readout"]["position"] in (4, 5), it["id"])
    check("json_suffix_shift",
          js["suffix_tokens"] - sh["suffix_tokens"] == js["readout"]["position"] - sh["readout"]["position"],
          it["id"])
    check("two_suffix_same", tw["suffix_tokens"] == sh["suffix_tokens"], it["id"])

# devset correspondence (gold key)
dev = [json.loads(line) for line in open(f"{REPO}/src/ggufone/bench/devset.jsonl")]
check("devset_n", len(dev) == len(items) == 60, f"{len(dev)} vs {len(items)}")
check("devset_ids", [x["id"] for x in dev] == [it["id"] for it in items], "")
check("devset_types", all(a["type"] == b["type"] for a, b in zip(dev, items)), "")
check("devset_gold", all(a["gold"] == b["expected"] for a, b in zip(dev, items)), "")
tc = Counter(x["type"] for x in dev)
check("devset_counts", dict(tc) == {"choice": 24, "score": 18, "noul": 18}, str(dict(tc)))

# policy keys exactly
for it in items:
    check("policy_keys", sorted(it["ranked"]) == sorted(d["ranked_keys"]), it["id"])

print("\nchecks run per tag:", dict(counts))
print("FAILS:", dict(fails) or "NONE - all consistent")

# report newline capped rows count
cap_hits = sum(
    1 for it in items for shape in shapes
    for var, lab in it["shapes"][shape]["readout"]["labels"].items()
    if var == "newline" and sum(lab["first_mass"]) > 1.0
)
print("newline rows where the 1.0 cap actually binds:", cap_hits)
