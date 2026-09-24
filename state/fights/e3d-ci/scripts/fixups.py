#!/usr/bin/env python3
"""Fix-ups: softmax mapping by first token + devset gold mismatch details."""
import json
import math

REPO = "/var/home/rybens/workspace/ggufone"
d = json.load(open(f"{REPO}/docs/evidence/e3d/full.json"))
items = d["items"]

worst = 0.0
worst_where = None
bad = 0
for it in items:
    for pol, r in it["ranked"].items():
        shape, variant = pol.split("=", 1)
        lab = it["shapes"][shape]["readout"]["labels"][variant]
        # map label -> first token
        first = dict(zip(lab["texts"], lab["first_tokens"]))
        # sequence score keyed by parsed token list; map by first token
        score_by_first = {}
        for key, val in r["sequence_scores"].items():
            toks = json.loads(key)
            score_by_first[str(toks[0])] = val
        scores = [score_by_first[str(first[t])] for t in lab["texts"]]
        m = max(scores)
        ex = [math.exp(v - m) for v in scores]
        s = sum(ex)
        sm = [e / s for e in ex]
        probs = [r["probabilities"][t] for t in lab["texts"]]
        for a, b in zip(probs, sm):
            diff = abs(a - b)
            if diff > worst:
                worst = diff
                worst_where = (it["id"], pol, list(zip(probs, sm)))
        if any(abs(a - b) > 1e-6 for a, b in zip(probs, sm)):
            bad += 1

print("probs vs softmax(full-seq scores), max abs diff:", worst)
print("worst:", worst_where)
print("cells beyond 1e-6:", bad, "of 180")

# tolerance distribution
import collections
tol = collections.Counter()
for it in items:
    for pol, r in it["ranked"].items():
        shape, variant = pol.split("=", 1)
        lab = it["shapes"][shape]["readout"]["labels"][variant]
        first = dict(zip(lab["texts"], lab["first_tokens"]))
        score_by_first = {str(json.loads(k)[0]): v for k, v in r["sequence_scores"].items()}
        scores = [score_by_first[str(first[t])] for t in lab["texts"]]
        m = max(scores)
        ex = [math.exp(v - m) for v in scores]
        s = sum(ex)
        sm = [e / s for e in ex]
        probs = [r["probabilities"][t] for t in lab["texts"]]
        mx = max(abs(a - b) for a, b in zip(probs, sm))
        if mx <= 1e-12:
            tol["<=1e-12"] += 1
        elif mx <= 1e-9:
            tol["<=1e-9"] += 1
        elif mx <= 1e-6:
            tol["<=1e-6"] += 1
        elif mx <= 1e-4:
            tol["<=1e-4"] += 1
        else:
            tol[">1e-4"] += 1
print("tol histogram:", dict(tol))

# devset gold mismatches
dev = [json.loads(line) for line in open(f"{REPO}/src/ggufone/bench/devset.jsonl")]
mism = [(a["id"], a["type"], a.get("gold"), b["type"], b["expected"])
        for a, b in zip(dev, items) if a["gold"] != b["expected"] or a["type"] != b["type"]]
print("devset mismatches (id, dev type, dev gold, rec type, rec expected):")
for m_ in mism:
    print("  ", m_)
