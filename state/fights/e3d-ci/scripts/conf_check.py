"""Verify ranked.confidence == normalized_peak(probabilities) for all 180 cells."""
import json

d = json.load(open("/var/home/rybens/workspace/ggufone/.e3d/full.json"))
fails = 0
worst = 0.0
for it in d["items"]:
    for pol, r in it["ranked"].items():
        probs = list(r["probabilities"].values())
        k = len(probs)
        pmax = max(probs)
        want = max(0.0, min(1.0, (pmax - 1.0 / k) / (1.0 - 1.0 / k)))
        diff = abs(want - r["confidence"])
        worst = max(worst, diff)
        if diff > 1e-9:
            fails += 1
            print("FAIL", it["id"], pol, want, r["confidence"])
print("confidence check: fails =", fails, "| max diff =", worst)
