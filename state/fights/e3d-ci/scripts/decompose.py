"""Correctness decomposition over the shipped-low_mass subset vs the measured subset."""
import json

d = json.load(open("/var/home/rybens/workspace/ggufone/docs/evidence/e3d/full.json"))
items = d["items"]
L, M, T = [], [], []
for it in items:
    sh = it["ranked"]["shipped=bare"]
    tw = it["ranked"]["two_step_shipped=bare"]
    sub = L if sh["reliability"] == "low_mass" else M
    sub.append((it["id"], it["type"], sh["correct"], tw["correct"], sh["coverage"], tw["coverage"]))

for name, sub in (("shipped-low_mass subset (44)", L), ("shipped-measured subset (16)", M)):
    sc = sum(1 for _, _, a, b, _, _ in sub if a)
    tc = sum(1 for _, _, a, b, _, _ in sub if b)
    print(f"{name}: n={len(sub)}  shipped correct={sc}  two_step correct={tc}")

# type breakdown of the low subset
from collections import Counter
c = Counter((t, a, b) for _, t, a, b, _, _ in L)
print("\nlow subset by (type, shipped_correct, two_correct):", dict(c))
