"""Paired McNemar on the low_mass axis (shipped vs two_step / json_field), from the record."""
import json
import math

d = json.load(open("/var/home/rybens/workspace/ggufone/docs/evidence/e3d/full.json"))
items = d["items"]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


for pol in ("two_step_shipped=bare", "json_field=bare"):
    a_win = b_win = 0
    for it in items:
        base_low = it["ranked"]["shipped=bare"]["reliability"] == "low_mass"
        new_low = it["ranked"][pol]["reliability"] == "low_mass"
        if base_low and not new_low:
            a_win += 1  # only baseline low_mass
        if new_low and not base_low:
            b_win += 1  # only challenger low_mass
    print(f"{pol}: low_mass -> {'ok' if pol.startswith('two') else 'ok'}: "
          f"a_only(low)={a_win} b_only(low)={b_win} mcnemar p={mcnemar_exact(a_win, b_win):.6g}")

# also the same pairing for 'correct' on the reliability-restricted view:
# how many of the low_mass rows are correct vs not, per policy
for pol in ("shipped=bare", "two_step_shipped=bare", "json_field=bare"):
    rows = [it["ranked"][pol] for it in items]
    low = [r for r in rows if r["reliability"] == "low_mass"]
    ok = [r for r in rows if r["reliability"] != "low_mass"]
    print(f"{pol}: low_mass rows {len(low)} (correct {sum(r['correct'] for r in low)}), "
          f"measured rows {len(ok)} (correct {sum(r['correct'] for r in ok)})")
