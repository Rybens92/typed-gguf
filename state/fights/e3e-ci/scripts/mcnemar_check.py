#!/usr/bin/env python3
"""Cross-check of the two partitioned McNemar values quoted in the scorecard."""
import math


def mcnemar(b, c):
    n = b + c
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


for b, c, name in ((12, 3, "E3d json_field (12 vs 3)"), (13, 4, "E3e winner (13 vs 4)"),
                   (12, 5, "E3e winner, one item moved (12 vs 5)"),
                   (11, 3, "E3e instructed/role_split (11 vs 3)"),
                   (12, 4, "E3e shipped/role_split (12 vs 4)")):
    print(f"{name:<45} p = {mcnemar(b, c):.6f}")
