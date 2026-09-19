Agreement on the committed 60-item dev set, 95 % Wilson intervals; every column is cut to the items all three models measured. The mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | Tiel-Coder (this card, host, vulkan) | Δ (last − first) |
|---|---|---|---|---|
| overall | 0.633 (38/60) [0.507–0.744] | 0.517 (31/60) [0.393–0.638] | 0.517 (31/60) [0.393–0.638] | -0.117 |
| choice | 0.750 (18/24) [0.551–0.880] | 0.625 (15/24) [0.427–0.788] | 0.667 (16/24) [0.467–0.820] | -0.083 |
| noul | 0.889 (16/18) [0.672–0.969] | 0.389 (7/18) [0.203–0.614] | 0.389 (7/18) [0.203–0.614] | -0.500 |
| score | 0.222 (4/18) [0.090–0.452] | 0.500 (9/18) [0.290–0.710] | 0.444 (8/18) [0.246–0.663] | +0.222 |
| low_mass (below the floor) | 0.500 (6/12) [0.254–0.746] | 0.509 (29/57) [0.383–0.634] | 0.571 (8/14) [0.326–0.786] | +0.071 |
| measured (at or above the floor) | 0.667 (32/48) [0.525–0.783] | 0.667 (2/3) [0.208–0.939] | 0.500 (23/46) [0.361–0.639] | -0.167 |

| coverage of the answers | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | Tiel-Coder (this card, host, vulkan) | — |
|---|---|---|---|---|
| all rows | median 2.56e-01 · min 9.96e-03 · max 8.94e-01 · below the 0.10 floor 12/60 | median 2.34e-02 · min 1.86e-03 · max 2.00e-01 · below the 0.10 floor 57/60 | median 2.58e-01 · min 7.88e-03 · max 8.20e-01 · below the 0.10 floor 14/60 | — |

Paired comparison: 60 dev items measured by all three models (dropped 0, 0, 0 unpaired row(s) in report order).
Coverage split: 4B default (E2, 60 items, CPU): 48/60 rows `measured`; Occamy 1.0 (E3, chunks, vulkan): 3/60 rows `measured`; Tiel-Coder (this card, host, vulkan): 46/60 rows `measured`.
