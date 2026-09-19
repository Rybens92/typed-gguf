Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | delta |
|---|---|---|---|
| overall | 0.633 (38/60) [0.507–0.744] | 0.517 (31/60) [0.393–0.638] | -0.117 |
| choice | 0.750 (18/24) [0.551–0.880] | 0.625 (15/24) [0.427–0.788] | -0.125 |
| noul | 0.889 (16/18) [0.672–0.969] | 0.389 (7/18) [0.203–0.614] | -0.500 |
| score | 0.222 (4/18) [0.090–0.452] | 0.500 (9/18) [0.290–0.710] | +0.278 |
| low_mass (below the floor) | 0.500 (6/12) [0.254–0.746] | 0.509 (29/57) [0.383–0.634] | +0.009 |
| measured (at or above the floor) | 0.667 (32/48) [0.525–0.783] | 0.667 (2/3) [0.208–0.939] | +0.000 |

`Occamy 1.0 (E3, chunks, vulkan)` is worse than `4B default (E2, 60 items, CPU)` by -0.117 overall (0.633 -> 0.517); the `measured` row is the one to read first.

Paired comparison: 60 dev items measured by both models (dropped 0 unpaired baseline row(s) and 0 unpaired challenger row(s) so the two sides ask the same questions).

