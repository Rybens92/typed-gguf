Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | delta |
|---|---|---|---|
| overall | 0.500 (10/20) [0.299–0.701] | 0.450 (9/20) [0.258–0.658] | -0.050 |
| choice | 0.429 (3/7) [0.158–0.750] | 0.571 (4/7) [0.250–0.842] | +0.143 |
| noul | 1.000 (6/6) [0.610–1.000] | 0.167 (1/6) [0.030–0.564] | -0.833 |
| score | 0.143 (1/7) [0.026–0.513] | 0.571 (4/7) [0.250–0.842] | +0.429 |
| low_mass (below the floor) | 0.333 (1/3) [0.061–0.792] | 0.450 (9/20) [0.258–0.658] | +0.117 |
| measured (at or above the floor) | 0.529 (9/17) [0.310–0.738] | — | — |

`Occamy 1.0 (E3, chunks, vulkan)` is worse than `4B default (E2, 60 items, CPU)` by -0.050 overall (0.500 -> 0.450); the `measured` row is the one to read first.

Paired comparison: 20 dev items measured by both models (dropped 40 unpaired baseline row(s) and 0 unpaired challenger row(s) so the two sides ask the same questions).

