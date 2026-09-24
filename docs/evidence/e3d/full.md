# E3c — cue shapes that put the readout mid-answer (`spark2_5`)

- card `t_6c119626` · model `Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes, sha256 `5c2c3c190e4337e1…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   271.16 MiB
sched_reserve: graph nodes  = 1266
sched_reserve: graph splits = 1
sched_reserve: reserve took 3.58 ms, sched copies = 1`
- dev items: c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14, s15, s16, s17, s18, n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11, n12, n13, n14, n15, n16, n17, n18 (choice 24, noul 18, score 18) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T17:04:48Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `two_step_shipped`, `json_field`
- wall: 2792.0 s (one model load 967 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| c01 | `shipped` | cue | `
` | 0.9536 | 0.0000 | ok |
| c01 | `two_step_shipped` | advanced | `t` | 0.8593 | 0.0000 | ok |
| c01 | `json_field` | cue | `t` | 0.5068 | 0.0000 | ok |
| c02 | `shipped` | cue | `
` | 0.9362 | 0.0000 | ok |
| c02 | `two_step_shipped` | advanced | `b` | 0.9975 | 0.0000 | ok |
| c02 | `json_field` | cue | `b` | 0.9711 | 0.0000 | ok |
| c03 | `shipped` | cue | `
` | 0.9933 | 0.0000 | ok |
| c03 | `two_step_shipped` | advanced | `mobile` | 0.9942 | 0.0000 | ok |
| c03 | `json_field` | cue | `mobile` | 0.9989 | 0.0000 | ok |
| c04 | `shipped` | cue | `
` | 0.9805 | 0.0000 | ok |
| c04 | `two_step_shipped` | advanced | `conf` | 0.8825 | 0.0000 | ok |
| c04 | `json_field` | cue | `conf` | 0.9752 | 0.0000 | ok |
| c05 | `shipped` | cue | `
` | 0.8772 | 0.0000 | ok |
| c05 | `two_step_shipped` | advanced | `security` | 0.9968 | 0.0000 | ok |
| c05 | `json_field` | cue | `security` | 0.9937 | 0.0000 | ok |
| c06 | `shipped` | cue | `
` | 0.6146 | 0.0000 | ok |
| c06 | `two_step_shipped` | advanced | `fixed` | 0.8985 | 0.0000 | ok |
| c06 | `json_field` | cue | `fixed` | 0.9989 | 0.0000 | ok |
| c07 | `shipped` | cue | `
` | 0.9597 | 0.0000 | ok |
| c07 | `two_step_shipped` | advanced | `termin` | 0.9709 | 0.0000 | ok |
| c07 | `json_field` | cue | `termin` | 0.9995 | 0.0000 | ok |
| c08 | `shipped` | cue | `
` | 0.9968 | 0.0000 | ok |
| c08 | `two_step_shipped` | advanced | `business` | 0.9518 | 0.0000 | ok |
| c08 | `json_field` | cue | `business` | 0.8634 | 0.0000 | ok |
| c09 | `shipped` | cue | `
` | 0.6298 | 0.0000 | ok |
| c09 | `two_step_shipped` | advanced | `sp` | 0.9969 | 0.0000 | ok |
| c09 | `json_field` | cue | `sp` | 1.0000 | 0.0000 | ok |
| c10 | `shipped` | cue | `
` | 0.9957 | 0.0000 | ok |
| c10 | `two_step_shipped` | advanced | `comp` | 0.9841 | 0.0000 | ok |
| c10 | `json_field` | cue | `comp` | 0.9997 | 0.0000 | ok |
| c11 | `shipped` | cue | `
` | 0.8955 | 0.0000 | ok |
| c11 | `two_step_shipped` | advanced | `ex` | 0.9940 | 0.0000 | ok |
| c11 | `json_field` | cue | `ex` | 0.9999 | 0.0000 | ok |
| c12 | `shipped` | cue | `
` | 0.9817 | 0.0000 | ok |
| c12 | `two_step_shipped` | advanced | `backend` | 0.4844 | 0.0000 | ok |
| c12 | `json_field` | cue | `front` | 0.9993 | 0.0000 | ok |
| c13 | `shipped` | cue | `
` | 0.9567 | 0.0000 | ok |
| c13 | `two_step_shipped` | advanced | `part` | 0.9103 | 0.0000 | ok |
| c13 | `json_field` | cue | `part` | 0.9992 | 0.0000 | ok |
| c14 | `shipped` | cue | `
` | 0.9869 | 0.0000 | ok |
| c14 | `two_step_shipped` | advanced | `cap` | 0.9615 | 0.0000 | ok |
| c14 | `json_field` | cue | `cap` | 0.9997 | 0.0000 | ok |
| c15 | `shipped` | cue | `
` | 0.8965 | 0.0000 | ok |
| c15 | `two_step_shipped` | advanced | `card` | 0.9463 | 0.0000 | ok |
| c15 | `json_field` | cue | `card` | 0.9988 | 0.0000 | ok |
| c16 | `shipped` | cue | `
` | 0.9788 | 0.0000 | ok |
| c16 | `two_step_shipped` | advanced | `tr` | 0.9847 | 0.0000 | ok |
| c16 | `json_field` | cue | `tr` | 1.0000 | 0.0000 | ok |
| c17 | `shipped` | cue | `
` | 0.9979 | 0.0000 | ok |
| c17 | `two_step_shipped` | advanced | `mid` | 0.9198 | 0.0000 | ok |
| c17 | `json_field` | cue | `mid` | 0.9972 | 0.0000 | ok |
| c18 | `shipped` | cue | `
` | 0.9379 | 0.0000 | ok |
| c18 | `two_step_shipped` | advanced | `stand` | 0.7282 | 0.0000 | ok |
| c18 | `json_field` | cue | `stand` | 0.7294 | 0.0000 | ok |
| c19 | `shipped` | cue | `configuration` | 0.4962 | 0.0000 | ok |
| c19 | `two_step_shipped` | advanced | `<｜end▁of▁sentence｜>` | 0.9992 | 0.0000 | ok |
| c19 | `json_field` | cue | `configuration` | 0.9979 | 0.0000 | ok |
| c20 | `shipped` | cue | `
` | 0.9005 | 0.0000 | ok |
| c20 | `two_step_shipped` | advanced | `cop` | 0.9912 | 0.0000 | ok |
| c20 | `json_field` | cue | `cop` | 0.9998 | 0.0000 | ok |
| c21 | `shipped` | cue | `
` | 0.9909 | 0.0000 | ok |
| c21 | `two_step_shipped` | advanced | `w` | 0.3222 | 0.0000 | ok |
| c21 | `json_field` | cue | `archive` | 0.7588 | 0.0000 | ok |
| c22 | `shipped` | cue | `
` | 0.9284 | 0.0000 | ok |
| c22 | `two_step_shipped` | advanced | `s` | 0.8865 | 0.0000 | ok |
| c22 | `json_field` | cue | `s` | 0.9999 | 0.0000 | ok |
| c23 | `shipped` | cue | `
` | 0.8663 | 0.0000 | ok |
| c23 | `two_step_shipped` | advanced | `identity` | 0.9988 | 0.0000 | ok |
| c23 | `json_field` | cue | `identity` | 1.0000 | 0.0000 | ok |
| c24 | `shipped` | cue | `
` | 0.8012 | 0.0000 | ok |
| c24 | `two_step_shipped` | advanced | `run` | 0.9871 | 0.0000 | ok |
| c24 | `json_field` | cue | `run` | 1.0000 | 0.0000 | ok |
| s01 | `shipped` | cue | `
` | 0.7622 | 0.0000 | ok |
| s01 | `two_step_shipped` | advanced | `3` | 0.8102 | 0.0000 | ok |
| s01 | `json_field` | cue | `3` | 0.8065 | 0.0000 | ok |
| s02 | `shipped` | cue | `
` | 0.9574 | 0.0000 | ok |
| s02 | `two_step_shipped` | advanced | `1` | 0.4768 | 0.0000 | ok |
| s02 | `json_field` | cue | `1` | 0.5632 | 0.0000 | ok |
| s03 | `shipped` | cue | `
` | 0.7280 | 0.0000 | ok |
| s03 | `two_step_shipped` | advanced | `4` | 0.5416 | 0.0000 | ok |
| s03 | `json_field` | cue | `4` | 0.5394 | 0.0000 | ok |
| s04 | `shipped` | cue | `3` | 0.3553 | 0.0000 | ok |
| s04 | `two_step_shipped` | advanced | `<｜end▁of▁sentence｜>` | 0.9998 | 0.0000 | ok |
| s04 | `json_field` | cue | `3` | 0.8313 | 0.0000 | ok |
| s05 | `shipped` | cue | `
` | 0.9433 | 0.0000 | ok |
| s05 | `two_step_shipped` | advanced | `1` | 0.4629 | 0.0000 | ok |
| s05 | `json_field` | cue | `1` | 0.5454 | 0.0000 | ok |
| s06 | `shipped` | cue | `
` | 0.8585 | 0.0000 | ok |
| s06 | `two_step_shipped` | advanced | `3` | 0.8609 | 0.0000 | ok |
| s06 | `json_field` | cue | `3` | 0.8192 | 0.0000 | ok |
| s07 | `shipped` | cue | `
` | 0.8632 | 0.0000 | ok |
| s07 | `two_step_shipped` | advanced | `1` | 0.7236 | 0.0000 | ok |
| s07 | `json_field` | cue | `1` | 0.8049 | 0.0000 | ok |
| s08 | `shipped` | cue | `
` | 0.8167 | 0.0000 | ok |
| s08 | `two_step_shipped` | advanced | `1` | 0.9111 | 0.0000 | ok |
| s08 | `json_field` | cue | `1` | 0.8334 | 0.0000 | ok |
| s09 | `shipped` | cue | `
` | 0.7656 | 0.0000 | ok |
| s09 | `two_step_shipped` | advanced | `1` | 0.8702 | 0.0000 | ok |
| s09 | `json_field` | cue | `1` | 0.9773 | 0.0000 | ok |
| s10 | `shipped` | cue | `
` | 0.8246 | 0.0000 | ok |
| s10 | `two_step_shipped` | advanced | `1` | 0.5135 | 0.0000 | ok |
| s10 | `json_field` | cue | `3` | 0.6882 | 0.0000 | ok |
| s11 | `shipped` | cue | `
` | 0.9270 | 0.0000 | ok |
| s11 | `two_step_shipped` | advanced | `1` | 0.5427 | 0.0000 | ok |
| s11 | `json_field` | cue | `3` | 0.6388 | 0.0000 | ok |
| s12 | `shipped` | cue | `
` | 0.9590 | 0.0000 | ok |
| s12 | `two_step_shipped` | advanced | `2` | 0.5088 | 0.0000 | ok |
| s12 | `json_field` | cue | `3` | 0.3878 | 0.0000 | ok |
| s13 | `shipped` | cue | `
` | 0.7316 | 0.0000 | ok |
| s13 | `two_step_shipped` | advanced | `2` | 0.8217 | 0.0000 | ok |
| s13 | `json_field` | cue | `0` | 0.5328 | 0.0000 | ok |
| s14 | `shipped` | cue | `
` | 0.9097 | 0.0000 | ok |
| s14 | `two_step_shipped` | advanced | `1` | 0.8099 | 0.0000 | ok |
| s14 | `json_field` | cue | `1` | 0.7651 | 0.0000 | ok |
| s15 | `shipped` | cue | `
` | 0.9255 | 0.0000 | ok |
| s15 | `two_step_shipped` | advanced | `1` | 0.5850 | 0.0000 | ok |
| s15 | `json_field` | cue | `2` | 0.8951 | 0.0000 | ok |
| s16 | `shipped` | cue | `
` | 0.7901 | 0.0000 | ok |
| s16 | `two_step_shipped` | advanced | `1` | 0.6519 | 0.0000 | ok |
| s16 | `json_field` | cue | `1` | 0.3127 | 0.0000 | ok |
| s17 | `shipped` | cue | `
` | 0.9157 | 0.0000 | ok |
| s17 | `two_step_shipped` | advanced | `2` | 0.4356 | 0.0000 | ok |
| s17 | `json_field` | cue | `2` | 0.6995 | 0.0000 | ok |
| s18 | `shipped` | cue | `
` | 0.9306 | 0.0000 | ok |
| s18 | `two_step_shipped` | advanced | `0` | 0.4519 | 0.0000 | ok |
| s18 | `json_field` | cue | `0` | 0.8226 | 0.0000 | ok |
| n01 | `shipped` | cue | `
` | 0.9431 | 0.0000 | ok |
| n01 | `two_step_shipped` | advanced | `yes` | 0.7689 | 0.0000 | ok |
| n01 | `json_field` | cue | `yes` | 0.9990 | 0.0000 | ok |
| n02 | `shipped` | cue | `
` | 0.8679 | 0.0000 | ok |
| n02 | `two_step_shipped` | advanced | `yes` | 0.9197 | 0.0000 | ok |
| n02 | `json_field` | cue | `yes` | 0.9154 | 0.0000 | ok |
| n03 | `shipped` | cue | `
` | 0.7550 | 0.0000 | ok |
| n03 | `two_step_shipped` | advanced | `yes` | 0.9299 | 0.0000 | ok |
| n03 | `json_field` | cue | `yes` | 0.9769 | 0.0000 | ok |
| n04 | `shipped` | cue | `
` | 0.9859 | 0.0000 | ok |
| n04 | `two_step_shipped` | advanced | `yes` | 0.8893 | 0.0000 | ok |
| n04 | `json_field` | cue | `yes` | 0.9991 | 0.0000 | ok |
| n05 | `shipped` | cue | `
` | 0.9869 | 0.0000 | ok |
| n05 | `two_step_shipped` | advanced | `yes` | 0.7485 | 0.0000 | ok |
| n05 | `json_field` | cue | `yes` | 0.9957 | 0.0000 | ok |
| n06 | `shipped` | cue | `
` | 0.8867 | 0.0000 | ok |
| n06 | `two_step_shipped` | advanced | `yes` | 0.6215 | 0.0000 | ok |
| n06 | `json_field` | cue | `no` | 0.7606 | 0.0000 | ok |
| n07 | `shipped` | cue | `
` | 0.8347 | 0.0000 | ok |
| n07 | `two_step_shipped` | advanced | `yes` | 0.8372 | 0.0000 | ok |
| n07 | `json_field` | cue | `yes` | 0.9797 | 0.0000 | ok |
| n08 | `shipped` | cue | `
` | 0.9620 | 0.0000 | ok |
| n08 | `two_step_shipped` | advanced | `yes` | 0.7255 | 0.0000 | ok |
| n08 | `json_field` | cue | `yes` | 0.5724 | 0.0000 | ok |
| n09 | `shipped` | cue | `
` | 0.9156 | 0.0000 | ok |
| n09 | `two_step_shipped` | advanced | `no` | 0.5909 | 0.0000 | ok |
| n09 | `json_field` | cue | `no` | 0.9937 | 0.0000 | ok |
| n10 | `shipped` | cue | `
` | 0.9807 | 0.0000 | ok |
| n10 | `two_step_shipped` | advanced | `yes` | 0.9478 | 0.0000 | ok |
| n10 | `json_field` | cue | `yes` | 0.9635 | 0.0000 | ok |
| n11 | `shipped` | cue | `
` | 0.8053 | 0.0000 | ok |
| n11 | `two_step_shipped` | advanced | `yes` | 0.4777 | 0.0000 | ok |
| n11 | `json_field` | cue | `yes` | 0.7235 | 0.0000 | ok |
| n12 | `shipped` | cue | `
` | 0.9490 | 0.0000 | ok |
| n12 | `two_step_shipped` | advanced | `no` | 0.8468 | 0.0000 | ok |
| n12 | `json_field` | cue | `no` | 0.9977 | 0.0000 | ok |
| n13 | `shipped` | cue | `
` | 0.7132 | 0.0000 | ok |
| n13 | `two_step_shipped` | advanced | `yes` | 0.7295 | 0.0000 | ok |
| n13 | `json_field` | cue | `yes` | 0.9485 | 0.0000 | ok |
| n14 | `shipped` | cue | `
` | 0.9506 | 0.0000 | ok |
| n14 | `two_step_shipped` | advanced | `no` | 0.7998 | 0.0000 | ok |
| n14 | `json_field` | cue | `no` | 0.9959 | 0.0000 | ok |
| n15 | `shipped` | cue | `
` | 0.6664 | 0.0000 | ok |
| n15 | `two_step_shipped` | advanced | `no` | 0.5907 | 0.0000 | ok |
| n15 | `json_field` | cue | `yes` | 0.8853 | 0.0000 | ok |
| n16 | `shipped` | cue | `
` | 0.4915 | 0.0000 | ok |
| n16 | `two_step_shipped` | advanced | `yes` | 0.5723 | 0.0000 | ok |
| n16 | `json_field` | cue | `no` | 0.9287 | 0.0000 | ok |
| n17 | `shipped` | cue | `
` | 0.9420 | 0.0000 | ok |
| n17 | `two_step_shipped` | advanced | `no` | 0.6241 | 0.0000 | ok |
| n17 | `json_field` | cue | `no` | 0.9950 | 0.0000 | ok |
| n18 | `shipped` | cue | `
` | 0.6901 | 0.0000 | ok |
| n18 | `two_step_shipped` | advanced | `yes` | 0.9268 | 0.0000 | ok |
| n18 | `json_field` | cue | `yes` | 0.9932 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape |  | `bare` coverage |
|---|---|---|
| c01 | `shipped` |  | 1.144e-02 |
| c01 | `two_step_shipped` |  | 9.559e-01 |
| c01 | `json_field` |  | 9.994e-01 |
| c02 | `shipped` |  | 3.048e-02 |
| c02 | `two_step_shipped` |  | 9.979e-01 |
| c02 | `json_field` |  | 9.997e-01 |
| c03 | `shipped` |  | 1.687e-03 |
| c03 | `two_step_shipped` |  | 9.942e-01 |
| c03 | `json_field` |  | 9.999e-01 |
| c04 | `shipped` |  | 1.571e-02 |
| c04 | `two_step_shipped` |  | 9.584e-01 |
| c04 | `json_field` |  | 9.997e-01 |
| c05 | `shipped` |  | 8.205e-02 |
| c05 | `two_step_shipped` |  | 9.968e-01 |
| c05 | `json_field` |  | 9.997e-01 |
| c06 | `shipped` |  | 1.755e-01 |
| c06 | `two_step_shipped` |  | 9.933e-01 |
| c06 | `json_field` |  | 9.999e-01 |
| c07 | `shipped` |  | 1.462e-02 |
| c07 | `two_step_shipped` |  | 9.736e-01 |
| c07 | `json_field` |  | 9.996e-01 |
| c08 | `shipped` |  | 5.833e-04 |
| c08 | `two_step_shipped` |  | 9.941e-01 |
| c08 | `json_field` |  | 9.965e-01 |
| c09 | `shipped` |  | 3.412e-01 |
| c09 | `two_step_shipped` |  | 9.969e-01 |
| c09 | `json_field` |  | 1.000e+00 |
| c10 | `shipped` |  | 1.147e-07 |
| c10 | `two_step_shipped` |  | 9.841e-01 |
| c10 | `json_field` |  | 9.997e-01 |
| c11 | `shipped` |  | 7.294e-02 |
| c11 | `two_step_shipped` |  | 9.943e-01 |
| c11 | `json_field` |  | 9.999e-01 |
| c12 | `shipped` |  | 9.297e-03 |
| c12 | `two_step_shipped` |  | 9.487e-01 |
| c12 | `json_field` |  | 9.999e-01 |
| c13 | `shipped` |  | 8.968e-05 |
| c13 | `two_step_shipped` |  | 9.106e-01 |
| c13 | `json_field` |  | 9.992e-01 |
| c14 | `shipped` |  | 5.879e-03 |
| c14 | `two_step_shipped` |  | 9.615e-01 |
| c14 | `json_field` |  | 9.997e-01 |
| c15 | `shipped` |  | 1.851e-02 |
| c15 | `two_step_shipped` |  | 9.476e-01 |
| c15 | `json_field` |  | 9.989e-01 |
| c16 | `shipped` |  | 2.148e-03 |
| c16 | `two_step_shipped` |  | 9.848e-01 |
| c16 | `json_field` |  | 1.000e+00 |
| c17 | `shipped` |  | 1.435e-04 |
| c17 | `two_step_shipped` |  | 9.398e-01 |
| c17 | `json_field` |  | 9.995e-01 |
| c18 | `shipped` |  | 4.660e-02 |
| c18 | `two_step_shipped` |  | 9.884e-01 |
| c18 | `json_field` |  | 9.991e-01 |
| c19 | `shipped` |  | 4.967e-01 |
| c19 | `two_step_shipped` |  | 4.268e-08 |
| c19 | `json_field` |  | 9.979e-01 |
| c20 | `shipped` |  | 9.525e-02 |
| c20 | `two_step_shipped` |  | 9.912e-01 |
| c20 | `json_field` |  | 9.998e-01 |
| c21 | `shipped` |  | 4.554e-03 |
| c21 | `two_step_shipped` |  | 7.348e-01 |
| c21 | `json_field` |  | 9.953e-01 |
| c22 | `shipped` |  | 6.145e-02 |
| c22 | `two_step_shipped` |  | 8.865e-01 |
| c22 | `json_field` |  | 9.999e-01 |
| c23 | `shipped` |  | 4.502e-02 |
| c23 | `two_step_shipped` |  | 9.988e-01 |
| c23 | `json_field` |  | 1.000e+00 |
| c24 | `shipped` |  | 1.620e-01 |
| c24 | `two_step_shipped` |  | 9.873e-01 |
| c24 | `json_field` |  | 1.000e+00 |
| s01 | `shipped` |  | 4.904e-02 |
| s01 | `two_step_shipped` |  | 8.664e-01 |
| s01 | `json_field` |  | 9.936e-01 |
| s02 | `shipped` |  | 1.354e-02 |
| s02 | `two_step_shipped` |  | 9.788e-01 |
| s02 | `json_field` |  | 9.989e-01 |
| s03 | `shipped` |  | 6.893e-02 |
| s03 | `two_step_shipped` |  | 7.903e-01 |
| s03 | `json_field` |  | 9.804e-01 |
| s04 | `shipped` |  | 7.143e-01 |
| s04 | `two_step_shipped` |  | 4.298e-07 |
| s04 | `json_field` |  | 9.777e-01 |
| s05 | `shipped` |  | 2.800e-02 |
| s05 | `two_step_shipped` |  | 6.989e-01 |
| s05 | `json_field` |  | 7.525e-01 |
| s06 | `shipped` |  | 1.023e-01 |
| s06 | `two_step_shipped` |  | 9.769e-01 |
| s06 | `json_field` |  | 9.946e-01 |
| s07 | `shipped` |  | 1.004e-01 |
| s07 | `two_step_shipped` |  | 9.899e-01 |
| s07 | `json_field` |  | 9.657e-01 |
| s08 | `shipped` |  | 8.836e-02 |
| s08 | `two_step_shipped` |  | 9.744e-01 |
| s08 | `json_field` |  | 8.957e-01 |
| s09 | `shipped` |  | 1.402e-01 |
| s09 | `two_step_shipped` |  | 9.107e-01 |
| s09 | `json_field` |  | 9.970e-01 |
| s10 | `shipped` |  | 6.634e-02 |
| s10 | `two_step_shipped` |  | 9.660e-01 |
| s10 | `json_field` |  | 8.826e-01 |
| s11 | `shipped` |  | 5.480e-02 |
| s11 | `two_step_shipped` |  | 7.861e-01 |
| s11 | `json_field` |  | 9.907e-01 |
| s12 | `shipped` |  | 1.178e-02 |
| s12 | `two_step_shipped` |  | 9.354e-01 |
| s12 | `json_field` |  | 7.903e-01 |
| s13 | `shipped` |  | 2.341e-01 |
| s13 | `two_step_shipped` |  | 9.718e-01 |
| s13 | `json_field` |  | 9.723e-01 |
| s14 | `shipped` |  | 7.166e-02 |
| s14 | `two_step_shipped` |  | 9.352e-01 |
| s14 | `json_field` |  | 9.790e-01 |
| s15 | `shipped` |  | 3.216e-02 |
| s15 | `two_step_shipped` |  | 9.519e-01 |
| s15 | `json_field` |  | 9.510e-01 |
| s16 | `shipped` |  | 1.287e-01 |
| s16 | `two_step_shipped` |  | 7.525e-01 |
| s16 | `json_field` |  | 6.508e-01 |
| s17 | `shipped` |  | 5.320e-02 |
| s17 | `two_step_shipped` |  | 8.794e-01 |
| s17 | `json_field` |  | 9.867e-01 |
| s18 | `shipped` |  | 2.844e-02 |
| s18 | `two_step_shipped` |  | 6.883e-01 |
| s18 | `json_field` |  | 9.905e-01 |
| n01 | `shipped` |  | 2.754e-02 |
| n01 | `two_step_shipped` |  | 9.240e-01 |
| n01 | `json_field` |  | 1.000e+00 |
| n02 | `shipped` |  | 6.351e-02 |
| n02 | `two_step_shipped` |  | 9.901e-01 |
| n02 | `json_field` |  | 9.999e-01 |
| n03 | `shipped` |  | 2.303e-01 |
| n03 | `two_step_shipped` |  | 9.897e-01 |
| n03 | `json_field` |  | 9.999e-01 |
| n04 | `shipped` |  | 6.077e-03 |
| n04 | `two_step_shipped` |  | 9.771e-01 |
| n04 | `json_field` |  | 1.000e+00 |
| n05 | `shipped` |  | 1.204e-03 |
| n05 | `two_step_shipped` |  | 8.277e-01 |
| n05 | `json_field` |  | 9.999e-01 |
| n06 | `shipped` |  | 4.256e-02 |
| n06 | `two_step_shipped` |  | 9.869e-01 |
| n06 | `json_field` |  | 9.998e-01 |
| n07 | `shipped` |  | 1.512e-01 |
| n07 | `two_step_shipped` |  | 9.937e-01 |
| n07 | `json_field` |  | 9.999e-01 |
| n08 | `shipped` |  | 2.575e-02 |
| n08 | `two_step_shipped` |  | 9.806e-01 |
| n08 | `json_field` |  | 9.998e-01 |
| n09 | `shipped` |  | 1.058e-02 |
| n09 | `two_step_shipped` |  | 9.933e-01 |
| n09 | `json_field` |  | 1.000e+00 |
| n10 | `shipped` |  | 1.453e-02 |
| n10 | `two_step_shipped` |  | 9.908e-01 |
| n10 | `json_field` |  | 9.999e-01 |
| n11 | `shipped` |  | 1.137e-01 |
| n11 | `two_step_shipped` |  | 5.353e-01 |
| n11 | `json_field` |  | 9.998e-01 |
| n12 | `shipped` |  | 2.192e-02 |
| n12 | `two_step_shipped` |  | 9.650e-01 |
| n12 | `json_field` |  | 9.999e-01 |
| n13 | `shipped` |  | 2.745e-01 |
| n13 | `two_step_shipped` |  | 9.863e-01 |
| n13 | `json_field` |  | 9.998e-01 |
| n14 | `shipped` |  | 3.212e-02 |
| n14 | `two_step_shipped` |  | 9.783e-01 |
| n14 | `json_field` |  | 1.000e+00 |
| n15 | `shipped` |  | 2.605e-01 |
| n15 | `two_step_shipped` |  | 9.611e-01 |
| n15 | `json_field` |  | 9.996e-01 |
| n16 | `shipped` |  | 2.814e-02 |
| n16 | `two_step_shipped` |  | 9.692e-01 |
| n16 | `json_field` |  | 9.999e-01 |
| n17 | `shipped` |  | 2.099e-02 |
| n17 | `two_step_shipped` |  | 9.755e-01 |
| n17 | `json_field` |  | 9.999e-01 |
| n18 | `shipped` |  | 2.949e-01 |
| n18 | `two_step_shipped` |  | 9.904e-01 |
| n18 | `json_field` |  | 9.999e-01 |

## 3. Coverage per item, shape × label variant

| shape | label | c01 | c02 | c03 | c04 | c05 | c06 | c07 | c08 | c09 | c10 | c11 | c12 | c13 | c14 | c15 | c16 | c17 | c18 | c19 | c20 | c21 | c22 | c23 | c24 | s01 | s02 | s03 | s04 | s05 | s06 | s07 | s08 | s09 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | s17 | s18 | n01 | n02 | n03 | n04 | n05 | n06 | n07 | n08 | n09 | n10 | n11 | n12 | n13 | n14 | n15 | n16 | n17 | n18 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 1.144e-02* | 3.048e-02* | 1.687e-03* | 1.571e-02* | 8.205e-02* | 1.755e-01 | 1.462e-02* | 5.833e-04* | 3.412e-01 | 1.147e-07* | 7.294e-02* | 9.297e-03* | 8.968e-05* | 5.879e-03* | 1.851e-02* | 2.148e-03* | 1.435e-04* | 4.660e-02* | 4.967e-01 | 9.525e-02* | 4.554e-03* | 6.145e-02* | 4.502e-02* | 1.620e-01 | 4.904e-02* | 1.354e-02* | 6.893e-02* | 7.143e-01 | 2.800e-02* | 1.023e-01 | 1.004e-01 | 8.836e-02* | 1.402e-01 | 6.634e-02* | 5.480e-02* | 1.178e-02* | 2.341e-01 | 7.166e-02* | 3.216e-02* | 1.287e-01 | 5.320e-02* | 2.844e-02* | 2.754e-02* | 6.351e-02* | 2.303e-01 | 6.077e-03* | 1.204e-03* | 4.256e-02* | 1.512e-01 | 2.575e-02* | 1.058e-02* | 1.453e-02* | 1.137e-01 | 2.192e-02* | 2.745e-01 | 3.212e-02* | 2.605e-01 | 2.814e-02* | 2.099e-02* | 2.949e-01 | 16/60 | 4.502e-02 |
| `shipped` | `space` | 2.244e-05* | 9.132e-05* | 1.808e-04* | 8.595e-06* | 3.981e-04* | 4.107e-02* | 7.122e-03* | 7.199e-05* | 6.530e-04* | 2.797e-08* | 1.293e-03* | 4.040e-04* | 1.735e-03* | 1.142e-03* | 1.971e-02* | 9.847e-05* | 6.494e-05* | 3.839e-03* | 6.017e-03* | 5.397e-04* | 1.119e-04* | 4.865e-04* | 9.591e-04* | 2.197e-03* | 8.456e-04* | 5.668e-05* | 1.106e-03* | 5.676e-04* | 5.565e-04* | 1.358e-04* | 4.765e-05* | 6.644e-04* | 4.226e-04* | 3.833e-04* | 2.076e-04* | 1.027e-04* | 2.024e-04* | 6.722e-04* | 2.229e-04* | 1.036e-04* | 9.555e-05* | 2.481e-04* | 2.330e-04* | 9.485e-04* | 9.317e-04* | 1.454e-04* | 3.359e-05* | 5.961e-04* | 3.037e-04* | 2.540e-04* | 4.299e-04* | 7.272e-05* | 8.291e-04* | 3.739e-04* | 4.688e-04* | 2.880e-04* | 1.894e-03* | 1.177e-03* | 3.256e-04* | 5.359e-04* | 0/60 | 4.040e-04 |
| `shipped` | `caps` | 6.037e-04* | 8.570e-04* | 4.630e-04* | 1.535e-03* | 1.249e-02* | 6.995e-02* | 3.800e-03* | 2.780e-04* | 1.302e-03* | 4.054e-07* | 3.082e-03* | 1.018e-03* | 1.465e-04* | 2.208e-03* | 3.464e-02* | 6.176e-03* | 7.559e-05* | 1.748e-03* | 2.195e-02* | 1.693e-03* | 1.307e-03* | 1.832e-03* | 4.690e-03* | 1.148e-02* | 4.904e-02* | 1.354e-02* | 6.893e-02* | 7.143e-01 | 2.800e-02* | 1.023e-01 | 1.004e-01 | 8.836e-02* | 1.402e-01 | 6.634e-02* | 5.480e-02* | 1.178e-02* | 2.341e-01 | 7.166e-02* | 3.216e-02* | 1.287e-01 | 5.320e-02* | 2.844e-02* | 2.266e-03* | 1.967e-02* | 4.500e-03* | 1.843e-03* | 3.915e-04* | 5.042e-03* | 3.000e-03* | 1.222e-03* | 3.528e-03* | 8.172e-04* | 1.108e-02* | 2.116e-03* | 2.104e-03* | 2.606e-03* | 1.154e-02* | 8.545e-03* | 2.710e-03* | 4.587e-03* | 6/60 | 4.690e-03 |
| `shipped` | `newline` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9957 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9831 | 1.0000 | 1.0000 | 60/60 | 1.0000 |
| `shipped` | `long` | 1.144e-02* | 3.048e-02* | 1.687e-03* | 1.571e-02* | 8.205e-02* | 1.755e-01 | 1.462e-02* | 5.833e-04* | 3.412e-01 | 1.147e-07* | 7.294e-02* | 9.297e-03* | 8.968e-05* | 5.879e-03* | 1.851e-02* | 2.148e-03* | 1.435e-04* | 4.660e-02* | 4.967e-01 | 9.525e-02* | 4.554e-03* | 6.145e-02* | 4.502e-02* | 1.620e-01 | 4.980e-05* | 1.414e-07* | 7.250e-05* | 9.013e-06* | 8.925e-07* | 1.369e-06* | 5.848e-07* | 9.070e-07* | 3.387e-06* | 6.473e-03* | 1.006e-06* | 5.003e-06* | 1.939e-08* | 6.907e-07* | 4.656e-06* | 1.237e-05* | 9.903e-07* | 7.118e-07* | 2.754e-02* | 6.351e-02* | 2.303e-01 | 6.077e-03* | 1.204e-03* | 4.256e-02* | 1.512e-01 | 2.575e-02* | 1.058e-02* | 1.453e-02* | 1.137e-01 | 2.192e-02* | 2.745e-01 | 3.212e-02* | 2.605e-01 | 2.814e-02* | 2.099e-02* | 2.949e-01 | 10/60 | 1.144e-02 |
| `two_step_shipped` | `bare` | 9.559e-01 | 9.979e-01 | 9.942e-01 | 9.584e-01 | 9.968e-01 | 9.933e-01 | 9.736e-01 | 9.941e-01 | 9.969e-01 | 9.841e-01 | 9.943e-01 | 9.487e-01 | 9.106e-01 | 9.615e-01 | 9.476e-01 | 9.848e-01 | 9.398e-01 | 9.884e-01 | 4.268e-08* | 9.912e-01 | 7.348e-01 | 8.865e-01 | 9.988e-01 | 9.873e-01 | 8.664e-01 | 9.788e-01 | 7.903e-01 | 4.298e-07* | 6.989e-01 | 9.769e-01 | 9.899e-01 | 9.744e-01 | 9.107e-01 | 9.660e-01 | 7.861e-01 | 9.354e-01 | 9.718e-01 | 9.352e-01 | 9.519e-01 | 7.525e-01 | 8.794e-01 | 6.883e-01 | 9.240e-01 | 9.901e-01 | 9.897e-01 | 9.771e-01 | 8.277e-01 | 9.869e-01 | 9.937e-01 | 9.806e-01 | 9.933e-01 | 9.908e-01 | 5.353e-01 | 9.650e-01 | 9.863e-01 | 9.783e-01 | 9.611e-01 | 9.692e-01 | 9.755e-01 | 9.904e-01 | 58/60 | 9.736e-01 |
| `two_step_shipped` | `space` | 1.253e-03* | 3.187e-06* | 7.854e-05* | 2.922e-04* | 1.321e-04* | 5.534e-04* | 1.393e-02* | 6.517e-04* | 1.380e-04* | 3.451e-04* | 5.302e-06* | 4.546e-05* | 1.136e-04* | 2.609e-04* | 5.505e-03* | 7.467e-05* | 1.078e-04* | 3.246e-04* | 8.195e-06* | 4.150e-04* | 2.228e-04* | 4.400e-05* | 4.667e-05* | 3.435e-05* | 7.990e-05* | 2.946e-05* | 2.018e-05* | 4.414e-06* | 7.927e-05* | 1.324e-05* | 1.024e-05* | 2.299e-05* | 1.161e-05* | 3.243e-05* | 1.258e-04* | 1.812e-05* | 3.669e-06* | 2.636e-05* | 2.413e-05* | 3.467e-05* | 3.545e-05* | 2.990e-05* | 1.393e-04* | 3.989e-05* | 3.853e-05* | 8.055e-05* | 2.463e-04* | 5.651e-05* | 1.854e-05* | 2.998e-05* | 7.718e-05* | 2.303e-05* | 6.928e-04* | 1.398e-04* | 3.773e-05* | 7.038e-05* | 8.021e-05* | 1.186e-04* | 3.980e-05* | 2.124e-05* | 0/60 | 5.651e-05 |
| `two_step_shipped` | `caps` | 9.044e-03* | 4.637e-04* | 3.048e-04* | 4.538e-04* | 9.967e-04* | 3.679e-03* | 3.619e-03* | 4.533e-04* | 1.052e-03* | 1.452e-02* | 2.651e-03* | 2.558e-03* | 8.246e-02* | 1.749e-02* | 6.721e-03* | 2.456e-03* | 8.950e-04* | 1.053e-03* | 1.582e-08* | 2.350e-03* | 1.939e-01 | 9.591e-02* | 7.850e-05* | 1.062e-02* | 8.664e-01 | 9.788e-01 | 7.903e-01 | 4.298e-07* | 6.989e-01 | 9.769e-01 | 9.899e-01 | 9.744e-01 | 9.107e-01 | 9.660e-01 | 7.861e-01 | 9.354e-01 | 9.718e-01 | 9.352e-01 | 9.519e-01 | 7.525e-01 | 8.794e-01 | 6.883e-01 | 6.450e-02* | 8.269e-03* | 9.462e-03* | 1.963e-02* | 1.494e-01 | 1.108e-02* | 4.843e-03* | 1.667e-02* | 4.627e-03* | 8.398e-03* | 3.754e-01 | 2.537e-02* | 4.531e-03* | 1.592e-02* | 2.962e-02* | 2.598e-02* | 1.757e-02* | 7.139e-03* | 20/60 | 1.667e-02 |
| `two_step_shipped` | `newline` | 8.544e-04* | 1.038e-04* | 2.656e-04* | 2.479e-04* | 3.106e-05* | 1.789e-05* | 4.405e-05* | 4.426e-05* | 2.218e-05* | 1.800e-05* | 2.348e-05* | 1.401e-04* | 2.154e-04* | 4.192e-04* | 5.874e-04* | 1.772e-03* | 1.214e-04* | 5.116e-05* | 3.007e-03* | 4.092e-03* | 6.275e-02* | 1.372e-04* | 5.173e-06* | 1.536e-05* | 2.068e-03* | 1.439e-04* | 1.811e-04* | 6.064e-04* | 3.499e-03* | 1.797e-04* | 4.336e-04* | 1.261e-04* | 2.618e-04* | 8.930e-05* | 9.610e-04* | 9.872e-04* | 3.911e-04* | 2.948e-04* | 2.176e-04* | 2.590e-03* | 2.921e-04* | 5.572e-04* | 3.925e-03* | 5.819e-04* | 2.515e-04* | 2.938e-03* | 1.665e-03* | 5.181e-04* | 1.240e-03* | 5.650e-04* | 1.215e-03* | 1.405e-04* | 1.856e-02* | 5.015e-03* | 7.940e-03* | 2.351e-03* | 3.345e-03* | 6.722e-04* | 8.519e-04* | 2.250e-03* | 0/60 | 4.336e-04 |
| `two_step_shipped` | `long` | 9.559e-01 | 9.979e-01 | 9.942e-01 | 9.584e-01 | 9.968e-01 | 9.933e-01 | 9.736e-01 | 9.941e-01 | 9.969e-01 | 9.841e-01 | 9.943e-01 | 9.487e-01 | 9.106e-01 | 9.615e-01 | 9.476e-01 | 9.848e-01 | 9.398e-01 | 9.884e-01 | 4.268e-08* | 9.912e-01 | 7.348e-01 | 8.865e-01 | 9.988e-01 | 9.873e-01 | 5.358e-05* | 9.147e-08* | 1.278e-05* | 3.457e-09* | 1.446e-05* | 7.868e-08* | 1.329e-06* | 5.652e-07* | 2.752e-07* | 1.929e-03* | 1.379e-05* | 7.966e-06* | 1.936e-09* | 9.563e-07* | 2.544e-06* | 9.867e-06* | 6.103e-07* | 1.537e-06* | 9.240e-01 | 9.901e-01 | 9.897e-01 | 9.771e-01 | 8.277e-01 | 9.869e-01 | 9.937e-01 | 9.806e-01 | 9.933e-01 | 9.908e-01 | 5.353e-01 | 9.650e-01 | 9.863e-01 | 9.783e-01 | 9.611e-01 | 9.692e-01 | 9.755e-01 | 9.904e-01 | 41/60 | 9.611e-01 |
| `json_field` | `bare` | 0.9994 | 0.9997 | 0.9999 | 0.9997 | 0.9997 | 0.9999 | 0.9996 | 0.9965 | 1.0000 | 0.9997 | 0.9999 | 0.9999 | 0.9992 | 0.9997 | 0.9989 | 1.0000 | 0.9995 | 0.9991 | 0.9979 | 0.9998 | 0.9953 | 0.9999 | 1.0000 | 1.0000 | 0.9936 | 0.9989 | 0.9804 | 0.9777 | 0.7525 | 0.9946 | 0.9657 | 0.8957 | 0.9970 | 0.8826 | 0.9907 | 0.7903 | 0.9723 | 0.9790 | 0.9510 | 0.6508 | 0.9867 | 0.9905 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 0.9999 | 0.9998 | 0.9999 | 0.9998 | 1.0000 | 0.9999 | 0.9998 | 0.9999 | 0.9998 | 1.0000 | 0.9996 | 0.9999 | 0.9999 | 0.9999 | 60/60 | 0.9997 |
| `json_field` | `space` | 2.071e-05* | 2.524e-05* | 9.532e-06* | 1.335e-05* | 2.076e-05* | 2.143e-05* | 2.352e-04* | 1.442e-04* | 2.651e-05* | 1.293e-04* | 5.310e-06* | 8.141e-06* | 5.105e-05* | 4.049e-05* | 2.499e-04* | 1.863e-05* | 2.237e-05* | 9.531e-05* | 4.511e-05* | 3.551e-05* | 2.561e-05* | 3.450e-06* | 2.756e-06* | 6.357e-06* | 1.221e-05* | 1.076e-05* | 4.499e-05* | 1.885e-05* | 5.428e-05* | 7.884e-06* | 1.399e-05* | 2.731e-05* | 2.647e-06* | 3.254e-05* | 9.164e-06* | 8.946e-05* | 7.877e-06* | 3.276e-05* | 1.046e-05* | 2.613e-04* | 8.533e-06* | 1.515e-05* | 3.304e-06* | 3.395e-06* | 2.684e-06* | 3.428e-06* | 2.797e-06* | 4.223e-06* | 2.272e-06* | 4.040e-06* | 1.929e-06* | 2.022e-06* | 4.028e-06* | 2.728e-06* | 3.495e-06* | 2.277e-06* | 4.907e-06* | 2.415e-06* | 2.117e-06* | 2.137e-06* | 0/60 | 1.076e-05 |
| `json_field` | `caps` | 1.051e-05* | 1.240e-05* | 1.299e-06* | 1.880e-05* | 1.250e-05* | 2.733e-06* | 2.021e-05* | 1.012e-04* | 1.204e-06* | 7.928e-05* | 1.875e-05* | 7.579e-06* | 6.432e-04* | 6.603e-05* | 5.479e-04* | 5.813e-06* | 9.345e-06* | 1.411e-05* | 8.290e-06* | 3.244e-05* | 1.333e-04* | 5.679e-05* | 1.458e-06* | 4.085e-06* | 9.936e-01 | 9.989e-01 | 9.804e-01 | 9.777e-01 | 7.525e-01 | 9.946e-01 | 9.657e-01 | 8.957e-01 | 9.970e-01 | 8.826e-01 | 9.907e-01 | 7.903e-01 | 9.723e-01 | 9.790e-01 | 9.510e-01 | 6.508e-01 | 9.867e-01 | 9.905e-01 | 6.261e-06* | 2.925e-05* | 3.703e-05* | 1.026e-05* | 1.811e-05* | 5.153e-05* | 2.189e-05* | 1.104e-04* | 8.670e-06* | 3.110e-05* | 5.038e-05* | 2.394e-05* | 3.795e-05* | 1.337e-05* | 1.497e-04* | 2.078e-05* | 1.107e-05* | 4.804e-05* | 18/60 | 4.804e-05 |
| `json_field` | `newline` | 5.338e-06* | 2.859e-06* | 5.615e-07* | 7.817e-06* | 2.927e-06* | 9.977e-07* | 2.012e-06* | 1.334e-04* | 1.877e-06* | 8.487e-06* | 1.415e-06* | 8.608e-07* | 1.869e-06* | 9.400e-07* | 9.450e-06* | 2.138e-06* | 7.371e-07* | 5.313e-06* | 2.490e-06* | 2.586e-07* | 2.479e-06* | 4.425e-07* | 1.910e-07* | 1.521e-06* | 1.226e-04* | 3.671e-04* | 2.586e-04* | 1.610e-04* | 4.285e-04* | 9.595e-05* | 3.521e-04* | 4.953e-04* | 8.653e-05* | 5.060e-04* | 1.420e-04* | 1.392e-04* | 8.634e-05* | 1.680e-04* | 1.507e-04* | 3.249e-04* | 2.541e-04* | 4.595e-05* | 1.453e-06* | 3.261e-06* | 1.155e-06* | 6.432e-07* | 1.080e-06* | 7.602e-06* | 2.685e-06* | 2.513e-06* | 6.701e-07* | 2.109e-06* | 8.624e-06* | 5.016e-07* | 2.957e-06* | 6.267e-07* | 5.976e-06* | 3.887e-06* | 1.572e-06* | 1.336e-06* | 0/60 | 2.957e-06 |
| `json_field` | `long` | 9.994e-01 | 9.997e-01 | 9.999e-01 | 9.997e-01 | 9.997e-01 | 9.999e-01 | 9.996e-01 | 9.965e-01 | 1.000e+00 | 9.997e-01 | 9.999e-01 | 9.999e-01 | 9.992e-01 | 9.997e-01 | 9.989e-01 | 1.000e+00 | 9.995e-01 | 9.991e-01 | 9.979e-01 | 9.998e-01 | 9.953e-01 | 9.999e-01 | 1.000e+00 | 1.000e+00 | 5.825e-03* | 8.968e-06* | 1.369e-02* | 1.590e-02* | 1.975e-01 | 1.311e-04* | 2.350e-02* | 1.014e-01 | 7.829e-04* | 4.040e-01 | 5.512e-03* | 5.336e-03* | 2.081e-02* | 1.193e-02* | 3.139e-03* | 2.561e-01 | 1.212e-03* | 2.693e-03* | 1.000e+00 | 9.999e-01 | 9.999e-01 | 1.000e+00 | 9.999e-01 | 9.998e-01 | 9.999e-01 | 9.998e-01 | 1.000e+00 | 9.999e-01 | 9.998e-01 | 9.999e-01 | 9.998e-01 | 1.000e+00 | 9.996e-01 | 9.999e-01 | 9.999e-01 | 9.999e-01 | 46/60 | 9.997e-01 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 8.834e-02 | 4.502e-02 | 16/60 | 0/60 | — |
| `two_step_shipped` | advanced | 9.031e-01 | 9.736e-01 | 58/60 | 0/60 | content |
| `json_field` | cue | 0.9788 | 0.9997 | 60/60 | 0/60 | — |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 60 | 42 | 0.700 | 0.575–0.801 | 44/60 | 0.0450 | 59/60 |
| `two_step_shipped=bare` | 60 | 46 | 0.767 | 0.646–0.856 | 2/60 | 0.9736 | 58/60 |
| `json_field=bare` | 60 | 51 | 0.850 | 0.739–0.919 | 0/60 | 0.9997 | 60/60 |

