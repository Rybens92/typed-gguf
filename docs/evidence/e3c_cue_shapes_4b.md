# E3c — cue shapes that put the readout mid-answer (`spark2_5`)

- card `t_6c119626` · model `Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes, sha256 `5c2c3c190e4337e1…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   271.20 MiB
sched_reserve: graph nodes  = 1266
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.12 ms, sched copies = 1`- dev items: c01, s01, n01, c02, s02, n02 (choice 2, noul 2, score 2) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T15:42:17Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 201.9 s (one model load 677 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| c01 | `shipped` | cue | `
` | 0.9536 | 0.0000 | ok |
| c01 | `answer_is` | cue | `
` | 0.7104 | 0.0000 | ok |
| c01 | `answer_colon` | cue | `
` | 0.8568 | 0.0000 | ok |
| c01 | `wybieram_pl` | cue | `
` | 0.8078 | 0.0000 | ok |
| c01 | `json_field` | cue | `t` | 0.5928 | 0.0000 | ok |
| c01 | `two_step_shipped` | advanced | `t` | 0.8580 | 0.0000 | ok |
| c01 | `two_step_answer_is` | advanced | `t` | 0.9101 | 0.0000 | ok |
| s01 | `shipped` | cue | `
` | 0.7622 | 0.0000 | ok |
| s01 | `answer_is` | cue | `3` | 0.7873 | 0.0000 | ok |
| s01 | `answer_colon` | cue | `3` | 0.6271 | 0.0000 | ok |
| s01 | `wybieram_pl` | cue | `3` | 0.8036 | 0.0000 | ok |
| s01 | `json_field` | cue | `3` | 0.8100 | 0.0000 | ok |
| s01 | `two_step_shipped` | advanced | `3` | 0.7367 | 0.0000 | ok |
| s01 | `two_step_answer_is` | advanced | `<｜end▁of▁sentence｜>` | 0.6164 | 0.0000 | ok |
| n01 | `shipped` | cue | `
` | 0.9431 | 0.0000 | ok |
| n01 | `answer_is` | cue | `
` | 0.6011 | 0.0000 | ok |
| n01 | `answer_colon` | cue | `
` | 0.9772 | 0.0000 | ok |
| n01 | `wybieram_pl` | cue | `
` | 0.9281 | 0.0000 | ok |
| n01 | `json_field` | cue | `yes` | 0.9991 | 0.0000 | ok |
| n01 | `two_step_shipped` | advanced | `yes` | 0.7656 | 0.0000 | ok |
| n01 | `two_step_answer_is` | advanced | `yes` | 0.7394 | 0.0000 | ok |
| c02 | `shipped` | cue | `
` | 0.9362 | 0.0000 | ok |
| c02 | `answer_is` | cue | `
` | 0.6121 | 0.0000 | ok |
| c02 | `answer_colon` | cue | `
` | 0.7451 | 0.0000 | ok |
| c02 | `wybieram_pl` | cue | `
` | 0.7175 | 0.0000 | ok |
| c02 | `json_field` | cue | `b` | 0.9703 | 0.0000 | ok |
| c02 | `two_step_shipped` | advanced | `b` | 0.9975 | 0.0000 | ok |
| c02 | `two_step_answer_is` | advanced | `b` | 0.9619 | 0.0000 | ok |
| s02 | `shipped` | cue | `
` | 0.9574 | 0.0000 | ok |
| s02 | `answer_is` | cue | `1` | 0.7061 | 0.0000 | ok |
| s02 | `answer_colon` | cue | `1` | 0.7886 | 0.0000 | ok |
| s02 | `wybieram_pl` | cue | `3` | 0.5796 | 0.0000 | ok |
| s02 | `json_field` | cue | `1` | 0.5538 | 0.0000 | ok |
| s02 | `two_step_shipped` | advanced | `1` | 0.4811 | 0.0000 | ok |
| s02 | `two_step_answer_is` | advanced | `<｜end▁of▁sentence｜>` | 0.8745 | 0.0000 | ok |
| n02 | `shipped` | cue | `
` | 0.8679 | 0.0000 | ok |
| n02 | `answer_is` | cue | `
` | 0.6394 | 0.0000 | ok |
| n02 | `answer_colon` | cue | `
` | 0.9757 | 0.0000 | ok |
| n02 | `wybieram_pl` | cue | `
` | 0.9249 | 0.0000 | ok |
| n02 | `json_field` | cue | `yes` | 0.9353 | 0.0000 | ok |
| n02 | `two_step_shipped` | advanced | `yes` | 0.9280 | 0.0000 | ok |
| n02 | `two_step_answer_is` | advanced | `no` | 0.7370 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape |  | `bare` coverage |
|---|---|---|
| c01 | `shipped` |  | 1.144e-02 |
| c01 | `answer_is` |  | 6.453e-06 |
| c01 | `answer_colon` |  | 4.419e-06 |
| c01 | `wybieram_pl` |  | 3.674e-05 |
| c01 | `json_field` |  | 9.994e-01 |
| c01 | `two_step_shipped` |  | 9.457e-01 |
| c01 | `two_step_answer_is` |  | 9.493e-01 |
| s01 | `shipped` |  | 4.904e-02 |
| s01 | `answer_is` |  | 9.871e-01 |
| s01 | `answer_colon` |  | 9.084e-01 |
| s01 | `wybieram_pl` |  | 9.972e-01 |
| s01 | `json_field` |  | 9.913e-01 |
| s01 | `two_step_shipped` |  | 8.259e-01 |
| s01 | `two_step_answer_is` |  | 1.371e-06 |
| n01 | `shipped` |  | 2.754e-02 |
| n01 | `answer_is` |  | 3.526e-03 |
| n01 | `answer_colon` |  | 3.671e-05 |
| n01 | `wybieram_pl` |  | 8.434e-05 |
| n01 | `json_field` |  | 1.000e+00 |
| n01 | `two_step_shipped` |  | 9.356e-01 |
| n01 | `two_step_answer_is` |  | 9.949e-01 |
| c02 | `shipped` |  | 3.048e-02 |
| c02 | `answer_is` |  | 1.237e-05 |
| c02 | `answer_colon` |  | 9.293e-05 |
| c02 | `wybieram_pl` |  | 2.260e-03 |
| c02 | `json_field` |  | 9.997e-01 |
| c02 | `two_step_shipped` |  | 9.979e-01 |
| c02 | `two_step_answer_is` |  | 9.902e-01 |
| s02 | `shipped` |  | 1.354e-02 |
| s02 | `answer_is` |  | 9.995e-01 |
| s02 | `answer_colon` |  | 9.983e-01 |
| s02 | `wybieram_pl` |  | 9.934e-01 |
| s02 | `json_field` |  | 9.988e-01 |
| s02 | `two_step_shipped` |  | 9.728e-01 |
| s02 | `two_step_answer_is` |  | 9.524e-07 |
| n02 | `shipped` |  | 6.351e-02 |
| n02 | `answer_is` |  | 3.411e-03 |
| n02 | `answer_colon` |  | 4.177e-05 |
| n02 | `wybieram_pl` |  | 3.171e-05 |
| n02 | `json_field` |  | 9.999e-01 |
| n02 | `two_step_shipped` |  | 9.898e-01 |
| n02 | `two_step_answer_is` |  | 9.974e-01 |

## 3. Coverage per item, shape × label variant

| shape | label | c01 | s01 | n01 | c02 | s02 | n02 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 0.0114* | 0.0490* | 0.0275* | 0.0305* | 0.0135* | 0.0635* | 0/6 | 0.0305 |
| `shipped` | `space` | 2.244e-05* | 8.456e-04* | 2.330e-04* | 9.132e-05* | 5.668e-05* | 9.485e-04* | 0/6 | 2.330e-04 |
| `shipped` | `caps` | 0.0006* | 0.0490* | 0.0023* | 0.0009* | 0.0135* | 0.0197* | 0/6 | 0.0135 |
| `shipped` | `newline` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6/6 | 1.0000 |
| `shipped` | `long` | 1.144e-02* | 4.980e-05* | 2.754e-02* | 3.048e-02* | 1.414e-07* | 6.351e-02* | 0/6 | 2.754e-02 |
| `answer_is` | `bare` | 6.453e-06* | 9.871e-01 | 3.526e-03* | 1.237e-05* | 9.995e-01 | 3.411e-03* | 2/6 | 3.526e-03 |
| `answer_is` | `space` | 1.977e-01 | 6.643e-07* | 5.303e-02* | 3.729e-01 | 8.883e-08* | 2.222e-02* | 2/6 | 5.303e-02 |
| `answer_is` | `caps` | 8.588e-07* | 9.871e-01 | 2.358e-04* | 1.375e-06* | 9.995e-01 | 2.770e-04* | 2/6 | 2.770e-04 |
| `answer_is` | `newline` | 1.0000 | 0.0641* | 1.0000 | 1.0000 | 0.0006* | 1.0000 | 4/6 | 1.0000 |
| `answer_is` | `long` | 6.453e-06* | 8.651e-09* | 3.526e-03* | 1.237e-05* | 4.240e-10* | 3.411e-03* | 0/6 | 1.237e-05 |
| `answer_colon` | `bare` | 4.419e-06* | 9.084e-01 | 3.671e-05* | 9.293e-05* | 9.983e-01 | 4.177e-05* | 2/6 | 9.293e-05 |
| `answer_colon` | `space` | 3.324e-03* | 4.903e-06* | 2.374e-03* | 1.392e-01 | 1.533e-07* | 4.652e-03* | 1/6 | 3.324e-03 |
| `answer_colon` | `caps` | 9.251e-06* | 9.084e-01 | 1.548e-05* | 2.510e-05* | 9.983e-01 | 1.234e-05* | 2/6 | 2.510e-05 |
| `answer_colon` | `newline` | 1.0000 | 0.4524 | 1.0000 | 1.0000 | 0.0063* | 1.0000 | 5/6 | 1.0000 |
| `answer_colon` | `long` | 4.419e-06* | 3.352e-08* | 3.671e-05* | 9.293e-05* | 1.350e-10* | 4.177e-05* | 0/6 | 3.671e-05 |
| `wybieram_pl` | `bare` | 3.674e-05* | 9.972e-01 | 8.434e-05* | 2.260e-03* | 9.934e-01 | 3.171e-05* | 2/6 | 2.260e-03 |
| `wybieram_pl` | `space` | 8.095e-02* | 1.516e-07* | 1.094e-02* | 5.101e-02* | 1.561e-07* | 6.939e-03* | 0/6 | 1.094e-02 |
| `wybieram_pl` | `caps` | 2.163e-05* | 9.972e-01 | 6.908e-06* | 2.906e-04* | 9.934e-01 | 6.690e-06* | 2/6 | 2.906e-04 |
| `wybieram_pl` | `newline` | 1.0000 | 0.0138* | 1.0000 | 1.0000 | 0.0248* | 1.0000 | 4/6 | 1.0000 |
| `wybieram_pl` | `long` | 3.674e-05* | 7.666e-09* | 8.434e-05* | 2.260e-03* | 5.528e-10* | 3.171e-05* | 0/6 | 3.674e-05 |
| `json_field` | `bare` | 0.9994 | 0.9913 | 1.0000 | 0.9997 | 0.9988 | 0.9999 | 6/6 | 0.9997 |
| `json_field` | `space` | 2.257e-05* | 1.124e-05* | 3.056e-06* | 2.582e-05* | 1.147e-05* | 3.219e-06* | 0/6 | 1.147e-05 |
| `json_field` | `caps` | 1.136e-05* | 9.913e-01 | 6.521e-06* | 1.256e-05* | 9.988e-01 | 2.488e-05* | 2/6 | 2.488e-05 |
| `json_field` | `newline` | 4.910e-06* | 1.208e-04* | 1.322e-06* | 3.104e-06* | 3.277e-04* | 2.720e-06* | 0/6 | 4.910e-06 |
| `json_field` | `long` | 9.994e-01 | 8.119e-03* | 1.000e+00 | 9.997e-01 | 6.142e-06* | 9.999e-01 | 4/6 | 9.997e-01 |
| `two_step_shipped` | `bare` | 0.9457 | 0.8259 | 0.9356 | 0.9979 | 0.9728 | 0.9898 | 6/6 | 0.9728 |
| `two_step_shipped` | `space` | 1.664e-03* | 9.633e-05* | 1.212e-04* | 3.071e-06* | 3.318e-05* | 4.149e-05* | 0/6 | 9.633e-05 |
| `two_step_shipped` | `caps` | 0.0100* | 0.8259 | 0.0537* | 0.0005* | 0.9728 | 0.0081* | 2/6 | 0.0537 |
| `two_step_shipped` | `newline` | 0.0007* | 0.0021* | 0.0032* | 0.0001* | 0.0002* | 0.0009* | 0/6 | 0.0009 |
| `two_step_shipped` | `long` | 9.457e-01 | 5.248e-05* | 9.356e-01 | 9.979e-01 | 1.162e-07* | 9.898e-01 | 4/6 | 9.457e-01 |
| `two_step_answer_is` | `bare` | 9.493e-01 | 1.371e-06* | 9.949e-01 | 9.902e-01 | 9.524e-07* | 9.974e-01 | 4/6 | 9.902e-01 |
| `two_step_answer_is` | `space` | 2.561e-03* | 1.880e-05* | 6.186e-05* | 8.304e-04* | 1.725e-06* | 3.313e-05* | 0/6 | 6.186e-05 |
| `two_step_answer_is` | `caps` | 3.138e-02* | 1.371e-06* | 1.699e-03* | 3.175e-03* | 9.524e-07* | 3.205e-04* | 0/6 | 1.699e-03 |
| `two_step_answer_is` | `newline` | 3.721e-04* | 2.889e-02* | 1.586e-04* | 2.535e-05* | 1.041e-03* | 7.684e-06* | 0/6 | 3.721e-04 |
| `two_step_answer_is` | `long` | 9.493e-01 | 5.887e-09* | 9.949e-01 | 9.902e-01 | 1.025e-10* | 9.974e-01 | 4/6 | 9.902e-01 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 0.0326 | 0.0305 | 0/6 | 0/6 | — |
| `answer_is` | cue | 3.323e-01 | 3.526e-03 | 2/6 | 0/6 | — |
| `answer_colon` | cue | 3.178e-01 | 9.293e-05 | 2/6 | 0/6 | — |
| `wybieram_pl` | cue | 3.322e-01 | 2.260e-03 | 2/6 | 0/6 | — |
| `json_field` | cue | 0.9982 | 0.9997 | 6/6 | 0/6 | — |
| `two_step_shipped` | advanced | 0.9446 | 0.9728 | 6/6 | 0/6 | content |
| `two_step_answer_is` | advanced | 6.553e-01 | 9.902e-01 | 4/6 | 0/6 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 6 | 3 | 0.500 | 0.188–0.812 | 6/6 | 0.0305 | 6/6 |
| `two_step_shipped=bare` | 6 | 5 | 0.833 | 0.436–0.970 | 0/6 | 0.9728 | 6/6 |
| `answer_is=bare` | 6 | 4 | 0.667 | 0.300–0.903 | 4/6 | 0.0035 | 5/6 |

