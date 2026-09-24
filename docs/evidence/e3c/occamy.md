# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   501.91 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 1
sched_reserve: reserve took 79.70 ms, sched copies = 1`
- dev items: c01, c02, s01, s02, n01, n02 (choice 2, noul 2, score 2) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T15:58:15Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 2027.6 s (one model load 67671 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| c01 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| c01 | `answer_is` | cue | `<|im_end|>` | 0.9997 | 0.9997 | W_CUE_REFUSED |
| c01 | `answer_colon` | cue | `<|im_end|>` | 0.9875 | 0.9875 | W_CUE_REFUSED |
| c01 | `wybieram_pl` | cue | `<|im_end|>` | 0.9998 | 0.9998 | W_CUE_REFUSED |
| c01 | `json_field` | cue | `<|im_end|>` | 0.9878 | 0.9878 | W_CUE_REFUSED |
| c01 | `two_step_shipped` | advanced | `
` | 0.5441 | 0.0000 | ok |
| c01 | `two_step_answer_is` | advanced | `
` | 0.7232 | 0.0000 | ok |
| c02 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| c02 | `answer_is` | cue | `<|im_end|>` | 0.9997 | 0.9997 | W_CUE_REFUSED |
| c02 | `answer_colon` | cue | `<|im_end|>` | 0.9856 | 0.9856 | W_CUE_REFUSED |
| c02 | `wybieram_pl` | cue | `<|im_end|>` | 0.9999 | 0.9999 | W_CUE_REFUSED |
| c02 | `json_field` | cue | `<|im_end|>` | 0.9964 | 0.9964 | W_CUE_REFUSED |
| c02 | `two_step_shipped` | advanced | `

` | 1.0000 | 0.0000 | ok |
| c02 | `two_step_answer_is` | advanced | `
` | 0.8824 | 0.0000 | ok |
| s01 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| s01 | `answer_is` | cue | `<|im_end|>` | 0.9963 | 0.9963 | W_CUE_REFUSED |
| s01 | `answer_colon` | cue | `<|im_end|>` | 0.9981 | 0.9981 | W_CUE_REFUSED |
| s01 | `wybieram_pl` | cue | `<|im_end|>` | 0.9030 | 0.9030 | W_CUE_REFUSED |
| s01 | `json_field` | cue | `<|im_end|>` | 0.9997 | 0.9997 | W_CUE_REFUSED |
| s01 | `two_step_shipped` | advanced | `
` | 0.8896 | 0.0001 | ok |
| s01 | `two_step_answer_is` | advanced | `

` | 0.9436 | 0.0000 | ok |
| s02 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| s02 | `answer_is` | cue | `<|im_end|>` | 0.9841 | 0.9841 | W_CUE_REFUSED |
| s02 | `answer_colon` | cue | `<|im_end|>` | 0.9783 | 0.9783 | W_CUE_REFUSED |
| s02 | `wybieram_pl` | cue | `<|im_end|>` | 0.5643 | 0.5643 | W_CUE_REFUSED |
| s02 | `json_field` | cue | `<|im_end|>` | 0.9990 | 0.9990 | W_CUE_REFUSED |
| s02 | `two_step_shipped` | advanced | `

` | 0.9981 | 0.0003 | ok |
| s02 | `two_step_answer_is` | advanced | `<|im_end|>` | 0.9968 | 0.9968 | W_CUE_REFUSED |
| n01 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| n01 | `answer_is` | cue | `<|im_end|>` | 0.9996 | 0.9996 | W_CUE_REFUSED |
| n01 | `answer_colon` | cue | `<|im_end|>` | 0.9994 | 0.9994 | W_CUE_REFUSED |
| n01 | `wybieram_pl` | cue | `<|im_end|>` | 0.9998 | 0.9998 | W_CUE_REFUSED |
| n01 | `json_field` | cue | `<|im_end|>` | 0.9301 | 0.9301 | W_CUE_REFUSED |
| n01 | `two_step_shipped` | advanced | `

` | 1.0000 | 0.0000 | ok |
| n01 | `two_step_answer_is` | advanced | `

` | 0.8226 | 0.0000 | ok |
| n02 | `shipped` | cue | `<|im_end|>` | 0.9999 | 0.9999 | W_CUE_REFUSED |
| n02 | `answer_is` | cue | `<|im_end|>` | 0.9983 | 0.9983 | W_CUE_REFUSED |
| n02 | `answer_colon` | cue | `<|im_end|>` | 0.9938 | 0.9938 | W_CUE_REFUSED |
| n02 | `wybieram_pl` | cue | `<|im_end|>` | 0.9978 | 0.9978 | W_CUE_REFUSED |
| n02 | `json_field` | cue | `<|im_end|>` | 0.9130 | 0.9130 | W_CUE_REFUSED |
| n02 | `two_step_shipped` | advanced | `<|im_end|>` | 0.9337 | 0.9337 | W_CUE_REFUSED |
| n02 | `two_step_answer_is` | advanced | `

` | 0.9215 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape | `<|endoftext|>` | `<|im_end|>` | `eos` | `bare` coverage |
|---|---|---|---|---|---|
| c01 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 5.262e-08 |
| c01 | `answer_is` | 0.0000 | 0.9997 | 0.0000 | 4.544e-06 |
| c01 | `answer_colon` | 0.0000 | 0.9875 | 0.0000 | 1.491e-07 |
| c01 | `wybieram_pl` | 0.0000 | 0.9998 | 0.0000 | 6.823e-06 |
| c01 | `json_field` | 0.0000 | 0.9878 | 0.0000 | 1.155e-02 |
| c01 | `two_step_shipped` | 0.0000 | 0.0000 | 0.0000 | 1.947e-10 |
| c01 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 6.650e-11 |
| c02 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 7.421e-07 |
| c02 | `answer_is` | 0.0000 | 0.9997 | 0.0000 | 4.040e-07 |
| c02 | `answer_colon` | 0.0000 | 0.9856 | 0.0000 | 6.845e-08 |
| c02 | `wybieram_pl` | 0.0000 | 0.9999 | 0.0000 | 1.722e-08 |
| c02 | `json_field` | 0.0000 | 0.9964 | 0.0000 | 3.480e-03 |
| c02 | `two_step_shipped` | 0.0000 | 0.0000 | 0.0000 | 3.193e-13 |
| c02 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 1.572e-11 |
| s01 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 1.251e-06 |
| s01 | `answer_is` | 0.0000 | 0.9963 | 0.0000 | 1.889e-03 |
| s01 | `answer_colon` | 0.0000 | 0.9981 | 0.0000 | 1.353e-03 |
| s01 | `wybieram_pl` | 0.0000 | 0.9030 | 0.0000 | 9.626e-02 |
| s01 | `json_field` | 0.0000 | 0.9997 | 0.0000 | 1.839e-04 |
| s01 | `two_step_shipped` | 0.0000 | 0.0001 | 0.0000 | 1.148e-09 |
| s01 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 3.678e-06 |
| s02 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 6.154e-06 |
| s02 | `answer_is` | 0.0000 | 0.9841 | 0.0000 | 1.470e-02 |
| s02 | `answer_colon` | 0.0000 | 0.9783 | 0.0000 | 2.139e-02 |
| s02 | `wybieram_pl` | 0.0000 | 0.5643 | 0.0000 | 4.295e-01 |
| s02 | `json_field` | 0.0000 | 0.9990 | 0.0000 | 7.430e-04 |
| s02 | `two_step_shipped` | 0.0000 | 0.0003 | 0.0000 | 1.622e-07 |
| s02 | `two_step_answer_is` | 0.0000 | 0.9968 | 0.0000 | 1.002e-07 |
| n01 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 1.250e-06 |
| n01 | `answer_is` | 0.0000 | 0.9996 | 0.0000 | 5.902e-09 |
| n01 | `answer_colon` | 0.0000 | 0.9994 | 0.0000 | 1.325e-08 |
| n01 | `wybieram_pl` | 0.0000 | 0.9998 | 0.0000 | 4.457e-09 |
| n01 | `json_field` | 0.0001 | 0.9301 | 0.0000 | 6.880e-02 |
| n01 | `two_step_shipped` | 0.0000 | 0.0000 | 0.0000 | 1.793e-08 |
| n01 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 8.378e-12 |
| n02 | `shipped` | 0.0000 | 0.9999 | 0.0000 | 3.726e-06 |
| n02 | `answer_is` | 0.0000 | 0.9983 | 0.0000 | 5.486e-09 |
| n02 | `answer_colon` | 0.0000 | 0.9938 | 0.0000 | 4.178e-08 |
| n02 | `wybieram_pl` | 0.0000 | 0.9978 | 0.0000 | 4.279e-08 |
| n02 | `json_field` | 0.0000 | 0.9130 | 0.0000 | 8.491e-02 |
| n02 | `two_step_shipped` | 0.0000 | 0.9337 | 0.0000 | 5.536e-08 |
| n02 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 5.103e-12 |

## 3. Coverage per item, shape × label variant

| shape | label | c01 | c02 | s01 | s02 | n01 | n02 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 5.262e-08* | 7.421e-07* | 1.251e-06* | 6.154e-06* | 1.250e-06* | 3.726e-06* | 0/6 | 1.251e-06 |
| `shipped` | `space` | 3.417e-09* | 8.893e-09* | 7.894e-10* | 7.201e-10* | 4.843e-09* | 5.221e-08* | 0/6 | 4.843e-09 |
| `shipped` | `caps` | 4.828e-09* | 1.695e-08* | 1.251e-06* | 6.154e-06* | 1.707e-08* | 2.837e-08* | 0/6 | 2.837e-08 |
| `shipped` | `newline` | 5.188e-08* | 4.625e-09* | 3.461e-08* | 7.377e-09* | 1.373e-08* | 3.893e-07* | 0/6 | 3.461e-08 |
| `shipped` | `long` | 5.262e-08* | 7.421e-07* | 1.333e-09* | 1.133e-10* | 1.250e-06* | 3.726e-06* | 0/6 | 7.421e-07 |
| `answer_is` | `bare` | 4.544e-06* | 4.040e-07* | 1.889e-03* | 1.470e-02* | 5.902e-09* | 5.486e-09* | 0/6 | 4.544e-06 |
| `answer_is` | `space` | 2.867e-05* | 2.525e-06* | 2.024e-07* | 8.729e-07* | 1.917e-06* | 1.444e-06* | 0/6 | 1.917e-06 |
| `answer_is` | `caps` | 3.320e-07* | 8.865e-08* | 1.889e-03* | 1.470e-02* | 7.168e-10* | 8.178e-10* | 0/6 | 3.320e-07 |
| `answer_is` | `newline` | 2.014e-08* | 1.497e-08* | 3.863e-09* | 1.583e-08* | 3.024e-09* | 8.051e-09* | 0/6 | 1.497e-08 |
| `answer_is` | `long` | 4.544e-06* | 4.040e-07* | 1.376e-08* | 9.534e-09* | 5.902e-09* | 5.486e-09* | 0/6 | 1.376e-08 |
| `answer_colon` | `bare` | 1.491e-07* | 6.845e-08* | 1.353e-03* | 2.139e-02* | 1.325e-08* | 4.178e-08* | 0/6 | 1.491e-07 |
| `answer_colon` | `space` | 1.694e-07* | 2.245e-07* | 5.269e-07* | 9.211e-07* | 1.168e-05* | 1.432e-05* | 0/6 | 9.211e-07 |
| `answer_colon` | `caps` | 8.922e-08* | 1.516e-08* | 1.353e-03* | 2.139e-02* | 2.450e-09* | 8.010e-09* | 0/6 | 8.922e-08 |
| `answer_colon` | `newline` | 1.206e-08* | 3.303e-09* | 3.694e-08* | 4.064e-08* | 6.777e-09* | 4.840e-08* | 0/6 | 3.694e-08 |
| `answer_colon` | `long` | 1.491e-07* | 6.845e-08* | 5.384e-09* | 9.338e-09* | 1.325e-08* | 4.178e-08* | 0/6 | 4.178e-08 |
| `wybieram_pl` | `bare` | 6.823e-06* | 1.722e-08* | 9.626e-02* | 4.295e-01 | 4.457e-09* | 4.279e-08* | 1/6 | 6.823e-06 |
| `wybieram_pl` | `space` | 7.856e-05* | 5.150e-07* | 6.024e-04* | 3.373e-04* | 6.593e-07* | 9.635e-06* | 0/6 | 7.856e-05 |
| `wybieram_pl` | `caps` | 1.751e-07* | 1.460e-09* | 9.626e-02* | 4.295e-01 | 3.658e-09* | 1.087e-08* | 1/6 | 1.751e-07 |
| `wybieram_pl` | `newline` | 1.670e-08* | 1.473e-08* | 9.917e-08* | 2.776e-07* | 4.975e-09* | 1.307e-08* | 0/6 | 1.670e-08 |
| `wybieram_pl` | `long` | 6.823e-06* | 1.722e-08* | 7.545e-08* | 3.905e-07* | 4.457e-09* | 4.279e-08* | 0/6 | 7.545e-08 |
| `json_field` | `bare` | 0.0115* | 0.0035* | 0.0002* | 0.0007* | 0.0688* | 0.0849* | 0/6 | 0.0115 |
| `json_field` | `space` | 3.367e-05* | 6.975e-06* | 1.576e-06* | 2.882e-06* | 6.209e-05* | 6.571e-05* | 0/6 | 3.367e-05 |
| `json_field` | `caps` | 2.397e-05* | 6.029e-06* | 1.839e-04* | 7.430e-04* | 3.356e-05* | 4.224e-05* | 0/6 | 4.224e-05 |
| `json_field` | `newline` | 4.683e-05* | 5.641e-06* | 5.725e-06* | 3.711e-05* | 2.864e-04* | 1.011e-03* | 0/6 | 4.683e-05 |
| `json_field` | `long` | 1.155e-02* | 3.480e-03* | 6.313e-06* | 1.102e-04* | 6.880e-02* | 8.491e-02* | 0/6 | 1.155e-02 |
| `two_step_shipped` | `bare` | 1.947e-10* | 3.193e-13* | 1.148e-09* | 1.622e-07* | 1.793e-08* | 5.536e-08* | 0/6 | 1.793e-08 |
| `two_step_shipped` | `space` | 1.616e-09* | 2.151e-12* | 6.602e-07* | 2.801e-06* | 1.529e-06* | 5.595e-05* | 0/6 | 1.529e-06 |
| `two_step_shipped` | `caps` | 9.546e-11* | 2.452e-13* | 1.148e-09* | 1.622e-07* | 9.046e-10* | 9.331e-08* | 0/6 | 1.148e-09 |
| `two_step_shipped` | `newline` | 1.000e+00 | 3.296e-05* | 1.000e+00 | 6.453e-03* | 3.667e-06* | 6.817e-06* | 2/6 | 6.453e-03 |
| `two_step_shipped` | `long` | 1.947e-10* | 3.193e-13* | 3.347e-11* | 8.612e-13* | 1.793e-08* | 5.536e-08* | 0/6 | 1.947e-10 |
| `two_step_answer_is` | `bare` | 6.650e-11* | 1.572e-11* | 3.678e-06* | 1.002e-07* | 8.378e-12* | 5.103e-12* | 0/6 | 6.650e-11 |
| `two_step_answer_is` | `space` | 1.775e-09* | 2.638e-10* | 3.174e-06* | 3.601e-07* | 1.130e-09* | 2.056e-09* | 0/6 | 2.056e-09 |
| `two_step_answer_is` | `caps` | 7.481e-11* | 1.749e-11* | 3.678e-06* | 1.002e-07* | 6.818e-11* | 7.229e-11* | 0/6 | 7.481e-11 |
| `two_step_answer_is` | `newline` | 1.000e+00 | 1.000e+00 | 2.814e-01 | 3.194e-05* | 3.529e-01 | 1.558e-01 | 5/6 | 3.529e-01 |
| `two_step_answer_is` | `long` | 6.650e-11* | 1.572e-11* | 6.821e-10* | 7.932e-07* | 8.378e-12* | 5.103e-12* | 0/6 | 6.650e-11 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 2.196e-06 | 1.251e-06 | 0/6 | 6/6 | — |
| `answer_is` | cue | 2.766e-03 | 4.544e-06 | 0/6 | 6/6 | — |
| `answer_colon` | cue | 3.791e-03 | 1.491e-07 | 0/6 | 6/6 | — |
| `wybieram_pl` | cue | 8.763e-02 | 6.823e-06 | 1/6 | 6/6 | — |
| `json_field` | cue | 0.0283 | 0.0115 | 0/6 | 6/6 | — |
| `two_step_shipped` | advanced | 3.947e-08 | 1.793e-08 | 0/6 | 1/6 | content |
| `two_step_answer_is` | advanced | 6.297e-07 | 6.650e-11 | 0/6 | 1/6 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 6 | 1 | 0.167 | 0.030–0.564 | 6/6 | 0.0000 | 5/6 |
| `two_step_shipped=bare` | 6 | 3 | 0.500 | 0.188–0.812 | 6/6 | 0.0000 | 6/6 |
| `answer_is=bare` | 6 | 4 | 0.667 | 0.300–0.903 | 6/6 | 0.0000 | 5/6 |

