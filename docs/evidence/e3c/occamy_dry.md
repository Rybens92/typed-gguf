# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   501.91 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 1
sched_reserve: reserve took 79.70 ms, sched copies = 1`
- dev items: c01, c02 (choice 2) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T15:58:15Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 491.5 s (one model load 67671 ms, one context per item, all shapes of one item in one decode batch)

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

## 3. Coverage per item, shape × label variant

| shape | label | c01 | c02 | above floor | median |
|---|---|---|---|---|---|
| `shipped` | `bare` | 5.262e-08* | 7.421e-07* | 0/2 | 7.421e-07 |
| `shipped` | `space` | 3.417e-09* | 8.893e-09* | 0/2 | 8.893e-09 |
| `shipped` | `caps` | 4.828e-09* | 1.695e-08* | 0/2 | 1.695e-08 |
| `shipped` | `newline` | 5.188e-08* | 4.625e-09* | 0/2 | 5.188e-08 |
| `shipped` | `long` | 5.262e-08* | 7.421e-07* | 0/2 | 7.421e-07 |
| `answer_is` | `bare` | 4.544e-06* | 4.040e-07* | 0/2 | 4.544e-06 |
| `answer_is` | `space` | 2.867e-05* | 2.525e-06* | 0/2 | 2.867e-05 |
| `answer_is` | `caps` | 3.320e-07* | 8.865e-08* | 0/2 | 3.320e-07 |
| `answer_is` | `newline` | 2.014e-08* | 1.497e-08* | 0/2 | 2.014e-08 |
| `answer_is` | `long` | 4.544e-06* | 4.040e-07* | 0/2 | 4.544e-06 |
| `answer_colon` | `bare` | 1.491e-07* | 6.845e-08* | 0/2 | 1.491e-07 |
| `answer_colon` | `space` | 1.694e-07* | 2.245e-07* | 0/2 | 2.245e-07 |
| `answer_colon` | `caps` | 8.922e-08* | 1.516e-08* | 0/2 | 8.922e-08 |
| `answer_colon` | `newline` | 1.206e-08* | 3.303e-09* | 0/2 | 1.206e-08 |
| `answer_colon` | `long` | 1.491e-07* | 6.845e-08* | 0/2 | 1.491e-07 |
| `wybieram_pl` | `bare` | 6.823e-06* | 1.722e-08* | 0/2 | 6.823e-06 |
| `wybieram_pl` | `space` | 7.856e-05* | 5.150e-07* | 0/2 | 7.856e-05 |
| `wybieram_pl` | `caps` | 1.751e-07* | 1.460e-09* | 0/2 | 1.751e-07 |
| `wybieram_pl` | `newline` | 1.670e-08* | 1.473e-08* | 0/2 | 1.670e-08 |
| `wybieram_pl` | `long` | 6.823e-06* | 1.722e-08* | 0/2 | 6.823e-06 |
| `json_field` | `bare` | 0.0115* | 0.0035* | 0/2 | 0.0115 |
| `json_field` | `space` | 3.367e-05* | 6.975e-06* | 0/2 | 3.367e-05 |
| `json_field` | `caps` | 2.397e-05* | 6.029e-06* | 0/2 | 2.397e-05 |
| `json_field` | `newline` | 4.683e-05* | 5.641e-06* | 0/2 | 4.683e-05 |
| `json_field` | `long` | 0.0115* | 0.0035* | 0/2 | 0.0115 |
| `two_step_shipped` | `bare` | 1.947e-10* | 3.193e-13* | 0/2 | 1.947e-10 |
| `two_step_shipped` | `space` | 1.616e-09* | 2.151e-12* | 0/2 | 1.616e-09 |
| `two_step_shipped` | `caps` | 9.546e-11* | 2.452e-13* | 0/2 | 9.546e-11 |
| `two_step_shipped` | `newline` | 1.000e+00 | 3.296e-05* | 1/2 | 1.000e+00 |
| `two_step_shipped` | `long` | 1.947e-10* | 3.193e-13* | 0/2 | 1.947e-10 |
| `two_step_answer_is` | `bare` | 6.650e-11* | 1.572e-11* | 0/2 | 6.650e-11 |
| `two_step_answer_is` | `space` | 1.775e-09* | 2.638e-10* | 0/2 | 1.775e-09 |
| `two_step_answer_is` | `caps` | 7.481e-11* | 1.749e-11* | 0/2 | 7.481e-11 |
| `two_step_answer_is` | `newline` | 1.0000 | 1.0000 | 2/2 | 1.0000 |
| `two_step_answer_is` | `long` | 6.650e-11* | 1.572e-11* | 0/2 | 6.650e-11 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 3.974e-07 | 7.421e-07 | 0/2 | 2/2 | — |
| `answer_is` | cue | 2.474e-06 | 4.544e-06 | 0/2 | 2/2 | — |
| `answer_colon` | cue | 1.088e-07 | 1.491e-07 | 0/2 | 2/2 | — |
| `wybieram_pl` | cue | 3.420e-06 | 6.823e-06 | 0/2 | 2/2 | — |
| `json_field` | cue | 0.0075 | 0.0115 | 0/2 | 2/2 | — |
| `two_step_shipped` | advanced | 9.749e-11 | 1.947e-10 | 0/2 | 0/2 | content |
| `two_step_answer_is` | advanced | 4.111e-11 | 6.650e-11 | 0/2 | 0/2 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 2 | 0 | 0.000 | 0.000–0.658 | 2/2 | 0.0000 | 1/2 |
| `two_step_shipped=bare` | 2 | 1 | 0.500 | 0.095–0.905 | 2/2 | 0.0000 | 2/2 |
| `answer_is=bare` | 2 | 1 | 0.500 | 0.095–0.905 | 2/2 | 0.0000 | 1/2 |

