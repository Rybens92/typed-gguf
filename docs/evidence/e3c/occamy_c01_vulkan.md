# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `vulkan` (explicit) · threads 4 · `--gpu-layers` requested 7
- placement used: `{"note": "fit plan: 7 layer(s) offloaded, kv_type=auto", "n_gpu_layers": 7, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `ulkan_Host compute buffer size =   120.55 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 744 (with bs=512), 86 (with bs=1)
sched_reserve: reserve took 131.16 ms, sched copies = 1`
- dev items: c01 (choice 1) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T16:51:41Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 270.1 s (one model load 18104 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| c01 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| c01 | `answer_is` | cue | `<|im_end|>` | 0.9997 | 0.9997 | W_CUE_REFUSED |
| c01 | `answer_colon` | cue | `<|im_end|>` | 0.9914 | 0.9914 | W_CUE_REFUSED |
| c01 | `wybieram_pl` | cue | `<|im_end|>` | 0.9995 | 0.9995 | W_CUE_REFUSED |
| c01 | `json_field` | cue | `<|im_end|>` | 0.9922 | 0.9922 | W_CUE_REFUSED |
| c01 | `two_step_shipped` | advanced | `
` | 0.9582 | 0.0000 | ok |
| c01 | `two_step_answer_is` | advanced | `
` | 0.8917 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape | `<|endoftext|>` | `<|im_end|>` | `eos` | `bare` coverage |
|---|---|---|---|---|---|
| c01 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 6.502e-08 |
| c01 | `answer_is` | 0.0000 | 0.9997 | 0.0000 | 1.410e-05 |
| c01 | `answer_colon` | 0.0000 | 0.9914 | 0.0000 | 1.869e-07 |
| c01 | `wybieram_pl` | 0.0000 | 0.9995 | 0.0000 | 2.257e-05 |
| c01 | `json_field` | 0.0000 | 0.9922 | 0.0000 | 7.334e-03 |
| c01 | `two_step_shipped` | 0.0000 | 0.0000 | 0.0000 | 5.359e-13 |
| c01 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 3.283e-11 |

## 3. Coverage per item, shape × label variant

| shape | label | c01 | above floor | median |
|---|---|---|---|---|
| `shipped` | `bare` | 6.502e-08* | 0/1 | 6.502e-08 |
| `shipped` | `space` | 4.765e-09* | 0/1 | 4.765e-09 |
| `shipped` | `caps` | 5.983e-09* | 0/1 | 5.983e-09 |
| `shipped` | `newline` | 4.493e-08* | 0/1 | 4.493e-08 |
| `shipped` | `long` | 6.502e-08* | 0/1 | 6.502e-08 |
| `answer_is` | `bare` | 1.410e-05* | 0/1 | 1.410e-05 |
| `answer_is` | `space` | 3.199e-05* | 0/1 | 3.199e-05 |
| `answer_is` | `caps` | 3.262e-07* | 0/1 | 3.262e-07 |
| `answer_is` | `newline` | 1.200e-08* | 0/1 | 1.200e-08 |
| `answer_is` | `long` | 1.410e-05* | 0/1 | 1.410e-05 |
| `answer_colon` | `bare` | 1.869e-07* | 0/1 | 1.869e-07 |
| `answer_colon` | `space` | 2.794e-07* | 0/1 | 2.794e-07 |
| `answer_colon` | `caps` | 1.293e-07* | 0/1 | 1.293e-07 |
| `answer_colon` | `newline` | 2.756e-08* | 0/1 | 2.756e-08 |
| `answer_colon` | `long` | 1.869e-07* | 0/1 | 1.869e-07 |
| `wybieram_pl` | `bare` | 2.257e-05* | 0/1 | 2.257e-05 |
| `wybieram_pl` | `space` | 0.0002* | 0/1 | 0.0002 |
| `wybieram_pl` | `caps` | 5.073e-07* | 0/1 | 5.073e-07 |
| `wybieram_pl` | `newline` | 1.696e-08* | 0/1 | 1.696e-08 |
| `wybieram_pl` | `long` | 2.257e-05* | 0/1 | 2.257e-05 |
| `json_field` | `bare` | 0.0073* | 0/1 | 0.0073 |
| `json_field` | `space` | 2.004e-05* | 0/1 | 2.004e-05 |
| `json_field` | `caps` | 9.550e-06* | 0/1 | 9.550e-06 |
| `json_field` | `newline` | 4.226e-05* | 0/1 | 4.226e-05 |
| `json_field` | `long` | 0.0073* | 0/1 | 0.0073 |
| `two_step_shipped` | `bare` | 5.359e-13* | 0/1 | 5.359e-13 |
| `two_step_shipped` | `space` | 3.992e-12* | 0/1 | 3.992e-12 |
| `two_step_shipped` | `caps` | 1.925e-13* | 0/1 | 1.925e-13 |
| `two_step_shipped` | `newline` | 1.0000 | 1/1 | 1.0000 |
| `two_step_shipped` | `long` | 5.359e-13* | 0/1 | 5.359e-13 |
| `two_step_answer_is` | `bare` | 3.283e-11* | 0/1 | 3.283e-11 |
| `two_step_answer_is` | `space` | 5.490e-10* | 0/1 | 5.490e-10 |
| `two_step_answer_is` | `caps` | 5.378e-11* | 0/1 | 5.378e-11 |
| `two_step_answer_is` | `newline` | 1.0000 | 1/1 | 1.0000 |
| `two_step_answer_is` | `long` | 3.283e-11* | 0/1 | 3.283e-11 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 6.502e-08 | 6.502e-08 | 0/1 | 1/1 | — |
| `answer_is` | cue | 1.410e-05 | 1.410e-05 | 0/1 | 1/1 | — |
| `answer_colon` | cue | 1.869e-07 | 1.869e-07 | 0/1 | 1/1 | — |
| `wybieram_pl` | cue | 2.257e-05 | 2.257e-05 | 0/1 | 1/1 | — |
| `json_field` | cue | 0.0073 | 0.0073 | 0/1 | 1/1 | — |
| `two_step_shipped` | advanced | 5.359e-13 | 5.359e-13 | 0/1 | 0/1 | content |
| `two_step_answer_is` | advanced | 3.283e-11 | 3.283e-11 | 0/1 | 0/1 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0000 | 1/1 |
| `two_step_shipped=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |
| `answer_is=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |

