# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   501.27 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 1
sched_reserve: reserve took 99.39 ms, sched copies = 1`
- dev items: s01 (score 1) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T16:11:21Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 248.4 s (one model load 89664 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| s01 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| s01 | `answer_is` | cue | `<|im_end|>` | 0.9963 | 0.9963 | W_CUE_REFUSED |
| s01 | `answer_colon` | cue | `<|im_end|>` | 0.9981 | 0.9981 | W_CUE_REFUSED |
| s01 | `wybieram_pl` | cue | `<|im_end|>` | 0.9030 | 0.9030 | W_CUE_REFUSED |
| s01 | `json_field` | cue | `<|im_end|>` | 0.9997 | 0.9997 | W_CUE_REFUSED |
| s01 | `two_step_shipped` | advanced | `
` | 0.8896 | 0.0001 | ok |
| s01 | `two_step_answer_is` | advanced | `

` | 0.9436 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape | `<|endoftext|>` | `<|im_end|>` | `eos` | `bare` coverage |
|---|---|---|---|---|---|
| s01 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 1.251e-06 |
| s01 | `answer_is` | 0.0000 | 0.9963 | 0.0000 | 1.889e-03 |
| s01 | `answer_colon` | 0.0000 | 0.9981 | 0.0000 | 1.353e-03 |
| s01 | `wybieram_pl` | 0.0000 | 0.9030 | 0.0000 | 9.626e-02 |
| s01 | `json_field` | 0.0000 | 0.9997 | 0.0000 | 1.839e-04 |
| s01 | `two_step_shipped` | 0.0000 | 0.0001 | 0.0000 | 1.148e-09 |
| s01 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 3.678e-06 |

## 3. Coverage per item, shape × label variant

| shape | label | s01 | above floor | median |
|---|---|---|---|---|
| `shipped` | `bare` | 1.251e-06* | 0/1 | 1.251e-06 |
| `shipped` | `space` | 7.894e-10* | 0/1 | 7.894e-10 |
| `shipped` | `caps` | 1.251e-06* | 0/1 | 1.251e-06 |
| `shipped` | `newline` | 3.461e-08* | 0/1 | 3.461e-08 |
| `shipped` | `long` | 1.333e-09* | 0/1 | 1.333e-09 |
| `answer_is` | `bare` | 0.0019* | 0/1 | 0.0019 |
| `answer_is` | `space` | 2.024e-07* | 0/1 | 2.024e-07 |
| `answer_is` | `caps` | 0.0019* | 0/1 | 0.0019 |
| `answer_is` | `newline` | 3.863e-09* | 0/1 | 3.863e-09 |
| `answer_is` | `long` | 1.376e-08* | 0/1 | 1.376e-08 |
| `answer_colon` | `bare` | 0.0014* | 0/1 | 0.0014 |
| `answer_colon` | `space` | 5.269e-07* | 0/1 | 5.269e-07 |
| `answer_colon` | `caps` | 0.0014* | 0/1 | 0.0014 |
| `answer_colon` | `newline` | 3.694e-08* | 0/1 | 3.694e-08 |
| `answer_colon` | `long` | 5.384e-09* | 0/1 | 5.384e-09 |
| `wybieram_pl` | `bare` | 0.0963* | 0/1 | 0.0963 |
| `wybieram_pl` | `space` | 0.0006* | 0/1 | 0.0006 |
| `wybieram_pl` | `caps` | 0.0963* | 0/1 | 0.0963 |
| `wybieram_pl` | `newline` | 9.917e-08* | 0/1 | 9.917e-08 |
| `wybieram_pl` | `long` | 7.545e-08* | 0/1 | 7.545e-08 |
| `json_field` | `bare` | 0.0002* | 0/1 | 0.0002 |
| `json_field` | `space` | 1.576e-06* | 0/1 | 1.576e-06 |
| `json_field` | `caps` | 0.0002* | 0/1 | 0.0002 |
| `json_field` | `newline` | 5.725e-06* | 0/1 | 5.725e-06 |
| `json_field` | `long` | 6.313e-06* | 0/1 | 6.313e-06 |
| `two_step_shipped` | `bare` | 1.148e-09* | 0/1 | 1.148e-09 |
| `two_step_shipped` | `space` | 6.602e-07* | 0/1 | 6.602e-07 |
| `two_step_shipped` | `caps` | 1.148e-09* | 0/1 | 1.148e-09 |
| `two_step_shipped` | `newline` | 1.0000 | 1/1 | 1.0000 |
| `two_step_shipped` | `long` | 3.347e-11* | 0/1 | 3.347e-11 |
| `two_step_answer_is` | `bare` | 3.678e-06* | 0/1 | 3.678e-06 |
| `two_step_answer_is` | `space` | 3.174e-06* | 0/1 | 3.174e-06 |
| `two_step_answer_is` | `caps` | 3.678e-06* | 0/1 | 3.678e-06 |
| `two_step_answer_is` | `newline` | 0.2814 | 1/1 | 0.2814 |
| `two_step_answer_is` | `long` | 6.821e-10* | 0/1 | 6.821e-10 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 1.251e-06 | 1.251e-06 | 0/1 | 1/1 | — |
| `answer_is` | cue | 0.0019 | 0.0019 | 0/1 | 1/1 | — |
| `answer_colon` | cue | 0.0014 | 0.0014 | 0/1 | 1/1 | — |
| `wybieram_pl` | cue | 0.0963 | 0.0963 | 0/1 | 1/1 | — |
| `json_field` | cue | 0.0002 | 0.0002 | 0/1 | 1/1 | — |
| `two_step_shipped` | advanced | 1.148e-09 | 1.148e-09 | 0/1 | 0/1 | content |
| `two_step_answer_is` | advanced | 3.678e-06 | 3.678e-06 | 0/1 | 0/1 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0000 | 1/1 |
| `two_step_shipped=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0000 | 1/1 |
| `answer_is=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0019 | 1/1 |

