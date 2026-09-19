# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   501.52 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 1
sched_reserve: reserve took 19.57 ms, sched copies = 1`
- dev items: n01 (noul 1) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T16:33:01Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 767.4 s (one model load 295179 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| n01 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| n01 | `answer_is` | cue | `<|im_end|>` | 0.9996 | 0.9996 | W_CUE_REFUSED |
| n01 | `answer_colon` | cue | `<|im_end|>` | 0.9994 | 0.9994 | W_CUE_REFUSED |
| n01 | `wybieram_pl` | cue | `<|im_end|>` | 0.9998 | 0.9998 | W_CUE_REFUSED |
| n01 | `json_field` | cue | `<|im_end|>` | 0.9301 | 0.9301 | W_CUE_REFUSED |
| n01 | `two_step_shipped` | advanced | `

` | 1.0000 | 0.0000 | ok |
| n01 | `two_step_answer_is` | advanced | `

` | 0.8226 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape | `<|endoftext|>` | `<|im_end|>` | `eos` | `bare` coverage |
|---|---|---|---|---|---|
| n01 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 1.250e-06 |
| n01 | `answer_is` | 0.0000 | 0.9996 | 0.0000 | 5.902e-09 |
| n01 | `answer_colon` | 0.0000 | 0.9994 | 0.0000 | 1.325e-08 |
| n01 | `wybieram_pl` | 0.0000 | 0.9998 | 0.0000 | 4.457e-09 |
| n01 | `json_field` | 0.0001 | 0.9301 | 0.0000 | 6.880e-02 |
| n01 | `two_step_shipped` | 0.0000 | 0.0000 | 0.0000 | 1.793e-08 |
| n01 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 8.378e-12 |

## 3. Coverage per item, shape × label variant

| shape | label | n01 | above floor | median |
|---|---|---|---|---|
| `shipped` | `bare` | 1.250e-06* | 0/1 | 1.250e-06 |
| `shipped` | `space` | 4.843e-09* | 0/1 | 4.843e-09 |
| `shipped` | `caps` | 1.707e-08* | 0/1 | 1.707e-08 |
| `shipped` | `newline` | 1.373e-08* | 0/1 | 1.373e-08 |
| `shipped` | `long` | 1.250e-06* | 0/1 | 1.250e-06 |
| `answer_is` | `bare` | 5.902e-09* | 0/1 | 5.902e-09 |
| `answer_is` | `space` | 1.917e-06* | 0/1 | 1.917e-06 |
| `answer_is` | `caps` | 7.168e-10* | 0/1 | 7.168e-10 |
| `answer_is` | `newline` | 3.024e-09* | 0/1 | 3.024e-09 |
| `answer_is` | `long` | 5.902e-09* | 0/1 | 5.902e-09 |
| `answer_colon` | `bare` | 1.325e-08* | 0/1 | 1.325e-08 |
| `answer_colon` | `space` | 1.168e-05* | 0/1 | 1.168e-05 |
| `answer_colon` | `caps` | 2.450e-09* | 0/1 | 2.450e-09 |
| `answer_colon` | `newline` | 6.777e-09* | 0/1 | 6.777e-09 |
| `answer_colon` | `long` | 1.325e-08* | 0/1 | 1.325e-08 |
| `wybieram_pl` | `bare` | 4.457e-09* | 0/1 | 4.457e-09 |
| `wybieram_pl` | `space` | 6.593e-07* | 0/1 | 6.593e-07 |
| `wybieram_pl` | `caps` | 3.658e-09* | 0/1 | 3.658e-09 |
| `wybieram_pl` | `newline` | 4.975e-09* | 0/1 | 4.975e-09 |
| `wybieram_pl` | `long` | 4.457e-09* | 0/1 | 4.457e-09 |
| `json_field` | `bare` | 0.0688* | 0/1 | 0.0688 |
| `json_field` | `space` | 0.0001* | 0/1 | 0.0001 |
| `json_field` | `caps` | 3.356e-05* | 0/1 | 3.356e-05 |
| `json_field` | `newline` | 0.0003* | 0/1 | 0.0003 |
| `json_field` | `long` | 0.0688* | 0/1 | 0.0688 |
| `two_step_shipped` | `bare` | 1.793e-08* | 0/1 | 1.793e-08 |
| `two_step_shipped` | `space` | 1.529e-06* | 0/1 | 1.529e-06 |
| `two_step_shipped` | `caps` | 9.046e-10* | 0/1 | 9.046e-10 |
| `two_step_shipped` | `newline` | 3.667e-06* | 0/1 | 3.667e-06 |
| `two_step_shipped` | `long` | 1.793e-08* | 0/1 | 1.793e-08 |
| `two_step_answer_is` | `bare` | 8.378e-12* | 0/1 | 8.378e-12 |
| `two_step_answer_is` | `space` | 1.130e-09* | 0/1 | 1.130e-09 |
| `two_step_answer_is` | `caps` | 6.818e-11* | 0/1 | 6.818e-11 |
| `two_step_answer_is` | `newline` | 0.3529 | 1/1 | 0.3529 |
| `two_step_answer_is` | `long` | 8.378e-12* | 0/1 | 8.378e-12 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 1.250e-06 | 1.250e-06 | 0/1 | 1/1 | — |
| `answer_is` | cue | 5.902e-09 | 5.902e-09 | 0/1 | 1/1 | — |
| `answer_colon` | cue | 1.325e-08 | 1.325e-08 | 0/1 | 1/1 | — |
| `wybieram_pl` | cue | 4.457e-09 | 4.457e-09 | 0/1 | 1/1 | — |
| `json_field` | cue | 0.0688 | 0.0688 | 0/1 | 1/1 | — |
| `two_step_shipped` | advanced | 1.793e-08 | 1.793e-08 | 0/1 | 0/1 | content |
| `two_step_answer_is` | advanced | 8.378e-12 | 8.378e-12 | 0/1 | 0/1 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0000 | 1/1 |
| `two_step_shipped=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |
| `answer_is=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |

