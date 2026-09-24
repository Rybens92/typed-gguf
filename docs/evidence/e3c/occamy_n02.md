# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   501.52 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 1
sched_reserve: reserve took 15.16 ms, sched copies = 1`
- dev items: n02 (noul 1) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T16:42:44Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 202.9 s (one model load 80500 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
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
| n02 | `shipped` | 0.0000 | 0.9999 | 0.0000 | 3.726e-06 |
| n02 | `answer_is` | 0.0000 | 0.9983 | 0.0000 | 5.486e-09 |
| n02 | `answer_colon` | 0.0000 | 0.9938 | 0.0000 | 4.178e-08 |
| n02 | `wybieram_pl` | 0.0000 | 0.9978 | 0.0000 | 4.279e-08 |
| n02 | `json_field` | 0.0000 | 0.9130 | 0.0000 | 8.491e-02 |
| n02 | `two_step_shipped` | 0.0000 | 0.9337 | 0.0000 | 5.536e-08 |
| n02 | `two_step_answer_is` | 0.0000 | 0.0000 | 0.0000 | 5.103e-12 |

## 3. Coverage per item, shape × label variant

| shape | label | n02 | above floor | median |
|---|---|---|---|---|
| `shipped` | `bare` | 3.726e-06* | 0/1 | 3.726e-06 |
| `shipped` | `space` | 5.221e-08* | 0/1 | 5.221e-08 |
| `shipped` | `caps` | 2.837e-08* | 0/1 | 2.837e-08 |
| `shipped` | `newline` | 3.893e-07* | 0/1 | 3.893e-07 |
| `shipped` | `long` | 3.726e-06* | 0/1 | 3.726e-06 |
| `answer_is` | `bare` | 5.486e-09* | 0/1 | 5.486e-09 |
| `answer_is` | `space` | 1.444e-06* | 0/1 | 1.444e-06 |
| `answer_is` | `caps` | 8.178e-10* | 0/1 | 8.178e-10 |
| `answer_is` | `newline` | 8.051e-09* | 0/1 | 8.051e-09 |
| `answer_is` | `long` | 5.486e-09* | 0/1 | 5.486e-09 |
| `answer_colon` | `bare` | 4.178e-08* | 0/1 | 4.178e-08 |
| `answer_colon` | `space` | 1.432e-05* | 0/1 | 1.432e-05 |
| `answer_colon` | `caps` | 8.010e-09* | 0/1 | 8.010e-09 |
| `answer_colon` | `newline` | 4.840e-08* | 0/1 | 4.840e-08 |
| `answer_colon` | `long` | 4.178e-08* | 0/1 | 4.178e-08 |
| `wybieram_pl` | `bare` | 4.279e-08* | 0/1 | 4.279e-08 |
| `wybieram_pl` | `space` | 9.635e-06* | 0/1 | 9.635e-06 |
| `wybieram_pl` | `caps` | 1.087e-08* | 0/1 | 1.087e-08 |
| `wybieram_pl` | `newline` | 1.307e-08* | 0/1 | 1.307e-08 |
| `wybieram_pl` | `long` | 4.279e-08* | 0/1 | 4.279e-08 |
| `json_field` | `bare` | 0.0849* | 0/1 | 0.0849 |
| `json_field` | `space` | 0.0001* | 0/1 | 0.0001 |
| `json_field` | `caps` | 4.224e-05* | 0/1 | 4.224e-05 |
| `json_field` | `newline` | 0.0010* | 0/1 | 0.0010 |
| `json_field` | `long` | 0.0849* | 0/1 | 0.0849 |
| `two_step_shipped` | `bare` | 5.536e-08* | 0/1 | 5.536e-08 |
| `two_step_shipped` | `space` | 0.0001* | 0/1 | 0.0001 |
| `two_step_shipped` | `caps` | 9.331e-08* | 0/1 | 9.331e-08 |
| `two_step_shipped` | `newline` | 6.817e-06* | 0/1 | 6.817e-06 |
| `two_step_shipped` | `long` | 5.536e-08* | 0/1 | 5.536e-08 |
| `two_step_answer_is` | `bare` | 5.103e-12* | 0/1 | 5.103e-12 |
| `two_step_answer_is` | `space` | 2.056e-09* | 0/1 | 2.056e-09 |
| `two_step_answer_is` | `caps` | 7.229e-11* | 0/1 | 7.229e-11 |
| `two_step_answer_is` | `newline` | 0.1558 | 1/1 | 0.1558 |
| `two_step_answer_is` | `long` | 5.103e-12* | 0/1 | 5.103e-12 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 3.726e-06 | 3.726e-06 | 0/1 | 1/1 | — |
| `answer_is` | cue | 5.486e-09 | 5.486e-09 | 0/1 | 1/1 | — |
| `answer_colon` | cue | 4.178e-08 | 4.178e-08 | 0/1 | 1/1 | — |
| `wybieram_pl` | cue | 4.279e-08 | 4.279e-08 | 0/1 | 1/1 | — |
| `json_field` | cue | 0.0849 | 0.0849 | 0/1 | 1/1 | — |
| `two_step_shipped` | advanced | 5.536e-08 | 5.536e-08 | 0/1 | 1/1 | content |
| `two_step_answer_is` | advanced | 5.103e-12 | 5.103e-12 | 0/1 | 0/1 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0000 | 1/1 |
| `two_step_shipped=bare` | 1 | 0 | 0.000 | 0.000–0.793 | 1/1 | 0.0000 | 1/1 |
| `answer_is=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |

