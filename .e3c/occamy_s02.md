# E3c — cue shapes that put the readout mid-answer (`qwen35moe`)

- card `t_6c119626` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   501.91 MiB
sched_reserve: graph nodes  = 3787
sched_reserve: graph splits = 1
sched_reserve: reserve took 16.55 ms, sched copies = 1`
- dev items: s02 (score 1) · label variants: `bare`, `space`, `caps`, `newline`, `long` · generated 2026-09-19T16:16:20Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `answer_is`, `answer_colon`, `wybieram_pl`, `json_field`, `two_step_shipped`, `two_step_answer_is`
- wall: 317.4 s (one model load 107005 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| s02 | `shipped` | cue | `<|im_end|>` | 1.0000 | 1.0000 | W_CUE_REFUSED |
| s02 | `answer_is` | cue | `<|im_end|>` | 0.9841 | 0.9841 | W_CUE_REFUSED |
| s02 | `answer_colon` | cue | `<|im_end|>` | 0.9783 | 0.9783 | W_CUE_REFUSED |
| s02 | `wybieram_pl` | cue | `<|im_end|>` | 0.5643 | 0.5643 | W_CUE_REFUSED |
| s02 | `json_field` | cue | `<|im_end|>` | 0.9990 | 0.9990 | W_CUE_REFUSED |
| s02 | `two_step_shipped` | advanced | `

` | 0.9981 | 0.0003 | ok |
| s02 | `two_step_answer_is` | advanced | `<|im_end|>` | 0.9968 | 0.9968 | W_CUE_REFUSED |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape | `<|endoftext|>` | `<|im_end|>` | `eos` | `bare` coverage |
|---|---|---|---|---|---|
| s02 | `shipped` | 0.0000 | 1.0000 | 0.0000 | 6.154e-06 |
| s02 | `answer_is` | 0.0000 | 0.9841 | 0.0000 | 1.470e-02 |
| s02 | `answer_colon` | 0.0000 | 0.9783 | 0.0000 | 2.139e-02 |
| s02 | `wybieram_pl` | 0.0000 | 0.5643 | 0.0000 | 4.295e-01 |
| s02 | `json_field` | 0.0000 | 0.9990 | 0.0000 | 7.430e-04 |
| s02 | `two_step_shipped` | 0.0000 | 0.0003 | 0.0000 | 1.622e-07 |
| s02 | `two_step_answer_is` | 0.0000 | 0.9968 | 0.0000 | 1.002e-07 |

## 3. Coverage per item, shape × label variant

| shape | label | s02 | above floor | median |
|---|---|---|---|---|
| `shipped` | `bare` | 6.154e-06* | 0/1 | 6.154e-06 |
| `shipped` | `space` | 7.201e-10* | 0/1 | 7.201e-10 |
| `shipped` | `caps` | 6.154e-06* | 0/1 | 6.154e-06 |
| `shipped` | `newline` | 7.377e-09* | 0/1 | 7.377e-09 |
| `shipped` | `long` | 1.133e-10* | 0/1 | 1.133e-10 |
| `answer_is` | `bare` | 0.0147* | 0/1 | 0.0147 |
| `answer_is` | `space` | 8.729e-07* | 0/1 | 8.729e-07 |
| `answer_is` | `caps` | 0.0147* | 0/1 | 0.0147 |
| `answer_is` | `newline` | 1.583e-08* | 0/1 | 1.583e-08 |
| `answer_is` | `long` | 9.534e-09* | 0/1 | 9.534e-09 |
| `answer_colon` | `bare` | 0.0214* | 0/1 | 0.0214 |
| `answer_colon` | `space` | 9.211e-07* | 0/1 | 9.211e-07 |
| `answer_colon` | `caps` | 0.0214* | 0/1 | 0.0214 |
| `answer_colon` | `newline` | 4.064e-08* | 0/1 | 4.064e-08 |
| `answer_colon` | `long` | 9.338e-09* | 0/1 | 9.338e-09 |
| `wybieram_pl` | `bare` | 0.4295 | 1/1 | 0.4295 |
| `wybieram_pl` | `space` | 0.0003* | 0/1 | 0.0003 |
| `wybieram_pl` | `caps` | 0.4295 | 1/1 | 0.4295 |
| `wybieram_pl` | `newline` | 2.776e-07* | 0/1 | 2.776e-07 |
| `wybieram_pl` | `long` | 3.905e-07* | 0/1 | 3.905e-07 |
| `json_field` | `bare` | 0.0007* | 0/1 | 0.0007 |
| `json_field` | `space` | 2.882e-06* | 0/1 | 2.882e-06 |
| `json_field` | `caps` | 0.0007* | 0/1 | 0.0007 |
| `json_field` | `newline` | 3.711e-05* | 0/1 | 3.711e-05 |
| `json_field` | `long` | 0.0001* | 0/1 | 0.0001 |
| `two_step_shipped` | `bare` | 1.622e-07* | 0/1 | 1.622e-07 |
| `two_step_shipped` | `space` | 2.801e-06* | 0/1 | 2.801e-06 |
| `two_step_shipped` | `caps` | 1.622e-07* | 0/1 | 1.622e-07 |
| `two_step_shipped` | `newline` | 0.0065* | 0/1 | 0.0065 |
| `two_step_shipped` | `long` | 8.612e-13* | 0/1 | 8.612e-13 |
| `two_step_answer_is` | `bare` | 1.002e-07* | 0/1 | 1.002e-07 |
| `two_step_answer_is` | `space` | 3.601e-07* | 0/1 | 3.601e-07 |
| `two_step_answer_is` | `caps` | 1.002e-07* | 0/1 | 1.002e-07 |
| `two_step_answer_is` | `newline` | 3.194e-05* | 0/1 | 3.194e-05 |
| `two_step_answer_is` | `long` | 7.932e-07* | 0/1 | 7.932e-07 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 6.154e-06 | 6.154e-06 | 0/1 | 1/1 | — |
| `answer_is` | cue | 0.0147 | 0.0147 | 0/1 | 1/1 | — |
| `answer_colon` | cue | 0.0214 | 0.0214 | 0/1 | 1/1 | — |
| `wybieram_pl` | cue | 0.4295 | 0.4295 | 1/1 | 1/1 | — |
| `json_field` | cue | 0.0007 | 0.0007 | 0/1 | 1/1 | — |
| `two_step_shipped` | advanced | 1.622e-07 | 1.622e-07 | 0/1 | 0/1 | content |
| `two_step_answer_is` | advanced | 1.002e-07 | 1.002e-07 | 0/1 | 1/1 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

| policy | n | correct | agreement | 95% CI | low_mass | median coverage | ranked == coverage |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |
| `two_step_shipped=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0000 | 1/1 |
| `answer_is=bare` | 1 | 1 | 1.000 | 0.207–1.000 | 1/1 | 0.0147 | 1/1 |

