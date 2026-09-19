# E3b — `qwen35moe` label policy: candidate-mass coverage per variant

- card `t_6952f0dd` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend `vulkan` · threads 4 · `--gpu-layers` requested 7
- placement used: `{"note": "degraded after a backend allocation failure: 3 layer(s) offloaded, kv_type=f16", "n_gpu_layers": 3, "kv_type": "f16", "degraded": true, "attempts": ["n_gpu_layers=7 -> oom"], "warnings": ["W_BACKEND_OOM", "W_FIT_DOWNGRADE"]}`
- dev items: c01, c02, s01, s02, n01, n02 (choice 2, noul 2, score 2) · prefix variants: `shipped`, `kept` · generated 2026-09-19T13:51:15Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- wall: 1783.6 s (one model load 22198 ms, one context per item-prefix, one decode batch per prefix for all cue variants)

## 1. What the model puts at the cue (the row coverage is read from)

| item | cue | top tokens at the cue (full-vocab p) |
|---|---|---|
| c01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · ````` 0.0000 · `Answer` 0.0000 |
| c01 | `blank` | `<|im_end|>` 0.9999 · `</think>` 0.0000 · `<think>` 0.0000 · `<|im_start|>` 0.0000 |
| c01 | `explicit` | `<|im_end|>` 0.9996 · `Answer` 0.0003 · `</think>` 0.0000 · ````` 0.0000 |
| c02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `answer` 0.0000 |
| c02 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `technical` 0.0000 |
| c02 | `explicit` | `<|im_end|>` 0.9998 · `Answer` 0.0002 · `answer` 0.0000 · `</think>` 0.0000 |
| s01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `4` 0.0000 · `Answer` 0.0000 |
| s01 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `4` 0.0000 |
| s01 | `explicit` | `<|im_end|>` 0.9992 · `Answer` 0.0006 · `</think>` 0.0001 · `4` 0.0000 |
| s02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `3` 0.0000 · `Answer` 0.0000 |
| s02 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `<|im_start|>` 0.0000 |
| s02 | `explicit` | `<|im_end|>` 0.9992 · `Answer` 0.0007 · `</think>` 0.0000 · `<think>` 0.0000 |
| n01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `no` 0.0000 |
| n01 | `blank` | `<|im_end|>` 0.9999 · `<think>` 0.0001 · `</think>` 0.0000 · `Answer` 0.0000 |
| n01 | `explicit` | `<|im_end|>` 0.9976 · `Answer` 0.0020 · `no` 0.0001 · `</think>` 0.0001 |
| n02 | `shipped` | `<|im_end|>` 0.9999 · `Answer` 0.0000 · `</think>` 0.0000 · `no` 0.0000 |
| n02 | `blank` | `<|im_end|>` 0.9999 · `<think>` 0.0000 · `</think>` 0.0000 · `Answer` 0.0000 |
| n02 | `explicit` | `<|im_end|>` 0.9977 · `Answer` 0.0019 · `</think>` 0.0002 · `no` 0.0001 |

## 2. Coverage per item, cue × label variant

| cue | label | c01 | c02 | s01 | s02 | n01 | n02 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 8.068e-08* | 5.429e-07* | 7.318e-07* | 5.018e-06* | 1.522e-06* | 5.645e-06* | 0/6 | 1.522e-06 |
| `shipped` | `space` | 5.084e-09* | 7.350e-09* | 2.855e-10* | 1.433e-09* | 7.425e-09* | 6.495e-08* | 0/6 | 7.350e-09 |
| `shipped` | `caps` | 7.816e-09* | 1.029e-08* | 7.318e-07* | 5.018e-06* | 8.718e-09* | 3.691e-08* | 0/6 | 3.691e-08 |
| `shipped` | `newline` | 6.735e-08* | 1.882e-09* | 5.735e-09* | 9.914e-09* | 6.169e-08* | 1.907e-07* | 0/6 | 6.169e-08 |
| `shipped` | `long` | 8.068e-08* | 5.429e-07* | 5.532e-10* | 6.400e-11* | 1.522e-06* | 5.645e-06* | 0/6 | 5.429e-07 |
| `blank` | `bare` | 1.491e-06* | 3.184e-07* | 2.457e-07* | 6.754e-07* | 1.243e-06* | 8.350e-07* | 0/6 | 8.350e-07 |
| `blank` | `space` | 5.580e-08* | 1.187e-08* | 6.250e-11* | 3.475e-10* | 4.960e-09* | 2.792e-09* | 0/6 | 4.960e-09 |
| `blank` | `caps` | 1.558e-07* | 1.752e-08* | 2.457e-07* | 6.754e-07* | 1.903e-08* | 4.769e-09* | 0/6 | 1.558e-07 |
| `blank` | `newline` | 6.241e-10* | 3.487e-11* | 2.904e-10* | 4.510e-11* | 1.694e-09* | 1.066e-08* | 0/6 | 6.241e-10 |
| `blank` | `long` | 1.491e-06* | 3.184e-07* | 3.409e-10* | 2.104e-10* | 1.243e-06* | 8.350e-07* | 0/6 | 8.350e-07 |
| `explicit` | `bare` | 2.564e-07* | 1.316e-07* | 7.186e-06* | 9.869e-06* | 1.290e-04* | 5.504e-05* | 0/6 | 9.869e-06 |
| `explicit` | `space` | 2.357e-08* | 1.232e-08* | 3.514e-07* | 7.164e-08* | 2.519e-06* | 1.031e-06* | 0/6 | 3.514e-07 |
| `explicit` | `caps` | 4.697e-07* | 3.077e-08* | 7.186e-06* | 9.869e-06* | 2.364e-06* | 1.847e-06* | 0/6 | 2.364e-06 |
| `explicit` | `newline` | 2.423e-08* | 2.144e-08* | 1.255e-07* | 8.366e-09* | 1.976e-07* | 2.446e-07* | 0/6 | 1.255e-07 |
| `explicit` | `long` | 2.564e-07* | 1.316e-07* | 6.695e-09* | 6.480e-10* | 1.290e-04* | 5.504e-05* | 0/6 | 2.564e-07 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the cue (`readout.coverage_from_scale`), read from the cue row — so every label variant of one cue costs no extra forward pass.

## 3. Variant summary

| cue | label | mean coverage | median | above floor | low_mass share | best item | worst item |
|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 2.257e-06 | 1.522e-06 | 0/6 | 1.00 | n02 5.645e-06 | c01 8.068e-08 |
| `shipped` | `space` | 1.442e-08 | 7.350e-09 | 0/6 | 1.00 | n02 6.495e-08 | s01 2.855e-10 |
| `shipped` | `caps` | 9.689e-07 | 3.691e-08 | 0/6 | 1.00 | s02 5.018e-06 | c01 7.816e-09 |
| `shipped` | `newline` | 5.621e-08 | 6.169e-08 | 0/6 | 1.00 | n02 1.907e-07 | c02 1.882e-09 |
| `shipped` | `long` | 1.299e-06 | 5.429e-07 | 0/6 | 1.00 | n02 5.645e-06 | s02 6.400e-11 |
| `blank` | `bare` | 8.014e-07 | 8.350e-07 | 0/6 | 1.00 | c01 1.491e-06 | s01 2.457e-07 |
| `blank` | `space` | 1.264e-08 | 4.960e-09 | 0/6 | 1.00 | c01 5.580e-08 | s01 6.250e-11 |
| `blank` | `caps` | 1.864e-07 | 1.558e-07 | 0/6 | 1.00 | s02 6.754e-07 | n02 4.769e-09 |
| `blank` | `newline` | 2.225e-09 | 6.241e-10 | 0/6 | 1.00 | n02 1.066e-08 | c02 3.487e-11 |
| `blank` | `long` | 6.480e-07 | 8.350e-07 | 0/6 | 1.00 | c01 1.491e-06 | s02 2.104e-10 |
| `explicit` | `bare` | 3.359e-05 | 9.869e-06 | 0/6 | 1.00 | n01 1.290e-04 | c02 1.316e-07 |
| `explicit` | `space` | 6.682e-07 | 3.514e-07 | 0/6 | 1.00 | n01 2.519e-06 | c02 1.232e-08 |
| `explicit` | `caps` | 3.628e-06 | 2.364e-06 | 0/6 | 1.00 | s02 9.869e-06 | c02 3.077e-08 |
| `explicit` | `newline` | 1.036e-07 | 1.255e-07 | 0/6 | 1.00 | n02 2.446e-07 | s02 8.366e-09 |
| `explicit` | `long` | 3.074e-05 | 2.564e-07 | 0/6 | 1.00 | n01 1.290e-04 | s02 6.480e-10 |

## 4. The prefix control (shipped vs the template's own empty think block)

| item | prefix | prefix tokens | cue | top token (p) | `bare` coverage | `newline` coverage |
|---|---|---|---|---|---|---|
| c01 | `shipped` | 109 | `shipped` | `<|im_end|>` 1.0000 | 8.068e-08 | 6.735e-08 |
| c01 | `kept` | 113 | `shipped` | `<|im_end|>` 0.9988 | 3.646e-06 | 6.221e-06 |
| c02 | `shipped` | 99 | `shipped` | `<|im_end|>` 1.0000 | 5.429e-07 | 1.882e-09 |
| c02 | `kept` | 103 | `shipped` | `<|im_end|>` 0.9985 | 1.368e-05 | 4.613e-05 |

## 5. The ranked readout (the engine's decision under a policy)

| policy | n | correct | agreement | 95% CI | low_mass | median coverage |
|---|---|---|---|---|---|---|
| `shipped=bare` | 6 | 2 | 0.333 | 0.097–0.700 | 6/6 | 0.0000 |
| `shipped=newline` | 6 | 1 | 0.167 | 0.030–0.564 | 6/6 | 0.0000 |

## 6. Cross-check against `DecisionEngine` (shipped policy, `bare`)

| item | probe coverage | engine coverage | |Δ coverage| | probe winner | engine winner |
|---|---|---|---|---|---|
| c01 | 8.0684e-08 | 1.6435e-07 | 8.37e-08 | technical | support |

Worst |Δ coverage| over the cross-checked items: **8.37e-08** — the probe reads the same row with the same function; only the batch shape differs.

