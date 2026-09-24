# E3b — `qwen35moe` label policy: candidate-mass coverage per variant

- card `t_6952f0dd` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend `vulkan` · threads 4 · `--gpu-layers` requested 7
- placement used: `{"note": "degraded after a backend allocation failure: 0 layer(s) offloaded, kv_type=f16", "n_gpu_layers": 0, "kv_type": "f16", "degraded": true, "attempts": ["n_gpu_layers=7 -> oom", "n_gpu_layers=3 -> oom"], "warnings": ["W_BACKEND_OOM", "W_FIT_DOWNGRADE"]}`
- dev items: c01, s01, n01, c02, s02, n02, c03, s03, n03, c04 (choice 4, noul 3, score 3) · prefix variants: `shipped` · generated 2026-09-19T14:24:53Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- wall: 942.6 s (one model load 22261 ms, one context per item-prefix, one decode batch per prefix for all cue variants)

## 1. What the model puts at the cue (the row coverage is read from)

| item | cue | top tokens at the cue (full-vocab p) |
|---|---|---|
| c01 | `shipped` | `<|im_end|>` 0.9999 · `</think>` 0.0000 · ````` 0.0000 · `Answer` 0.0000 |
| s01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `4` 0.0000 · `Answer` 0.0000 |
| n01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `no` 0.0000 |
| c02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `technical` 0.0000 |
| s02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `3` 0.0000 · `Answer` 0.0000 |
| n02 | `shipped` | `<|im_end|>` 1.0000 · `Answer` 0.0000 · `</think>` 0.0000 · `no` 0.0000 |
| c03 | `shipped` | `<|im_end|>` 0.9996 · `</think>` 0.0003 · ````` 0.0000 · `Answer` 0.0000 |
| s03 | `shipped` | `<|im_end|>` 1.0000 · `4` 0.0000 · `</think>` 0.0000 · `Answer` 0.0000 |
| n03 | `shipped` | `<|im_end|>` 0.9998 · `</think>` 0.0001 · `Answer` 0.0000 · ````` 0.0000 |
| c04 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · ````` 0.0000 |

## 2. Coverage per item, cue × label variant

| cue | label | c01 | s01 | n01 | c02 | s02 | n02 | c03 | s03 | n03 | c04 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0/10 | 0.0000 |
| `shipped` | `space` | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0/10 | 0.0000 |
| `shipped` | `caps` | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0/10 | 0.0000 |
| `shipped` | `newline` | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0/10 | 0.0000 |
| `shipped` | `long` | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0/10 | 0.0000 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the cue (`readout.coverage_from_scale`), read from the cue row — so every label variant of one cue costs no extra forward pass.

## 3. Variant summary

| cue | label | mean coverage | median | above floor | low_mass share | best item | worst item |
|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 0.0000 | 0.0000 | 0/10 | 1.00 | s02 0.0000 | c04 0.0000 |
| `shipped` | `space` | 0.0000 | 0.0000 | 0/10 | 1.00 | n02 0.0000 | s03 0.0000 |
| `shipped` | `caps` | 0.0000 | 0.0000 | 0/10 | 1.00 | s02 0.0000 | c03 0.0000 |
| `shipped` | `newline` | 0.0000 | 0.0000 | 0/10 | 1.00 | n02 0.0000 | c02 0.0000 |
| `shipped` | `long` | 0.0000 | 0.0000 | 0/10 | 1.00 | n03 0.0000 | s02 0.0000 |

## 4. The prefix control (shipped vs the template's own empty think block)

No `--extra-prefix` in this run: only the shipped prompt shape was measured.

## 5. The ranked readout (the engine's decision under a policy)

| policy | n | correct | agreement | 95% CI | low_mass | median coverage |
|---|---|---|---|---|---|---|
| `shipped=bare` | 10 | 2 | 0.200 | 0.057–0.510 | 10/10 | 0.0000 |
| `shipped=newline` | 10 | 3 | 0.300 | 0.108–0.603 | 10/10 | 0.0000 |

## 6. Cross-check against `DecisionEngine` (shipped policy, `bare`)

No `--cross-check N` in this run: the probe's shipped-policy numbers were not confirmed against `DecisionEngine`.

