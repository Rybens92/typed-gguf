# E3b — `qwen35moe` label policy: candidate-mass coverage per variant

- card `t_6952f0dd` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend `vulkan` · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- dev items: s04, n04, c05, s05, n05, c06, s06, n06, c07, s07 (choice 3, noul 3, score 4) · prefix variants: `shipped` · generated 2026-09-19T14:46:56Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- wall: 897.3 s (one model load 33975 ms, one context per item-prefix, one decode batch per prefix for all cue variants)

## 1. What the model puts at the cue (the row coverage is read from)

| item | cue | top tokens at the cue (full-vocab p) |
|---|---|---|
| s04 | `shipped` | `<|im_end|>` 1.0000 · `4` 0.0000 · `Answer` 0.0000 · `</think>` 0.0000 |
| n04 | `shipped` | `<|im_end|>` 1.0000 · `Answer` 0.0000 · `</think>` 0.0000 · `no` 0.0000 |
| c05 | `shipped` | `<|im_end|>` 0.9999 · `</think>` 0.0001 · ````` 0.0000 · `Answer` 0.0000 |
| s05 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `0` 0.0000 · `4` 0.0000 |
| n05 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `no` 0.0000 · ````` 0.0000 |
| c06 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · ````` 0.0000 |
| s06 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `<think>` 0.0000 |
| n06 | `shipped` | `<|im_end|>` 1.0000 · `Answer` 0.0000 · `no` 0.0000 · `</think>` 0.0000 |
| c07 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `answer` 0.0000 |
| s07 | `shipped` | `<|im_end|>` 1.0000 · `Answer` 0.0000 · `</think>` 0.0000 · `answer` 0.0000 |

## 2. Coverage per item, cue × label variant

| cue | label | s04 | n04 | c05 | s05 | n05 | c06 | s06 | n06 | c07 | s07 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 6.664e-06* | 8.431e-07* | 1.315e-07* | 4.718e-07* | 1.270e-05* | 2.648e-09* | 9.517e-07* | 4.362e-06* | 4.798e-09* | 4.178e-08* | 0/10 | 8.431e-07 |
| `shipped` | `space` | 3.014e-09* | 3.055e-09* | 1.087e-08* | 2.798e-09* | 1.380e-07* | 2.466e-10* | 9.862e-10* | 1.328e-08* | 2.570e-10* | 1.222e-09* | 0/10 | 3.014e-09 |
| `shipped` | `caps` | 6.664e-06* | 1.931e-09* | 1.968e-08* | 4.718e-07* | 7.791e-08* | 3.004e-09* | 9.517e-07* | 2.073e-08* | 7.505e-10* | 4.178e-08* | 0/10 | 4.178e-08 |
| `shipped` | `newline` | 1.796e-08* | 7.931e-09* | 4.116e-08* | 2.164e-08* | 1.978e-07* | 1.028e-08* | 2.185e-08* | 5.443e-08* | 3.941e-09* | 4.116e-08* | 0/10 | 2.185e-08 |
| `shipped` | `long` | 1.548e-09* | 8.431e-07* | 1.315e-07* | 2.214e-09* | 1.270e-05* | 2.648e-09* | 3.457e-10* | 4.362e-06* | 4.798e-09* | 8.125e-11* | 0/10 | 4.798e-09 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the cue (`readout.coverage_from_scale`), read from the cue row — so every label variant of one cue costs no extra forward pass.

## 3. Variant summary

| cue | label | mean coverage | median | above floor | low_mass share | best item | worst item |
|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 2.617e-06 | 8.431e-07 | 0/10 | 1.00 | n05 1.270e-05 | c06 2.648e-09 |
| `shipped` | `space` | 1.738e-08 | 3.014e-09 | 0/10 | 1.00 | n05 1.380e-07 | c06 2.466e-10 |
| `shipped` | `caps` | 8.253e-07 | 4.178e-08 | 0/10 | 1.00 | s04 6.664e-06 | c07 7.505e-10 |
| `shipped` | `newline` | 4.182e-08 | 2.185e-08 | 0/10 | 1.00 | n05 1.978e-07 | c07 3.941e-09 |
| `shipped` | `long` | 1.804e-06 | 4.798e-09 | 0/10 | 1.00 | n05 1.270e-05 | s07 8.125e-11 |

## 4. The prefix control (shipped vs the template's own empty think block)

No `--extra-prefix` in this run: only the shipped prompt shape was measured.

## 5. The ranked readout (the engine's decision under a policy)

| policy | n | correct | agreement | 95% CI | low_mass | median coverage |
|---|---|---|---|---|---|---|
| `shipped=bare` | 10 | 4 | 0.400 | 0.168–0.687 | 10/10 | 0.0000 |
| `shipped=newline` | 10 | 6 | 0.600 | 0.313–0.832 | 10/10 | 0.0000 |

## 6. Cross-check against `DecisionEngine` (shipped policy, `bare`)

No `--cross-check N` in this run: the probe's shipped-policy numbers were not confirmed against `DecisionEngine`.

