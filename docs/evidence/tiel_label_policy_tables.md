# E3b — `qwen35moe` label policy: candidate-mass coverage per variant

- card `t_6952f0dd` · model `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` (22,360,476,736 bytes, sha256 `9286a94c453c6a40…`)
- runtime `/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend `vulkan` · threads 4 · `--gpu-layers` requested 9
- placement used: `{"note": "fit plan: 9 layer(s) offloaded, kv_type=auto", "n_gpu_layers": 9, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- dev items: c01, c02, s01, s02, n01, n02 (choice 2, noul 2, score 2) · prefix variants: `shipped` · generated 2026-09-19T16:42:45Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- wall: 168.4 s (one model load 25879 ms, one context per item-prefix, one decode batch per prefix for all cue variants)

## 1. What the model puts at the cue (the row coverage is read from)

| item | cue | top tokens at the cue (full-vocab p) |
|---|---|---|
| c01 | `shipped` | `</think>` 0.5145 · `<|im_end|>` 0.4453 · `en` 0.0042 · `</` 0.0021 |
| c01 | `blank` | `<think>` 0.6813 · `<|im_end|>` 0.1567 · `</think>` 0.1365 · `<tool_call>` 0.0035 |
| c01 | `explicit` | `</think>` 0.8940 · `<|im_end|>` 0.0270 · `</` 0.0166 · `no` 0.0034 |
| c02 | `shipped` | `</think>` 0.7835 · `<|im_end|>` 0.2083 · `</` 0.0035 · ````` 0.0004 |
| c02 | `blank` | `<|im_end|>` 0.5939 · `</think>` 0.1751 · `<think>` 0.1354 · `</` 0.0098 |
| c02 | `explicit` | `<|im_end|>` 0.6027 · ````` 0.0775 · `</think>` 0.0464 · `</` 0.0201 |
| s01 | `shipped` | `<|im_end|>` 0.8340 · `</think>` 0.1453 · `<think>` 0.0022 · `</` 0.0011 |
| s01 | `blank` | `</think>` 0.4806 · `<|im_end|>` 0.3763 · `<think>` 0.1197 · `</` 0.0038 |
| s01 | `explicit` | `<|im_end|>` 0.5655 · `</think>` 0.1943 · `<think>` 0.1646 · `A` 0.0072 |
| s02 | `shipped` | `<|im_end|>` 0.5275 · `</think>` 0.4351 · ````` 0.0208 · `<think>` 0.0031 |
| s02 | `blank` | `</think>` 0.5926 · `<|im_end|>` 0.2292 · `<think>` 0.1686 · `<tool_call>` 0.0012 |
| s02 | `explicit` | `<|im_end|>` 0.3261 · `</think>` 0.2944 · `<think>` 0.1579 · ````` 0.0376 |
| n01 | `shipped` | `<|im_end|>` 0.9780 · `</think>` 0.0128 · `no` 0.0036 · `<think>` 0.0008 |
| n01 | `blank` | `<|im_end|>` 0.7739 · `</think>` 0.1836 · `<think>` 0.0341 · `<tool_call>` 0.0025 |
| n01 | `explicit` | `</think>` 0.7831 · `<|im_end|>` 0.1340 · `no` 0.0417 · `A` 0.0072 |
| n02 | `shipped` | `<|im_end|>` 0.8779 · `</think>` 0.0537 · `<think>` 0.0320 · `<tool_call>` 0.0073 |
| n02 | `blank` | `<think>` 0.9129 · `<|im_end|>` 0.0395 · `</think>` 0.0302 · `<tool_call>` 0.0061 |
| n02 | `explicit` | `</think>` 0.4906 · `<think>` 0.1582 · `<|im_end|>` 0.0319 · `wo` 0.0312 |

## 2. Coverage per item, cue × label variant

| cue | label | c01 | c02 | s01 | s02 | n01 | n02 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 4.088e-06* | 1.004e-06* | 7.721e-04* | 3.426e-04* | 3.651e-03* | 2.689e-03* | 0/6 | 7.721e-04 |
| `shipped` | `space` | 6.774e-07* | 2.858e-07* | 1.332e-06* | 4.964e-06* | 1.500e-05* | 3.750e-05* | 0/6 | 4.964e-06 |
| `shipped` | `caps` | 3.143e-07* | 2.859e-07* | 7.721e-04* | 3.426e-04* | 1.942e-06* | 3.945e-06* | 0/6 | 3.945e-06 |
| `shipped` | `newline` | 2.169e-05* | 8.953e-08* | 3.560e-06* | 2.223e-06* | 3.987e-07* | 2.374e-07* | 0/6 | 2.223e-06 |
| `shipped` | `long` | 4.088e-06* | 1.004e-06* | 4.088e-06* | 6.886e-07* | 3.651e-03* | 2.689e-03* | 0/6 | 4.088e-06 |
| `blank` | `bare` | 1.246e-06* | 1.115e-06* | 2.318e-04* | 6.543e-05* | 3.457e-04* | 6.676e-04* | 0/6 | 2.318e-04 |
| `blank` | `space` | 7.667e-08* | 5.412e-08* | 1.564e-06* | 2.355e-07* | 3.866e-06* | 2.609e-06* | 0/6 | 1.564e-06 |
| `blank` | `caps` | 1.821e-07* | 1.351e-06* | 2.318e-04* | 6.543e-05* | 6.115e-06* | 1.047e-06* | 0/6 | 6.115e-06 |
| `blank` | `newline` | 1.984e-06* | 1.612e-05* | 1.376e-05* | 1.128e-06* | 2.094e-07* | 1.622e-06* | 0/6 | 1.984e-06 |
| `blank` | `long` | 1.246e-06* | 1.115e-06* | 2.109e-06* | 3.014e-07* | 3.457e-04* | 6.676e-04* | 0/6 | 2.109e-06 |
| `explicit` | `bare` | 2.410e-05* | 3.783e-06* | 5.260e-04* | 2.303e-03* | 4.169e-02* | 2.991e-02* | 0/6 | 2.303e-03 |
| `explicit` | `space` | 7.186e-06* | 4.829e-06* | 2.841e-05* | 2.443e-04* | 3.327e-04* | 5.386e-04* | 0/6 | 2.443e-04 |
| `explicit` | `caps` | 1.536e-05* | 5.663e-06* | 5.260e-04* | 2.303e-03* | 1.440e-03* | 8.666e-04* | 0/6 | 8.666e-04 |
| `explicit` | `newline` | 1.948e-05* | 3.400e-06* | 6.137e-07* | 2.314e-05* | 3.561e-07* | 2.145e-06* | 0/6 | 3.400e-06 |
| `explicit` | `long` | 2.410e-05* | 3.783e-06* | 8.394e-06* | 1.427e-06* | 4.169e-02* | 2.991e-02* | 0/6 | 2.410e-05 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the cue (`readout.coverage_from_scale`), read from the cue row — so every label variant of one cue costs no extra forward pass.

## 3. Variant summary

| cue | label | mean coverage | median | above floor | low_mass share | best item | worst item |
|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 1.243e-03 | 7.721e-04 | 0/6 | 1.00 | n01 3.651e-03 | c02 1.004e-06 |
| `shipped` | `space` | 9.960e-06 | 4.964e-06 | 0/6 | 1.00 | n02 3.750e-05 | c02 2.858e-07 |
| `shipped` | `caps` | 1.869e-04 | 3.945e-06 | 0/6 | 1.00 | s01 7.721e-04 | c02 2.859e-07 |
| `shipped` | `newline` | 4.700e-06 | 2.223e-06 | 0/6 | 1.00 | c01 2.169e-05 | c02 8.953e-08 |
| `shipped` | `long` | 1.058e-03 | 4.088e-06 | 0/6 | 1.00 | n01 3.651e-03 | s02 6.886e-07 |
| `blank` | `bare` | 2.188e-04 | 2.318e-04 | 0/6 | 1.00 | n02 6.676e-04 | c02 1.115e-06 |
| `blank` | `space` | 1.401e-06 | 1.564e-06 | 0/6 | 1.00 | n01 3.866e-06 | c02 5.412e-08 |
| `blank` | `caps` | 5.098e-05 | 6.115e-06 | 0/6 | 1.00 | s01 2.318e-04 | c01 1.821e-07 |
| `blank` | `newline` | 5.804e-06 | 1.984e-06 | 0/6 | 1.00 | c02 1.612e-05 | n01 2.094e-07 |
| `blank` | `long` | 1.697e-04 | 2.109e-06 | 0/6 | 1.00 | n02 6.676e-04 | s02 3.014e-07 |
| `explicit` | `bare` | 1.241e-02 | 2.303e-03 | 0/6 | 1.00 | n01 4.169e-02 | c02 3.783e-06 |
| `explicit` | `space` | 1.927e-04 | 2.443e-04 | 0/6 | 1.00 | n02 5.386e-04 | c02 4.829e-06 |
| `explicit` | `caps` | 8.594e-04 | 8.666e-04 | 0/6 | 1.00 | s02 2.303e-03 | c02 5.663e-06 |
| `explicit` | `newline` | 8.189e-06 | 3.400e-06 | 0/6 | 1.00 | s02 2.314e-05 | n01 3.561e-07 |
| `explicit` | `long` | 1.194e-02 | 2.410e-05 | 0/6 | 1.00 | n01 4.169e-02 | s02 1.427e-06 |

## 4. The prefix control (shipped vs the template's own empty think block)

No `--extra-prefix` in this run: only the shipped prompt shape was measured.

## 5. The ranked readout (the engine's decision under a policy)

No policy was ranked in this run: coverage was measured alone (`--rank cue=label` runs the full readout).

## 6. Cross-check against `DecisionEngine` (shipped policy, `bare`)

No `--cross-check N` in this run: the probe's shipped-policy numbers were not confirmed against `DecisionEngine`.

