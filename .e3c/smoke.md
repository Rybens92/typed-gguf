# E3c — cue shapes that put the readout mid-answer (`spark2_5`)

- card `t_6c119626` · model `Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes, sha256 `5c2c3c190e4337e1…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu` (explicit) · threads 4 · `--gpu-layers` requested 0
- placement used: `{"note": "no layers offloaded: the weights stay on the host (n_gpu_layers=0, kv_type=auto)", "n_gpu_layers": 0, "kv_type": "auto", "degraded": false, "attempts": [], "warnings": []}`
- engine device log (tail): `sched_reserve:        CPU compute buffer size =   271.17 MiB
sched_reserve: graph nodes  = 1266
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.83 ms, sched copies = 1`- dev items: c01 (choice 1) · label variants: `bare` · generated 2026-09-19T15:40:17Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- shapes: `shipped`, `two_step_shipped`
- wall: 39.4 s (one model load 7518 ms, one context per item, all shapes of one item in one decode batch)

## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)

| item | shape | readout row | top token at the readout | top-token mass | turn-closer mass | verdict |
|---|---|---|---|---|---|---|
| c01 | `shipped` | cue | `
` | 0.9536 | 0.0000 | ok |
| c01 | `two_step_shipped` | advanced | `t` | 0.8675 | 0.0000 | ok |

`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant turn instead of answering, so *no* label rendering can reach the floor here. `turn-closer mass` is the largest mass any catalogue closer holds at that row.

## 2. The EOT side: turn-closer mass vs the candidate's mass

| item | shape |  | `bare` coverage |
|---|---|---|
| c01 | `shipped` |  | 1.144e-02 |
| c01 | `two_step_shipped` |  | 9.506e-01 |

## 3. Coverage per item, shape × label variant

| shape | label | c01 | above floor | median |
|---|---|---|---|---|
| `shipped` | `bare` | 0.0114* | 0/1 | 0.0114 |
| `two_step_shipped` | `bare` | 0.9506 | 1/1 | 0.9506 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the shape's readout row (`readout.coverage_from_scale`) — every label variant of one shape costs no extra forward pass.

## 4. Shape summary (the `bare` label the engine ships)

| shape | readout row | mean `bare` coverage | median | above floor | refused items | advance rule |
|---|---|---|---|---|---|---|
| `shipped` | cue | 0.0114 | 0.0114 | 0/1 | 0/1 | — |
| `two_step_shipped` | advanced | 0.9506 | 0.9506 | 1/1 | 0/1 | content |

## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking

No shape was ranked in this run: coverage alone was measured (`--rank <shape>=<label>` runs the engine's full readout).

