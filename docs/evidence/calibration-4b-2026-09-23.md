# Live calibration of the default 4B — Spark-X2.5-4B-Q8_0 (2026-09-23)
Repo `/var/home/rybens/workspace/ggufone` @ `d9e41d0` (main, clean); model
`~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf` (4 375 021 152 B, sha256 `5c2c3c19…d9dea2`); pinned
b11026 **CPU** bundle `/work/t603-runtime/b11026-linux-x64-cpu`; `HOME=/var/home/rybens` (this box's
own data home, not a scratch one). The committed 60-item dev set was re-measured through the serving
path (`open_model` + `ModelSession` + `DecisionEngine`, what `ask`/`run` use), split per type 2/3 fit
· 1/3 held out; only the types whose held-out ECE improved were stored.

## 1. Run — exit 0, 51.5 min
    typed-gguf calibrate --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --threads 2 --json --out docs/evidence/calibration-4b-2026-09-23_rows.json

| type | fit/holdout | mode | temp | fit ECE before→after | held-out ECE before→after | verdict |
|---|---|---|---|---|---|---|
| `choice` | 16 / 8 | `normalized_peak` | 1.0 | 0.1040 → 0.1040 | 0.1919 → 0.1919 | **no calibration applied** — the held-out split did not improve |
| `noul` | 12 / 6 | `normalized_peak` | **0.0500** | 0.0769 → 0.0000 | 0.0355 → 0.0000 | **applied** — held-out ECE 0.0355 → 0.0000 at temperature 0.0500 |
| `score` | 12 / 6 | **`margin`** | **1.6475** | 0.3946 → 0.2852 | 0.2158 → 0.2026 | **applied** — held-out ECE 0.2158 → 0.2026 at temperature 1.6475 |

`params_hash` = `sha256:f09fe20e1169dcb13a2e1561bf9c742b90165276ba3c2438f906f1f973e3fc81`
## 2. Reproducibility — the fit re-run from the rows, no model touched (183 ms)
    typed-gguf calibrate --from-report docs/evidence/calibration-4b-2026-09-23_rows.json --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --dry-run --json

Same `params_hash` (`sha256:f09fe20e…73e3fc81`); a third copy of that number is in the store.
## 3. Store — `/var/home/rybens/.local/share/typed-gguf/calibration.json`
Schema `typed_gguf.calibration-store/v1`: one entry `file:Spark-X2.5-4B-Q8_0.gguf:4375021152`,
`accepted_types: ["noul","score"]`, `created_at 2026-09-23T17:55:02Z`. Machine state in the data
home — this file is not committed.
## 4. Applied at readout — three live asks on the 4B, warm host, all exit 0
Store written 17:55:02, host cold-spawned 17:56:46: all three answered `"calibrated": true` with the
same `params_hash`; no `keep stop` was needed (the "host before the store" ordering was not
exercised). a1 choice+score → `"calibration": {"source": "…/typed-gguf/calibration.json", "applied":
true, "model": "file:Spark-X2.5-4B-Q8_0.gguf:4375021152", "params_hash": "sha256:f09fe20e…3fc81",
"temperatures": {"score": 1.64755}, "confidence_modes": {"score": "margin"}, "accepted_types":
["noul","score"]}` — the `choice` answer went through untouched (the type the gate refused).
a2 noul+score → `"temperatures": {"noul": 0.05, "score": 1.64755}`; a3 choice+noul → `{"noul": 0.05}`.
## 5. Honest reading
`noul` was accepted at the temperature grid's **edge** (0.05 is its minimum) and `score` by +0.0132 —
both on 6-item held-out splits: what this dev set supports, not a claim about other prompts or states
(E2.5 §5.1's caveat). The measured run is **CPU**: the Vulkan bundle cannot place this model on this
box right now (`E_BACKEND_OOM`, 3/3 attempts) — the CPU bundle is what every live 4B run here used.
