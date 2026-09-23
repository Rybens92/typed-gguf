# t_287e0d18 — live placement evidence (Vulkan bundle, Spark-X2.5-4B-Q8_0, this box)

Recipe: `HOME=/var/home/rybens`, `TYPED_GGUF_RUNTIME_DIR=…/ggufone/runtime/b11026-linux-x64-vulkan`,
`python3 tg.py ask --state @… --choice … --score … --noul … --model … --keep-alive 0 --format native`
(same env shape as the calibration card t_b67f9c49).

| run | command shape | exit | wall | placement it chose |
|---|---|---|---|---|
| `ask` | one ask, GPU as found (1353 MiB held) | 0 | 22.4 s | `n_gpu_layers=36`, `kv_type` rung really used **q4_0** (`engine.kv_type=q8_0`), `n_ctx=41984`, `Vulkan0` compute buffer 261 MiB, `effective_backend=vulkan`. Warnings `W_KV_TYPE_DOWNGRADE, W_FIT_DOWNGRADE, W_BACKEND_OOM` — the two OOM'd rungs are published, not hidden. |
| `pair-a` / `pair-b` | two asks at once (busier desktop) | 0 / 0 | 36.6 s / 47.8 s | both: the **load** ladder degraded to `n_gpu_layers=18` (`attempts: ["n_gpu_layers=36 -> oom"]`, note "degraded after a backend allocation failure: 18 layer(s) offloaded"), then the ctx settled on `q4_0`. |
| `q4-a` / `q4-b` | two asks, `b` with `--kv-type q4_0 --no-fit-cache`, aimed at the ctx rungs | 0 / 0 | 9.1 s / 14.8 s | `a`: `36` layers, kv `q4_0`. `b`: the planner itself went honest-CPU-only (`n_gpu_layers=0`, `budget_bytes=0`, `n_ctx` shrunk to 4096, `W_CTX_BELOW_STANDARD`) because the device was held by `a` — a truthful CPU placement, no ladder needed. |

`ask.stderr` (the interesting line, with the failed allocation that used to be a hard
`E_BACKEND_OOM`):

```
ggml_vulkan: Device memory allocation of size 1050607616 failed.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
~llama_context:    Vulkan0 compute buffer size is 261.0000 MiB, matches expectation of 261.0000 MiB
```

So on this box, in these hours, the **context** ladder was needed only down to its kv rungs (the
GPU was either roomy or already reduced by the loader's own ladder); the *new* rungs (smaller
`n_ctx`, re-placed layers, CPU-only) are pinned offline instead — `tests/test_fit_placement_ladder.py`
asserts the exact walk through the production `ModelSession`:

```
ctx kv_type=f16 n_ctx=32768 n_gpu_layers=36 -> oom
ctx kv_type=q8_0 n_ctx=32768 n_gpu_layers=36 -> oom
ctx kv_type=q4_0 n_ctx=32768 n_gpu_layers=36 -> oom
ctx kv_type=q4_0 n_ctx=16384 n_gpu_layers=36 -> oom
ctx kv_type=q4_0 n_ctx=4096  n_gpu_layers=36 -> oom
ctx kv_type=q4_0 n_ctx=4096  n_gpu_layers=18 -> oom      (the model is re-loaded here)
ctx kv_type=q4_0 n_ctx=4096  n_gpu_layers=0  -> ok       (CPU-only: no device allocation at all)
```

What the fix contributed to the *live* runs above: `ask`'s response carries `W_BACKEND_OOM` for the
two rungs whose allocation failed before the one that worked (`session._note_context_rung`) — the
pre-fix session published only `W_KV_TYPE_DOWNGRADE`/`W_FIT_DOWNGRADE` there; and `pair-a/b` publish
the loader's rung walk in `attempts`.

## P2 — `--fit-target` on the planner `calibrate` now feeds

`fit.diff` is one live pair on this box (the plan `calibrate` hands its flag to is the same
`cli.fit_plan_for` call — pinned by
`tests/test_fit_placement_ladder.py::test_calibrate_passes_fit_target_and_n_seq_max_into_the_plan`):

| field | `fit --json` | `fit --fit-target 4500 --json` |
|---|---|---|
| `n_gpu_layers` | 36 | 15 |
| `n_ctx` | 47621 (`grown`) | 4096 (`shrunk`) |
| `kv_type` | q8_0 | q4_0 |
| `budget_bytes` | 5846859776 (5575 MiB) | 2237661184 (2134 MiB) |
| warnings | `W_KV_TYPE_DOWNGRADE` | + `W_CTX_BELOW_STANDARD`, `W_FIT_DOWNGRADE` |

Before the fix `calibrate --fit-target 4500 --dry-run` produced the *left* column (the card's own
repro: "plan identical to a run without it"); the CLI gate now asserts the right one.

`nvidia-smi` before/after each run is in `smi-before.txt` / `smi-after.txt` (and
`pair-smi-after.txt`): 1353 MiB → 1207 MiB held at the measurement points, no model resident after
the runs.
