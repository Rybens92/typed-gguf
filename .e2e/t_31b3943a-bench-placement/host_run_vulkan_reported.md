# Requirement 4 — the operator-host run, as reported by @bots-coordinator

Provenance: this is **not** a capture from this container. The operator host (RTX 3060 Ti, Vulkan
bundle `b11026`, `~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf`) ran the command below; the report
reached the card as a comment by the `worker` profile (coordinator) on 2026-09-18 and is copied
here verbatim. The raw JSONs live **on the host** in `~/.ggufone-host-gate-2026-09-18/`
(`host_latency_vulkan.json`, `host_determinism_vulkan.json`, `host_latency.json`) and were not
copied into this sandbox (no bind mount of the host home); this container has no `/dev/dri`, so
nothing here can re-measure the accelerated placement.

## The command

```
GGUFONE_BENCH_RUNTIME_DIR=~/.local/share/ggufone/runtime/b11026-linux-x64-vulkan uv run ggufone bench \
  --suite latency --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --backend vulkan --gpu-layers -1 \
  --runs 3 --threads 4 --json --out /tmp/host_latency_vulkan.json
```
→ **exit 0**. VRAM free: 5522 MiB before / 5495 MiB after.

## placement (the requirement-4 acceptance subject)

```
requested: "n_gpu_layers=-1"
used: {n_gpu_layers: -1, degraded: false, attempts: [], kv_type: "auto",
       note: "all layers requested: n_gpu_layers=-1 (kv_type=auto)", warnings: []}
```

**Accelerated placement confirmed, no degradation** (an `E_INTERNAL` is not).

## Table head (backend=vulkan, p50)

| row | value |
|---|---|
| `model_load_ms` | **1094.1** (p95 1199.9) |
| prefill 256 / 2048 / 8192 tok | **2436.0 / 2786.0 / 2504.9 tok/s** (105 / 735 / 3270 ms) |
| per question, 2 / 4 / 10 candidates | **92 / 157 / 234 ms** |
| warm cache `questions_ms` | **129.4** (p95 130.2); `prefill_reused: true`, `prefill_ms` 0.0 |
| load amortisation | serve **412 ms**/req · one-shot 1506 ms/req (load 1094 ms) |

## Bonus — A-E2-5 for the accelerated path

`--suite determinism --backend vulkan --gpu-layers -1 --threads 1` → `ok: true`, digests
`sha256:d9978816…` ×3 → **`identical: true`**, placement `n_gpu_layers=-1`.

## Two notes from the coordinator (recorded here, resolved in the evidence document)

1. `--backend auto` **without** `--gpu-layers` measured **CPU only** on this GPU host (placement
   requested `n_gpu_layers=0`, all rows `backend: cpu`, zero `vulkan` occurrences in the report).
   "If deliberate, one line in the docs saves the next reader the surprise; if not, it is the same
   contract hole you already fixed for `-1`, seen from the default side."
2. The "not GPU-specific" correction is **confirmed independently**: fresh worktrees at `fff127e`
   and `4e1d549` fail the recorded `commands.reproduce` with the same `AttributeError` — also with
   `--backend cpu`. For the E2-provenance consequences the coordinator filed AUDIT `t_78f5ea7a`
   (auditor): bounded re-run + verdict + regeneration recommendation. *That audit, not this card,
   closes the E2 question.*
