# BENCHMARKS — measured tables for E2 (latency, throughput, quality, calibration, determinism)

Every table below is produced by one command, on the box described in §0, and stored as JSON in
`docs/evidence/e2_*.json`. Tags follow SPEC.md: **[executed]** = measured by this repository,
right now; **[recon]** = quoted from the coordinator's reconnaissance notes and *not* re-run by
us; **[target]** = a number we intend to hit later.

```
# one suite, table on stdout, JSON report written next to it
GGUFONE_RUNTIME_DIR=<bundle> python3 tools/e2_reproduce.py --suite latency \
    --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --threads 4 --runs 5 \
    --out docs/evidence/e2_latency.json

# all five suites at once (each report goes to <out-dir>/e2_<suite>.json)
GGUFONE_RUNTIME_DIR=<bundle> python3 tools/e2_reproduce.py --suite all \
    --model <path.gguf> --threads 4 --out-dir docs/evidence
```

The same suites are reachable through the CLI (`uv run ggufone bench --suite <name> ...`); the
`reproduce:` line inside every rendered table *is* the exact command for that report.

**A benchmark never touches the registry and never opens a socket** (A-E2-7): `--model` takes a
path on disk (an alias is a hard `E_BENCH_MODEL` error that says so), the runtime is a local
llama.cpp bundle, and the dev set ships inside the package. `tests/test_bench.py` asserts it with
the network and the registry store poisoned.

## 0. The box and the models

<!-- TABLE:host -->

| what | value |
|---|---|
| runtime | pinned `llama-b11026-bin-ubuntu-x64` (CPU) at `/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu` |
| models | `Spark-X2.5-4B-Q8_0.gguf` (4 375 021 152 B, `spark2_5`, Q8_0) · `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (558 772 480 B, `qwen35`, Q4_K_M) |
| placement | `n_gpu_layers=0` — CPU only; **no GPU is reachable from this container** (`/dev/dri` absent) |
| backends measured | CPU. Vulkan and CUDA are reported as `measured: false` with their reason (A-E2-2) — see §3.4 |
| threads | 4 for the primary tables (measured best here, §3.5), 2 for the 0.8B model |

Two caveats that a reader must keep in mind, because they dominate every number below:

1. **The container sees 24 CPUs but is capped at 2 CPU-seconds per second** (`/sys/fs/cgroup/cpu.max
   = 200000 100000`). llama.cpp defaults to `threads = os.cpu_count() = 24`, which *looks*
   reasonable and is 5–20× slower than `threads = 4` — measured, §3.5. Every published row prints
   the threads it used.
2. **The box is shared.** Other agents run builds and test sweeps on the same quota; a `p95` here
   is "this box, today", not a hardware spec. That is exactly what the `p50`/`p95` pair is for.

## 1. Headline: the two questions from the E2 card

### 1.1 Does a fresh CLI process pay Vulkan pipeline/shader compilation?

The recon observed **23–30 s on the first Vulkan call** and 14–20 ms on a warm decision
**[recon]** — i.e. a one-shot CLI process pays a fixed cost that a server would amortise.

What this box can and cannot answer, stated honestly:

* **Cannot**: re-measure a real GPU shader compile. This container has no GPU (`/dev/dri` is
  absent) and the only local bundle is CPU-only, so there is no Vulkan device to compile for.
  A software Vulkan device (Mesa `lavapipe`) is installed; §3.6 attempts it and tags the result
  `[executed: lavapipe]` — a software rasteriser measures the *pipeline-cache* mechanism, not the
  GPU's compile time.
* **Can, and did**: measure the *shape* of the same effect on CPU — the per-process cost of
  loading the model and reserving the compute graph, and what a long-lived process saves. The
  `model_load` and `load amortisation` tables in §3.1 answer "serve vs one-shot CLI"
  quantitatively for this box:

<!-- TABLE:headline_amortisation -->

*Conclusion for (a):* a fresh process pays `model_load_ms` (weights mapping + graph/sched
reserve) which is **not** amortised across calls; on GPU/Vulkan a fresh *process* additionally
pays pipeline creation, while the *driver's* shader cache (`~/.cache/mesa_shader_cache` for Mesa,
`~/.nv/ComputeCache` for NVIDIA) can survive process exit — so the fix is not a warm-up inside the
process, it is a warm-up **plus** a persistent driver cache, exactly as SPEC R2 prescribes
(`W_VULKAN_WARMUP`). `ggufone serve` keeps the model loaded; a one-shot `ggufone run` pays the
load every time (measured: `saved_ms_per_request` above).

### 1.2 Is `waves = 8` for 5 forks intended grouping, or an accounting bug?

**Intended, and now documented — but the name is doing extra work.** `usage.waves` counts the
**`llama_decode` batches a decision takes after the prefill**, not "batches of forks":

```
waves = Σ over candidate groups ( 1 suffix decode + (max candidate sequence length - 1) step decodes )
```

so for `forks = 5` in **one** question whose candidates are multi-token labels, `waves` is
`1 + (maxLen - 1)`; for several questions it is the sum (each question pays its own suffix
decode, because E2 decodes one question group at a time). It is *not* `ceil(forks / (n_seq_max-1))`
and it is not a suffix-length bucket. The suite prints the decomposition, so a published row can
be audited:

<!-- TABLE:wave_accounting -->

`usage.forks` counts the fork *operations* (`llama_memory_seq_cp` calls, one per candidate plus
the head), `usage.decode_steps` counts candidate tokens read, and `usage.waves` counts batches —
three different things that the coordinator's single line `forks=5 / waves=8 / decode_steps=19`
conflated. Nothing in the engine is mis-grouped; the response field is dense, and this document
plus `engine.bench.plane`-style accounting in the report is the fix (a rename would break the
frozen wire shape of SPEC §2.5).

## 2. Quality on our own dev set (A-E2-3) and calibration (A-E2-4)

**Provenance (S-10).** The dev set is `src/ggufone/bench/devset.jsonl`: **60 items authored for
this repository** (24 `choice`, 18 `score`, 18 `noul`), each one state of ≤ 200 tokens with a
single defensible answer, one question per item. No vendor evaluation set, no scraped benchmark,
no model output is reused; every record carries `provenance: authored for ggufone E2 …`.
`devset.validate()` enforces the contract (count, types, gold inside the criteria, ≤ 200 words,
no duplicate states, no two candidates sharing a first word) and `tests/test_bench.py` runs it.

**Agreement** = the candidate with the **highest probability** is the gold candidate (for
`choice` the option name, for `score` the argmax level, for `noul` the `yes`/`no` argmax). That is
the discrete decision the caller acts on; the probability *shape* is the calibration suite's
business. Report-only in v1 (SPEC S-11): there is no minimum, the number is the measurement.

<!-- TABLE:quality -->

## 3. The measurements

### 3.1 Latency (A-E2-1)

`ggufone bench --suite latency`: model load, prefill throughput at {256, 2k, 8k} tokens,
per-question ms at {2, 4, 10} candidates, wave scaling N = 1..16 questions, the warm state cache,
and the `serve` vs one-shot comparison. `p50`/`p95` are the interpolated percentiles of
`--runs 5` samples of the same request; `tok/s` is summarised per run (tokens ÷ seconds) and never
as a ratio of two medians. Every decision row is warm (`prefill_reused: true`), so the tables
isolate the question phase.

<!-- TABLE:latency_spark -->

<!-- TABLE:latency_qwen -->

### 3.2 Throughput per backend (A-E2-2)

One row per documented backend on the same model. A backend with no local bundle is reported with
its reason rather than dropped silently; non-CPU backends offload all layers (`--gpu-layers`)
unless told otherwise, and the placement is printed in the row.

<!-- TABLE:throughput -->

### 3.3 Determinism (A-E2-5)

Three repeats per backend of one request (choice + score + noul), `threads=1`, compared
byte-for-byte after stripping `timings` (A5). A backend whose repeats differ would make the suite
exit non-zero.

<!-- TABLE:determinism -->

### 3.4 The recon numbers, side by side (A-E2-6)

<!-- TABLE:recon -->

Reading the deltas: the recon's CPU prefill figure (~2k tokens in 5.9 s ≈ 350 tok/s **[recon]**)
is **two orders of magnitude** above what this container measures for the same 4B model. Two
things differ and both are measurable: (i) the recon ran on the *host*, on a box whose 24 cores
are real, while this container is capped at 2 CPU-seconds per second (measured: `cgroup quota 2.0`
in every report's `host` block); (ii) prefill is FLOP-bound at ~2·params per token, so a 4B Q8_0
model at 2 cores cannot reach hundreds of tok/s — the numbers in §3.1 are the honest floor for
this environment. The Vulkan figures (0.43 s prefill, 14–20 ms decisions, 23–30 s first call)
cannot be reproduced here at all: no GPU is reachable from the container (§3.6). None of these
deltas is silent: every recon row is tagged and every executed row names its box.

### 3.5 The `--threads` effect

The container advertises 24 CPUs and is capped at 2; llama.cpp's default (`threads = cpu_count`)
is therefore the *worst* setting on this box. Measured with the same 256-token prefill and the
same decision tables, `--runs 3`:

<!-- TABLE:threads -->

This is the explanation for the coordinator's "two identical requests, the second *slower*"
observation: with `threads = 24` on a 2-CPU quota, run-to-run variance is dominated by
oversubscription, not by anything the engine does.

### 3.6 Vulkan status on this box

<!-- TABLE:vulkan -->

## 4. What these tables deliberately do not claim

* **No cross-backend equality** (SPEC R8): determinism is pinned to (runtime, backend,
  `threads=1`); CPU and Vulkan are compared for *speed*, never for identical bits.
* **No vendor-parity claim**: the dev set is ours and agreement is reported as a measurement.
* **No escalation, no calibration applied**: `max_escalations` stays 0 in E2 (S-7) and the
  calibration suite *measures* ECE for the three confidence modes; fitting and applying a
  temperature is E2.5's card (`ggufone calibrate`).
* **No CUDA**: no CUDA device and no CUDA bundle exist in this container; the throughput table
  says so per row instead of omitting the backend.
