# BENCHMARKS — measured tables for E2 (latency, throughput, quality, calibration, determinism)

> **Provenance (audit `t_78f5ea7a`, 2026-09-18).** The `reproduce:` lines below **fail on the commits
> that shipped them** (`fff127e`, `4e1d549`: `AttributeError: 'Placement' object has no attribute
> 'kv_type'`, exit 4) and work from `8d4fc9f` onward; the values are genuine (float-precision re-runs,
> 163 consistency checks). Note `docs/evidence/e2_provenance_note.md`; source of record
> `state/fights/e2-provenance/scorecard.md`.

Every table below is produced by one command, on the box described in §0, and stored as JSON in
`docs/evidence/e2_*.json`. Tags follow SPEC.md: **[executed]** = measured by this repository,
right now; **[recon]** = quoted from the coordinator's reconnaissance notes and *not* re-run by
us; **[host]** = measured on the operator host (GPU present) by the coordinator's run, not re-run in
this container; **[target]** = a number we intend to hit later.

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

| what | value |
|---|---|
| platform | `Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41` |
| CPUs seen | 24 |
| cgroup CPU quota | 2.0 CPU-seconds/s |
| cgroup memory limit | 8.0 GiB |
| runtime | pinned `llama-b11026-bin-ubuntu-x64` (CPU) |
| model | `Spark-X2.5-4B-Q8_0.gguf` — 4,375,021,152 B, arch `spark2_5`, None |

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

| model | model_load_ms (p50) | serve ms/request (p50) | one-shot ms/request | saved |
|---|---|---|---|---|
| Spark-X2.5-4B | 855.004 | 9,757.844 | 10,612.848 | 855.004 |
| Qwen3.5-0.8B | 1,571.673 | 1,343.235 | 2,914.909 | 1,571.673 |

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

| request | forks | waves | decode steps | ms (p50) |
|---|---|---|---|---|
| choice, 2 candidates | 2 | 2 | 4 | 6,141.113 |
| choice, 4 candidates | 4 | 2 | 8 | 8,542.577 |
| choice, 10 candidates | 10 | 2 | 20 | 14,923.724 |
| the accounting request (5 candidates, n_seq_max=6) | 5 | 2 | — | — |

`wave_accounting` of the published run: `groups=1`, `suffix_decodes=1`, `step_decodes=1`, `waves=2`.

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

**Two items stay in the set although they are judgement calls** (`c01`, checkout 500s → the
`technical` team; `c06`, "added a fix" → the `fixed` changelog section): the set is frozen before
the first measurement and is *not* tuned against a result — re-wording an item after seeing a
model miss it would make every later number unfalsifiable. They are flagged here instead.

**Reported with the agreement, because it changes how the number should be read** — §2.1 has the
analysis, and the two tables below are the data it is drawn from.

### 2.1 What the misses look like (both models, 60 items each)

The per-item rows in `docs/evidence/e2_quality.json` / `e2_qwen_quality.json` show *systematic*
behaviour, and it is worth more than the headline number:

* **`score` collapses to level 0 — for both models.** `Spark-X2.5-4B` and `Qwen3.5-0.8B` answer
  level `"0"` on 17 of 18 `score` items each, both score 4/18, and **13 of the 14 misses are the
  same items**. A 5× larger model does not move this number at all, so it is *not* a capability
  gap: the candidate labels of a `score` question are the level numbers (`"0"`…`"K-1"`) and the
  restricted readout is picking the first digit. That is a readout-policy finding — the thing
  `docs/TEMPLATES.md`'s per-family label policy and E2.5's calibration are for — and it is the
  single most useful measurement in this document.
* **`noul` has a yes-bias in the small model.** All 7 of the 0.8B's noul misses are `no` items
  answered `yes` (0.63–0.81 on the wrong side); the 4B gets 16/18 with no such pattern.
* **`choice` is where model size shows.** Per item: 12 correct in both, 6 only with the 4B, 1 only
  with the 0.8B, 5 missed by both — the 4B's 18/24 against the 0.8B's 13/24, and most of the
  difference is questions that need world knowledge (a copyleft clause, an incident owner).
* **The coverage/reliability flag separates on the 4B and does not on the 0.8B** — measured:
  4B 0.667 (`ok`) vs 0.500 (`low_mass`), 0.8B 0.452 vs 0.500. Read that as the flag's honest
  scope: it catches answers read off a candidate-set tail (e.g. the 4B's `c01`, p = 0.769 on the
  wrong option with 3.7 % coverage, and `c13`, 1.6 %), and it does **not** catch a confident
  wrong answer with healthy coverage (the 4B's `c06`/`c07`, coverage 0.66–0.70). It is a mass
  diagnostic, not a correctness oracle — which is exactly why E2 measures agreement *and*
  calibration instead of trusting either one.

| model | reliability | items | correct | agreement | 95% CI | note |
|---|---|---|---|---|---|---|
| Spark-X2.5-4B | `ok` | 48 | 32 | 0.6667 | 0.5254 – 0.7832 | candidate mass dominates the row |
| Spark-X2.5-4B | `low_mass` | 12 | 6 | 0.5000 | 0.2538 – 0.7462 | most of the row's mass sits outside the candidate set — the restricted softmax is reading a tail |
| Qwen3.5-0.8B | `ok` | 42 | 19 | 0.4524 | 0.3122 – 0.6005 | candidate mass dominates the row |
| Qwen3.5-0.8B | `low_mass` | 18 | 9 | 0.5000 | 0.2903 – 0.7097 | most of the row's mass sits outside the candidate set — the restricted softmax is reading a tail |

### quality — Spark-X2.5-4B-Q8_0.gguf

- generated: 2026-09-18T07:16:32Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=4
- reproduce: `uv run ggufone bench --suite quality --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --backend auto --runs 1 --threads 4 --json`

**exact-match agreement**

| type | n | correct | agreement | 95% CI |
|---|---|---|---|---|
| choice | 24 | 18 | 0.7500 | 0.5510 – 0.8800 |
| noul | 18 | 16 | 0.8889 | 0.6720 – 0.9690 |
| score | 18 | 4 | 0.2222 | 0.0900 – 0.4522 |
| overall | 60 | 38 | 0.6333 | 0.5068 – 0.7438 |

- agreement = the highest-probability candidate equals the gold candidate (the discrete decision), measured per question type and overall with 95% Wilson intervals; report-only in v1 (SPEC S-11).

## 3. The measurements

### 3.0 Which suite ran on which model

The five suites are the same code path for both models; this table says exactly what was run, so a
missing cell is visible instead of implied. Every cell that is blank has its command listed below
the table — the suites are `--model`-parametric and nothing about them is model-specific.

| suite | Spark-X2.5-4B-Q8_0 | Qwen3.5-0.8B-UD-Q4_K_XL |
|---|---|---|
| `latency` | yes (§3.1, 256/2k rows) | yes (§3.1, 256/2k/8k rows) |
| `throughput` | yes (§3.2) | not run — `--suite throughput --model <qwen>` |
| `quality` | yes (60 items, §2) | yes (60 items, §2) |
| `calibration` | not run — see below | yes (§3.3) |
| `determinism` | yes (§3.3) | not run — `--suite determinism --model <qwen>` |

The 4B's `calibration` run is the one cell that was *deliberately* dropped: the suite's evidence
(reliability bins, ECE, the three confidence modes, r(confidence, coverage)) is model-agnostic,
its per-item rows are already in the 4B quality report, and the 60-item 4B run costs ~18 more
minutes on this box; the 0.8B's numbers are published instead, with its command.

### 3.1 Latency (A-E2-1)

`ggufone bench --suite latency`: model load, prefill throughput at {256, 2k, 8k} tokens,
per-question ms at {2, 4, 10} candidates, wave scaling N = 1..16 questions, the warm state cache,
and the `serve` vs one-shot comparison. `p50`/`p95` are the interpolated percentiles of
`--runs 5` samples of the same request; `tok/s` is summarised per run (tokens ÷ seconds) and never
as a ratio of two medians. Every decision row is warm (`prefill_reused: true`), so the tables
isolate the question phase.

**Cost note (why the two tables have different row sets).** On this 2-CPU container the 4B
model's **8k-token prefill costs ≈ 26 minutes per sample** — measured, not assumed: a run of
`--sizes 256,2048,8192 --runs 5` spent 52 minutes in the 8k row before it was stopped at 2 of 5
samples. The suite is identical in both tables and `--sizes` only selects which rows are
measured, so the **three-size** prefill table is published on the 0.8B model (all three sizes fit
inside `--runs 5`, 8k ≈ 2.5 min per sample) and the 4B table publishes the 256/2k rows. A reader
with real cores reproduces the missing 4B row with the printed command plus `--sizes 8192`.

### latency — Spark-X2.5-4B-Q8_0.gguf

- generated: 2026-09-18T07:37:18Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=5 threads=4
- reproduce: `uv run ggufone bench --suite latency --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --backend auto --runs 5 --threads 4 --json`

**model load (ms)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| model_load_ms | 5 | 855.004 | 983.163 | 734.869 | 1,005.676 |

**prefill**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 5 | 31,706.761 | 41,915.468 | 25,893.846 | 43,494.498 |
| 2048 | 5 | 248,300.702 | 268,261.935 | 227,275.077 | 268,466.515 |

**prefill throughput (tok/s)**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 5 | 8.074 | 9.579 | 5.886 | 9.887 |
| 2048 | 5 | 8.248 | 8.948 | 7.629 | 9.011 |

**per question**

| candidates | waves | forks | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 5 | 6,141.113 | 6,419.314 | 5,370.510 | 6,443.273 |
| 4 | 2 | 4 | 5 | 8,542.577 | 11,221.168 | 7,271.997 | 11,854.647 |
| 10 | 2 | 10 | 5 | 14,923.724 | 18,782.525 | 11,099.255 | 19,735.482 |

**wave scaling (N questions)**

| questions | waves | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| 1 | 1 | 5 | 7,594.447 | 12,573.907 | 4,080.342 | 13,439.045 |
| 2 | 2 | 5 | 7,394.049 | 13,879.688 | 6,587.373 | 15,377.806 |
| 3 | 3 | 5 | 14,711.329 | 33,216.033 | 10,392.902 | 37,444.471 |
| 4 | 4 | 5 | 14,632.800 | 81,802.283 | 13,694.008 | 97,687.586 |
| 5 | 5 | 5 | 24,073.554 | 201,117.425 | 18,354.013 | 211,755.717 |
| 6 | 6 | 5 | 21,513.931 | 111,521.437 | 20,380.661 | 133,939.670 |
| 7 | 7 | 5 | 114,289.027 | 178,195.216 | 19,297.427 | 191,017.310 |
| 8 | 8 | 5 | 41,486.986 | 43,192.379 | 34,147.662 | 43,431.414 |
| 9 | 9 | 5 | 46,950.933 | 48,982.510 | 43,790.522 | 49,282.973 |
| 10 | 10 | 5 | 44,425.097 | 51,586.516 | 28,174.696 | 53,084.336 |
| 11 | 11 | 5 | 203,447.621 | 486,197.718 | 68,307.479 | 529,448.001 |
| 12 | 12 | 5 | 66,541.337 | 77,050.610 | 58,750.020 | 77,961.817 |
| 13 | 13 | 5 | 64,177.134 | 77,981.744 | 48,643.622 | 78,406.651 |
| 14 | 14 | 5 | 86,845.894 | 251,232.297 | 51,373.442 | 290,696.123 |
| 15 | 15 | 5 | 71,118.109 | 94,824.842 | 68,556.840 | 99,912.936 |
| 16 | 16 | 5 | 72,488.106 | 110,387.467 | 59,677.483 | 111,333.938 |

**warm cache (state reuse)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| prefill_ms | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| questions_ms | 5 | 7,978.160 | 8,459.720 | 7,577.175 | 8,486.949 |

**load amortisation (serve vs one-shot)**

| path | ms per request |
|---|---|
| serve (model already loaded) | 9,757.844 |
| one-shot CLI (load each call) | 10,612.848 |
| model_load_ms | 855.004 |

- `waves` counts the decode batches a decision takes after the prefill (here: 1 suffix decode per question group + 1 per extra candidate token); five single-token candidates in one question are 1 wave, not ceil(5 / n_seq_max): {'groups': 1, 'suffix_decodes': 1, 'step_decodes': 1, 'waves': 2}.
- every measured call reports `prefill_reused: true` once the prefix state cache is warm, which is why the decision tables isolate the question phase.

**Reading the 4B table (and why the p95s are wild).** This box is shared: the same 256-token
prefill measured 8.1 tok/s in the published run and 12.9 tok/s in the far quieter threads probe of
§3.5, and the wave-scaling rows carry outliers (N = 5 at 201 s, N = 11 at 203 s) between *lower*
neighbours (N = 12–13 at 64–66 s) — contention, not engine behaviour: `usage.waves` is monotone in
N and so is the underlying cost (≈ 4–5 s per extra question at this speed). Read the `p50` as an
upper bound for this container at this hour and the *ratios* (2 → 4 → 10 candidates, N → N + 1) as
the signal; the 0.8B table below, measured in the same window, is 25–60× faster with
correspondingly smaller outliers.


### latency — Qwen3.5-0.8B-UD-Q4_K_XL.gguf

- generated: 2026-09-18T05:54:37Z
- host: Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.41 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=5 threads=2
- reproduce: `uv run ggufone bench --suite latency --model /var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --backend auto --runs 5 --threads 2 --json`

**model load (ms)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| model_load_ms | 5 | 1,571.673 | 2,518.973 | 992.566 | 2,645.827 |

**prefill**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 5 | 9,443.904 | 10,391.614 | 8,526.632 | 10,598.579 |
| 2048 | 5 | 55,240.973 | 67,101.257 | 50,525.492 | 69,941.665 |
| 8192 | 5 | 250,811.546 | 321,099.395 | 216,356.185 | 322,341.620 |

**prefill throughput (tok/s)**

| tokens | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 256 | 5 | 27.107 | 29.896 | 24.154 | 30.024 |
| 2048 | 5 | 37.074 | 40.215 | 29.282 | 40.534 |
| 8192 | 5 | 32.662 | 37.619 | 25.414 | 37.863 |

**per question**

| candidates | waves | forks | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 5 | 833.886 | 874.365 | 768.123 | 876.112 |
| 4 | 2 | 4 | 5 | 1,125.088 | 1,439.058 | 1,087.908 | 1,478.421 |
| 10 | 2 | 10 | 5 | 2,221.644 | 2,543.329 | 1,836.712 | 2,571.118 |

**wave scaling (N questions)**

| questions | waves | n | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| 1 | 1 | 5 | 653.667 | 921.289 | 448.219 | 966.161 |
| 2 | 2 | 5 | 1,478.926 | 1,663.013 | 1,301.385 | 1,667.214 |
| 3 | 3 | 5 | 2,198.045 | 3,101.855 | 2,100.313 | 3,184.668 |
| 4 | 4 | 5 | 3,244.997 | 3,634.245 | 2,399.344 | 3,713.203 |
| 5 | 5 | 5 | 3,766.558 | 4,147.747 | 3,045.215 | 4,169.800 |
| 6 | 6 | 5 | 4,703.782 | 5,049.311 | 4,377.850 | 5,105.823 |
| 7 | 7 | 5 | 4,991.353 | 5,337.537 | 4,086.272 | 5,369.841 |
| 8 | 8 | 5 | 6,989.073 | 7,656.278 | 5,171.169 | 7,819.743 |
| 9 | 9 | 5 | 9,248.717 | 22,929.696 | 7,194.017 | 26,253.310 |
| 10 | 10 | 5 | 8,401.411 | 10,535.499 | 6,540.778 | 10,896.590 |
| 11 | 11 | 5 | 12,875.971 | 14,191.363 | 11,463.999 | 14,197.734 |
| 12 | 12 | 5 | 13,948.431 | 16,604.623 | 10,736.038 | 16,918.372 |
| 13 | 13 | 5 | 11,112.752 | 15,014.153 | 9,312.274 | 15,972.972 |
| 14 | 14 | 5 | 13,288.185 | 16,031.074 | 9,790.845 | 16,507.120 |
| 15 | 15 | 5 | 12,306.517 | 15,985.169 | 12,072.694 | 16,595.219 |
| 16 | 16 | 5 | 13,108.831 | 19,455.936 | 11,687.492 | 20,721.094 |

**warm cache (state reuse)**

| row | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| prefill_ms | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| questions_ms | 5 | 1,281.379 | 1,784.839 | 955.888 | 1,821.991 |

**load amortisation (serve vs one-shot)**

| path | ms per request |
|---|---|
| serve (model already loaded) | 1,343.235 |
| one-shot CLI (load each call) | 2,914.909 |
| model_load_ms | 1,571.673 |

- `waves` counts the decode batches a decision takes after the prefill (here: 1 suffix decode per question group + 1 per extra candidate token); five single-token candidates in one question are 1 wave, not ceil(5 / n_seq_max): {'groups': 1, 'suffix_decodes': 1, 'step_decodes': 1, 'waves': 2}.
- every measured call reports `prefill_reused: true` once the prefix state cache is warm, which is why the decision tables isolate the question phase.


### 3.2 Throughput per backend (A-E2-2)

One row per documented backend on the same model. A backend with no local bundle is reported with
its reason rather than dropped silently; non-CPU backends offload all layers (`--gpu-layers`)
unless told otherwise, and the placement is printed in the row.

**How `--backend` and `--gpu-layers` resolve** (established from the operator's Vulkan host, card
`t_31b3943a`; the evidence is `docs/evidence/e2_fix_t_31b3943a_bench_placement.md` §8–9):

* `--backend auto` resolves to **every locally installed bundle**, in `DEFAULT_BACKENDS` order
  (`cpu`, `vulkan`, `cuda`). The single-backend suites (latency, quality, calibration, determinism)
  measure the **first** of them, which is `cpu` — a GPU box that also carries a CPU bundle therefore
  measures CPU when it asks for `auto`. That is deliberate (the primary tables stay
  CPU-reproducible), and since this card the report says so: `backend_selection = {requested,
  selected, available, missing}`, a rendered `- backend selection: csv of the local bundles` line
  when there was more than one to choose from, and a note naming the flag that measures the
  accelerator instead — `--backend vulkan`, or `--suite throughput` (every local backend, one
  report).
* `--gpu-layers` is derived when it is not passed: `0` for a CPU bundle, `-1` ("every layer") for an
  accelerator bundle. An explicit `--gpu-layers N` always wins.
* A placement that cannot load *degrades* through the ladder (fewer layers → smaller `kv_type` →
  CPU-only) and the row prints `placement.requested` next to `placement.used`
  (`n_gpu_layers`, `kv_type`, `degraded`, `attempts`) — a degraded retry offloads fewer layers than
  the flags asked, and the table must say so.

| model | backend | placement | prefill tok/s (p50) | decision ms (p50) | load ms (p50) | decision tok/s (p50) | unavailable |
|---|---|---|---|---|---|---|---|
| Spark-X2.5-4B | cpu | `n_gpu_layers=0` | 6.756 | 8,010.987 | 802.017 | 0.9986 | |
| Spark-X2.5-4B | vulkan | — | — | — | — | — | no local llama.cpp bundle carries libggml-vulkan.so (benchmarks never download one: run `ggufone init --backend vulkan` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |
| Spark-X2.5-4B | cuda | — | — | — | — | — | no local llama.cpp bundle carries libggml-cuda.so (benchmarks never download one: run `ggufone init --backend cuda` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |

### 3.3 Determinism (A-E2-5)

Three repeats per backend of one request (choice + score + noul), `threads=1`, compared
byte-for-byte after stripping `timings` (A5). A backend whose repeats differ would make the suite
exit non-zero. The `digest` column is a **within-tree witness**: it is `sha256` over the whole
timings-stripped response body, so it moves when the response envelope changes — it did after
`8d4fc9f` (`engine.placement`/`engine.fit` grew; the request bytes and the decode are unchanged).
Compare digests only within one tree state (`docs/evidence/e2_provenance_note.md` §4).

| model | backend | threads | repeats | identical | digest |
|---|---|---|---|---|---|
| Spark-X2.5-4B | cpu | 1 | 3 | yes | `sha256:7ab32e5880d35e04a609659…` |
| Spark-X2.5-4B | cuda | — | — | not measured | no local llama.cpp bundle carries libggml-cuda.so (benchmarks never download one: run `ggufone init --backend cuda` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |
| Spark-X2.5-4B | vulkan | — | — | not measured | no local llama.cpp bundle carries libggml-vulkan.so (benchmarks never download one: run `ggufone init --backend vulkan` or point GGUFONE_BENCH_RUNTIME_DIR at extracted bundles) |

### 3.4 The recon numbers, side by side (A-E2-6)

| quantity | recon (host, as quoted) | this container (measured) | delta / why |
|---|---|---|---|
| prefill ~2k tokens, CPU | 5.9 s ≈ 350 tok/s **[recon]** | 248,300.7 ms p50 ≈ 8.25 tok/s **[executed]** | the container is capped at 2 CPU-seconds/s (`cgroup_cpu_max: 2.0`) and prefill is FLOP-bound (~2·params per token); the recon ran on the host's real cores |
| prefill ~2k tokens, Vulkan | 0.43 s **[recon]** | not measurable here — no GPU device in this container | needs a Vulkan device; `--suite throughput --backend vulkan` reports the reason instead of inventing a number |
| warm decision (4 candidates) | 14–20 ms **[recon]** | 8,542.6 ms p50 **[executed]** | same cause (CPU quota) plus a different engine generation: E2's decision phase is measured warm (`prefill_reused: true`) but on 2 CPUs |
| first Vulkan call (shader compile) | 23–30 s **[recon]** | not measurable here (no GPU) | see §3.6; a software Vulkan device only shows the pipeline-cache *mechanism* |
| model load | 0.70–0.72 s **[recon]** | 855.0 ms p50 **[executed]** | page-cache warm on the host; this box also re-reads a 4.4 GB file under a shared I/O path |
| decisions after the fit fix (Vulkan, host) | 2.4 s questions_ms **[recon]** | see §3.7 | `waves` is unchanged by the fit fix; the drop is per-batch cost (q8_0 KV + full offload) |

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
is therefore the *worst* setting on this box. The table is the five-point probe measured on this
box (Spark-X2.5-4B-Q8_0, CPU, 78-token prefix, one choice question with 4 candidates, warm prefix
state cache) — **not** the suite's own sweep: that sweep is the same code path
(`--suite latency --threads N --sizes 256`, `--runs 3`) and was cut from the campaign's last hour
to keep the published tables coming; `docs/evidence/e2_threads_probe.json` carries the raw rows and
the exact command, and `spark_t{1,4,24}_latency.json` names what the suite would have produced.
The suite's own `p50`s at `--threads 4` (8.1 tok/s prefill on 256 tokens) are consistent with the
probe's 12.9 tok/s measured when the box was quieter — the *shape* is the finding:

| threads | prefill ms (78 tok) | prefill tok/s | decision ms (4 candidates) | total ms | vs the best prefill |
|---|---|---|---|---|---|
| 1 | 11,314 | 6.90 | 4,403 | 15,718 | 3.83× |
| 2 | 9,384 | 8.30 | 4,100 | 13,486 | 4.61× |
| 4 | 6,040 | 12.90 | 3,748 | 9,790 | 7.17× |
| 8 | 8,900 | 8.80 | 5,599 | 14,501 | 4.89× |
| 24 | 43,461 | 1.80 | 73,429 | 116,892 | 1.00× |

`GGUFONE_RUNTIME_DIR=<bundle> python3 /tmp/probe1.py threads   # scratch probe, logs in /tmp/probe_threads.log` (model `/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf`, prefix 78 tokens, backend cpu).

Read it as: **`threads = cores_seen` is a trap.** 24 threads on a 2-CPU quota is 5× slower than 4
threads for prefill and 20× slower for the decision phase (73.4 s vs 3.7 s for the same four
candidates), because the work is split into more pieces than the quota can run in parallel and
every context switch is paid. `ggufone` leaves `threads` at the host's physical cores by default
(SPEC §2.2) — on this container that is the wrong choice by construction, which is why every
published row prints the threads it used and why `--threads 4` is pinned in the tables above.

### 3.6 Vulkan status on this box

**Not measured, and the attempt is on record.** This container has no GPU (`/dev/dri` is absent)
and only the CPU bundle, so:

* the `throughput` row for `vulkan` is `measured: false` with the reason printed next to it
  (§3.2) — the suite refuses to invent a number for a backend it cannot run;
* `tools/e2_vulkan_probe.py` was pointed at the pinned Vulkan bundle
  (`GGUFONE_VULKAN_RUNTIME_DIR=/work/e1a/home/runtime/b11026-linux-x64-vulkan`) with Mesa
  `lavapipe` as the Vulkan device. The model **loaded** and the context was created, then the
  first `llama_decode` returned `-1` (`E_DECODE_FAILED`) in every one of the three fresh
  processes — the software rasteriser in this container cannot run the pinned Vulkan backend's
  kernels. The machine-readable attempt is `docs/evidence/e2_vulkan_probe.json`
  (`[executed: lavapipe]`), including the empty shader-cache directories before and after
  (`~/.cache/mesa_shader_cache` / `MESA_SHADER_CACHE_DIR`): nothing was compiled because nothing
  decoded.

So the recon's 23–30 s first-call shader compilation and its 0.43 s Vulkan prefill stay
**[recon]**, and this document does not pretend otherwise. What *is* measurable about the
mechanism is in the CPU tables: the per-process load (§3.1 `load amortisation`) has exactly the
same shape — a fixed cost paid once per process, amortised by `serve` — and that is why SPEC R2
prescribes a warm-up **plus** a persistent driver cache rather than a warm-up alone.

### 3.7 The post-fix fit run on the host (the coordinator's reference point)

The fit fix (card `t_8cb0a05e`, commit `5410e42`) plans against **free** device memory at load
time, caps `--fit-target`, and degrades `kv_type` with `W_KV_TYPE_DOWNGRADE` instead of failing
the load. The coordinator's run on this box's host (Spark-X2.5-4B-Q8_0, **Vulkan**, 6201 MiB
free) is **[recon]** and reads:

| field | value | tag |
|---|---|---|
| `engine.kv_type` | `q8_0` (degraded from f16 by the fit plan) | **[recon]** |
| `engine.n_gpu_layers` | 36 (full offload) | **[recon]** |
| `timings.model_load_ms` | 1.4 s | **[recon]** |
| `timings.prefill_ms` (cold prefix) | 5.6 s | **[recon]** |
| `timings.questions_ms` | **2.4 s** (was 11.6–18.8 s before the fix) | **[recon]** |
| `timings.total_ms` | 8.1 s | **[recon]** |

**Why `questions_ms` fell by 4.8–7.8×, and what it is *not*:** the decision phase's *shape* did
not change. `usage.waves` counts the decode batches after the prefill
(`1` per candidate group for the question suffix, `+ (max candidate length − 1)` step decodes per
group) and depends only on the rendered candidate sequences and `n_seq_max` — never on `kv_type`
or `n_gpu_layers`. So the same request has the same `waves`/`decode_steps` before and after the
fit fix, and the whole difference is *cost per batch*: `q8_0` KV halves the KV bytes each decode
step moves, and 36 offloaded layers replace whatever was running on the CPU (or nothing at all —
the pre-fix run refused to load on a starved device). The two reports carry every field needed to
test that reading without a new run: `usage.waves`, `usage.decode_steps`, `engine.kv_type`,
`engine.n_gpu_layers`, `timings.questions_ms`.

The earlier "same request, second run slower (11.6 → 16.3 s)" observation has a measured
explanation on this box too: llama.cpp defaults to `threads = os.cpu_count()` = 24 while the
container is capped at 2 CPU-seconds/s, and §3.5 measures that setting as 5–20× slower than
`--threads 4` — run-to-run variance under oversubscription, not something the engine does.

### 3.8 The bench run on the operator host (E2 FIX t_31b3943a, requirement 4) **[host]**

The fix card's fourth requirement was the accelerated re-run: `ggufone bench` must reach the loader
on a GPU box. The coordinator's run (RTX 3060 Ti, Vulkan bundle `b11026`) — `--suite latency
--backend vulkan --gpu-layers -1 --runs 3 --threads 4`, exit 0, VRAM free 5522/5495 MiB — reports:

| field | value | tag |
|---|---|---|
| `placement.requested` | `n_gpu_layers=-1` | **[host]** |
| `placement.used` | `{n_gpu_layers: -1, degraded: false, attempts: [], kv_type: auto}` — all layers requested, **no degradation** | **[host]** |
| `model_load_ms` (p50 / p95) | 1094.1 / 1199.9 | **[host]** |
| prefill 256 / 2048 / 8192 tok | 2436.0 / 2786.0 / 2504.9 tok/s (105 / 735 / 3270 ms) | **[host]** |
| per question, 2 / 4 / 10 candidates | 92 / 157 / 234 ms | **[host]** |
| warm cache `questions_ms` | 129.4 (p95 130.2), `prefill_reused: true`, `prefill_ms` 0.0 | **[host]** |
| load amortisation | serve 412 ms/req · one-shot 1506 ms/req | **[host]** |
| `--suite determinism --backend vulkan --threads 1` | `ok: true`, digest `sha256:d9978816…` ×3, `identical: true` | **[host]** |

Raw JSONs live on the host (`~/.ggufone-host-gate-2026-09-18/`); the verbatim report, the placement
JSON and what this does *not* cover are in
`docs/evidence/e2_fix_t_31b3943a_bench_placement.md` §8 and
`.e2e/t_31b3943a-bench-placement/host_run_vulkan_reported.md`. These are the first Vulkan numbers with
a *working* bench path; the Vulkan rows of the E2 tables above were produced on this container
(**[recon]** / `measured: false`), and nothing here retroactively re-measures them.

## 4. What these tables deliberately do not claim

* **No cross-backend equality** (SPEC R8): determinism is pinned to (runtime, backend,
  `threads=1`); CPU and Vulkan are compared for *speed*, never for identical bits.
* **No vendor-parity claim**: the dev set is ours and agreement is reported as a measurement.
* **No escalation, no calibration applied in the E2 tables**: `max_escalations` stays 0 in E2 (S-7)
  and the calibration suite *measures* ECE for the three confidence modes. Fitting a temperature,
  the held-out acceptance gate, `--route auto` and the bounded escalation are E2.5
  (`ggufone calibrate`) and are measured in §5 below.
* **No CUDA**: no CUDA device and no CUDA bundle exist in this container; the throughput table
  says so per row instead of omitting the backend.

## 5. E2.5 — calibration, routing and escalation (measured)

Same box, same pinned runtime and the same 60-item dev set as §3/§4 (container: cpu, threads=2,
runtime `b11026-linux-x64-cpu`, 2 CPU-seconds/s quota). One command per table —
`tools/e2p5_reproduce.py <calibrate|route|escalate>` — and the raw JSON plus the live logs sit in
`docs/evidence/e2p5_*`. Gate mapping and narrative: `docs/evidence/e2p5_t_630f32a3_calibration.md`.

### 5.1 The fit and the gate — 0.8B, per-type split 2/3 fit · 1/3 held out

`ggufone calibrate` fits a temperature per (model, question type), fits **all three** confidence
statistics, and stores nothing it cannot defend on the held-out split. Measured (two dev-set
passes, identical `params_hash=sha256:dd995c81…b1bb`, 611.5 s for both passes):

| type | rows (fit/holdout) | statistic the parameter was accepted with | temperature | fit ECE | held-out ECE | stored? |
|---|---|---|---|---|---|---|
| `choice` | 16 / 8 | `normalized_peak` (identity) | 1.0000 | 0.2451 → 0.2451 | 0.2266 → 0.2266 | no — `margin` improved the held-out ECE by 0.0031, below the 0.0050 margin a switch needs |
| `noul` | 12 / 6 | `normalized_peak` (identity) | 1.0000 | 0.2627 → 0.2627 | 0.3843 → 0.3843 | no — the statistic is already honest |
| `score` | 12 / 6 | **`entropy`** | **1.6475** | 0.1154 → 0.0695 | 0.4672 → **0.4479** | **yes** — the default statistic overfit its fit split (0.1906 → 0.0265) and lost the held-out one (0.3732 → 0.4400); `entropy` won its held-out split by 0.0194 |

`calibration.json` therefore holds exactly `accepted_types: ["score"]`, and a live `ggufone run`
with that store answers with `calibrated: true`,
`calibration.temperatures: {"score": 1.64755}` and
`calibration.confidence_modes: {"score": "entropy"}` — the promoted statistic is visible to the
caller, and the two refused types go through the readout untouched.

### 5.2 `--route auto`

| registry | chosen | quant | kv_type | n_ctx | n_seq_max | placement | rejected |
|---|---|---|---|---|---|---|---|
| `qwen-0.8b` (0.51 GiB) | `qwen-0.8b` | — | f16 | 4096 | 5 | cpu (cpu-only box, 25 596 MiB RAM budget) | — |
| `qwen-0.8b` + `spark-4b` (4.07 GiB) | `spark-4b` | — | f16 | 4096 | 5 | cpu | `qwen-0.8b`: ranked below the bigger model that fits |

The plan carries the reason and one verdict per candidate (`engine.route.steps`), and the same
record lands in the audit log when `--audit DIR` is set.

### 5.3 Escalation (A-E2p5-5)

0.8B → 4B, the 20 dev items the policy flags at `threshold=0.5` (confidence < 0.5 or
`low_mass`), re-asked on the 4B (`--limit 20`; the shipped request default is `max_escalations=1`).
Rows: the calibration run's own primary pass (`--rows`), so the before/after are the same path.

| set | before | after | delta |
|---|---|---|---|
| whole dev set (60) | 32/60 = 0.5333 | 34/60 = 0.5667 | **+0.0333** (+2 items) |
| fit split (40) | 21/40 = 0.5250 | 23/40 = 0.5750 | +0.0500 |
| held-out split (20) | 11/20 = 0.5500 | 11/20 = 0.5500 | 0.0000 |

So: escalation improved the dev-set agreement by two items, both inside the fit split, and changed
nothing on the 20 held-out items. Every re-ask is logged with its trigger and its `was` → `now`
(`engine.escalations`, and `--audit DIR` persists the same record).

### 5.4 Notes and limitations

* **The bench loader is broken at this commit** (`ggufone bench` → `E_INTERNAL AttributeError:
  'Placement' object has no attribute 'kv_type'` from `degrade_ladder` in `session.py:250`; card
  `t_31b3943a`, **fixed by a sibling while this card ran** — the branch is rebased on that fix, and
  the 872-test suite above includes its `tests/test_bench_placement.py`). E2.5's live numbers were
  measured **through the serving path** (`open_model` + `ModelSession` — exactly what `run`/`ask`
  use) instead of the bench harness, which is also the more faithful distribution to calibrate;
  `ggufone calibrate` does not touch the bench path.
* **Absolute agreement differs between the two load paths**: the same 0.8B answers 32/60 through
  the serving path (E2.5) and 28/60 through the bench path (§4). Every §5 number is measured inside
  one path, so the §5.3 delta is apples-to-apples.
* **The per-type held-out splits are thin** (6–8 items), so the verdict is sensitive to small row
  drift: an earlier live pass of the same model produced the same probabilities but a different
  `coverage` float, and the `score` verdict moved with it — which is *why* the params digest now
  covers only the fields the fit reads. The smallest honest dev-set extension is +12 items per type
  (doubling the held-out split per type to 12–16).
* **No GPU on the measuring box**, so the router's device-budget branch is exercised offline
  (injected VRAM numbers, the free-VRAM margin, the KV-floor fallback) and only its CPU branch live.
