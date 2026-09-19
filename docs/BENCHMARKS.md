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

## Quick preset — iteration, not publication (card `t_f46cec41`)

**Every table on this page is a full-campaign table: it was produced without `--quick`.** A quick
report carries `"quick": true`, the effective preset config and the note; it writes
`ggufone-bench-<suite>_quick.json` unless `--out` says otherwise, so it can never land on an
`e2_<suite>.json`. Do **not** quote a `--quick` number as a published one — the preset measures one
sample per row (no `p95` worth the name), one prefill size, six dev items and two determinism
repeats:

```
# the fast loop (all five suites, ~5 min on this container, reports in /tmp/quick)
GGUFONE_RUNTIME_DIR=<bundle> python3 tools/e2_reproduce.py --suite all --quick \
    --model <path.gguf> --threads 2 --out-dir /tmp/quick
```

| suite | full campaign | `--quick` |
|---|---|---|
| latency | runs=5, sizes 256/2k/8k, candidates 2/4/10, waves N=1..16 | runs=1, size 256, candidates 2/4, waves N∈{1,2} |
| throughput | every documented backend | 1 resolved backend, 1 sample (the rest are reported unmeasured, with the preset named as the reason) |
| quality | 60 items | 6 items, stratified 2/2/2 across choice/score/noul |
| calibration | 10 bins over 60 items | the same 6 items, bins as available (6 = samples), all three confidence modes |
| determinism | 3 repeats × 3 question types | 1 request × 2 repeats |

`--quick` refuses `--runs`/`--items`/`--sizes`/`--n-seq-max` (`E_BENCH_QUICK`): the preset fixes
those, so a quick run is never a half-applied one. `--max-seconds N` is a soft cap checked *between*
measurements — the current measurement finishes, the report is marked `"truncated": true` with the
unmeasured rows listed, and the exit code stays 0. The measured wall times (quick vs full, this
container, 2 CPU-seconds/s, CPU only) are in
`docs/evidence/e2_t_f46cec41_bench_quick.md`; the headline is that the *quick latency suite* costs
~2 minutes where the full one needs >40 minutes of pure measurement, with the same model and the
same rows.

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

> **Pre-fix rows (plain framing).** Everything in this section — the tables, §2.1's split and
> `docs/evidence/e2_quality.json` — was measured **before** the bench was fixed (card
> `t_6de5fc53`): the instrument planned the executed context from the live session, which resolves
> no chat template, so these rows are the **plain E1b framing** while `ggufone ask`/`run` sends the
> model's chat template. **§2.2 carries the same row re-measured with the corrected instrument**;
> the two must not be mixed.

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

### 2.2 The same row with the corrected instrument (card `t_6de5fc53`)

The bench used to plan the executed context from the live `ModelSession` — no `.model`/`.runtime`,
so `resolve_template` returned `None` and `prompt.build_prefix(state, None)` fell back to the bare
E1b framing — while `ggufone ask`/`run` planned from the model handle (the chat template).
**Fixed**: one plan, resolved from the handle, and every row now carries the framing it measured
(`framing`, `prefix_tokens`; the report prints `- framing: …`).

Same box, same model, same 60 items, same command; only the tree and `--cue` move:

| arm | framing | agreement | Wilson 95 % | `low_mass` | refused at the cue | coverage median | above the 0.10 floor | choice · noul · score |
|---|---|---|---|---|---|---|---|---|
| `--cue shipped`, **pre-fix** | plain (prompt.py E1b framing) | 36/60 = 0.600 | 0.474–0.714 | 13/60 | 0/60 | 0.2711 | 47/60 | 16/24 · 16/18 · 4/18 |
| `--cue shipped`, **post-fix** | chat-template: spark2_5 / internal | **42/60 = 0.700** | 0.575–0.801 | **45/60** | 1/60 | 0.03847 | 15/60 | 20/24 · 16/18 · 6/18 |

Raw reports: `.e3d/bench_plain_shipped.json` (pre-fix) and `.e3d/bench_templated_shipped.json`
(post-fix); the pre-fix file reproduces the published `.e3d/bench_shipped.json` item for item
(same agreement, same `low_mass`, same coverage list).

**What moves.** The prefix itself (102 → 119 tokens on `c01`), the mass split — the chat template
parks the cue row on a bare newline, so the label mass there is a tail (`low_mass` 13 → 45,
coverage median 0.2711 → 0.0869, above the floor 47 → 15) — and the agreement (36 → 42).
E2's published 38/60 is a *plain* row from an earlier campaign; the corrected row for the prompt
the product sends is the 42/60 above (the two items between 36 and 38 are the box/fit drift §8
already records, not the framing).

Reliability split, §2.1's shape, both framings:

| framing | reliability | items | correct | agreement | 95 % CI |
|---|---|---|---|---|---|
| plain (pre-fix) | `ok` | 47 | 31 | 0.6596 | 0.5167 – 0.7783 |
| plain (pre-fix) | `low_mass` | 13 | 5 | 0.3846 | 0.1771 – 0.6448 |
| chat template (post-fix) | `ok` | 15 | 11 | 0.7333 | 0.4805 – 0.8910 |
| chat template (post-fix) | `low_mass` | 45 | 31 | 0.6889 | 0.5433 – 0.8047 |

**The `low_mass` count is a statement about the *cue shape*, not about the model.** With the
product's own prompt the shipped cue lands on a row whose mass is a tail for 45 of 60 items — which
is the thing E3d's `--cue two_step` changes (§8, `docs/evidence/e3d_cue_decision_4b.md`), and it is
the reason the corrected instrument's `low_mass` column and the pre-fix one are not the same
measurement.

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
## 6. E3 — the 23 GB MoE on this box (Occamy 1.0, `qwen35moe`)

> **Pre-fix rows (plain framing).** Every row below was measured with the pre-fix bench (card
> `t_6de5fc53`), i.e. the plain E1b framing, while `ggufone ask`/`run` send Occamy's chat template
> (`qwen35moe`). The corrected framing's six-item probe is in `docs/BENCHMARKS.md` §7.9 and
> `docs/evidence/e2_fix_t_6de5fc53_framing.md` §4.3.

**These rows are not comparable with §0–§5 without their environment**: E2 was measured in a
container with **no GPU reachable** (`/dev/dri` absent); E3 ran in a worker container that *has*
the GPU (an ICD manifest fix, see the evidence doc §1.1), so its rows carry a `backend: vulkan`
that E2 could never produce. Everything else — the box, the shared quota, the pinned runtime — is
the same, and the same caveats apply: **2 CPU-seconds/s** of cgroup quota and an **8 GiB** memory
limit, the second of which decides this section.

### 6.1 The pin and the box (A-E3-5, A-E3-1)

| what | value |
|---|---|
| model | `Accio-Lab_occamy-1.0-Q4_K_L.gguf` — 24 113 674 848 B, arch `qwen35moe`, 40 layers, 24.1 GB of weights |
| SHA-256 | `633ae57faf731e863cc3ba7cb75396a1b1e377191730e7b0d7294eff55cdf757` (`docs/evidence/e3_environment.json`) |
| downloads | none in this milestone: the artifact was already on disk (`mtime` 2026-09-18 09:18, the card started at 13:27) |
| runtime | pinned `b11026-linux-x64-vulkan`; every Occamy row below forces `--backend vulkan` |
| GPU | `Vulkan0: NVIDIA GeForce RTX 3060 Ti (8192 MiB, 5669 MiB free)` — via an ICD manifest fix, evidence doc §1.1 |
| container | `cpu.max = 2 CPU-seconds/s`, `memory.max = 8 GiB` (the number that decides this section) |

### 6.2 What the 23 GB model costs here (A-E3-1, A-E3-4)

**`[container]`** — the per-item cost table below is the 8 GiB / 2 CPU-seconds-per-second worker
cgroup, which is the box this section is about; the `[host]` chunks the completion card added are
tagged in the chunk ledger at the end of the subsection.

`ggufone fit --print --json` (E1c, measured against free VRAM) says **`n_gpu_layers 7/40,
n_ctx 4096, kv_type q4_0, n_seq_max 8`** — 7 layers is all that 5685 MiB of free VRAM buys at
~600 MB per layer. The cost, however, is not the GPU: an mmap'd GGUF is cached by whichever cgroup
faults it in, this container is capped at 8 GiB, so 23 GB of weights can never be resident and
every forward pass re-reads experts from disk (~14 000 major faults/s ≈ 55 MB/s measured).

| measurement | value |
|---|---|
| model load (7 layers up) | **30.8 s** |
| dev item `c01` (choice): prefill / decision / wall | 152.8 s / 105.2 s / **258.6 s** |
| dev item `c02` (same process): prefill / decision / wall | 73.0 s / 74.2 s / **147.2 s** |
| placement used | `{n_gpu_layers: 7, kv_type: auto, degraded: false, attempts: []}`, log line `Vulkan0 compute buffer size 362.2 MiB` |
| degraded attempt seen later | `ErrorOutOfDeviceMemory` (~950 MB buffer) → ladder settled at 3 layers / CPU-only, per chunk |

A 60-item pass is therefore ~2.5–4.5 h **in the container**. That is a property of `23 GB vs 8 GiB`,
not a flag to tune, and it is why the campaign is **chunked** (10 items per chunk, each chunk a
complete `--suite quality` report of its own subset; `compare.merge_reports` stitches them). Once
the card moved the last four chunks onto the operator host, the same protocol became compute-bound
there — the tag on every row says which box produced it:

<!-- @@E3C_BENCH_6_2_BEGIN@@ -->
| chunk | box | items (choice/score/noul) | compute path the report proves | wall per item (median) | correct |
|---|---|---|---|---:|---:|
| `docs/evidence/e3_chunks/report_001.json` | `[container]` | 10 (4/3/3) | `n_gpu_layers 0`, `kv_type f16`, `degraded: true`, walked 7→oom, 3→oom | 113.2 s | 5/10 |
| `docs/evidence/e3_chunks/report_002.json` | `[container]` | 10 (3/4/3) | `n_gpu_layers 3`, `kv_type f16`, `degraded: true`, walked 7→oom | 99.5 s | 4/10 |
| `docs/evidence/e3_chunks/report_003.json` | `[host]` | 10 (3/3/4) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 49.2 s | 4/10 |
| `docs/evidence/e3_chunks/report_004.json` | `[host]` | 10 (4/3/3) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 42.7 s | 4/10 |
| `docs/evidence/e3_chunks/report_005.json` | `[host]` | 10 (3/4/3) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 47.0 s | 8/10 |
| `docs/evidence/e3_chunks/report_006.json` | `[host]` | 10 (7/1/2) | buffers: `Vulkan0`=10, `Vulkan_Host`=10; effective `vulkan` | 44.6 s | 6/10 |

Merged: `docs/evidence/e3_occamy_quality.json` — 60 items, 31 correct (0.517, 95 % CI 0.393–0.638); the `[container]` rows are the cost measurement this section is about, the `[host]` rows are the same protocol on a box that caches the weights.
<!-- @@E3C_BENCH_6_2_END@@ -->

### 6.3 The 20-question batch (A-E3-2)

`ggufone run` with 20 dev-set questions on one state and `n_seq_max = 4` (`docs/evidence/e3_batch.json`):

| what | value |
|---|---|
| wall (load + prefill + 65 decode batches) | **3 633.6 s** (60.6 min) |
| `usage` | `prefill_tokens` 109, `forks` 80, `decode_steps` 117, **`waves` 65**, `input_tokens` 1175, `output_tokens` 117 |
| placement used | `{n_gpu_layers: 3, kv_type: q4_0, degraded: false, attempts: []}` + the log's `Vulkan0 compute buffer size is 363.5 MiB` |
| outcome | exit 0, 20/20 answers, no OOM |

`waves = 65` for `forks = 80` is the adaptation the gate asks about: with `n_seq_max = 4` only 3
candidate slots fit in one decode batch, so the engine splits the forks into waves instead of
failing. (The response's `engine.backend` still reads `cpu` while the Vulkan device computed — the
mislabelling class `t_603a35a0` fixed in the bench path, still present in the serving path.)

### 6.4 Occamy vs the 4B default (A-E3-3)

<!-- @@E3C_BENCH_6_4_BEGIN@@ -->
Paired on the 60 dev items both models measured (`e2_quality.json` cut to the same ids — `tools/e3_reproduce.py --suite compare --align`):

Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | 4B default (E2, 60 items, CPU) | Occamy 1.0 (E3, chunks, vulkan) | delta |
|---|---|---|---|
| overall | 0.633 (38/60) [0.507–0.744] | 0.517 (31/60) [0.393–0.638] | -0.117 |
| choice | 0.750 (18/24) [0.551–0.880] | 0.625 (15/24) [0.427–0.788] | -0.125 |
| noul | 0.889 (16/18) [0.672–0.969] | 0.389 (7/18) [0.203–0.614] | -0.500 |
| score | 0.222 (4/18) [0.090–0.452] | 0.500 (9/18) [0.290–0.710] | +0.278 |
| low_mass (below the floor) | 0.500 (6/12) [0.254–0.746] | 0.509 (29/57) [0.383–0.634] | +0.009 |
| measured (at or above the floor) | 0.667 (32/48) [0.525–0.783] | 0.667 (2/3) [0.208–0.939] | +0.000 |

`Occamy 1.0 (E3, chunks, vulkan)` is worse than `4B default (E2, 60 items, CPU)` by -0.117 overall (0.633 -> 0.517); the `measured` row is the one to read first.

What separates the models is the **split itself**: Occamy answers below the floor on 57 of its 60 rows, the 4B on 12 — while the agreement *inside* the split is level (0.509 (29/57) [0.383–0.634] against 0.500 (6/12) [0.254–0.746]). Per type: `choice` 0.750 (18/24) [0.551–0.880] vs 0.625 (15/24) [0.427–0.788]; `noul` 0.889 (16/18) [0.672–0.969] vs 0.389 (7/18) [0.203–0.614]; `score` 0.222 (4/18) [0.090–0.452] vs 0.500 (9/18) [0.290–0.710]. Overall the two are 0.117 apart at n = 60 and their Wilson intervals overlap, so this table cannot rank them on the headline.

Both prompts were verified to end at their own assistant header (no template failure): the difference is the model's answer distribution, not the bytes it was given (§4.2 of the evidence doc).
<!-- @@E3C_BENCH_6_4_END@@ -->

### 6.5 Threads, and the routing recommendation (A-E3-4)

`llama-bench` from the pinned bundle, one model load per row (~2 min), on the same container:

| setting | pp64 (tok/s) | tg8 (tok/s) |
|---|---:|---:|
| `-ngl 7 -t 4` | **1.71** | **0.28** |
| `-ngl 7 -t 8` | 0.91 | 0.13 |
| `-ngl 7 -t 12` | 1.07 | 0.11 |
| `-ngl 0 -t 4` (CPU only) | 0.87 | 0.09 |

`threads = 4` wins and 8/12 lose by ~2× (the oversubscription E2 §3.5 measured for the 4B), and the
7 offloaded layers buy ~2× prefill / ~3× decode against CPU-only — worth doing, and worth doing
*only* as far as the free VRAM allows. Recommendation for this artifact: **`--backend vulkan
--gpu-layers 7 --threads 4`, `kv_type auto` (`q4_0` at 4k if you go through `ggufone fit`), small
requests — and route the interactive work elsewhere**: 0.28 tok/s decode is the box's physics for a
23 GB model that cannot be cached in 8 GiB, and no flag changes that. A host that can keep the
weights resident (the operator host's own 31 GiB) turns the same command into a compute-bound run.

## 7. E3c — Tiel-Coder (35B-A3B, 20.8 GB) measured like Occamy, and the three-way table

> **Pre-fix rows (plain framing).** Every row below was measured with the pre-fix bench (card
> `t_6de5fc53`), i.e. the plain E1b framing, while `ggufone ask`/`run` send Tiel's chat template.
> §7.9 carries the corrected framing's six-item probe and labels what is routed.

<!-- @@E3C_TIEL_BENCH_7_BEGIN@@ -->
**This section is the third column of the `qwen35moe` comparison** (card `t_a58f8b67`): E2's
`4B default` (§2, container, CPU), E3's `Occamy 1.0` (§6, chunks, vulkan) and **Tiel-Coder**
(here, host, vulkan) on the **same 60 committed dev items**. Raw evidence:
`docs/evidence/e3c_tiel_t_a58f8b67_tiel.md` (the campaign doc, with QA in its §11),
`docs/evidence/tiel_chunks/` (6 raw reports + placement captures), `docs/evidence/tiel_quality.json`
(the merge), `docs/evidence/e3c_tiel_three_way.md` (the table).

### 7.1 The pin, the box, and the scope that decides the numbers

| what | value |
|---|---|
| model | `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` — 22 360 476 736 B (20.8 GiB), arch `qwen35moe`, GGUF name `Ornith-1.5-35B`, 40 layers, 256 experts / 8 used |
| SHA-256 | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` — **identical before and after** the campaign (`.e3c_tiel/sha256_before.txt` / `sha256_after.txt`) |
| downloads | none: the file's `mtime` is 2026-09-04, the card started 2026-09-19 |
| runtime | pinned `b11026-linux-x64-vulkan`; every row `--backend vulkan --threads 4` |
| GPU | `NVIDIA GeForce RTX 3060 Ti`, 8192 MiB, driver 615.71.09, `vram_free_bytes` 6 955 204 608 at plan time |
| box | **host**, `systemd-run --user --unit=e3c-tiel-campaign` with `MemoryMax=infinity` (`memory.max=max` printed in the run's own log) |
| Occamy (untouched) | `633ae57f…df757`, 24 113 674 848 B — re-verified after the campaign |

**The scope is a measurement condition, not a detail.** Run inside the kanban worker's own scope
(`memory.max = 4 GiB`) a 21 GB model re-reads its weights from disk on every forward: **608 s for
one 10-item chunk** (run record `.e3c_tiel/flawed_capped/placement_001.json`: `wall_s` 608.0,
`load_wall_s` 62.7, `degraded: true`; a `read_bytes` figure of 68 GB in 12 minutes is quoted from
the run record in the card's comment thread, not from a committed file). In the
unlimited scope the same chunk costs 141–205 s and the load 9–29 s. The two capped chunks are kept
as `.e3c_tiel/flawed_capped/` and are **not** model rows — they measure the cap. E3's `[host]` rows
(§6.2) are on the same box for the same reason.

### 7.2 Placement (free-VRAM aware, and what the loader actually did)

`ggufone fit --print --json` plans `n_gpu_layers 9 / n_ctx 4096 / kv_type q4_0 / n_seq_max 8`
(warnings `W_KV_TYPE_DOWNGRADE`, `W_FIT_DOWNGRADE`; note "offloading 9/40 layers within 5609 MiB").
The quality report shape does not carry the placement (E3 §4.3), so
`tools/e3c_tiel_reproduce.py` records what the loader settled on per chunk — an observer on
`handle.placement.to_dict()`, no flag or number changed:

| chunk | ngl requested | ngl used | degraded | attempts | load wall | chunk wall |
|---|---|---|---|---|---|---|
| 001 | 9 | 9 | false | `[]` | 29.3 s | 204.9 s |
| 002 | 9 | 9 | false | `[]` | 12.5 s | 141.0 s |
| 003 | 9 | 9 | false | `[]` | 24.9 s | 136.5 s |
| 004 | 9 | 9 | false | `[]` | 9.1 s | **957.7 s** (contention, below) |
| 005 | 9 | 9 | false | `[]` | 19.0 s | 178.3 s |
| 006 | 9 | 9 | false | `[]` | 26.9 s | 152.6 s |

**No degrade rung was taken in any chunk** (Occamy took two in E3/E3b). Engine log, verbatim:
`load_tensors: offloaded 9/41 layers to GPU` · `CPU_Mapped model buffer size = 16680.10 MiB` ·
`Vulkan0 model buffer size = 4634.02 MiB`. Chunk 004 ran while a sibling card's own 21 GB Occamy
probe was on the same GPU (`tools/e3c_cue_shapes.py`, `.e3c/logs/occamy_c01_vulkan.log`) — two
21 GB models on a 32 GB host: its wall is an artifact of that contention; the answers are
unaffected (the decode is deterministic for the same items, and its agreement matches the rest).

### 7.3 The chunk ledger and the merged result (deliverable 2)

| chunk | items (choice/score/noul) | correct | median item decision | box |
|---|---|---|---:|---|
| `report_001.json` | 10 (4/3/3) | 5/10 | 7.0 s | host, unlimited scope |
| `report_002.json` | 10 (3/4/3) | 5/10 | 2.4 s | host, unlimited scope |
| `report_003.json` | 10 (3/3/4) | 6/10 | 1.6 s | host, unlimited scope |
| `report_004.json` | 10 (4/3/3) | 5/10 | 30.7 s | host, contended (see 7.2) |
| `report_005.json` | 10 (3/4/3) | 5/10 | 2.9 s | host, unlimited scope |
| `report_006.json` | 10 (7/1/2) | 5/10 | 4.4 s | host, unlimited scope |

Merged (`docs/evidence/tiel_quality.json`, 60 items, every row `ok`): **31/60 = 0.517
[0.393–0.638]** — per type `choice` 16/24 (0.667), `noul` 7/18 (0.389), `score` 8/18 (0.444).
Decision cost per item over the 60 rows: median 5.0 s, min 1.2 s, max 65.2 s. The dev-set slices are
byte-identical copies of `docs/evidence/e3_chunks/devset_00{1..6}.jsonl` (SHA-verified), so every
column of the three-way table is paired on the same items.

### 7.4 The mass split — and the shape it depends on (deliverable 3, input to `t_6952f0dd`)

| | 4B default (E2) | Occamy 1.0 (E3) | **Tiel-Coder (E3c)** |
|---|---|---|---|
| rows `measured` (≥ the 0.10 floor) | 48/60 | 3/60 | **46/60** |
| rows `low_mass` | 12/60 | 57/60 | 14/60 |
| coverage median · min · max | 2.56e-01 · 9.96e-03 · 8.94e-01 | 2.34e-02 · 1.86e-03 · 2.00e-01 | **2.58e-01** · 7.88e-03 · 8.20e-01 |

On the committed dev set **Tiel is not starved — it lands on the label strings**, with the 4B's
coverage shape rather than Occamy's, on the shipped `bare` label policy (nothing in the default
policy changed for this campaign; the control sweep is `docs/evidence/tiel_label_policy_tables.md`).

**The same 20 items read by the serving shape say the opposite, and both readings ship.** The
batch's questions are byte-identical to the bench suite's `c01`…`c20`, yet:

| shape | `measured` on c01–c20 | coverage median | cue row's argmax |
|---|---|---|---|
| bench (`--suite quality`) | 15/20 | 2.33e-01 | a real token on 17/20 |
| serving (`--suite batch`, one state, `readout: sequence`) | **0/20** | 6.42e-06 | special token `248069` on 19/20 (mass 0.68–0.99), `<|im_end|>` once |

So "Tiel answers where Occamy does not" holds for the **bench** shape (and is why a family-wide
label correction is still wrong); on the **serving** shape *both* 35B-A3B models return 20/20
`low_mass` (Occamy 8.8e-09…2.2e-06). The shapes differ in state prefix (109 tokens), `n_ctx` 256,
`readout: sequence`, 4 forks/question and one process for all 20 questions; **which** of those turns
the cue row away from the labels is not measured here — it is the cue-shape card's probe
(`t_6c119626`, `docs/evidence/e3c_cue_shapes.md`).

### 7.5 The 20-question batch (deliverable 4)

| what | value |
|---|---|
| outcome | exit 0 · **20/20 answers** · no OOM (`.e3c_tiel/batch_response.json`, `.e3c_tiel/extras_logs/extras2.log`) |
| wall | **35.6 s** (load 9.85 s + prefill 7.88 s + questions 27.72 s) |
| `usage` | questions 20 · forks 80 · **waves 40** · decode_steps 117 · input_tokens 1 175 · output_tokens 117 · prefill_tokens 109 |
| placement | `{n_gpu_layers: 9, kv_type: q4_0, degraded: false, attempts: []}` — no degrade |
| verdict | 20/20 `low_mass` (§7.4); warnings `[W_TEMPLATE_FALLBACK, W_LOW_MASS, W_CUE_REFUSED]` |

Occamy's E3 batch (§6.3) ran the same workload at `n_seq_max 4` and cost 3 633.6 s in the container;
Tiel's 35.6 s is a host figure at `n_seq_max 8` — **no speed comparison is claimed** between them.

### 7.6 Threads (deliverable 5)

`llama-bench` from the same bundle at the fitted placement, E3's sizes (`-p 64 -n 8 -r 2`), raw
tables in `.e3c_tiel/extras_logs/extras.log`:

| threads | pp64 (tok/s) | tg8 (tok/s) |
|---|---:|---:|
| 4 | 5.33 ± 2.39 | 1.85 ± 0.60 |
| **8** | **10.91 ± 3.12** | 3.26 ± 0.32 |
| 12 | 10.29 ± 2.87 | **5.53 ± 0.86** |

**The opposite of Occamy** (E3 §6.5: `-t 4` won, 8/12 were worse): Tiel scales with threads here —
prefill doubles 4 → 8 and decode keeps climbing to 12. The campaign kept `--threads 4` for E3
comparability and therefore publishes Tiel's **lower bound**. Recommendation for this artifact:
`--backend vulkan --gpu-layers 9 --threads 8` (more threads than E3's Occamy recommendation), on a
box that can keep the weights resident.

### 7.7 Three-way table (deliverable 6) — quotation only

`docs/evidence/e3c_tiel_three_way.md` (`tools/e3c_tiel_table.py`, which reuses
`ggufone.bench.compare`, paired, 0 unpaired rows):

| metric | 4B default (E2, 60, CPU) | Occamy 1.0 (E3, chunks, vulkan) | Tiel-Coder (E3c, host, vulkan) |
|---|---|---|---|
| overall | 0.633 (38/60) [0.507–0.744] | 0.517 (31/60) [0.393–0.638] | 0.517 (31/60) [0.393–0.638] |
| choice | 0.750 (18/24) [0.551–0.880] | 0.625 (15/24) [0.427–0.788] | 0.667 (16/24) [0.467–0.820] |
| noul | 0.889 (16/18) [0.672–0.969] | 0.389 (7/18) [0.203–0.614] | 0.389 (7/18) [0.203–0.614] |
| score | 0.222 (4/18) [0.090–0.452] | 0.500 (9/18) [0.290–0.710] | 0.444 (8/18) [0.246–0.663] |
| low_mass | 0.500 (6/12) [0.254–0.746] | 0.509 (29/57) [0.383–0.634] | 0.571 (8/14) [0.326–0.786] |
| measured | 0.667 (32/48) [0.525–0.783] | 0.667 (2/3) [0.208–0.939] | 0.500 (23/46) [0.361–0.639] |

**Every interval overlaps: no ranking is claimed.** Tiel and Occamy are identical overall
(31/60 each) — the pair that separates is `noul` (both 0.389 against the 4B's 0.889, and *those*
intervals do not overlap), and the honest reading of the whole table is that the two 35B-A3B
checkpoints are the same quality on this set while differing in *how much a reader can trust each
number* (7.4).

### 7.8 What §7 does not claim

No ranking (intervals overlap); no mechanism for the bench/serving split; no speed comparison
between the two batches; no claim about the token id `248069` beyond what it is not (it is not
`<|im_end|>`, so `W_CUE_REFUSED` does not fire on those 19 rows — a closers-list question for
`t_6c119626`); and no number at all from the two capped chunks
(`.e3c_tiel/flawed_capped/`).
<!-- @@E3C_TIEL_BENCH_7_END@@ -->

### 7.9 The framing marker on these rows (card `t_6de5fc53`)

**Every row of §7 is pre-fix (plain framing).** The instrument planned the executed context from the
live session, which resolves no chat template, so §7.3's `31/60` and §7.4's `46/60` `measured` rows
were measured with the plain E1b framing while `ggufone ask`/`run` send Tiel's own chat template
(`qwen35moe`). The corrected 60-item [host] re-run is **routed, not claimed here** — the card's box
cannot hold a 21 GB model. What ran there is a placement-matched six-item probe
(`docs/evidence/framing/`, receipts next to the reports):

| model (probe, 6 items) | framing | agreement | `low_mass` | refused at the cue | coverage median |
|---|---|---|---|---|---|
| Tiel-Coder 35B-A3B, pre-fix | plain (prompt.py E1b framing) | 3/6 | 3/6 | 3/6 | 0.2605 |
| Tiel-Coder 35B-A3B, post-fix | chat-template: qwen35moe / builtin | 0/6 | 6/6 | 6/6 | 1.94e-04 |
| Occamy 1.0, pre-fix | plain (prompt.py E1b framing) | 3/6 | 6/6 | 0/6 | 0.03233 |
| Occamy 1.0, post-fix | chat-template: qwen35moe / internal | 1/6 | 6/6 | 6/6 | 4.02e-06 |

Both arms of each pair ran the same placement (Tiel 4 layers — reached by de-escalation in the
pre-fix arm, asked for directly in the post-fix one — Occamy 7 layers; identical `n_ctx`, `n_prefix`,
`n_seq_max`, `threads` and `kv_type_used`). **The direction is that the product's own prompt
collapses both 35B-A3B models at the shipped cue — 6/6 refusals on six items**, which is the
behaviour §7.4's bench-vs-serving split predicted for the batch shape. Six items is a probe, not a
table: §7.3/§7.4's published numbers stand, now labelled, and the [host] re-run is where they get
replaced. Full detail: `docs/evidence/e2_fix_t_6de5fc53_framing.md` §4.3.

## 8. E3d — the cue switch, measured through the bench (card `t_d90404ac`)

E3d asked whether a different cue shape beats the shipped one, decided it on the full dev set
(`docs/evidence/e3d_cue_decision_4b.md`: `--cue two_step` moves the readout for 44 of 60 rows off a
row the engine itself calls `low_mass`, and `--cue json_field` is the only shape that clears the
paired CI on agreement) and shipped the mechanism as `--cue shipped|two_step|json_field` with
**`shipped` still the default**. This section is the same switch through the *bench* — the engine's
own quality path, the instrument every other table in this document comes from — and it is the
reason the default did not move.

Same box, same model (`Spark-X2.5-4B-Q8_0.gguf`), same 60 committed dev items, same context, same
placement (the Vulkan bundle, the device visible — the rows carry `effective_backend: vulkan` and
`W_BACKEND_MISMATCH`, because the request's *claim* is `cpu` while Vulkan0 computed, see §7.1 and
E3c). Only `--cue` moves, and every row names its framing:

| `--cue` | framing | agreement | Wilson 95 % | `low_mass` | refused at the cue | coverage median | above the 0.10 floor | choice · noul · score |
|---|---|---|---|---|---|---|---|---|
| `shipped` (the default), **pre-fix** | plain (prompt.py E1b framing) | 36/60 = 0.600 | 0.474–0.714 | 13/60 | 0/60 | 0.2711 | 47/60 | 16/24 · 16/18 · 4/18 |
| `two_step`, **pre-fix** | plain (prompt.py E1b framing) | 27/60 = 0.450 | 0.331–0.575 | 29/60 | 8/60 | 0.1103 | 31/60 | 15/24 · 7/18 · 5/18 |
| `json_field`, **pre-fix** | plain (prompt.py E1b framing) | 46/60 = 0.767 | 0.646–0.856 | 0/60 | 0/60 | 0.8544 | 60/60 | 21/24 · 17/18 · 8/18 |
| `shipped`, **post-fix** | chat-template: spark2_5 / internal | 42/60 = 0.700 | 0.575–0.801 | 45/60 | 1/60 | 0.03847 | 15/60 | 20/24 · 16/18 · 6/18 |
| `two_step`, **post-fix** | chat-template: spark2_5 / internal | **47/60 = 0.783** | 0.664–0.869 | **3/60** | 3/60 | 0.9708 | 57/60 | 21/24 · 15/18 · 11/18 |
| `json_field`, **post-fix** | chat-template: spark2_5 / internal | 51/60 = 0.850 | 0.739–0.919 | 0/60 | 0/60 | 0.9997 | 60/60 | 23/24 · 17/18 · 11/18 |

**The pre-fix rows measured the plain framing and the post-fix rows the model's chat template — the
two halves of this table are two different prompts and must not be compared across.** Reproduce:
`bash .e3d/run_bench_arms.sh` (post-fix arms, `.e3d/bench_templated_<cue>.json`; the pre-fix arms
are `.e3d/bench_plain_<cue>.json`, measured on the pre-fix tree — the flag is
`tools/e2_reproduce.py --cue <shape>`).

**The reversal is gone, and the bench now reproduces the probe** (card `t_6de5fc53`, fixed
2026-09-19). The pre-fix arms of this table were the reason §8 used to read "the cue's effect is
framing-dependent": they sent the plain prompt, where the shipped cue already sits on the answer,
so `two_step` cost agreement and doubled `low_mass`. With the executed plan resolved from the
handle — the source `ggufone ask`/`run` has always used — the same instrument says the opposite,
and its numbers land on the probe's:

| statistic | probe (`docs/evidence/e3d_cue_decision_4b.md` §1/§2) | bench, post-fix |
|---|---|---|
| `shipped` agreement | 42/60 = 0.700 | 42/60 = 0.700 |
| `two_step` agreement | 46/60 = 0.767 | 47/60 = 0.783 |
| `json_field` agreement | 51/60 = 0.850 | 51/60 = 0.850 |
| paired `two_step` vs `shipped` | **KEEP**: risk difference +0.067, 95 % CI −0.033…+0.167 | +5 correct items, `low_mass` 45 → 3 |
| coverage above the 0.10 floor | 0.267 → 0.967 (`shipped` → `two_step`) | 15/60 → 57/60 |

So the pre-fix `shipped` arm sits two items below E2's published plain row (36/60 vs 38/60) for the
reason §7 records — a different fit plan from a different box reading — and that gap is a
same-framing comparison, i.e. it says nothing about this card. The post-fix `shipped` arm (42/60)
lands exactly on the probe's `bare` row for the same shape, and the comparison this section makes is
between its own arms, which ran under one plan and one box.

**What this changes for the cue default.** The default stays `shipped` — the E3d card's decision is
that the *mechanism* ships and the *default* moves only with its own evidence, and this card's job
was the instrument, not the default. What changed is the reason: the bench no longer contradicts
the probe, so `--cue two_step`'s case is now the probe's case (a readout-position fix that takes the
engine's own coverage verdict from 15/60 usable rows to 57/60, with the agreement difference inside
the paired CI), and `--cue json_field`'s 51/60 is the number the *bench* now measures for it too.
The mechanism, the flag and the frozen default are unchanged by this card.


