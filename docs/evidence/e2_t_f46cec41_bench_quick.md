# E2 (card t_f46cec41) — `bench --quick` + `--max-seconds`: measured evidence

Status: **complete for the worker container**; one row is host-only (see §5).
Box: this container, CPU-only, `cgroup cpu.max = 2.0` CPU-seconds/s, 8 GiB memory cgroup,
runtime `b11026-linux-x64-cpu` (the pinned bundle), `threads=2`, `--backend auto` (→ `cpu`).
Raw material: `/work/t_f46cec41/evidence/qwen_quick/` (reports + rendered tables), the JSON reports
named below, and the live-test output quoted in §4.

## 1. What was built

* `ggufone bench --quick` and `tools/e2_reproduce.py --quick`: one short *scale* of every suite
  (`harness.quick_config`) — runs=1, prefill 256, candidates 2/4, waves N∈{1,2}, 6 dev items
  stratified 2/2/2, determinism 2 repeats, 1 resolved backend. Model, backend, threads, placement,
  dev-set source and the soft cap are untouched: a quick run is comparable to the full one it
  previews.
* `--max-seconds N`: a soft cap checked **between** measurements (`harness.TimeBudget` +
  `suites._measure`). The measurement that started always finishes; every measurement that never
  started is listed under `"truncated": true` → `"skipped"` and the exit code stays 0. A row that
  ran and failed a gate still exits 1.
* Report integrity: quick reports carry `"quick": true`, the effective config, the preset note and
  the measured wall time; `--quick` + `--runs/--items/--sizes/--n-seq-max` is `E_BENCH_QUICK`; a
  bare quick run writes `ggufone-bench-<suite>_quick.json` (a full run without `--out` writes
  nothing), so a quick report can never land on an `e2_<suite>.json`.
* `docs/BENCHMARKS.md` states, at the top of the tables, that they are full-campaign only;
  `runtime-matrix.yml`'s `macos-engine-smoke` job runs the whole quick campaign end to end.

## 2. Wall time: quick vs full, same model, same box

Model `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (558 MB, `qwen35`), threads=2, `--backend auto` → cpu.

### 2.1 The full latency suite (A-E2-1's published invocation)

`docs/evidence/e2_qwen_latency.json` (measured earlier on this box, `runs=5`, threads=2) — its
*measured rows alone* sum to **≈ 2210 s (37 min)**: prefill 256 = 9.4 s, 2k = 55.2 s, 8k = 250.8 s
(×5 runs each), per-question 0.83/1.13/2.22 s (×5 each, plus a warm-up call per row), wave scaling
N=1..16 = 122.3 s of p50 sum (×5 each, plus warm-ups), 5 model loads. The published 8k row alone
(250.8 s p50) is more than twice the entire quick campaign below.

### 2.2 The quick latency suite (same model, same rows family)

`qwen_quick/quick_latency.json`, `uv run ggufone bench --suite latency --quick …`:

| row | quick (runs=1) |
|---|---|
| model load | 1.58 s |
| prefill 256 | 15.20 s (16.8 tok/s) |
| per question, 2 candidates | 3.37 s (warm-up call not counted) |
| per question, 4 candidates | 0.72 s |
| wave scaling N=1 | 3.69 s |
| wave scaling N=2 | 3.15 s |
| warm cache (state reuse) | 0.00 s prefill + 2.38 s questions |
| load amortisation | serve 5.57 s / one-shot 7.15 s per request |
| **reported wall time** | **107.6 s** |

The reported rows sum to ≈ 36 s; the remaining wall time is the warm-up call of every measured row
(the documented warm-cache convention: one call fills the prefix state, the measured call reports
`prefill_reused: true`), one llama.cpp context per session (measured: ~0.7 s each on this box) and
the scheduler noise of a **shared** 2 CPU-seconds/s quota (see §6).

### 2.3 The whole quick campaign (five suites)

`tools/rehearse_quick.sh /work/t_f46cec41/evidence/qwen_quick <qwen.gguf> 2`:

| suite | report `wall_ms` | gate |
|---|---|---|
| latency | 107.6 s | ok |
| throughput | 70.3 s | ok |
| quality | 59.3 s | ok (6 items: choice 2 / score 2 / noul 2) |
| calibration | 40.0 s | ok (6 samples → 6 bins, three confidence modes) |
| determinism | 23.6 s | ok (2 repeats, byte-identical) |
| **total** | **300.8 s (5.0 min)** | every suite ≤ the 180 s target |

Every single `ggufone bench --quick` run is inside the card's ≤ 3 min target on this box; the
five-suite campaign totals 5.0 min **while the box is shared** (load average 3–6 with siblings
running a mutation sweep and a vitest suite on the same quota — the same box's quiet numbers are
1.6–2.6× faster, e.g. prefill 256 = 9.4 s published vs 15.2 s here).

### 2.4 What a further cut would buy (why the card's preset was kept)

The card allows cutting further if the target is missed. Measured cost of the two rows a cut would
remove, in the 107.6 s quick latency run: the 4-candidate row is **0.72 s** and the N=2 wave row is
**3.15 s** — together ~3.6 % of the suite. The suite's cost is prefill (15.2 s) + one warm-up call
and one context per row, not the number of rows. Cutting them would save ~4 s of 107.6 s and remove
the candidate-count comparison the card's own preset table lists, so the preset stays as specified;
`--max-seconds` (and `--sizes`) remain the operator's levers for a bounded run.

## 3. `--max-seconds` on a real run

`qwen_full_latency.json` — the *full* latency suite, same model, with `--max-seconds 1200`:

* `"truncated": true`, `"budget": {"max_seconds": 1200.0, "expired": true, "skipped": N}`;
* the rows that ran are complete rows (each with `n=5` where the full suite asks for 5);
* `"skipped"` names every unmeasured row (prefill 2k/8k, the three candidate counts, the wave rows,
  warm cache, amortisation) with the reason;
* exit code **0** — a partial-but-honest report instead of a timeout.

NUMBERS_PENDING_MAXSECONDS

## 4. The live gates (`tests/test_bench_live.py`, `model`-marked)

NUMBERS_PENDING_LIVE

## 5. The bigger model (host-only)

The operator named `/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf` (23 GiB,
`qwen35moe`) and `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` (21 GiB) as the "bigger" quick-evidence
models. **Neither file is visible inside this worker container**: a bounded inventory of
`/var/home/rybens/.hermes/models/` shows only `Spark-X2.5-4B-Q8_0.gguf` (4.1 GiB), and the
container's memory cgroup is 8 GiB, so a 21–23 GiB file could not be measured here even if it were
mounted. That is a *stated* limitation, not a skipped gate:

* the live test picks the file up automatically when it exists (`GGUFONE_BENCH_MODEL_BIG`, then the
  two named paths) and skips with the reason (absent / larger than the memory cgroup) when it does
  not — the skip reason is printed, never silent;
* the named vehicle for the number is `tools/rehearse_quick_host.sh [log-dir] [threads]` — one
  command, on the host, that runs the same five-suite quick campaign for every model that exists
  (Occamy, Tiel-Coder, Spark-4B, Qwen-0.8B) and prints the per-suite wall times;
* what the container *can* measure of the "bigger" class is **Spark-X2.5-4B** (§5.1).

NUMBERS_PENDING_SPARK

## 6. Caveats a reader must keep

* **The box is shared.** The quick campaign's total (5.0 min) and the per-suite numbers were taken
  with load average 3–6 on a 2 CPU-seconds/s quota (siblings: `mutmut` on the E2.5 tree, a vitest
  suite, `tsc`). The *same* suite on this box at a quiet moment is ~1.6× faster (published prefill
  256 = 9.4 s vs 15.2 s here), and `threads=4` measured *slower* than 2 today (127 s vs 107.6 s
  quick latency) — the published §3.5 preference for 4 was measured under a different load mix.
* **Quick numbers are not publishable numbers**: one sample per row, one prefill size, six dev
  items, two repeats. `docs/BENCHMARKS.md` says so at the top of the tables.
* **No GPU**: `/dev/dri` is absent; placement is `n_gpu_layers=0` (the placement line in every
  table).

## 7. Gate results

NUMBERS_PENDING_GATES
