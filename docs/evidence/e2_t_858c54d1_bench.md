# E2 — benchmarks: harness, dev set, five suites, published tables

Card `t_858c54d1` · branch `main` (this repo has no remote; commits are local on the shared tree)
· Tier **M** (default — the card declares none) · report schema `ggufone.bench/v1`

Every claim below is a command plus its real output. The machine-readable reports live in
`docs/evidence/e2_*.json`; the published tables are `docs/BENCHMARKS.md`, and one command
regenerates each of them (`tools/e2_reproduce.py --suite <name>`, printed inside every table).

Environment for every live number: pinned runtime
`/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu`, the pinned default model
`XHToken/Spark-X2.5-4B-GGUF:Q8_0` (4 375 021 152 B, sha256 `5c2c3c19…9dea2`) and
`unsloth/Qwen3.5-0.8B-GGUF` UD-Q4_K_XL (558 772 480 B). Container: 24 CPUs seen,
**2 CPU-seconds/s cgroup quota**, no GPU (`/dev/dri` absent).

## 0. What landed

| area | files |
|---|---|
| harness: statistics (interpolated p50/p95, Wilson, reliability bins, ECE, Pearson), the `ModelLike` seam, backend resolution, the timings-stripped digest, the markdown renderer | `src/ggufone/bench/harness.py` |
| the five suites (`latency`, `throughput`, `quality`, `calibration`, `determinism`) + the wave-accounting decomposition | `src/ggufone/bench/suites.py` |
| the committed dev set (60 items) and its contract (`validate`, `gold_key`, `request_for`) | `src/ggufone/bench/devset.py`, `src/ggufone/bench/devset.jsonl` |
| `ggufone bench --suite …` (flags, exit codes, `--out`) | `src/ggufone/cli.py` |
| one command per published table | `tools/e2_reproduce.py` |
| gates: 40 offline tests + 4 live (`model`-marked) tests | `tests/test_bench.py`, `tests/test_bench_live.py` |
| the model-free seam the suites are tested through | `tests/fake_engine.py` (`BenchModel`) |
| published tables + provenance | `docs/BENCHMARKS.md` |
| a non-linux CI job that runs the engine | `.github/workflows/runtime-matrix.yml` |

## 1. Gate table (A-E2-1 … A-E2-8)

<!-- GATES -->

## 2. The two headline questions

<!-- HEADLINES -->

## 3. Quality, calibration and determinism

<!-- QUALITY -->

## 4. Honest limits of this evidence

* The container has **no GPU**: Vulkan/CUDA throughput rows are `measured: false` with the
  reason; the recon's Vulkan numbers cannot be reproduced here (SPEC R7/R8 behaviour).
* The box is shared and capped at 2 CPU-seconds/s; `p95` values are "this box, today".
* E2 is report-only for quality (S-11): no minimum agreement is claimed.
