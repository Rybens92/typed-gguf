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

| gate | claim | evidence |
|---|---|---|
| **A-E2-1** | `--suite latency`: model_load, prefill tok/s at {256, 2k, 8k}, per-question ms at {2,4,10} candidates, wave scaling N=1..16, warm cache; p50/p95 over ≥5 runs | `uv run pytest -q tests/test_bench.py` → 41 passed (the suite's shape, the percentile definition and the wave accounting are pinned offline); live: `tools/e2_reproduce.py --suite latency` on both models → `docs/evidence/e2_latency.json`, `docs/evidence/e2_qwen_latency.json` (§3.1 tables; the 4B 8k row is 26 min/sample on this box and is published on the 0.8B model, cost note in §3.1) |
| **A-E2-2** | `--suite throughput` per backend, same model | `tools/e2_reproduce.py --suite throughput --backend all` → one row per documented backend, `measured: false` + the reason where no bundle exists (§3.2) |
| **A-E2-3** | `--suite quality` on the committed dev set (≥50 items, ≤200 tokens, ≥3 types): per-type + overall exact match with Wilson CIs | `devset.validate()` in `tests/test_bench.py` (60 items: 24 choice / 18 score / 18 noul, ≤200 words, gold inside the criteria, provenance); live per-item run → `docs/evidence/e2_quality.json` + the 4B run (§2 tables). The ≤200-*token* budget is re-measured with the real vocabulary in `tests/test_bench_live.py` |
| **A-E2-4** | `--suite calibration`: reliability bins, ECE, confidence/coverage correlation | `tools/e2_reproduce.py --suite calibration` → 10 equal-width bins, ECE per bin set, Pearson r(confidence, coverage), and all three confidence modes recomputed from the same stored distributions (§3.3) |
| **A-E2-5** | `--suite determinism`: 3 repeats, byte-identical after stripping `timings`, per backend | `tools/e2_reproduce.py --suite determinism --backend all --threads 1` → 3 digests per backend with `identical: true` (§3.3); the digest contract (timings-only deltas never move the hash) is pinned in `tests/test_bench.py` |
| **A-E2-6** | recon numbers re-measured side by side and tagged | §3.4 table: every recon row next to this box's measurement with the delta explained; the coordinator's post-fix Vulkan run is cited in §3.7 |
| **A-E2-7** | no registry, no network | `test_no_suite_touches_the_registry_or_the_network` poisons `socket.*` and `store.{load_registry,resolve,registry_path,data_home}` and runs **all five suites** green; `test_the_bench_modules_never_import_the_registry_at_module_level` (AST) and `test_the_model_must_be_a_file_never_an_alias` (`E_BENCH_MODEL`) |
| **A-E2-8** | `runtime-matrix.yml` gains one non-linux job that runs the engine | `.github/workflows/runtime-matrix.yml` → `macos-engine-smoke`: pinned Metal bundle + a 0.5B GGUF, then `bench --suite latency`, `--suite determinism` (asserts one digest) and `--suite quality --items 6` |
| **exit codes** | `bench` is no longer a stub | `test_the_engine_commands_are_no_longer_stubs[bench]` (moved out of the frozen list): no `--suite` → 2 `E_BENCH_SUITE`; unknown suite → 2; `--model` alias/non-file → 2 `E_BENCH_MODEL`; failed suite gate → 1 |

## 2. The two headline questions

**(b) `waves = 8` for `forks = 5` is the decode-batch count, not a fork bucket.** `usage.waves` is the number of `llama_decode` calls a decision issues after the prefill: one per candidate group (the question suffix) plus one per extra candidate token, summed over questions. The suite prints that decomposition:
* Spark-X2.5-4B: `groups=1`, `suffix_decodes=1`, `step_decodes=1`, `waves=2` for the canonical 5-candidate request at `n_seq_max=6`; the per-question rows in §3.1 print `waves`/`forks`/`decode_steps` for 2/4/10 candidates.
* Qwen3.5-0.8B: `groups=1`, `suffix_decodes=1`, `step_decodes=1`, `waves=2` for the canonical 5-candidate request at `n_seq_max=6`; the per-question rows in §3.1 print `waves`/`forks`/`decode_steps` for 2/4/10 candidates.

**(a) A fresh process pays the load, a server amortises it.** Measured (`load_amortisation` in the latency reports):
* Spark-X2.5-4B: `model_load_ms` p50 = 855.0 ms; `serve` = 9,757.8 ms per request; one-shot CLI = 10,612.8 ms per request (5 calls each).
* Qwen3.5-0.8B: `model_load_ms` p50 = 1,571.7 ms; `serve` = 1,343.2 ms per request; one-shot CLI = 2,914.9 ms per request (5 calls each).

On a GPU the same shape holds for shader compilation (SPEC R2): the pipeline cache is per *process* unless the driver's on-disk cache is warm — §3.6 of `BENCHMARKS.md`, and `tools/e2_vulkan_probe.py` for the mechanism on lavapipe.

## 3. Quality, calibration and determinism

* **Spark-X2.5-4B** — overall 38/60 = 0.633 (95 % CI 0.507–0.744); per type: choice 18/24 = 0.750, noul 16/18 = 0.889, score 4/18 = 0.222.
* **Qwen3.5-0.8B** — overall 28/60 = 0.467 (95 % CI 0.346–0.591); per type: choice 13/24 = 0.542, noul 11/18 = 0.611, score 4/18 = 0.222.

## 5. Mutation testing (Tier M)

```
cd <tree> && HOME=... UV_CACHE_DIR=... \
  uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 3
uv run --extra dev --with mutmut python tools/mutmut_driver.py export-cicd-stats
```

Scope (the pair this card left in `pyproject.toml`): `source_paths = ["src/ggufone/bench"]` (4
files: `harness.py`, `suites.py`, `devset.py`, `__init__.py`) with the bench gate as the test
selection (`tests/test_bench.py`). Two `BlockingIOError: Resource temporarily unavailable`
(pid cap, shared box) interrupted the sweep; the retry loop resumed from mutmut's cache, so the
result below covers the whole scope in three attempts.

```
mutants/…/mutmut-cicd-stats.json
{"killed": 1843, "survived": 1228, "total": 3171, "no_tests": 100, "skipped": 0,
 "suspicious": 0, "timeout": 0, "check_was_interrupted_by_user": 0, "segfault": 0}
killed / (killed + survived) = 1843 / 3071 = 60.0 %
```

Tier M takes a **soft threshold**: the run is required, the score is reported, survivors are
findings for the next card rather than a gate. The distribution is what a reviewer should read,
and it is lopsided by design of the code, not by accident of the tests:

| survivors | where | why it survives |
|---:|---|---|
| 426 | `harness.render_report` | the markdown renderer: the gate pins that every table's cells line up (`test_every_rendered_table_has_columns_that_line_up`) and that the latency section renders, but not the wording of every line — most survivors are cosmetic (labels, separators) |
| ~300 | the suites' per-part helpers (`_run_quality`, `_per_question_rows`, `_wave_scaling_rows`, `_throughput_row`, `_determinism_request`, …) | the rows are asserted for *shape* (n ≥ runs, columns present, `prefill_reused`), not for every metadata field; a mutant that changes `"backend"` to `"BACKEND"` in a row does not fail anything |
| ~150 | `devset.validate/load/…` | the gate asserts the *committed* set is valid and that a few injected problems are reported, not that every branch of the validator is reached |
| 40 | **`harness` statistics** (`percentile`, `ratio_summarise`, `wilson_interval`, `ece`, `pearson`) | the tests pin *documented examples* (textbook Wilson, a hand-computed ECE, degenerate Pearson, the interpolated percentile at several q) — the survivors are the branches those examples do not touch, e.g. `ratio_summarise`'s zero-denominator path beyond one case. **This is the cluster worth pinning next.** |
| ~120 | `LiveModel`, `classify_runtime`, `spec_for`, report metadata | the live path (`LiveModel`) is deliberately outside the offline gate: it runs in `tests/test_bench_live.py` and in every published table, so the mutation gate cannot see it |
| 100 | `no tests` (not "survived") | uncovered lines (`DevItem.to_json`, some error branches) — counted separately, never as kills |

## 6. Live gates on this box

```
GGUFONE_RUNTIME_DIR=<bundle> GGUFONE_BENCH_MODEL=<path.gguf> \
  uv run pytest -q --run-network tests/test_bench_live.py -s
→ 4 passed in 244.15s
```

* the dev set fits the 200-**token** budget on a real vocabulary (re-read with the model's own
  tokenizer, not the word proxy the offline gate uses);
* the latency suite runs end to end on a real model (model load > 0, prefill tok/s, warm cache
  with `prefill_reused: true`, model facts from the GGUF header);
* determinism holds on this box (3 repeats, one digest, `ok: true`);
* the quality suite answers six real dev items, with each answer's probabilities summing to 1.

The three suites whose output is wall-clock independent were run next to this mutation sweep; the
published tables are the ones in `docs/BENCHMARKS.md`, regenerated with `tools/e2_reproduce.py`.

## 7. Honest limits of this evidence

* The container has **no GPU**: Vulkan/CUDA throughput rows are `measured: false` with the
  reason; the recon's Vulkan numbers cannot be reproduced here (SPEC R7/R8 behaviour).
* The box is shared and capped at 2 CPU-seconds/s; `p95` values are "this box, today".
* E2 is report-only for quality (S-11): no minimum agreement is claimed.
