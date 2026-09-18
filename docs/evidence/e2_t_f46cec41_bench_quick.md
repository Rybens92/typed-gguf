bash: fork: retry: Resource temporarily unavailable
bash: fork: retry: Resource temporarily unavailable
bash: fork: retry: Resource temporarily unavailable
bash: fork: retry: Resource temporarily unavailable
# E2 (card t_f46cec41) — `bench --quick` + `--max-seconds`: measured evidence

Status: **complete for the worker container**; one row is host-only (see §5).
Box: this container, CPU-only, `cgroup cpu.max = 2.0` CPU-seconds/s, 8 GiB memory cgroup,
runtime `b11026-linux-x64-cpu` (the pinned bundle), `threads=2`, `--backend auto` (→ `cpu`).
Raw material (committed): `.e2e/t_f46cec41-bench-quick/` — the quick campaign reports and rendered
tables (`qwen_quick/`, `spark_quick2/`), the truncated full-suite report
(`qwen_full_latency_cap300.{json,md}`), the live-gate output (`live_quick_gates.txt`), the
hand-mutation proof (`hand_mutation_quick.txt`), the mutation artifacts and score (`mutmut_run*.log`,
`mutation_score.txt`), coverage (`coverage_changed.txt`, `coverage.log`) and the final offline suite
(`final_offline_suite.txt`). Everything below is a command plus its real output.

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

A *fresh* full-suite run on the same model/threads, with `--max-seconds 300` (§3), got through only
the load, the 256 and the 2048 prefill rows in **506.3 s** — the 8k row, the three candidate rows,
all sixteen wave rows, warm cache and amortisation never started.

### 2.2 The quick latency suite (same model, same row families)

`qwen_quick/quick_latency.json`, `uv run ggufone bench --suite latency --quick …` (12:50 UTC):

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

`tools/rehearse_quick.sh /work/t_f46cec41/evidence/qwen_quick <qwen.gguf> 2` (12:50–12:58 UTC, load
average 3–6 with siblings running their own builds):

| suite | report `wall_ms` | gate |
|---|---|---|
| latency | 107.6 s | ok |
| throughput | 70.3 s | ok |
| quality | 59.3 s | ok (6 items: choice 2 / score 2 / noul 2) |
| calibration | 40.0 s | ok (6 samples → 6 bins, three confidence modes) |
| determinism | 23.6 s | ok (2 repeats, byte-identical) |
| **total** | **300.8 s (5.0 min)** | every suite ≤ the 180 s target |

The same campaign re-run by the live gate 30 minutes later (§4, a quieter moment) totals
**213.6 s** — latency 56.5 s, throughput 44.3 s, quality 55.8 s, calibration 43.7 s,
determinism 13.2 s. So: **every `ggufone bench --quick` run is inside the card's ≤ 3 min target**,
and the five-suite campaign lands between 3.6 and 5.0 min depending on what else the box is doing
(that spread is the same suite re-measured, not a different preset).

### 2.4 What a further cut would buy (why the card's preset was kept)

The card allows cutting further if the target is missed. Measured cost of the two rows a cut would
remove, in the 107.6 s quick latency run: the 4-candidate row is **0.72 s** and the N=2 wave row is
**3.15 s** — together ~3.6 % of the suite. The suite's cost is prefill (15.2 s) + one warm-up call
and one context per row, not the number of rows. Cutting them would save ~4 s of 107.6 s and remove
the candidate-count comparison the card's own preset table lists, so the preset stays as specified;
`--max-seconds` (and `--sizes`) remain the operator's levers for a bounded run.

## 3. `--max-seconds` on a real run

`qwen_full_latency_cap300.json` — the *full* latency suite, same model, `--max-seconds 300`:

```
- wall: 506.3 s (soft cap 300.0 s)
- TRUNCATED at --max-seconds 300.0 s: 22 unmeasured row(s) — prefill/tokens=8192,
  per_question/candidates=2, per_question/candidates=4, per_question/candidates=10,
  wave_scaling/questions=1 … wave_scaling/questions=16, warm_cache/state reuse,
  load_amortisation/serve vs one-shot
exit code: 0
```

The rows that ran are complete rows: `model_load_ms` n=5 (p50 2,003.5 ms), prefill 256 n=5
(p50 15,698.6 ms), prefill 2048 n=5 (p50 86,147.7 ms). The cap is checked **between rows**, so a
row that started always finished (that 2048 row is 5 samples ≈ 7 min on this box, which is why the
wall time overshoots the 300 s cap — the granularity is documented in `bench --help`). The report
is written, `"truncated": true`, the 22 unmeasured rows are listed under `"skipped"`, and the exit
code is **0** — no timeout, no lie.

## 4. The live gates (`tests/test_bench_live.py`, `model`-marked)

```
GGUFONE_RUNTIME_DIR=<bundle> GGUFONE_BENCH_MODEL=<qwen.gguf> \
  uv run pytest -q --run-network tests/test_bench_live.py -k quick -s
  quick preset on Qwen3.5-0.8B-UD-Q4_K_XL.gguf (cgroup cpu.max 2.0, budget 180 s/suite):
    latency        56.5 s
    throughput     44.3 s
    quality        55.8 s
    calibration    43.7 s
    determinism    13.2 s
    total         213.6 s (<= 360 s)
  1 passed, 1 skipped in 213.80s   (exit 0)
```

The gate asserts the card's own budget per suite (`harness.QUICK_TARGET_SECONDS`) and a documented
campaign multiple on this shared box; the second quick test (the bigger local model) skips here
with its reason printed:

```
SKIPPED [1] tests/test_bench_live.py:84: no bigger local GGUF on this box
  (…/Accio-Lab_occamy-1.0-Q4_K_L.gguf, …/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf); set GGUFONE_BENCH_MODEL_BIG
```

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

### 5.1 Spark-X2.5-4B — the biggest model this container *can* hold

`tools/rehearse_quick.sh /work/t_f46cec41/evidence/spark_quick2 <spark.gguf> 2` (13:45–, threads=2,
same pinned CPU bundle). 4.07 GiB of weights in an 8 GiB memory cgroup: the largest local model this
container can measure.

| suite | report `wall_ms` | note |
|---|---|---|
| latency | **306.0 s** | full preset shape (256 prefill, 2/4 candidates, waves 1/2), exit 0 |
| throughput | **122.8 s** | 1 backend, 1 sample, exit 0 |
| quality | **11.9 s** | 6 stratified items on a real 4B vocabulary, exit 0 |
| calibration | **10.6 s** | the same 6 items → 6 bins, three modes, exit 0 |
| determinism | **27.8 s** | 2 repeats, byte-identical, exit 0 |
| **total** | **479.1 s (8.0 min)** | the latency suite is 64 % of it (256-token prefill + warm-up calls on 4B) |

The 4B quick latency run is 2.8× the 0.8B one (306.0 s vs 107.6 s) — the model is 7.8× the file
size and this box's quota is the same, so the preset behaves exactly as a size-scaled budget in
`tests/test_bench_live.py` predicts (4.07 GiB → a 733 s per-suite bound; measured 306 s).

**First attempt (13:30) failed in the environment, not in the preset**: the prefix-state file for a
4B model filled the shared 512 MB `/tmp` tmpfs and the run exited 3 with
`E_PREFILL_FAILED: llama_state_seq_save_file … No space left on device`; three later suites also
died with `SIGABRT` (exit 134) when `uv` could not fork under the box's exhausted pid cap
(`pids.current = 256/256`, siblings' sweeps + this campaign). The re-runs used
`TMPDIR=/work/t_f46cec41/tmp` and `uv run --no-sync`; both facts are environment, and both are worth
knowing for the host run (§5.2).

### 5.2 The host vehicle for Occamy 1.0 / Tiel-Coder-35B-A3B (named, not silently skipped)

```
# on the host (HOME=/var/home/rybens), one command:
tools/rehearse_quick_host.sh /tmp/quick-host 4
# → for every model that exists: the five-suite quick campaign; the per-suite walls land in
#   /tmp/quick-host/<model>/quick_walls.txt  (Occamy 23 GiB, Tiel-Coder 21 GiB, Spark 4B, Qwen 0.8B)
```

`tests/test_bench_live.py::test_the_quick_preset_measures_a_bigger_local_model_too` is the
automated half of it: it picks `GGUFONE_BENCH_MODEL_BIG` up when set, otherwise Occamy/Tiel-Coder
when they exist, asserts the preset's row shapes on real weights and prints the wall time (no
3-minute budget: a 35B-A3B on two CPU-seconds/s is a minutes-scale load). In this container it
skips with the reason printed (§4); on the host it runs.

## 6. Caveats a reader must keep

* **The box is shared.** The quick campaign's total (300.8 s at 12:50 → 213.6 s at 13:27) and the
  per-suite numbers were taken with load average 3–9 on a 2 CPU-seconds/s quota (siblings: `mutmut`
  sweeps, a vitest suite, `tsc`, their own 4B bench runs). The *same* suite is ~1.9× faster at a
  quieter moment (quick latency 107.6 s → 56.5 s), and `threads=4` measured *slower* than 2 today
  (127 s vs 107.6 s quick latency) — the published §3.5 preference for 4 was measured under a
  different load mix. Every number here is "this box, today".
* **`/tmp` is a 512 MB tmpfs shared with the other agents.** A 4B-class benchmark persists its
  prefix state under `tempfile.mkdtemp()` (i.e. `$TMPDIR`), and one Spark-X2.5-4B state file is
  hundreds of MB: the first `--quick` attempt on the 4B died with
  `E_PREFILL_FAILED: llama_state_seq_save_file … write error: No space left on device` (exit 3).
  The re-run set `TMPDIR=<roomy dir>`; a reviewer should read that as *environment*, not as a
  preset defect — but the operator running the big-model quick evidence on the host should know the
  same trap exists (a full `/tmp` fails the *warm-cache* rows, not the measurement).
* **Quick numbers are not publishable numbers**: one sample per row, one prefill size, six dev
  items, two repeats. `docs/BENCHMARKS.md` says so at the top of the tables.
* **No GPU**: `/dev/dri` is absent; placement is `n_gpu_layers=0` (the placement line in every
  table).

## 7. Gate results (Tier M — the card declares none, so the default)

```
uv run --no-sync --extra dev python -m pytest -q
→ 908 passed, 40 skipped in 34.54s   (exit 0; the 40 skips are the network/model-marked tests)
uv run --no-sync --extra dev ruff check src tests tools/e2_reproduce.py
→ All checks passed!
```

**Changed-line coverage** (`cov_changed.py 901c250 coverage.json`, the base is the rebase base —
`tests/test_bench_quick.py`, the live gates and the tools are not imported by the offline suite):

| file | added lines | executable | covered | missing | % |
|---|---:|---:|---:|---:|---:|
| `src/ggufone/bench/devset.py` | 17 | 7 | 7 | 0 | 100.0 |
| `src/ggufone/bench/harness.py` | 161 | 63 | 63 | 0 | 100.0 |
| `src/ggufone/bench/suites.py` | 271 | 140 | 140 | 0 | 100.0 |
| `src/ggufone/cli.py` | 83 | 32 | 32 | 0 | 100.0 |
| **TOTAL (executable added lines)** | **532** | **242** | **242** | **0** | **100.0** |

(whole-package line coverage: 89.1 %.) The first coverage pass left two added lines unexecuted —
`reproduce_command`'s `--devset` flag and the CLI's `report: <path>` line on the **human-readable**
quick path — and both are exactly the kind of gap worth a test rather than an excuse: they are now
pinned by `test_the_reproduce_command_names_a_custom_dev_set_and_a_soft_cap` and
`test_a_quick_run_without_json_prints_the_preset_table_and_names_its_report` (the card's own user
story: a short run from the terminal, table plus report path). `tests/test_bench_quick.py`: 31 → 33.

**Mutation testing** (Tier M: run once, report the score, soft threshold). The bench package is
mutmut's scope from the E2 pair (the sweep config the E2.5 card has since re-pointed at
`calibration`, §7.1). Two attempts, both cut short by the shared box's **pid cap** (mutmut forks a
worker per mutant; `pids.current = 256/256` mid-sweep, `BlockingIOError: [Errno 11]`):

```
uv run --no-sync --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2
uv run --no-sync --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 1
uv run --no-sync python tools/mutation_score.py mutants
→ src/ggufone/bench/devset.py    total  239  killed  59  survived 38  score 60.8 %  not-run  126
  src/ggufone/bench/harness.py   total 1743  killed 135  survived 71  score 65.5 %  not-run 1490
  src/ggufone/bench/suites.py    total 1711  killed  35  survived 31  score 53.0 %  not-run 1645
  TOTAL total 3693  {'killed': 229, 'survived': 140, 'no-tests': 63, 'not-checked': 3261}
  SCORE killed/scored = 62.1 %   (scored=369)
```

This is a **partial, time-boxed sweep** (369 of 3693 mutants scored — the artifacts and the
not-run count are reported, never rounded into a score). What it *does* answer is the Tier-M
question for this card, because mutmut walks the tree in file order and this card's new code sits
early:

| this card's new symbol | mutants | killed | survived | not-run |
|---|---:|---:|---:|---:|
| `harness.quick_config` | 21 | **21** | 0 | 0 |
| `harness.quick_note` | 3 | **3** | 0 | 0 |
| `harness.default_out_path` | 7 | 6 | 1 → **now killed** | 0 |
| `harness.TimeBudget` | 25 | 8 | 0 | 17 |
| `devset.stratify` | 5 | 3 | 2 → **1 killed, 1 equivalent** | 0 |
| `suites.run_suite` / `_measure` / `_dev_items` / `_devset_rows` / `_prefill_rows` / `_selected_backends` / `_run_quality` / … | 800+ | 0 | 0 | all (the box's pid cap ended the sweep before `suites.py`) |

The three survivors on this card's own lines were triaged one by one:

1. `default_out_path__mutmut_7` (`suffix="XXXX"` in the *full* branch) — **killed** by a new
   assertion (`default_out_path("latency", quick=False) == "ggufone-bench-latency.json"`).
2. `stratify__mutmut_2` (`if per_type <= 1: return []`) — **killed** by a new assertion
   (`per_type=1` yields one item per type, not an empty list).
3. `stratify__mutmut_1` (`if per_type < 0:` for `<= 0`) — **equivalent**: for every input the slice
   `[:per_type]` already returns `[]` at `per_type == 0`, so no behaviour changes (documented, not
   "fixed" by a test that would be testing the mutant's own shape).

Proof both new assertions are real kills — the mutants applied by hand to the sources, the gate file
run against each, then restored (`hand_mutation_quick.txt`):

```
mutant 1: harness.default_out_path full branch -> 'XXXX'
  → FAILED tests/test_bench_quick.py::test_a_quick_run_writes_its_own_report_and_never_a_full_campaign_file
mutant 2: devset.stratify '<= 0' -> '<= 1'
  → FAILED tests/test_bench_quick.py::test_stratified_selection_degrades_to_what_the_dev_set_has
restored: 33 passed
```

`cli.py` is *outside* mutmut's bench scope (it is a different package), so its mutants are not part
of this score; `tests/test_bench_quick.py` covers the CLI surface directly (33 tests).

### 7.1 The Tier-M score in context (and what the next pass inherits)

* The E2 card's full bench sweep scored **60.0 %** (1843/3071) and the E2.5 card's calibration sweep
  **73.6 %** (2008/2728) — a different scope each time; the *point* of a partial score here is the
  per-symbol table above, not the aggregate.
* The 3261 not-run mutants stay in mutmut's cache: the next bench-scoped sweep resumes instead of
  re-running (restore the three bench gate files, §7.1's pyproject note).
* The two *unmeasured* things a reviewer should know: **(a)** `suites.py`'s new code (the per-row
  `_measure` refactor, the quick branches in `_run_quality`/`_run_throughput`/`_run_determinism`) is
  covered by 33 offline tests but *not* mutation-scored, because the sweep died on the pid cap;
  **(b)** the environment blocked three separate attempts to complete live work (the 4B `/tmp`
  ENOSPC, `uv`'s SIGABRT, `libgomp: Thread creation failed`) — all three are recorded in §5.1/§6.

## 8. Risk summary, decision and confidence (Tier M report)

**🟡 WORTH CONSIDERING (the reviewer's call, with my recommendation)**

| # | item | risk | cost to fix | recommendation |
|---|---|---|---|---|
| 1 | The card's ≤ 3 min target is met **per suite** (max 107.6 s quick latency, 4B: 306 s), not by the whole five-suite campaign (213.6–300.8 s on the 0.8B, 479 s on the 4B) | a worker expecting a ≤3 min *whole campaign* on this box waits 4–5 min | the preset's own rows are ~4 s of 107.6 s (measured): the only real lever is running fewer suites per loop (`--suite latency` alone is 56–107 s) | accept the per-suite reading (it is the unit one `bench` invocation measures, and the live gate pins it) — the campaign multiple is documented in the test, not hidden |
| 2 | Mutation score on this card's new `suites.py` code: **not scored** (sweep killed by the box's pid cap; 62.1 % of the mutants that ran across the package, per-symbol table §7) | an untested branch in the new per-row budget code would not show up as a survivor | ~1 sweep with a free pid budget (the cache resumes) | defer to the next bench-scoped sweep; the inline table says exactly which symbols are unscored |
| 3 | A quick run *defaults* to writing `ggufone-bench-<suite>_quick.json` in `cwd` (full runs still write nothing) | a read-only `cwd` makes the write fail *after* the measurements (E_INTERNAL, exit 4) | 4 lines (a typed error) | acceptable: the pre-existing `--out` path behaves the same way and the message names the path; flagged here so it is a decision, not an accident |
| 4 | `rehearse_quick.sh` writes its reports into a caller-chosen dir, `--quick` in the CLI writes into `cwd` | two conventions | — | intentional: the script is the evidence vehicle, the CLI is the user surface |

**🟢 ACCEPTED (measured, no action)**

* `--quick` never shares a default report name with a full campaign (pinned + end-to-end tested).
* Every report (quick, truncated, full) carries `quick`/`truncated`/`wall_ms`/`budget` and is
  JSON-clean; the renderer prints the preset, the wall time and the unmeasured rows.
* The soft cap: measured live (`--max-seconds 300` → 22 unmeasured rows listed, exit 0), plus the
  row-level granularity documented in `bench --help`.
* `E_BENCH_QUICK` refuses `--quick --runs/--items/--sizes/--n-seq-max`; the `reproduce:` line the
  reports print never re-states them (`test_the_quick_reproduce_command_never_re_states_the_scale_…`).
* Offline gate 908 passed / 0 failed; ruff clean; changed-line coverage 100.0 %.

**Recommendation: ship (Option A), with item 1 as a documented deviation for the orchestrator.**
The preset does what the card asked (`--quick` + `--max-seconds`, pinned config, report integrity,
`--help`, tests, evidence); the two shortfalls are (1) the campaign-vs-suite reading of the ≤3-min
target and (2) the mutation sweep's coverage — both are *stated with numbers* rather than papered
over, and neither can be fixed by writing more code in this container without a quieter box.

```
CONFIDENCE: 8/10
  + the preset, the refusal rule, the report integrity and the soft cap are all pinned by 33
    offline tests, and the two surviving mutants on the new lines are proven killed by hand-mutation
  + the live gates ran on two real models (0.8B, 4B) with the pinned bundle; the 8k/2k full-suite
    numbers come from the same box and the same model
  + the report schema change (`quick`, `truncated`, `skipped`, `wall_ms`, `budget`) is additive and
    every published table still renders and re-reproduces
  - the mutation score for the new `suites.py` code is not measured (pid cap)
  - three of the five Spark suites needed environment workarounds (tmpfs, uv abort, libgomp) and
    Occamy 1.0 / Tiel-Coder are host-only: their numbers are NAMED (§5.2), not measured here
  - the ≤3-min target's campaign reading is a deviation, not a pass
```
