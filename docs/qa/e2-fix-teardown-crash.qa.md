# E2 FIX — the Vulkan child that SIGSEGVs at teardown (card t_57cc0179) — QA Report

Date: 2026-09-18 · Tier **M** (default; the card declares none) · Decision: **ship** — the card's
four requirements are measured (1+2+4 offline-gated, 3 by the repro recipe + the box's own raw
evidence; the residual live-run caveat is stated below with its window trace).
**Second pass, 2026-09-19: ship — as the *net*, with the remedy superseded by `t_97f1bc93`** (the
root cause was the NVIDIA ICD's own exit handler; `runtime/teardown.end_process` → `os._exit` now
ends the process before any exit handler runs). See the closing section.

Commits (the card's private clone, fast-forwarded into the shared tree): `385eab2` (detection +
retry + warning + the 40-gate file), `0de0536` (error-path/JSON/no-retry pins, ruff), `5fbc279`
(the repro recipe, the starved-device controls, the Tier-M sweep driver, the evidence doc), and the
final evidence commit (the live run + this report).

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none open.** The two candidates were both resolved inside this card:

* *"the crash is ours"* — it is not: a complete report proves our release sequence
  (`llama_free`/`llama_model_free`, called in `_throughput_row`'s `finally`, before the row is
  returned) ran to completion, and the signal arrives after it with no glibc line (unlike the
  `double free` of `t_dd62ec29`). Verdict: upstream (llama.cpp Vulkan backend / ICD teardown), and
  this card makes it **recoverable** — the withheld-row warning is only the fallback.
* *"the warning could be read as a backend mismatch"* — fixed in `suites.py` (the mismatch
  aggregation now runs over `isolation.verified_rows`) and pinned by
  `…test_the_determinism_suite_recovers_and_never_calls_the_crash_a_mismatch`.

🟡 WORTH CONSIDERING (3, recorded, none blocks)

* **Mutation (Tier M, soft threshold): 79.8 % over the 856 mutants the sweep scored** (killed 683,
  survived 173, **306 never run**) — `logs/mutation_score.txt` + `logs/mutation_survivors.txt`. The
  box's shared pid cgroup (256, with the E3 campaign, a sibling's own `tools/mutmut_driver.py` sweep
  and the voice-companion trees inside it) kills a `mutmut run` at `os.fork()` with `EAGAIN`; the
  driver re-runs until the meta has no pending mutants, but it ran out of clean windows, so the
  not-run count is reported and never folded into the score. Because a partial score cannot answer
  "does a *behavioural* mutant survive?", the diff also got a **hand-mutation table**
  (`logs/hand_mutants.txt`, `.e2e/…/hand_mutants.py`): 11 of 11 behavioural mutants — the halved
  layers, the "all layers" promotion, the KV rung, both conditions of the crash shape, the model's
  own placement, the starvation margin, the withheld row's warning, the retry loop itself, the
  recovered row's record and the suites' mismatch filter — are **KILLED**, each with the gate that
  killed it named, and a control mutant (a reworded closing sentence) **SURVIVES** as it must. Every
  splice restored the file byte-identically (sha256 before == after, printed per row). The survivors
  of the sweep itself are message/diagnostic pass-through of the same families `t_dd62ec29` triaged.
* **Two environment failures in the first full-suite pass**, both `tests/test_runtime_fallback.py`
  with `[Errno 11] Resource temporarily unavailable` in the child-probe helper — the pid cgroup at
  245/256, not a code path this card touches. Attribution, measured: the same files pass in the
  *same* tree when the cgroup is quiet (128 passed alongside the isolation gates:
  `logs/integrated_gates.txt`), and a shared-tree run's single `test_e3b_labels.py` failure came from
  a sibling's **uncommitted** `bench/labels.py` (that file passes at the landed commit: 36 passed in
  a clean worktree at `ac6345d`, 40 passed once the sibling finished editing).
* **GPU-window dependence of the live half.** The 4B shape needs the device left with ~3.2 GiB free
  (the operator's failing reading); the E3 campaign moves that by GiBs within seconds, so the
  recipe waits for the band and the repo'd raws carry the window trace. The reproduction *rate* on
  this box (2 of 5 identical single-bundle runs, exit 139 after a complete report) is measured by
  the sibling card `t_97f1bc93` from the root-cause side; my runs' own outcomes are the `pre_*`
  raws.

🟢 ACCEPTABLE

* The new gate file `tests/test_bench_teardown_crash.py` (**40 gates**) is RED on the parent tree
  (`f738315`: 40 failed, 0 passed — `logs/red_pretest.txt`) and green here; the `t_dd62ec29`
  containment file stays green **unmodified** (80 passed with both files selected), so the
  sibling's pins were reconciled by making the warning sentence carry the child's stderr tail, not
  by editing their file.
* Coverage of the changed modules: `bench/isolation.py` **100 %**, `bench/suites.py` 99 % (the same
  four pre-existing gaps the `t_dd62ec29` report lists).
* `ruff check src tests` clean.
* Behaviour is additive: a non-crashing run is byte-identical to before (no probe read, no retry,
  no extra process block keys); a crash gains one child and either a recovered row or the named
  warning.

## What could go wrong

* **A user reads a recovered row as a full-placement measurement.** The row carries its own
  `placement` / `placement_used` (`n_gpu_layers=16`) and `process.attempts` + `process.crash`, and
  the report carries `RECOVERED_AFTER_TEARDOWN_CRASH` — detectability: high.
* **The retry doubles a slow row's wall time.** Bounded by one extra child, and only for the crash
  shape; the `--max-seconds` budget still checks *between* rows, so a started unit finishes.
* **A driver that lies about free memory** mislabels "starved": the reading is the driver's own
  (source recorded in `crash.to_dict()["free_source"]`), and `starved=None` when unknown.

## Metrics

| metric | value |
|---|---|
| new gates | 40 (RED 40/40 on `f738315`) |
| containment gates | 45 (`t_dd62ec29`, unmodified, green) — 85 passed with both files |
| full offline suite (clean worktree at the landed commit `ac6345d`) | **1069 passed, 41 skipped, exit 0** |
| `bench/isolation.py` coverage | 100 % |
| `bench/suites.py` coverage | 99 % (4 pre-existing gaps) |
| ruff | clean |
| Tier-M mutation | 79.8 % of 856 scored (683 killed / 173 survived), 306 not run; **hand-mutation table 11/11 behavioural killed + 1/1 control survived** |
| live repro | recipe + controls in `logs/pre_*`; the crash itself on record in the `t_dd62ec29` raw and the sibling card `t_97f1bc93` (2 of 5 runs) |

## Second pass — 2026-09-19 (final): **ship, as the net**

Tier is still **M**; the first pass's Tier-M evidence (79.8 % of 856 scored mutants, 306 never run,
plus the hand-mutation table 11/11 behavioural killed + 1/1 control survived) stands unchanged — no
re-run was warranted once the remedy turned out to be superseded.

What changed, and what was re-measured on the integrated tree (`d82204f` = this card + `t_97f1bc93`) —
transcripts in `.e2e/t_57cc0179-vulkan-teardown/logs/`:

* the root cause is fixed one layer below this card: the SIGSEGV was the **NVIDIA ICD's own exit
  handler** (`libnvidia-glvkspirv → libnvidia-eglcore → __run_exit_handlers`, backtrace in the
  sibling card's evidence), and `runtime/teardown.end_process` now ends the process with `os._exit`
  before any exit handler runs — a Vulkan child exits **0/1 only**;
* **gates**: `tests/test_bench_isolation.py` + `tests/test_bench_teardown_crash.py` +
  `tests/test_cli_teardown.py` → **95 passed** (85 of them this card's; the containment file is still
  unmodified);
* **suite**: full offline run → **1082 passed, 43 skipped, exit 0**; **ruff**: clean;
* **coverage** over the whole suite: `bench/isolation.py` **100 %**, `bench/suites.py` **99 %**;
* **integration**: the documented two-bundle `--backend all` command publishes the `cpu` + `vulkan`
  rows, `withheld-rows=0`, `teardown-warnings=0` — the two cards' changes do not fight;
* **the net, deterministically** (`repro/fault_demo.sh`, injected SIGSEGV at the isolated child's
  exit): fires on `f738315` (withheld), **recovers** on `2e7eb6c` (`RECOVERED_AFTER_TEARDOWN_CRASH`,
  `process.attempts` carries the signal + the memory reading), **warns** on `2e7eb6c`/`always`
  (`W_BACKEND_CRASHED_AT_TEARDOWN` in the row's table cell), and is **inert** on `d82204f` — the fix
  below really does prevent the shape;
* the live half was stopped when the supersession became known: the band window never opened
  (`logs/band_window.txt`), the 0.8B rate batch saw 3 × exit 0 + 3 × typed `E_BACKEND_OOM` and **no
  signal**, and the two-bundle runs measured the refusal/degradation side; the *crash* itself stays on
  record from the sibling card (3 signals in 12 runs, with a libc backtrace).

Risk register for the retained net (none blocking):

* **a recovered row read as a full-placement measurement** — the row carries its own
  `placement`/`placement_used`, `process.attempts`, `process.crash`, and the report carries
  `RECOVERED_AFTER_TEARDOWN_CRASH`; detectability high;
* **the net is now nearly unreachable** — after `os._exit` there is almost no window between "the
  report is on disk" and "the process ends", so `W_BACKEND_CRASHED_AT_TEARDOWN` should be seen as
  defense-in-depth for out-of-tree bundles/other ICDs, not as an expected message; its failure mode is
  a *named* withheld row, never a silent one;
* **partial mutation score** — unchanged from the first pass; the behavioural question is answered by
  the hand-mutation table, not by the percentage.
