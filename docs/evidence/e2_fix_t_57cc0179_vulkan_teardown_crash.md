# E2 FIX (card t_57cc0179) — the Vulkan child that SIGSEGVs at teardown on a starved device

Follow-up to `t_dd62ec29` (`.e2e/t_dd62ec29-mixed-bundle-teardown/`, landed on `main` as `f738315`).
That card found the shape and contained it: an **isolated** Vulkan child writes its whole
`ggufone.bench/v1` report (`ok: true`), and *then* dies with **exit -11 (SIGSEGV)** when the device
is memory-starved; the row is withheld, the report is `ok: false`, the command exits 1. The crash
could not kill the campaign any more — but the measurement was lost, and on the operator's host
(where the E3 campaign shares the device) that shape is routine, not exotic. `t_dd62ec29` left it
unfixed on purpose; this card closes it from the bench side:

1. **detect** the shape and **retry it once** with the loader's next degraded placement;
2. when the retry crashes too, keep the containment and name it — `W_BACKEND_CRASHED_AT_TEARDOWN`,
   carrying both exit codes and the free-device-memory reading, in the row the table renders;
3. investigate the teardown ordering and publish a **minimal repro recipe** with a verdict;
4. prove a published table can never silently lose a row.

> **Status (2026-09-19) — superseded as the remedy, retained as the net.** The sibling card
> `t_97f1bc93` (landed on `main` as `d90bdf8`; evidence `docs/evidence/e2_fix_t_97f1bc93_vulkan_teardown.md`)
> found the root cause with a libc backtrace: the signal came from the **NVIDIA ICD's own exit
> handler** (`libnvidia-glvkspirv → libnvidia-eglcore → __run_exit_handlers`) — a third-party
> destructor at interpreter exit, with no ggml/llama.cpp/ggufone frame in it. ggufone now ends such a
> process itself (`runtime/teardown.end_process` → `os._exit` after both streams are flushed, from
> `cli.run` whenever a bundle is loaded), so the exit status of the shipped command can no longer be
> rewritten by the vendor's handler: `vendor_a.summary` measures 4/4 clean exits at 3.5–5.7 GiB free,
> and their live gate (`tests/test_bench_vulkan_teardown_live.py`) is green with **0/1 only**. The
> crash shape this card was opened for therefore **cannot arise from its original cause any more**,
> and no further GPU reproduction was run once that was known.
>
> What is retained — and what this card still pins — is the containment **net** for the *generic*
> shape "a child that died on a fatal signal after answering", whatever the cause (an out-of-tree
> bundle under `GGUFONE_RUNTIME_DIR`, another ICD, a heap fault after the report was written):
> `tests/test_bench_teardown_crash.py` (40 gates: the shape, the starvation reading, the one-rung
> retry, `W_BACKEND_CRASHED_AT_TEARDOWN` with both exit codes and the VRAM readings, the rendered
> table cell, and "no backend row can disappear from the table"), the untouched
> `tests/test_bench_isolation.py` containment (45 gates), and `suites.py`'s `verified_rows` filter
> that keeps a crash warning from ever being re-read as `W_BACKEND_MISMATCH`. The integration check
> on the integrated tree (`repro/integration_check.sh`, `logs/integrated_*`: exit 0, two rows
> measured, `withheld-rows=0`) proves the two cards' changes do not fight: the verified child now
> simply exits 0/1 and the row is published.

## The shape, from the raw that found it

`.e2e/t_dd62ec29-mixed-bundle-teardown/logs/after_mixed.raw` — the parent's own record of the
crash (`vram_before_controls.txt`: 8192 MiB total, **3272 MiB free** while the E3 campaign held the
rest):

* the child's report was complete (`ok: true`, `measured: true` was *not* published because the
  parent refused it): the last engine lines in the captured stderr are the context's own
  accounting — `~llama_context: Vulkan0 compute buffer size is 425.8354 MiB, matches expectation`;
* the process then died on **SIGSEGV** (child `exit_code: -11`), with **no** glibc line — unlike
  the mixed-bundle abort of `t_dd62ec29`, which printed `double free or corruption` (exit 134);
* the *same* bundle alone on a quieter device exits 0 with real device work
  (`logs/vulkan_alone.raw`, `effective_backend: "vulkan"`, `Vulkan0`/`Vulkan_Host` compute
  buffers).

## 1 + 2 — what the fix does (all offline-gated)

`src/ggufone/bench/isolation.py` (the module that owns the one-bundle-per-process seam) gained:

| seam | behaviour | gate |
|---|---|---|
| `teardown_crash(exit_code, report=…)` | the shape = a fatal signal (**-11/-6**) **and** a report the child did write. A child with no report crashed before it measured; a clean exit, a flagged child (exit 1) and a SIGKILL are not this shape | `tests/test_bench_teardown_crash.py::test_a_fatal_signal_with_a_complete_report_is_the_teardown_crash`, `…test_other_failures_are_not_the_teardown_crash`, `…test_a_child_that_wrote_no_report_is_not_this_shape` |
| `device_memory()` / `starved()` | the driver's own free/total device memory (`registry.recommend`, lazily imported) and whether the model's weights **plus** the 1 GiB `--fit-target` margin could fit at all. A box with no device (or a driver that raises) reads **unknown**, never 0 MiB | `…test_device_memory_reads_the_driver_probe`, `…test_no_device_is_unknown_never_zero`, `…test_the_crash_is_starved_only_when_the_weights_cannot_fit`, `…test_a_driver_that_raises_reads_as_an_unknown_device` |
| `retry_placement()` / `retry_config()` | the one retry's placement is the **loader's own ladder, one rung down** (`fit.degrade_ladder`'s first rung): half the layers the first attempt *really* ran; at `n_gpu_layers=0` the next KV rung instead; with no readable header, the rung that asks the device for nothing. The retry is the same documented `bench --backend <one>` command with the same scale flags — a config, not a private argv | `…test_the_retry_placement_is_the_loaders_own_degrade_rung`, `…test_the_retry_halves_the_layers_that_really_ran`, `…test_the_retry_reads_what_really_ran_out_of_the_childs_own_report`, `…test_the_retry_downgrades_the_kv_type_when_the_weights_are_already_on_the_host`, `…test_the_retry_placement_travels_as_a_bench_config` |
| `run_backend_child(…, attempts=2)` | the crash earns **exactly one** extra child; the retry is verified like the first attempt (exit code ↔ report, backend, bundle, echoed config). A recovered row keeps the report `ok: true` and carries `process.attempts` + `process.crash` | `…test_a_teardown_crash_is_retried_once_with_the_degraded_placement`, `…test_the_retry_happens_exactly_once`, `…test_a_clean_child_is_never_retried`, `…test_a_failure_that_is_not_the_shape_is_not_retried`, `…test_the_retry_child_is_verified_against_the_retry_placement`, `…test_a_recovered_row_keeps_the_report_ok` |
| `W_BACKEND_CRASHED_AT_TEARDOWN` | when the retry crashes too, the withheld row carries the warning: the child's exit code(s), the stderr tail, the free-VRAM reading at each attempt's start, the starvation verdict and the retry's placement — in `reason`, which `render_report` prints **in the table cell** (and in the report's `notes` through `isolation_note`) | `…test_a_failed_retry_keeps_the_containment_and_names_the_warning`, `…test_the_warning_reaches_the_rendered_table_not_only_the_notes` |
| `suites.py` | the mismatch aggregation now runs over `isolation.verified_rows`: a withheld row's crash warning can no longer be re-read as `W_BACKEND_MISMATCH`; a recovered row gets a `RECOVERED_AFTER_TEARDOWN_CRASH` note while the report stays `ok: true` | `…test_the_determinism_suite_recovers_and_never_calls_the_crash_a_mismatch` |

**Design decision — degrade, do not wait.** The card allows "a degraded placement … *or* after the
device frees". Waiting was rejected: the free-memory window on this box moves by GiBs within
seconds (measured: 235 MiB → 3509 MiB inside 90 s), so a wait would either spin for an unbounded
time or produce a row whose conditions nobody recorded. Degrading is bounded, and the row *says*
what it ran with (the child's own `placement_used`, now `n_gpu_layers=16` instead of all 32).

**Design decision — one retry, not a ladder walk.** The loader walks a whole ladder, but a bench
row is a measurement, not a service start: one degraded rung recovers the row in the observed
shape, and a second failure means something else is wrong (the withheld row says so).

## 3 — the teardown ordering: our release sequence, or llama.cpp's?

**Where the crash is in the process.** `bench/suites.py::_throughput_row` closes the model
(`harness.LiveModel.close()` → `session.ModelSession.close()` → `llama_free(ctx)`,
`ModelHandle.close()` → `llama_model_free(model)`) in its `finally`, **before** the row is returned
and therefore before the report is serialised to `--out` and printed. So a *complete report* proves
that **our release sequence returned normally** — `llama_free`/`llama_model_free` did not fault.
What is left after it is the library's own teardown: the Vulkan backend's device/buffer
destructors and the ICD's device destruction, which run when the process exits (`dlclose` /
static destructors / `vkDestroyDevice`).

**Verdict: upstream (the NVIDIA ICD's own exit handler), triggered by device pressure — and now fixed
at the source by the sibling card.** The evidence:

* **root cause, settled by the sibling card's libc backtrace (`t_97f1bc93`, 2026-09-19)**: the fault
  is `libnvidia-glvkspirv → libnvidia-eglcore → __run_exit_handlers`, i.e. the vendor ICD's *own*
  `atexit`-time destructor, not a ggml/llama.cpp/ggufone frame. That card's fix
  (`runtime/teardown.end_process` → `os._exit`) takes the third party's vote on the exit status away
  and is the reason this card's shape no longer occurs (see the status block above);

* ordering: the crash is strictly *after* our release path (above) — there is no our-code
  destructor left to run when the signal arrives;
* signature: a bare SIGSEGV with no glibc abort line, unlike every *our-side* teardown defect
  measured so far (`double free or corruption`, exit 134 — `t_603a35a0`/`t_dd62ec29`);
* pressure-dependence: the same command and the same bundle exit 0 on a device with room
  (`vulkan_alone.raw`), and the same *tree* dies only under starvation — the code path does not
  change, the device does;
* rate on the operator box: the sibling card `t_97f1bc93` measured **2 of 5 identical single-bundle
  runs** exiting 139 after a complete `ok: true` report (the same defect from the root-cause side,
  with a gdb/LD_PRELOAD backtrace harness); its raw logs are cited there.
* my own file-level backtrace attempt: `repro/` below runs the child under `gdb -batch -ex run -ex
  bt` with `PYTHONFAULTHANDLER=1`, so whichever frame the box produces is in the raw either way
  (see `logs/` — if the run could not be scheduled inside the box's memory window, this paragraph
  says so with the window trace attached, and the sibling card's backtrace stands as the C-frame
  evidence).

**The minimal repro recipes** (`.e2e/t_57cc0179-vulkan-teardown/repro/`, self-contained). The box
turned out to need *two* of them, because the defect lives in a narrow band and each recipe
documents one side of it:

1. `cc -O2 -o vram_hog vram_hog.c -lvulkan` — a dummy allocator: it takes `N MiB` of
   `DEVICE_LOCAL` memory on the discrete device and sleeps (no llama.cpp involved, so the pressure
   cannot come from the process under test).
2. **Starve before the child** — `sh starve_and_run.sh <tree> <tag> auto <keep_free> off <wait_min>`
   waits for the ambient window, brings the device to `keep_free` MiB, then runs
   `bench --suite throughput --model Qwen3.5-4B-Q4_0.gguf --backend vulkan --runs 1 --threads 4
   --sizes 64 --json` with `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`.
3. **Hold VRAM while the child exits** — `sh starve_at_teardown.sh <tree> <tag> <load_free> <leave_free>`
   waits for a *load* window, starts the one child, waits until its own stderr proves the context
   exists **and** goes quiet (the engine's Vulkan log stops once the allocation phase is over), and
   only *then* starts the dummy allocator with everything above `leave_free` MiB — so the child's
   placement succeeds and the device is tight at the exit.
4. `sh batch_teardown.sh <tree> <prefix> <runs> …` repeats (3) until a signal appears.
5. `sh measure_footprint.sh <tree> <tag> [model]` measures what one child really takes
   (nvidia-smi `used` before / peak / after, sampled while it runs).
6. **The confirmed-band rate driver** — `sh crashrate.sh <tree> <prefix> [runs] [keep_free|none]
   [wait_min]`: waits for the ambient window (the E3 campaign gives GiBs back in steps), holds the
   device **down to `keep_free`** with the dummy allocator, proves the hold landed with nvidia-smi
   *before* the child starts (`<tag>_hold_control.txt`, `confirmed=1` only inside ±256 MiB of the
   target), then repeats the documented single-bundle child and stops at the first fatal signal.
   This is the recipe that fixes (2)'s failure mode (a best-effort hold that never reaches the band);
   (3)/(4) are kept as the teardown-time variant.
7. **The retry driver** — `sh mixed_retry_run.sh <tree> <prefix> [runs] [keep_free]`: the same hold
   around the *two-bundle* `--backend all` command, i.e. the shape whose isolation seam owns this
   card's retry; it stops at the first run whose report shows `RECOVERED_AFTER_TEARDOWN_CRASH` or
   `W_BACKEND_CRASHED_AT_TEARDOWN`.
8. `sh integration_check.sh <tree> <tag>` — the two cards' intersection: the documented
   `--backend all` command on the integrated tree, expected to publish both rows with no withheld row
   and no warning (the child now ends itself, see the status block).
9. `sh fault_demo.sh <tree> retry-ok|always` — the retained net, demonstrated end to end with the
   real CLI and real child processes plus an **injected** SIGSEGV
   (`repro/fault_inject/sitecustomize.py` arms an `atexit` signal for the isolated `--backend vulkan`
   child only). It is deterministic, which is exactly what a flaky shape cannot be; the *natural*
   occurrence is the raws above and the sibling card's rate runs.

**What the recipes measured here (operator box, box shared with the E3 campaign):**

| raw | device at start | outcome |
|---|---|---|
| `logs/pre_starved.raw` | 185–298 MiB free | the child's **own fit ladder** walked `n_gpu_layers=-1 -> 16 -> 0` and every rung OOM'd: a **typed** `E_BACKEND_OOM` report (`ok: false`), exit 1 — *no* crash. This is the "device too full to load" side. |
| `logs/pre_small.raw` | 746 MiB free, 0.8B model | row measured, `Vulkan0` compute buffers, exit **0** — pressure alone is not the defect. |
| `logs/pre_teardown4b` (`pre_teardown_run*`) | 5.3–5.6 GiB free, 4B model, hog started after the context marker | the placement loaded (`loaded=1`); the hold then starved the child's **own remaining** allocation (`E_BACKEND_OOM` for a 0.45 GiB graph buffer), exit 1 — again typed, again no crash. The hold has to land *after* the child's last allocation, which is what (3) refines. |
| `logs/pre_foot_footprint.txt` | baseline 3236 MiB used | the child's own device footprint measured at **~2.78 GiB** (weights 2.45 + Vulkan0 context + graph) before the run was cut short by the box's pid cgroup (`python -m ggufone.runtime.probe_child` could not fork: `EAGAIN` → the CLI's typed exit 2). |

**Second session, 2026-09-19 — the card's last live attempt, stopped by the supersession.** The
device sat at 0.3–1.5 GiB free for the whole window (the E3b sweep held ~5.5 GiB, and `cgroup
memory.current` was at its 8 GiB ceiling), so the confirmed-band hold never got its window
(`logs/band_window.txt`: `free=276MiB wanted=3856MiB`), and every attempt that did run measured the
*refusal* side, never a signal:

| raw | device at start | outcome |
|---|---|---|
| `logs/livep_run1.raw`, `logs/livep_run2.raw` (recipe B, pre-fix tree) | 6.8 GiB free, hog started after the context marker | complete report, exit **0** — the hold lands after the child's last allocation, which is exactly why (3) cannot hit the band |
| `logs/mixed_run1.raw` (two-bundle, fixed tree) | 4.2 GiB free | both rows measured, exit 0, no withheld row, no warning |
| `logs/mixed_run2.raw` | 1.5 GiB free at the vulkan child | the child's **own** fit ladder walked `-1 → 16 → 0` (`W_BACKEND_OOM`, `W_FIT_DOWNGRADE`) and measured on the host — a typed degradation, no crash: the "device too full to load" side again |
| `logs/small_rate.txt` (0.8B model, 6 runs) | 1.0–1.4 GiB free | 3 × exit 0 with a complete report, 3 × exit 1 typed `E_BACKEND_OOM`; **no signal in any of the six** |
| `logs/band_rate.txt` + `band_run{1..4}.*` (the confirmed-band recipe, pre-fix tree) | window opened at 14:12:41 with **4126 MiB free** — inside the operator's band | **4 attempts, all exit 0 with complete reports** (device 4103 → 1749 → 1460 → 1468 MiB free; `band_run1_hold_control.txt`: `hold=505MiB confirmed=0`, the ambient device was already in the band so the hold never had to bite). No signal — and the batch was stopped here by the supersession |

**Verdict on the live half, final.** The crash itself was captured on this box from the *root-cause*
side, not from mine: the sibling card `t_97f1bc93` reproduced it (3 signals in 12 identical single
bundle runs — `logs/segv_bt_a-*.raw`, `crashrate_a-*.raw`, `red_parent-*.raw`) and pinned it with a
libc backtrace to the **NVIDIA ICD's own exit handler**. My own windows, by contrast, produced 0
signals in 12 attempts (2 recipe-B repeats, 2 two-bundle runs, 6 small-model runs, 4 confirmed-band
runs) — the flakiness is the whole story, and it is why the shape could not be *closed* from the
bench side by catching it, only by fixing its cause. That card owns the defect and its fix
(`os._exit` before any exit handler runs) is what makes the shape unreachable now. This card's
remaining live evidence is therefore about the *net*, not the bug: `logs/integrated_*` (the two
cards' intersection, measured green) and the deterministic injection below.

### The retained net, demonstrated — deterministic fault injection, three trees

The natural shape is flaky *and* (after `d90bdf8`) unreachable, so the net is demonstrated with a
fault injected **inside the child, at its own exit** (`repro/fault_inject/sitecustomize.py`: an
`atexit` handler that raises SIGSEGV, armed only when `GGUFONE_T57_FAULT` is set, the process is the
isolated `--backend vulkan` child, and — in `retry-ok` mode — the report is the first attempt's. The
parent is never faulted). Same command, same harness, one tree per row; every raw and every rendered
table is in `logs/fault_*`:

| tree | injection | outcome |
|---|---|---|
| `f738315` (before both cards: containment, no retry) | fires | the row is withheld (`ISOLATED_CHILD_FAILED`, exit 1) — the control that the injection is real, and the behaviour `t_dd62ec29` left behind (`logs/fault_retry-ok-red.raw`) |
| `2e7eb6c` (this card: retry, before the teardown fix) | fires | **recovered** (`logs/fault_retry-ok-mid.raw`, exit 0): attempt 1 exits `-11` after its complete `ok: true` report, the one retry measures the row, `process.attempts[0]` carries the signal, the placement and the 4.0/8.0 GiB reading, the report stays `ok: true`, and the table's note is `RECOVERED_AFTER_TEARDOWN_CRASH` (the retry's own placement travels in the row) |
| `2e7eb6c`, mode `always` (both attempts die) | fires twice | withheld + **named**: `logs/fault_always-mid.raw` (exit 1) and its rendered table carry `W_BACKEND_CRASHED_AT_TEARDOWN: the isolated child exited -11 … after writing a complete report` **in the row's table cell**, with the stderr tail attached (`logs/fault_always-mid_rendered.md`) |
| `d82204f` (this card + the teardown fix) | **inert** | the child ends itself (`runtime/teardown.end_process`) before `atexit` can run, so the injected signal never fires: exit 0/1 only, both rows published, no withheld row, no warning (`logs/fault_retry-ok-int.raw`) — the supersession, observed from this card's own harness |

That last row is the whole story of this card in one measurement: the net is still there and still
correct, and the thing it was built to catch is now prevented one layer below it.

## 4 — a published table can never silently lose a row

`render_report`'s throughput branch iterates *every* row of the report — measured, withheld,
missing-bundle — and the withheld ones print `not measured` plus their reason
(`tests/test_bench_quick.py`'s truncation gate and `t_dd62ec29`'s rows rely on the same
invariant). This card pins the crash case explicitly: with a crashed-and-not-recovered backend in
the report, the rendered table keeps one line per backend and that line carries the warning with
the exit code and the VRAM reading
(`…test_no_backend_row_can_disappear_from_the_rendered_table`, `…test_the_warning_reaches_the_rendered_table_not_only_the_notes`).

The E3-style *comparison* path (a sibling card's module, inspected on the shared tree at
`src/ggufone/bench/compare.py` — not owned by this card) partitions the same item rows and renders
every partition (`per_type` + the `low_mass`/`measured` split), and its one filter — the
id-mismatch between a baseline and a challenger — *reports the count it dropped* instead of hiding
it. So the invariant holds on both published-table paths; the pins above are the executable half.

## Gates

Filled by the run's own artifacts under `.e2e/t_57cc0179-vulkan-teardown/logs/` (see the README
there for the map). Summary, all measured on the integrated tree (this card rebased onto the
siblings' `56cfe5e`, landed as `ac6345d`):

* **new gate file** `tests/test_bench_teardown_crash.py`: 40 gates, **RED on the parent tree**
  (`f738315`: 40 failed, 0 passed — `logs/red_pretest.txt`) and green here;
* **the containment stays green unmodified**: `tests/test_bench_isolation.py` (45 gates) — with both
  files selected, **85 passed** (`logs/integrated_gates.txt`);
* **full offline suite**, clean worktree at the landed commit: **1069 passed, 41 skipped, exit 0**
  (`logs/full_suite_clean_head.txt`);
* **ruff**: `All checks passed!`;
* **coverage** of the changed modules over the bench/CLI gate files:
  `bench/isolation.py` **100 %**, `bench/suites.py` **99 %** (the same four pre-existing gaps
  `t_dd62ec29` reported);
* **Tier-M mutation** (soft threshold, one sweep): **79.8 %** of the **856** mutants the sweep scored
  (683 killed, 173 survived, **306 never run** — the pid cgroup's `EAGAIN` at `os.fork()`, reported
  and never folded in), `logs/mutation_score.txt` + `logs/mutation_survivors.txt`;
* **the hand-mutation table over the diff** (`logs/hand_mutants.txt`, `.e2e/…/hand_mutants.py`),
  because a partial score cannot answer "does a *behavioural* mutant survive?": **11 of 11**
  behavioural mutants are **KILLED**, each with the gate that killed it named (halved layers, the
  "all layers" promotion, the KV rung, both halves of the crash shape, the model's own placement,
  the starvation margin, the withheld row's warning, the retry loop, the recovered row's record, the
  suites' mismatch filter), and the control mutant (a reworded closing sentence) **SURVIVES** as it
  must. Every splice restored the file byte-identically (`sha256` printed per row);
* the box's own failures during this run are recorded with their attribution: the pid cgroup hit
  245–256/256 several times, which surfaces as `Cannot fork` in the shell, `BlockingIOError: [Errno
  11]` in a sweep, and `probe_failed` in `tests/test_runtime_fallback.py`/`test_runtime_contract.py`
  — those files pass in the same tree when the cgroup is quiet (128 passed with the isolation gates,
  `logs/integrated_gates.txt`), and the one `test_e3b_labels.py` failure of a shared-tree run came
  from a sibling's *uncommitted* `bench/labels.py` (that file passes at the landed commit: 36 passed).

**Second pass, 2026-09-19 — on the integrated tree (`d82204f`, i.e. this card + `t_97f1bc93`)**, once
the root-cause fix had landed and the live campaign was stopped:

* the gate files together with the teardown card's: `pytest -q tests/test_bench_isolation.py
  tests/test_bench_teardown_crash.py tests/test_cli_teardown.py` → **95 passed** (85 of them this
  card's: 45 containment + 40 new; the other 10 are `t_97f1bc93`'s);
* the full offline suite: **1082 passed, 43 skipped, exit 0** (`logs/full_suite_integrated.txt`);
* `ruff check src tests`: `All checks passed!` (`logs/ruff_integrated.txt`);
* coverage over the *whole* offline suite: `bench/isolation.py` **100 %**, `bench/suites.py` **99 %**
  (`logs/coverage_full.txt`; with only this card's two gate files selected the two modules read
  100 %/47 % — `logs/coverage_gates.txt` — because the rest of `suites.py` is other suites'
  functionality);
* the two cards' intersection: `logs/integrated_*.raw`/`_rendered.md` — exit 0, `cpu` + `vulkan`
  rows published, `withheld-rows=0`, `teardown-warnings=0`.

## How to re-run

```
cd /work/t57cc-ggufone            # or any checkout of the card's commit
.venv/bin/python -m pytest -q tests/test_bench_isolation.py tests/test_bench_teardown_crash.py
.venv/bin/python -m pytest -q tests/test_bench_isolation.py tests/test_bench_teardown_crash.py \
    tests/test_cli_teardown.py                          # with the teardown card's gates
# the retained net, deterministically (any box, no device needed):
sh .e2e/t_57cc0179-vulkan-teardown/repro/fault_demo.sh <tree> retry-ok    # recovered row
sh .e2e/t_57cc0179-vulkan-teardown/repro/fault_demo.sh <tree> always      # named warning + table
# the two cards' intersection on the documented mixed command:
sh .e2e/t_57cc0179-vulkan-teardown/repro/integration_check.sh <tree> <tag>
# the starved-device recipes (need a real device and a window; see the README):
sh .e2e/t_57cc0179-vulkan-teardown/repro/crashrate.sh /work/t57cc-red band 4 3600 40
sh .e2e/t_57cc0179-vulkan-teardown/repro/starve_and_run.sh /work/t57cc-red pre_starved auto 3272 gdb 30
sh .e2e/t_57cc0179-vulkan-teardown/repro/mixed_retry_run.sh <tree> mixed 3 3600
```

The three trees the fault injection was measured on: `/work/t57cc-red` (`f738315`, before both
cards), `/work/t57cc-mid` (`2e7eb6c`, this card only) and the integrated clone (`d82204f`, both).
