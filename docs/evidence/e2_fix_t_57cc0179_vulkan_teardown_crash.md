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

**Verdict: upstream (llama.cpp + the Vulkan ICD), triggered by device pressure — and contained by
this card's retry.** The evidence:

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

**What the recipes measured here (operator box, box shared with the E3 campaign):**

| raw | device at start | outcome |
|---|---|---|
| `logs/pre_starved.raw` | 185–298 MiB free | the child's **own fit ladder** walked `n_gpu_layers=-1 -> 16 -> 0` and every rung OOM'd: a **typed** `E_BACKEND_OOM` report (`ok: false`), exit 1 — *no* crash. This is the "device too full to load" side. |
| `logs/pre_small.raw` | 746 MiB free, 0.8B model | row measured, `Vulkan0` compute buffers, exit **0** — pressure alone is not the defect. |
| `logs/pre_teardown4b` (`pre_teardown_run*`) | 5.3–5.6 GiB free, 4B model, hog started after the context marker | the placement loaded (`loaded=1`); the hold then starved the child's **own remaining** allocation (`E_BACKEND_OOM` for a 0.45 GiB graph buffer), exit 1 — again typed, again no crash. The hold has to land *after* the child's last allocation, which is what (3) refines. |
| `logs/pre_foot_footprint.txt` | baseline 3236 MiB used | the child's own device footprint measured at **~2.78 GiB** (weights 2.45 + Vulkan0 context + graph) before the run was cut short by the box's pid cgroup (`python -m ggufone.runtime.probe_child` could not fork: `EAGAIN` → the CLI's typed exit 2). |

**Honest status of my own live runs.** I did **not** land the SIGSEGV inside my own windows: the
E3 campaign holds 7+ GiB of the 8 GiB board for long stretches, the pid cgroup sat at 245–255 of
256 for much of the run (my first attempt died with `Cannot fork`, the last one with the probe
child's `EAGAIN`), and the band that both lets the 4B placement through *and* leaves the device
tight at the exit is narrow. What is *not* missing is the crash itself: it is recorded twice on this
box — the `t_dd62ec29` raw above (one complete report, then `exit -11`, no glibc line) and the
sibling card `t_97f1bc93`'s rate run (**2 of 5 identical single-bundle runs** exiting 139 after a
complete `ok: true` report, with its own gdb/`LD_PRELOAD` backtrace harness). Closing paragraph of
the verdict therefore: **upstream and contained**; the vehicle that turns my recipe into a local
crash capture is a run of (3) on a box where the campaign leaves ≥5.4 GiB free for a minute — the
script waits for that window and reports the trace either way (`logs/*_window.txt`).

## 4 — a published table can never silently lose a row

`render_report`'s throughput branch iterates *every* row of the report — measured, withheld,
missing-bundle — and the withheld ones print `not measured` plus their reason
(`tests/test_bench_quick.py`'s truncation gate and `t_dd62ec29`'s rows rely on the same
invariant). This card pins the crash case explicitly: with a crashed-and-not-recovered backend in
the report, the rendered table keeps one line per backend and that line carries the warning with
the exit code and the VRAM reading
(`…test_no_backend_row_can_disappear_from_the_rendered_table`, `…test_the_warning_reaches_the_rendered_table_not_only_the_notes`).

## Gates

Filled by the run's own artifacts under `.e2e/t_57cc0179-vulkan-teardown/logs/` (see the README
there for the map). Summary:

* new gate file `tests/test_bench_teardown_crash.py`: 40 gates, **RED on the parent tree**
  (`f738315`: 40 failed, 0 passed — `logs/red_pretest.txt`) and green on this tree;
* `tests/test_bench_isolation.py` (the `t_dd62ec29` containment, 45 gates) stays green
  **unmodified** — with both files selected: 80 passed;
* full offline suite, ruff and coverage numbers: `logs/`.

## How to re-run

```
cd /work/t57cc-ggufone            # or any checkout of the card's commit
.venv/bin/python -m pytest -q tests/test_bench_isolation.py tests/test_bench_teardown_crash.py
sh .e2e/t_57cc0179-vulkan-teardown/repro/starve_and_run.sh /work/t57cc-red pre_starved auto 3272 gdb 30
```
