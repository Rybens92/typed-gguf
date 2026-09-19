# t_57cc0179 — E2 FIX: the Vulkan child that SIGSEGVs at teardown (raw material)

The readable version is `docs/evidence/e2_fix_t_57cc0179_vulkan_teardown_crash.md`; every claim
there is checkable from here. The code lives on the shared tree's `main`
(`src/ggufone/bench/isolation.py`, `src/ggufone/bench/suites.py`,
`tests/test_bench_teardown_crash.py`) — and, as of 2026-09-19, the defect it was opened for is fixed
one layer below it by `t_97f1bc93` (`src/ggufone/runtime/teardown.py`, `cli.run` → `os._exit`): the
card is **superseded as a remedy and retained as the net**. See the status block in the evidence doc.

Box: the operator host's container — cgroup quota 2.0, pid cgroup 256 (shared with the E3 campaign,
the voice-companion trees and other kanban workers), 8 GiB memory ceiling, GPU passed through with
`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`. Bundles: `GGUFONE_RUNTIME_DIR` = the pinned
`linux-x64-cpu` asset at `/work/t603-runtime`, `GGUFONE_BENCH_RUNTIME_DIR` = the installed Vulkan
b11026 bundle under `/var/home/rybens/.local/share/ggufone/runtime`. Models: `Qwen3.5-4B-Q4_0.gguf`
(2.4 GiB weights) and `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (0.53 GiB).

The trees the measurements ran against:

* `/work/t57cc-red` — a `git worktree` pinned at `f738315` (the parent commit: the `t_dd62ec29`
  containment, no retry, no teardown fix). `PYTHONPATH=<tree>/src` selects the tree under the run's
  interpreter (verified: `ggufone.__file__` and `hasattr(isolation, "teardown_crash")`).
* `/work/t57cc-mid` — a `git worktree` pinned at `2e7eb6c` (this card's fix, before `t_97f1bc93`'s).
* `/work/t57cc-ggufone` — this card's private clone; it carried `main` and is the tree the final
  numbers were measured on (`d82204f`, both cards). A sibling's `git add -A` sweeps WIP in the shared
  tree, so the work is done here and fast-forwarded into the shared tree.

| file | what it is |
|---|---|
| `repro/vram_hog.c` | the dummy allocator: holds N MiB of `DEVICE_LOCAL` memory on the discrete device and sleeps — the pressure comes from *outside* the process under test |
| `repro/starve_and_run.sh` | recipe A: starve the device **before** the child (the "too full to load" side: the loader's own fit ladder refuses with a typed `E_BACKEND_OOM`) |
| `repro/starve_at_teardown.sh` | recipe B: let the placement load, then **hold VRAM while the child exits**; the 5th arg `gdb` adds a C backtrace |
| `repro/batch_teardown.sh` | repeats recipe B until a *signal* appears (the crash shape), stopping at the first one |
| `repro/hold_and_run.sh` | recipe A with a **confirmed band**: the hold is proven with nvidia-smi (±256 MiB of the target) *before* the child starts, and the device is sampled every 2 s while it runs |
| `repro/crashrate.sh` | the band + repetition driver: waits for the ambient window, holds to `keep_free`, repeats the documented single-bundle child, stops at the first fatal signal. This is the recipe that fixes recipe A's failure mode |
| `repro/mixed_retry_run.sh` | the same hold around the two-bundle `bench --backend all` command; stops when a report shows `RECOVERED_AFTER_TEARDOWN_CRASH` or `W_BACKEND_CRASHED_AT_TEARDOWN` |
| `repro/integration_check.sh` | the two cards' intersection: the documented mixed command on the integrated tree, expected to publish every row with no withheld row and no warning |
| `repro/fault_demo.sh` + `repro/fault_inject/sitecustomize.py` | the retained net, end to end, with an **injected** SIGSEGV at the isolated child's exit (armed by `GGUFONE_T57_FAULT`, `retry-ok` = first attempt only). Deterministic, no device needed |
| `repro/measure_footprint.sh` | what one child really takes on the device (nvidia-smi `used` before / peak / after, sampled while it runs) |
| `render_from_raw.py` | renders a `.raw` run log's markdown tables through `harness.render_report` (the file a table claim is checked against) |
| `mutmut_sweep.sh` | the Tier-M sweep driver: `--max-children 2` (the pyproject key is inert in mutmut 3.8), a pid-cgroup guard, and re-runs until the meta has no pending mutants |
| `hand_mutants.py` | the hand-mutation table over the diff (behavioural mutants + a control), because a partial sweep score cannot answer "does a behavioural mutant survive?" |
| `starved_child.sh` | the tiny probe child the starve recipes use to prove the tree under test is the one loaded |

Logs (`logs/`):

| group | what it is |
|---|---|
| `pre_*` | the first session's live runs (recipes A/B/batch) on 2026-09-18: exit codes, stdout/stderr, the hog's control trace, the VRAM snapshots and the window traces |
| `livep_*`, `mixed_*`, `small_rate.txt`, `band_rate.txt`, `band_window.txt` | the second session's live runs (2026-09-19) on a box held by the E3b sweep: recipe-B repeats, two-bundle runs, the 0.8B rate batch, and the band driver's empty window trace |
| `integrated.*`, `integrated_*` | the integration check on the integrated tree (`d82204f`): exit 0, both rows published, `withheld-rows=0` |
| `fault_retry-ok-red.*` | the injected signal on `f738315`: withheld row (the injection is real) |
| `fault_retry-ok-mid.*` | the injected signal on `2e7eb6c`: recovered row (`RECOVERED_AFTER_TEARDOWN_CRASH`, `process.attempts`) |
| `fault_always-mid.*` | both attempts die on `2e7eb6c`: withheld row + `W_BACKEND_CRASHED_AT_TEARDOWN` in the rendered table cell |
| `fault_retry-ok-int.*` | the injected signal on `d82204f`: **inert** (the child ends itself before `atexit`), rows published |
| `red_pretest.txt` | the new gate file against the **parent tree**: 40 failed, 0 passed |
| `full_suite*.txt`, `ruff*.txt`, `coverage*.txt`, `mutmut.out`, `mutation_*.txt`, `hand_mutants.txt` | the gate/suite/lint/coverage/mutation transcripts (first session; the `*_integrated.txt` pair is the 2026-09-19 pass on `d82204f`) |

Card-side helpers used while producing these numbers live in `/work/t57cc-scratch/` (`census.py`
for the mutmut meta, `meta.py`, `whoami.py` for the tree check, `show_report.py` for reading a
`.raw` report).
