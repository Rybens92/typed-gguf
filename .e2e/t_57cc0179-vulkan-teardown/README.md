# t_57cc0179 — E2 FIX: the Vulkan child that SIGSEGVs at teardown (raw material)

The readable version is `docs/evidence/e2_fix_t_57cc0179_vulkan_teardown_crash.md`; every claim
there is checkable from here. The code lives on the card's branch of the shared tree
(`src/ggufone/bench/isolation.py`, `src/ggufone/bench/suites.py`,
`tests/test_bench_teardown_crash.py`).

Box: the operator host's container — cgroup quota 2.0, pid cgroup 256 (shared with the
voice-companion stryker tree, the E3 campaign and other kanban workers: my first live attempt died
with `Cannot fork`), GPU passed through with `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`.
Bundles: `GGUFONE_RUNTIME_DIR` = the pinned `linux-x64-cpu` asset at `/work/t603-runtime`,
`GGUFONE_BENCH_RUNTIME_DIR` = the installed Vulkan b11026 bundle under
`/var/home/rybens/.local/share/ggufone/runtime`. Models: `Qwen3.5-4B-Q4_0.gguf` (2.4 GiB weights)
and `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (0.53 GiB).

The two trees:

* `/work/t57cc-red` — a `git worktree` pinned at `f738315` (the parent commit: the `t_dd62ec29`
  containment, no retry). `PYTHONPATH=<tree>/src` selects the tree under the run's interpreter
  (verified: `ggufone.__file__` and `hasattr(isolation, "teardown_crash")`).
* `/work/t57cc-ggufone` — this card's private clone (a sibling's `git add -A` sweeps WIP in the
  shared tree, so the work is done here and fast-forwarded into the shared tree at the end).

| file | what it is |
|---|---|
| `repro/vram_hog.c` | the dummy allocator: holds N MiB of `DEVICE_LOCAL` memory on the discrete device and sleeps — the pressure comes from *outside* the process under test |
| `repro/starve_and_run.sh` | the minimal repro recipe: waits for the ambient memory window, brings the device to ~N MiB free with the hog, runs one Vulkan bench child, records nvidia-smi before/during/after + stdout/stderr/exit code (5th arg `gdb` adds a C backtrace) |
| `mutmut_sweep.sh` | the Tier-M sweep driver: re-runs `mutmut run` until the meta has no pending mutants (the box's pid cap kills a run at `os.fork` with `EAGAIN`) |
| `logs/red_pretest.txt` | the new gate file against the **parent tree**: 40 failed, 0 passed |
| `logs/` (`pre_*`) | the live runs of the repro recipe: exit codes, the child's stdout/stderr, the hog's control trace, the VRAM snapshots |
| `logs/mutmut.out` | the Tier-M sweep's raw output (attempts, per-attempt census, survivor list) |

Card-side helpers used while producing these numbers live in `/work/t57cc-scratch/` (`census.py`
for the mutmut meta, `meta.py`, `whoami.py` for the tree check).
