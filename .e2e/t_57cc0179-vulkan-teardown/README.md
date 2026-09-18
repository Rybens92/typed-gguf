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
| `repro/starve_and_run.sh` | recipe A: starve the device **before** the child (the "too full to load" side: the loader's own fit ladder refuses with a typed `E_BACKEND_OOM`) |
| `repro/starve_at_teardown.sh` | recipe B: let the placement load, then **hold VRAM while the child exits** — the card's own suggestion; the 5th arg `gdb` adds a C backtrace |
| `repro/batch_teardown.sh` | repeats recipe B until a *signal* appears (the crash shape), stopping at the first one |
| `repro/measure_footprint.sh` | what one child really takes on the device (nvidia-smi `used` before / peak / after) |
| `mutmut_sweep.sh` | the Tier-M sweep driver: `--max-children 2` (the pyproject key is inert in mutmut 3.8), a pid-cgroup guard, and re-runs until the meta has no pending mutants |
| `logs/red_pretest.txt` | the new gate file against the **parent tree**: 40 failed, 0 passed |
| `logs/full_suite.txt` | `pytest -q` on this tree (the two host-reading gates that failed there pass in isolation — see the QA note) |
| `logs/` (`pre_*`) | the live runs of the recipes: exit codes, the child's stdout/stderr, the hog's control trace, the VRAM snapshots, the window traces |
| `logs/mutmut.out` | the Tier-M sweep's raw output (attempts, per-attempt census, survivor list) |

Card-side helpers used while producing these numbers live in `/work/t57cc-scratch/` (`census.py`
for the mutmut meta, `meta.py`, `whoami.py` for the tree check).
