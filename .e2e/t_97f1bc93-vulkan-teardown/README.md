# t_97f1bc93 — E2 FIX: a single Vulkan bundle's teardown SIGSEGV (raw material)

The readable version lives in `docs/evidence/e2_fix_t_97f1bc93_vulkan_teardown.md`. Every claim
there is checkable from here. All runs use the same venv and the same environment; a RED run
differs only in `PYTHONPATH` (`/work/t97-red/src` = the parent commit `f738315`, a `git worktree`).

Box: the operator host's container, `cgroup quota 2.0`, GPU passed through,
`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` in every command (that ICD's `library_path` is
`/usr/lib64/libEGL_nvidia.so.0`; without it no Vulkan row sees a device). Bundle:
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan`. Model:
`Qwen3.5-4B-Q4_0.gguf` — the card's own model (the small one is used only by the box's other cards).

## Root cause and reproduction (2026-09-18)

| file | what it is |
|---|---|
| `logs/baseline_4b_vulkan.raw` / `.exit` | the first reproduction: the operator's command, one bundle, 4B — the whole `ok: true` report and then **exit 139** |
| `logs/crashrate_a.summary` (+ `crashrate_a-*.raw`) | four more runs of the same command on the parent tree: **139**, 0, 0, 0 (free device memory before each run, in the summary) |
| `logs/segv_bt_a.summary` + `logs/segv_bt_a-2.raw` | six runs with the SIGSEGV handler preloaded: 0, **139**, 1, 1, 1, 0 — the 139 raw carries the **backtrace** (`SEGV_BT`, NVIDIA ICD frames from `__run_exit_handlers`) |
| `logs/segv_bt_a-3.raw` | the *other* regime: free memory in the 2.5–3.4 GB band → a typed `E_BACKEND_OOM` row (`measured: false` + reason + the backend's own `failed to allocate Vulkan0 buffer …` line), **exit 1**, `ok: false` |
| `logs/gdb_1.gdb` | the same command under gdb: the process exits **normally** — the crash is timing-dependent and gdb perturbs it (`core_pattern` is read-only here, so a core was not an option) |
| `logs/vendor_a.*` | control: the bundle's **own** `llama-bench`, same model, same ICD, 4 runs — all **exit 0** (no ggufone code involved) |
| `logs/vram_samples.txt`, `logs/mem_samples.txt` | device free memory (2 s) and the container's cgroup accounting (3 s) across the reproduction window |
| `logs/proxy_fidelity.*` | `proxy_fidelity.py` on the parent tree: `MAIN-RETURNED-0` then the proxy's SIGSEGV → **exit 139**, with the usage text already complete on stdout |

## Controls and the fix's gates

| file | what it is |
|---|---|
| `logs/red_parent.summary` (+ `red_parent-*.raw`) | the parent worktree (`/work/t97-red`), the card's command, 3 runs: 0, **139**, 0 |
| `logs/green_parent.summary` (+ `green_parent-*.raw`) | this tree, the same command, four runs: **0, 0, 0, 0** (see `red_min.sh`) |
| `logs/red_test_file.txt` | `tests/test_cli_teardown.py` against the **parent tree**: **6 failed, 1 passed** |
| `logs/red_pressure.hog` / `.summary` | the pressure driver's first attempt, stopped by the box's pid cap (empty summary) |
| `logs/green_gate_file.txt` | the 7+3-gate file on this tree (fresh, on the landing tree) |
| `logs/live_gate_green.txt` | the card's live gate on the **fitting** regime: pressure child ready, 1556 MiB free — **1 passed in 176 s**, CLI exit 0 |
| `logs/live_gate_green2.txt` | the live gate re-run on a **fully starved** device (488 MiB free at start, the pressure child itself got `E_BACKEND_OOM`): the measured run exits **1** with a parseable report and a typed reason — 1 passed in 47 s |
| `logs/verify_integrated.txt` | ruff + the gate file + the **full offline suite** on the rebased, integrated tree |

## Quality gates

| file | what it is |
|---|---|
| `logs/green_full_suite.txt` | the full offline suite on the rebased tree (pre-integration base) |
| `logs/ruff.txt` | `ruff check src tests` on the committing tree |
| `logs/coverage_changed.txt` | `coverage.sh`'s transcript: subprocess-aware coverage of the changed modules |
| `logs/mutmut.out` + `logs/mutmut_census.txt` + `logs/mutmut_survivors.txt` | the Tier-M sweep of `runtime/teardown.py` (driver retried per `mutmut_sweep.sh`-style loop, `--max-children 2`) |
| `logs/hand_mutants.txt` | the hand-mutation table: 7 behavioural mutants KILLED (each with its failing gate), 2 controls SURVIVED, every splice restored byte-identically |

## Vehicles

| file | what it is |
|---|---|
| `segv_bt.c` → `segv_bt.so` (gitignored) | the 44-line `LD_PRELOAD` SIGSEGV backtrace handler: `cc -shared -fPIC -O2 -o segv_bt.so segv_bt.c` |
| `proxy_fidelity.py` | the faithful offline proxy: an `atexit` callback that raises SIGSEGV, then the CLI — the same order the ICD's handler runs in |
| `hog.py` | the pressure child: loads a real GGUF through ggufone's loader with an exact layer count and holds it (`--layers`, `--hold`) |
| `coverage.sh` | the subprocess-aware coverage run (`COVERAGE_PROCESS_START` + a `sitecustomize.py`; see the script's header for why the child processes cannot all report) |
| `hand_mutants.py` | the hand-mutation table's driver (apply one mutation, run the gate file, restore, print the sha256s) |
| `pressure.sh`, `repeat.sh`, `red_control.sh`, `red_min.sh`, `vendor_control.sh` | the drivers behind the raws above (bundle paths, env, `PYTHONPATH`; `red_min.sh` is the fork-light one used while the box's pid cap was tight) |
| `sample_vram.sh`, `mem_sample.sh` | the two samplers (`nvidia-smi` / cgroup), fork-light on purpose |
| `gdb_target.sh` | the gdb wrapper for the target command (it perturbs the crash away; kept for the record) |
| `teardown_probe.py` | the four teardown variants (`none`, `backend_free`, `exit_fast`, `close_then_free`) the remedy was chosen against |

Reproduce (from the repository root):

```bash
PARENT_TREE=/work/t97-red bash .e2e/t_97f1bc93-vulkan-teardown/red_min.sh red_parent 3    # RED
LD_PRELOAD=$PWD/.e2e/t_97f1bc93-vulkan-teardown/segv_bt.so \
  bash .e2e/t_97f1bc93-vulkan-teardown/repeat.sh segv_bt_a 3                              # + backtrace
PARENT_TREE=$PWD bash .e2e/t_97f1bc93-vulkan-teardown/red_min.sh green_parent 4           # GREEN
sh .e2e/t_97f1bc93-vulkan-teardown/coverage.sh                                            # coverage
.venv/bin/python .e2e/t_97f1bc93-vulkan-teardown/hand_mutants.py                          # hand table
VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json \
GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \
GGUFONE_BENCH_MODEL=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf \
  .venv/bin/python -m pytest -q --run-network tests/test_bench_vulkan_teardown_live.py -s   # live gate
```
