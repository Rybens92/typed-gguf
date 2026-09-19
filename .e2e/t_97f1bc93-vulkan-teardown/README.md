# t_97f1bc93 — E2 FIX: a single Vulkan bundle's teardown SIGSEGV (raw material)

The readable version lives in `docs/evidence/e2_fix_t_97f1bc93_vulkan_teardown.md`. Every claim
there is checkable from here. All runs use the same venv and the same environment; a RED run
differs only in `PYTHONPATH` (`/work/t97-red/src` = the parent commit `f738315`, a `git worktree`).

Box: the operator host's container, `cgroup quota 2.0`, GPU passed through,
`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` in every command (that ICD's `library_path` is
`/usr/lib64/libEGL_nvidia.so.0`; without it no Vulkan row sees a device). Bundle:
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan`. Model:
`Qwen3.5-4B-Q4_0.gguf` — the card's own model (the small one is used only by the box's other cards).

| file | what it is |
|---|---|
| `logs/baseline_4b_vulkan.raw` / `.exit` | the first reproduction: the operator's command, one bundle, 4B — the whole `ok: true` report and then **exit 139** |
| `logs/crashrate_a.summary` | four more runs of the same command on the parent tree: **139**, 0, 0, 0 (free device memory before each run, in the summary) |
| `logs/segv_bt_a.summary` + `logs/segv_bt_a-2.raw` | six runs with the SIGSEGV handler preloaded: 0, **139**, 1, 1, 1, 0 — the 139 raw carries the **backtrace** (`SEGV_BT`, NVIDIA ICD frames from `__run_exit_handlers`) |
| `logs/segv_bt_a-3.raw` | the *other* regime: free memory in the 2.5–3.4 GB band → a typed `E_BACKEND_OOM` row (`measured: false` + reason + the backend's own `failed to allocate Vulkan0 buffer …` line), **exit 1**, `ok: false` |
| `logs/gdb_1.gdb` | the same command under gdb: the process exits **normally** — the crash is timing-dependent and gdb perturbs it (`core_pattern` is read-only here, so a core was not an option) |
| `logs/vendor_a.*` | control: the bundle's **own** `llama-bench`, same model, same ICD, 4 runs — all **exit 0** (no ggufone code involved) |
| `logs/red_parent.summary` | the parent worktree (`/work/t97-red`), the card's command, 3 runs: 0, **139**, 0 |
| `logs/green_parent.summary` | this tree, the same command, four runs: **0, 0, 0, 0** (see `red_min.sh`) |
| `logs/red_test_file.txt` | `tests/test_cli_teardown.py` against the **parent tree**: **6 failed, 1 passed** (the `main()` purity pin passes on both trees) |
| `logs/live_gate_green.txt` | the card's live gate on this tree (pressure child ready, 1556 MiB free): **1 passed in 175.97 s**, the CLI exit 0 |
| `logs/proxy_fidelity.*` | `proxy_fidelity.py` on the parent tree: `MAIN-RETURNED-0` then the proxy's SIGSEGV → **exit 139**, with the usage text already complete on stdout |
| `logs/vram_samples.txt`, `logs/mem_samples.txt` | device free memory (2 s) and the container's cgroup accounting (3 s) across the reproduction window |
| `logs/green_full_suite.txt`, `logs/ruff.txt` | the offline suite (fresh) and `ruff check src tests` on the committing tree |
| `logs/coverage_changed.txt` | `coverage run -m pytest` over the new gate files + report for the changed modules |
| `logs/mutmut.out` + `logs/mutation_score.txt` + `logs/survivor_triage.txt` | the Tier-M sweep of `src/ggufone/runtime/teardown.py` with `tests/test_cli_teardown.py` |
| `segv_bt.c` → `segv_bt.so` (gitignored) | the 44-line `LD_PRELOAD` SIGSEGV backtrace handler: `cc -shared -fPIC -O2 -o segv_bt.so segv_bt.c` |
| `proxy_fidelity.py` | the faithful offline proxy: an `atexit` callback that raises SIGSEGV, then the CLI — the same order the ICD's handler runs in |
| `hog.py` | the pressure child: loads a real GGUF through ggufone's loader with an exact layer count and holds it (`--layers`, `--hold`) |
| `pressure.sh`, `repeat.sh`, `red_control.sh`, `red_min.sh`, `vendor_control.sh` | the drivers behind the raws above (bundle paths, env, `PYTHONPATH`; `red_min.sh` is the fork-light one used while the box's pid cap was tight) |
| `sample_vram.sh`, `mem_sample.sh` | the two samplers (`nvidia-smi` / cgroup), fork-light on purpose |
| `teardown_probe.py` | the four teardown variants (`none`, `backend_free`, `exit_fast`, `close_then_free`) the remedy was chosen against |

Reproduce (from the repository root):

```bash
PARENT_TREE=/work/t97-red bash .e2e/t_97f1bc93-vulkan-teardown/red_min.sh red_parent 3    # RED
LD_PRELOAD=$PWD/.e2e/t_97f1bc93-vulkan-teardown/segv_bt.so \
  bash .e2e/t_97f1bc93-vulkan-teardown/repeat.sh segv_bt_a 3                              # + backtrace
PARENT_TREE=$PWD bash .e2e/t_97f1bc93-vulkan-teardown/red_min.sh green_parent 4           # GREEN
```
