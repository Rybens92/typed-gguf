# E2 FIX — a single Vulkan bundle's exit status is its report's (card t_97f1bc93)

Branch `main` (no remote; local commits on the shared tree) · Tier **M** (the card declares M; the
remedy is two functions in one new module and its gate file runs in seconds per mutant, so no
upgrade) · evidence schema `ggufone.evidence.vulkan-teardown/v1`

Box: the operator host's container (`cgroup quota 2.0` on a 24-CPU box; the E3 campaign owned the
GPU throughout these runs), GPU passed through, `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`
— an ICD json whose `library_path` is `/usr/lib64/libEGL_nvidia.so.0` (NVIDIA 615.71.09; without it
no Vulkan row can see a device). Bundle: the installed Vulkan
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan`. Model:
`Qwen3.5-4B-Q4_0.gguf` — the card's own model, and the shape that crashes (`--backend vulkan`
defaults to `n_gpu_layers=-1`, i.e. ~4.6 GB of device memory on an 8 GB card that other tenants were
also using).

Commits (this card, landed on the shared tree's `main` at `2e7eb6c`):

| commit | what |
|---|---|
| `d90bdf8` | `test`+`fix`: `runtime/teardown.py`, `cli.run`, the two entry points, the 7 child-process gates, the live gate, the console-script target |
| `4293e94` | `test`+raws: the 3 in-process gates (so a sweep and a coverage run see the remedy), `.e2e/t_97f1bc93-vulkan-teardown/` |
| `b1150df` | `gates`: the transcripts — gate file, full suite, ruff, coverage, the Tier-M sweep and the hand-mutation table |
| *(the commit that carries this document)* | `evidence`: this document, the QA report, the index fix |

Raw material for every claim below: `.e2e/t_97f1bc93-vulkan-teardown/` (index in its `README.md`).
The RED control is a `git worktree` of the parent commit (`/work/t97-red`, `f738315`) driven with the
same interpreter and the same environment (`PYTHONPATH=<tree>/src`), so only `src/` differs between a
RED and a GREEN run.

## 0. What was wrong (reproduced here, then fixed)

One bundle, one process, one report — and then a signal:

```
$ GGUFONE_RUNTIME_DIR=…/b11026-linux-x64-vulkan \
  VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json \
    python -m ggufone bench --suite throughput --model …/Qwen3.5-4B-Q4_0.gguf \
      --backend vulkan --runs 1 --threads 4 --sizes 64 --json
…
  "ok": true,
  "truncated": false,
  "skipped": [],
  "wall_ms": 89323.057
}                                                                    exit 139   (SIGSEGV)
```

(`logs/baseline_4b_vulkan.raw`; the report is complete and `ok: true` — the process died *after* it.)
The card's own symptom, in the isolated child of `--backend all`, is the same event seen through the
isolation layer: the row was withheld with `process.exit_code: -11` and
`the isolated child exited -11 (SIGSEGV; 139 in a shell) while its own report says 'ok: True'`
(`.e2e/t_dd62ec29-mixed-bundle-teardown/logs/after_mixed.raw`). This card is about the crash itself:
one bundle, no isolation, no second library in the picture.

It is intermittent — **4 crashes in 14 runs** of exactly this command on the parent tree, at every
level of free device memory from 5.6 GB down to 2.5 GB:

| batch | runs (exit codes) | notes |
|---|---|---|
| `logs/baseline_4b_vulkan.*` | **139** | no pressure child; the first reproduction |
| `logs/crashrate_a.*` | **139**, 0, 0, 0 | 3427 / 3359 / 3412 / 3461 MiB free before each run |
| `logs/segv_bt_a.*` | 0, **139**, 1, 1, 1, 0 | the 1s are the typed `E_BACKEND_OOM` (see §1.3) |
| `logs/red_parent.*` | 0, **139**, 0 | the parent worktree, the card's exact command |

## 1. Where the fault is — not ggufone, not llama.cpp

### 1.1 A backtrace, captured where gdb cannot help

The crash happens at process teardown, after the report: under gdb the same command exited
*normally* (`logs/gdb_1.gdb`, "Inferior 1 … exited normally") and `core_pattern` is read-only in
this container, so a core is not available. The capture used a 44-line `LD_PRELOAD` handler
(`segv_bt.c`, built with the box's `cc`): it prints the fault address and a libc backtrace and then
exits 139, so the crash's own status is preserved for the caller. The result
(`logs/segv_bt_a-2.raw`):

```
######## SEGV_BT: signal 11 (Segmentation fault) fault_address=0x18 ########
/work/t97-ggufone/.e2e/t_97f1bc93-vulkan-teardown/segv_bt.so(+0x1227)
/lib/x86_64-linux-gnu/libc.so.6(+0x3fdf0)
/usr/lib64/libnvidia-glvkspirv.so.615.71.09(+0x363b2)
/usr/lib64/libnvidia-eglcore.so.615.71.09(+0xcf48df)
/usr/lib64/libnvidia-eglcore.so.615.71.09(+0xcf6db5)
/usr/lib64/libnvidia-eglcore.so.615.71.09(+0x97edea)
/lib/x86_64-linux-gnu/libc.so.6(+0x92b7b)          <- __run_exit_handlers
/lib/x86_64-linux-gnu/libc.so.6(+0x1107f8)         <- __libc_start_main
```

The frames are the **NVIDIA driver's own exit handler**, reached from libc's exit-handler path: the
EGL core (`libnvidia-eglcore`) calling into the shader compiler (`libnvidia-glvkspirv`), dereferencing
a null-ish pointer (`fault_address=0x18`). There is no `libggml`, no `libllama` and no ggufone frame
in it. This is the class of code `runtime.isolated` already refuses to trust for probes — "the
third-party destructors that run at interpreter exit are not ours to trust" (`isolated.py`) — seen
from the side of the process that cannot dodge them by running in a child. The frames were published
to the sibling card `t_57cc0179` (its QA cites this card's rate and asks for exactly this evidence).

### 1.2 Controls

* **The vendor's own tool does not show it** — `llama-bench` from the same bundle, same model, same
  ICD, `-ngl 99 -p 64`, with the same SIGSEGV handler preloaded: 4 runs, all **exit 0**
  (`logs/vendor_a.*`). So the crash needs the shape ggufone's run creates (several contexts and
  their KV caches on a device that is nearly full), not merely "any Vulkan user on this box".
  Honest bound: those 4 runs were at ~3.5 GB free, i.e. *less* pressure than the crashing ggufone
  runs — this control rules out "the ICD crashes for everyone here", not "ggufone allocates
  unusually much".
* **The device's own pressure is real, and ggufone already answers it when it is extreme** — with
  free memory in the 2.5–3.4 GB band the same command does not crash at all: the fit ladder refuses
  the placement and reports a typed `E_BACKEND_OOM` row (`measured: false` + reason + the backend's
  own `failed to allocate Vulkan0 buffer of size 446799360` line, `ok: false`, exit 1 —
  `logs/segv_bt_a-3.raw`). The crash sits in the *other* regime: the placement fits, the numbers are
  real, and the driver dies while cleaning up.
* **The offline proxy reproduces the shape deterministically** — `proxy_fidelity.py` registers an
  `atexit` callback that raises SIGSEGV (what the ICD's handler does) and then calls the CLI: on the
  parent tree the process prints the usage, reports `MAIN-RETURNED-0`, and dies **139** with
  `THIRD-PARTY-TEARDOWN-RAN` on stderr (`logs/proxy_fidelity.parent.txt`). That is the same
  contradiction the card reports, without a device.

## 2. The remedy

`src/ggufone/runtime/teardown.py` (new, stdlib-only) carries the policy and the evidence above:

* `engine_loaded()` — has this process dlopened a local bundle (i.e. do its destructors exist here)?
* `end_process(code)` — flush both streams, then `os._exit(code)`: the exit handlers the crash lives
  in never run.

`cli.run()` (new, in `cli.py`) is the **process** entry point: `main()`'s code, then `end_process`
when a bundle is loaded, else `SystemExit` like any other CLI. `main()` is untouched — it still
returns a code and never ends the process, so every in-process caller (`tests/`, the API) is
unaffected. The two process entry points now use it: `python -m ggufone` (`__main__.py`) and the
installed console script (`[project.scripts] ggufone = "ggufone.cli:run"`).

What it does **not** do: it runs only after the command has produced its answer. An exception, a
signal during the run, a withheld row — everything that happens *before* the report still reaches the
shell exactly as it did (the card's own `E_BACKEND_OOM` rows and the isolation layer's withheld rows
are unaffected). What is taken away is one third party's vote on the exit status of a run that
already succeeded.

The alternative the card allows — "refuse the placement at load time" — was measured and rejected:
the crashing runs *fit*, measured real device work and reported `ok: true`
(`logs/segv_bt_a-2.raw`: `Vulkan0`/`Vulkan_Host` compute buffers, `effective_backend: "vulkan"`), so a
refusal would have to reject runs that are correct, on a device whose free memory the driver reports
as unknown. A typed refusal for the *unfittable* case already exists (`E_BACKEND_OOM`, §1.2).

## 3. Gates

### 3.1 Offline (`tests/test_cli_teardown.py`, 10 gates)

Seven gates spawn a child process that installs the faithful proxy (an `atexit` callback raising
SIGSEGV) and runs the CLI through `cli.run`, through `runpy` of `ggufone.__main__` — i.e.
`python -m ggufone`'s own module — and through the console script's declared target; they assert that
the process ends with the CLI's code, that the streams were flushed (a piped stdout is
block-buffered: only an explicit flush keeps the usage text) and that the proxy never ran. Three
more drive the same functions *in this* process with the exit spied on, so a mutation run and a
coverage run see the remedy (the child-process gates execute it in another process).

| tree | result |
|---|---|
| parent (`/work/t97-red` @ `f738315`) | **6 failed, 1 passed** (`logs/red_test_file.txt`) — every gate that depends on the remedy fails; the `main()` purity pin passes on both trees by design (the 3 in-process gates postdate that RED run and are the same assertions as their child-process twins) |
| this tree, pre-integration base `cbaed9a` | **10 passed** in 5–9 s (`logs/green_gate_file.txt`) |
| this tree, rebased onto `2e7eb6c` (`0702a46`), and again in the landed shared tree | **10 passed** in 9.20 s (`logs/verify_integrated.txt`), and **10 passed** against the landed commits (card comment) |

### 3.2 Live (`tests/test_bench_vulkan_teardown_live.py`, `--run-network`)

The card's requirement 3, executed twice, in the two regimes the box offered: the operator's
command, the pinned bundle, the 4B model, and a pressure child of its own that holds 12 layers of
the same model on the same device. Assertions: the exit code is 0 or 1, the report file exists and
parses, the exit code *is* the report's `ok`, `--json` on stdout equals the file's bytes, and the row
either corroborates its bundle and device set or says in a typed reason why it was not measured.

| run | result |
|---|---|
| this tree, pressure child ready (1556 MiB free) — the **fitting** regime | **exit 0, 1 passed in 176 s** (`logs/live_gate_green.txt`; the E3 campaign held ~2 GB too) |
| this tree, device **already starved** (488 MiB free at start; the pressure child itself got `E_BACKEND_OOM`) — the **typed-failure** regime | **exit 1, report parseable, typed reason, 1 passed in 47 s** (`logs/live_gate_green2.txt`) |
| parent-tree command, same shape, no pressure child | **0, 139, 0** — the crash, twice seen (`logs/red_parent.*`) |
| this tree, same shape, four runs | **0, 0, 0, 0** (`logs/green_parent.*`) |

### 3.3 Why a live gate can only assert the guarantee

The driver's crash is intermittent (roughly a third of the runs here) and depends on the box's other
tenants; no test can force it. What the live gate can do — and now does — is bound the outcome: under
the same command and pressure, the exit status is the report's. The RED side of it is the parent
tree's measured crash rate above, plus the deterministic offline proxy.

## 4. Quality

| gate | result |
|---|---|
| full offline suite (integrated tree `0702a46`, `HOME=/var/home/rybens`, no `VK_DRIVER_FILES`) | **1083 passed, 42 skipped, exit 0** in 173 s (`logs/verify_integrated.txt`); the landed commits' `src/`, `tests/` and `docs/` are byte-identical to that tree — the only difference is `pyproject.toml`'s mutmut comment block, which no test reads (§6) |
| ruff (`python -m ruff check src tests`) | All checks passed (`logs/verify_integrated.txt`) |
| coverage, changed modules (subprocess-aware) | `runtime/teardown.py` **100 %** (13/13), `__main__.py` **100 %** (1/1), `cli.run()` 3/4 statements — the missing one is the `os._exit` call itself: a process that ends there cannot write coverage data, which is the fix working; the call's own function is 100 % and the child-process gate proves the branch by exit code (`logs/coverage_changed.txt`) |
| Tier-M mutation (`runtime/teardown.py`, `tests/test_cli_teardown.py`) | **4/4 killed, 0 survived, 0 pending** (`logs/mutmut.out`, `logs/mutmut_census.txt`); the 4 are mutmut 3.8's `None`-substitution mutants, all killed |
| hand-mutation table (the behavioural mutants the tool cannot emit) | **7/7 KILLED** — drop the flush, drop `os._exit`, loader always false, loader always true, silent 0, `__main__` back to `main`, console script back to `main` — each with its failing gate named; **2 controls SURVIVED** (the equivalent `int(code)` → `code`, and a docstring wording change); every splice restored byte-identically (`logs/hand_mutants.txt`) |
| live gate | exit 0 and exit 1 in the two regimes above, report parseable in both |

## 5. Reproducing the box side

```
# RED: the parent tree, the card's command, until it dies (about every third run)
PARENT_TREE=/work/t97-red bash .e2e/t_97f1bc93-vulkan-teardown/red_min.sh red_parent 3

# the crash, with a backtrace instead of a bare 139
LD_PRELOAD=$PWD/.e2e/t_97f1bc93-vulkan-teardown/segv_bt.so \
  bash .e2e/t_97f1bc93-vulkan-teardown/repeat.sh segv_bt_a 3

# GREEN: the same command on this tree — 0/1, never a signal
PARENT_TREE=$PWD bash .e2e/t_97f1bc93-vulkan-teardown/red_min.sh green_parent 4

# the card's live gate (needs a window with a few hundred MiB to a few GiB free)
VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json \
GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \
GGUFONE_BENCH_MODEL=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf \
  .venv/bin/python -m pytest -q --run-network tests/test_bench_vulkan_teardown_live.py -s

# coverage of the changed modules, and the hand-mutation table
sh .e2e/t_97f1bc93-vulkan-teardown/coverage.sh
.venv/bin/python .e2e/t_97f1bc93-vulkan-teardown/hand_mutants.py
```

## 6. Landing and coordination

* The four commits land on the shared tree's `main` at `2e7eb6c` while two sibling cards
  (`t_57cc0179`, `t_6952f0dd`) keep working in the same checkout. To leave their working-tree edits
  alone, this card's `pyproject.toml` change is the **console-script line only**: the
  `[tool.mutmut]` block is a hotspot every card retargets, and `t_6952f0dd` had an uncommitted
  retarget of it (`source_paths = ["src/ggufone/bench/labels.py"]`) in the tree at landing time.
  This card's Tier-M sweep pair is therefore recorded here instead:
  `source_paths = ["src/ggufone/runtime/teardown.py"]` with
  `pytest_add_cli_args_test_selection = ["tests/test_cli_teardown.py"]` — the exact pair the sweep
  in §4 ran with (`logs/mutmut.out`), for the next sweep to restore. (A first landing attempt with
  the pair retargeted in the commit was refused by the sibling's dirty hunks, exactly as the
  three-way merge is supposed to; the sweep never depends on the shared file's current value.)
* The console-script target *is* pinned by a gate
  (`test_the_console_script_points_at_the_process_entry_point`), so the entry point cannot drift
  back to `main` silently even though the mutmut bookkeeping lives in this document.
* The backtrace in §1.1 answers the sibling `t_57cc0179`'s request on this card's thread; its QA
  report cites this card's crash rate ("2 of 5 identical single-bundle runs").
* The `W_BACKEND_CRASHED_AT_TEARDOWN` recovery that `t_57cc0179` added is the layer *above* this fix:
  with `cli.run` in place, a child that dies at teardown can no longer contradict the report it
  wrote; the recovery remains the answer for the residual case where the ICD's handler runs before
  the report is complete.
