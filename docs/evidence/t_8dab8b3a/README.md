# t_8dab8b3a — the first live matrix run's two failures, fixed at the root

Run `36159785190` (head `7c72c4b`) executed steps no earlier run had reached. Two of them failed on
first contact; both are fixed here, in the product — not in the workflow, not per-platform:

* **F1** `linux-cpu` / *the placement retry answers a typed row, never E_INTERNAL*
  (job `108153215436`) stopped **before** the OOM path: `RuntimeMissingError: E_RUNTIME_SYMBOLS: the
  bundle at /tmp/fake-bundle cannot name its CPU device (ggml_backend_dev_by_name('CPU') is
  missing)`. The fixture lagged behind a gate that landed after it (`cpu` rows are CPU-pinned —
  card t_55de5779 — so production resolves the bundle's CPU *device* before the ladder). It also
  carried an assertion that could never pass: `'3 placement(s)'` demanded of a *bench* row, while
  the three rungs belong to the ladder (`fit_oom_probe`), and a CPU-pinned row has exactly one.
* **F2** `windows-cpu` / *`bench --suite latency` end to end (llama.dll loads and computes)*
  (job `108153215424`) answered `E_MODEL_ARCH_UNSUPPORTED: the unknown build runtime at
  D:\a\_temp\typed-gguf-rt has no implementation for architecture 'qwen2' (looked for
  llama_model_qwen2 in libllama.so)` — every leg of the capability path was ELF-shaped.

macOS is untouched (it went green end-to-end in 7m25s, job `108153215494`).

---

## The empirical homework (F2) — one line first

**`llama.dll` (pinned `llama-b11026-bin-win-cpu-x64.zip`, b11026, 3 167 232 bytes) carries
`llama_model_qwen2` 6 times, ZERO of them NUL-terminated (the ELF form), 2 of them as MSVC RTTI
type descriptors (`<?AUgraph@llama_model_qwen2@@`, `<?AUllama_model_qwen2@@`) — and the string
`11026` occurs in NO file of the zip at all, so the build number can only come from
`llama-cli.exe --version`.**

The pinned Linux `libllama.so.0` (4 607 392 bytes) is the mirror image: 7 NUL-terminated forms
(Itanium typeinfo tails), 0 decorated, and no build string either (its banner is likewise the only
source). Re-measured for this receipt:

```
llama.dll (PE, pinned win-cpu-x64): 3167232 bytes
  llama_model_qwen2        : 6
  ...qwen2 + NUL (ELF tail): 0
  ...qwen2 + @@ (MSVC RTTI): 2
  build-string matches     : []
libllama.so.0 (ELF, pinned linux-x64-cpu): 4607392 bytes
  llama_model_qwen2        : 115
  ...qwen2 + NUL (ELF tail): 7
  ...qwen2 + @@ (MSVC RTTI): 0
  build-string matches     : []
win-cpu.zip 18439911 bytes: b'11026' -> 0
```
`docs/evidence/t_8dab8b3a/logs/byte_claims.txt` (`/workspace/t_8dab8b3a/verify_claims.py`).

Also measured on the real archive: the zip root holds `llama.dll`, `llama-cli.exe`,
`llama-cli-impl.dll`, `llama-common.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu-x64.dll` — the
names the tests now use as their Windows constants, and `llama-cli.exe` (not the DLL) is what prints
`version: 0.4.1-dev (build 11026, commit b49650adb)`.

## F2-a — `build_number` reads a real build on Windows

`capability.build_number(runtime_dir, *, system=None)`: the CLI is the *platform's*
(`finder.tool_name(role, system)` → `llama-cli.exe`), and its banner is parsed by the product's own
`parse_build`; the byte-scan fallback reads the library this platform really has
(`finder.library_names(system)["llama"]` → `llama.dll`) instead of a hardcoded `libllama.so`.
Pinned by `test_build_number_reads_the_windows_cli_banner` (fake bundle built from the archive's
real file names, `build=11026`, plus `parse_build(REAL_CLI_BANNER) == 11026`) and
`test_build_number_scans_the_windows_library_it_actually_has` (no CLI → the scan must read
`llama.dll`, not a Linux name that is not there).

## F2-b — `supports_arch` gives a REAL PE capability answer

The scan now accepts **both** decorations the ABI really uses, each documented with the measured
evidence (`_ARCH_NAME_BOUNDARIES = (b"\x00", b"@@")`):

* `\x00` — the class name ends an Itanium/ELF symbol (Linux/macOS; 7 in the pinned `libllama.so.0`);
* `@@` — the class name is inside an MSVC-decorated type descriptor (Windows PE; 2 in the pinned
  `llama.dll`). The ELF-only form is exactly what made the first live run say
  "no implementation for architecture 'qwen2'".

Never a hardcoded `true`: the answer is read from the bundle's own bytes, a bundle without the class
still answers `False`, and `test_arch_scan_reads_a_windows_decorated_name` asserts both directions
(the decorated bytes make `qwen2` supported; the same bytes must NOT make an unimplemented arch look
supported). Linux/macOS behaviour is unchanged — the ELF form is what those bundles carry.

## F2-c — no user-facing text hardcodes `libllama.so`

`require_arch`'s message interpolates the platform's library name
(`… looked for llama_model_llama in llama.dll`); `ProbeResult` gained `system` with `cli_name` /
`lib_name` properties, and the "cannot determine the build" text names the platform's CLI and
library; `ctypes_binding`'s `E_RUNTIME_SYMBOLS` message names the file the user actually has;
`doctor`'s details (`runtime.files`, `runtime.sha_recorded`) follow the same table.
Pinned by `test_require_arch_names_the_library_that_is_really_there` (asserts `llama.dll` is named
*and* `libllama.so` is absent) and
`test_the_build_failure_text_names_the_platform_tool_and_library`.

## F2-d — the distribution check is platform-aware

`finder` now separates **roles** (`llama-cli`) from **file names** (`llama-cli.exe`,
`llama.dll`): `tool_name` / `tool_names` / `library_names` / `library_glob` are one naming table, and
`finder.required_files(lock, system=...)` maps `runtime.lock`'s canonical SONAMEs onto it (Linux
returns the lock's list unchanged; an unknown name is passed through, never dropped). Callers:
`capability.probe_runtime`, `cli.doctor_checks`, `install`'s staging check, `ctypes_binding`'s
binding table.

The Windows doctor result, from the product's own `cli.doctor_checks()` under a patched platform
(`logs/windows_report.txt`):

```
doctor exit 2 status 'warnings' (deep probe skipped)
  ok    runtime.present          /tmp/win-receipt/typed-gguf-rt
  ok    runtime.files            all required libraries present: llama.dll, ggml.dll, ggml-base.dll
  warn  runtime.symbols          symbol probe skipped (TYPED_GGUF_DEEP_PROBE=0)
  ok    runtime.build            build b11026
  ok    runtime.fit_params       llama-fit-params --help exit 0
  ok    runtime.backends         backends: cpu (driveable here: cpu)
  warn  runtime.accelerator      no accelerator in this bundle (expected 'vulkan'); CPU works but decoding is slower
  warn  runtime.sha_recorded     runtime.json has no llama.dll SHA-256 (re-run `typed-gguf init --force`)
  warn  model.present            no model in the registry; run `typed-gguf models pull` (default model is pinned)
```

`runtime.files` is **ok** and names the DLLs; the old `runtime.loadable` failure is gone (it was the
Linux SONAMEs refusing a complete bundle); `runtime.build` reads `b11026`. What genuinely still
cannot be green on a Windows runner stays a **reported** warning —
`logs/windows_report_deep.txt` shows the same report with the deep symbol probe injected as resolved
(`runtime.symbols: resolved 34/34 required symbols`), which is what a Windows runner reports when it
can dlopen the PE.

`tools/matrix_windows_doctor.py` is inverted from the parent card's pin (it pinned the *gap*; this
card pins the *closed* truth). It now refuses a report with any `fail` check, an exit code that is
not `0`/`2`, a `runtime.files` detail that went back to the Linux names, a build number that was not
read, a bundle that was never found — and it annotates **every** run with the warnings that
remain (`::warning::` naming each check, "reported, never hidden"). 13 gates; RED first: the old
tool refused the fixed report (`logs/red_windows_doctor.txt`).

## F1 — the fixture satisfies the CPU-device gate, and BOTH OOM worlds are judged

The fixture (`tools/fixtures/fit_oom_bundle.c`) now exports `ggml_backend_dev_by_name("CPU")` (a real
address inside the library, which is all production does with the handle) and keeps the pre-fix
shape behind `-DTYPED_GGUF_FIXTURE_NO_CPU_DEVICE`, so the refusal this card fixes stays replayable
in the suite (`tests/test_fit_oom_fixture.py`, which compiles the fixture with `cc` and drives
production's own dlopen path: `ctypes_binding.load_libraries` + `cpu_device`).

RED → GREEN, the step's own commands (`logs/f1_red_green.txt`, and the exact YAML block run verbatim
in `logs/f1_step_exact.txt`):

```
=== block A: the live step head 7c72c4b (the pre-fix fixture) ===
pre-fix bench exit=1
pre-fix row: {'backend': 'cpu', 'measured': False, 'placement': 'n_gpu_layers=4 (cpu compute pinned)'}
pre-fix reason: RuntimeMissingError: E_RUNTIME_SYMBOLS: the bundle at /tmp/fb-prefix cannot name its
  CPU device (ggml_backend_dev_by_name('CPU') is missing); a CPU-pinned load ...

=== block B: the fixed head (this card) ===
fixed bench exit=1   (1 = nothing was measured: a reported row, not a crash)
fixed row: {'backend': 'cpu', 'measured': False, 'placement': 'n_gpu_layers=4 (cpu compute pinned)'}
fixed reason: BackendOomError: E_BACKEND_OOM: llama.cpp could not allocate device memory for the fit
  plan (n_gpu_layers=4, kv_type=auto, needed ~1010 MiB); ... tried 1 placement(s) down to CPU-only,
  none fit: n_gpu_layers=0 -> oom; ...

=== block C: the ladder world (tools/fit_oom_probe.py, same bundle) ===
probe exit=0
ladder code: E_BACKEND_OOM exit_code: 3
ladder message: E_BACKEND_OOM: ... tried 3 placement(s) down to CPU-only, none fit:
  n_gpu_layers=36 -> oom; n_gpu_layers=18 -> oom; n_gpu_layers=0 -> oom; ...
```

**About the `3 placement(s)` in the old assertion** (this is the F1 argument, with the evidence):
the count belongs to the *ladder* world, and that world is what carries it — `tried 3 placement(s)
down to CPU-only` is in block C, from the same fake bundle, through production's own degradation
policy. The *bench* row resolves to the `cpu` backend, which is CPU-pinned (card t_55de5779):
`placement: n_gpu_layers=4 (cpu compute pinned)` — no rungs to walk, so it honestly reports 1. The
old assertion asked a CPU-pinned row for a GPU ladder's count, and nothing caught it because the
step had never reached the row at all. Both worlds are now judged as **data** by
`tools/matrix_oom_row.py`, each against its own count, with the CPU-pin fact asserted as the reason
the counts differ, and with the crash this step exists for (`E_INTERNAL` / `AttributeError`) a hard
refusal. Gates: green on the fixed head (`fake-OOM row OK`), RED on the pre-fix row
(`fake-OOM row RED: 3 problem(s)`, `logs/oom_row_live_fixed.txt`).

---

## Gates (verbatim)

| gate | result |
| --- | --- |
| `env -u PYTHONPATH uv run --extra dev pytest -q` (card's exact gate) | `1 failed, 1905 passed, 58 skipped` — the one failure is `tests/test_bench_prompt_parity.py::test_a_row_records_which_framing_it_measured`, i.e. **this container has no llama.cpp bundle** (`BenchError: none of the requested backends (cpu) has a local llama.cpp bundle`). **Proved pre-existing**: the same test fails identically at the parent revision `7c72c4b` in a clean worktree with its own venv, with no card code involved (`logs/parent_rev_parity_failure.txt`); with a bundle dir it passes (row 2). `logs/full_suite.txt` |
| same + `TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-023` | **`1906 passed, 58 skipped`** — 0 failed. `logs/full_suite_bundle_dir.txt` |
| same + `TYPED_GGUF_TEST_BLOCK_NET=1` (the keep-gate shape the parent card ran) | **`1903 passed, 61 skipped`** — 0 failed. `logs/full_suite_ci_gates.txt` |
| `uv run python -m pytest tests/test_matrix_windows_doctor.py tests/test_runtime_matrix.py -q` | **`36 passed`**. `logs/gate_windows_doctor_and_matrix.txt` |
| F1 repro (the workflow step's exact commands, run verbatim from the YAML) | `STEP_EXIT=0` — both worlds answered the typed row. `logs/f1_step_exact.txt` |
| `uv run --extra dev ruff check src tests tools docs .github` | `All checks passed!` |
| workflow YAML + every shell block | YAML OK, 5 jobs, 23 shell blocks `bash -n` clean, 5 pwsh blocks untouched (bodies unchanged). `logs/workflow_yaml.txt`, `logs/step_syntax.txt` |
| per-file gates | `test_capability.py` 40 passed (RED: 8 failed/30 passed), `test_matrix_windows_doctor.py` 13 passed (RED: 7 failed), `test_matrix_oom_row.py` 13 passed (RED: 13 failed — the tool absent), `test_fit_oom_fixture.py` 3 passed (new: real `cc` build, real ladder), matrix+doctor+oom_row+pins 92 passed |
| mutation (Tier M, `capability.py` + `finder.py`) | 846 mutants: 416 killed / 204 survived / 226 no-tests / 0 timeouts → **419 killed** after the triage's three new gates (soft threshold; triage in `logs/mutmut_triage.txt`) |

## Decisions (the card's slack), and why

* **Probe mechanism (F2-b): a byte scan for the two real ABI decorations**, not a ctypes probe and
  not an export-table parser. It is the same evidence on every platform (the class was compiled into
  *this* library), it needs no loader on a host that may refuse the PE, and it is what the pinned
  DLL actually answers — measured, not assumed. A ctypes probe would have needed a Windows host to
  be *testable*, i.e. an untestable gate on the box that runs the tests.
* **Build-string source (F2-a): the CLI banner, with the library scan as fallback.** `11026` occurs
  in no file of the zip, so there is no honest alternative; the fallback stays for bundles that ship
  no CLI, and it reads the platform's own library.
* **F1 fixture approach: satisfy the gate, and make the refusal replayable.**
  `-DTYPED_GGUF_FIXTURE_NO_CPU_DEVICE` is a test-only knob that rebuilds the pre-fix shape, so the
  suite can prove the refusal instead of narrating it.
* **The step's judge is a tool, not an inline `python -c`.** The old one-liner was unreadable and
  never executed; both worlds are now judged by a tested tool that its own gate file pins
  (13 gates), the same pattern the parent card used for the doctor.
* **Tier M / mutation:** the sweep is retargeted at this card's surface
  (`capability.py` + `finder.py`, with `test_capability.py` + `test_pins.py`) in `pyproject.toml`;
  see the mutation section below.

## Mutation (Tier M — one sweep on the changed modules, soft threshold)

`uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2`
(`logs/mutmut_sweep.txt`, survivors in `logs/mutmut_survivors.txt`):

**846 mutants — 416 killed, 204 survived, 226 no-tests, 0 timeouts** (killed ÷ tested = 67.1%).
This selection is deliberately narrow (two gate files, per the repo's Tier-M convention on a
pid-capped shared box), so the score is not a whole-package number; the triage is what makes it
readable (`logs/mutmut_triage.txt`, `logs/mutmut_findings.txt`):

* **26 survivors mutate a line this card added** (the rest are pre-existing lines inside the same
  functions, or mutants whose gates live outside this selection).
* Three of those 26 pointed at **genuine gate holes on this card's own surface** and now die —
  the triage added one test each and re-ran those mutants scoped
  (`logs/mutmut_scoped.txt`; 416 → **419 killed**):
  * `finder.required_files`: the documented "a name the table does not know is passed through
    unchanged" had no test (`…or True` survived) →
    `test_required_files_passes_through_a_name_the_table_does_not_know`;
  * `capability.supports_arch`'s build floor (`current = None` made every arch look supported) and
    `require_arch`'s explicit-`build` path →
    `test_the_arch_scan_honours_the_build_floor` (both sides of the lock's b10828 floor — the test
    also corrected this card's own wrong assumption that the floor was b11026);
  * (the third is `require_arch`'s `build=current` threading, killed by the same test.)
* The remaining on-surface survivors are **benign/equivalent** or belong to the narrow selection:
  `finder.required_files`'s `.lower()`/`== "linux"` mutants survive because the callee
  (`library_names`) normalizes case and the role map is idempotent for the pinned names (both are
  no-tests in the bar, not real behaviour changes); `arch_symbol_names(runtime_dir, …)`'s mutants
  survive because that parameter is unused. Nothing on-surface is an unguarded behaviour change.

## What the next live run must show (watch items)

1. `windows-cpu`: `runtime.files` ok naming `llama.dll` / `ggml.dll` / `ggml-base.dll`,
   `runtime.build` = `b11026`, and the doctor step's annotation `::warning::` listing only
   `runtime.accelerator` / `runtime.sha_recorded` / `model.present` (whatever this runner really
   warns about) — no `fail` check.
2. `windows-cpu` engine smoke: the arch pre-flight passes on the DLL's decorated descriptor (this
   is the step that answered `E_MODEL_ARCH_UNSUPPORTED` in run `36159785190`).
3. `linux-cpu`: the fake-OOM step prints `fake-OOM row OK`, with the bench row's own `1
   placement(s)` and the ladder's `3 placement(s)` (the tool fails if either drifts).
4. `windows-cpu`'s `runtime.accelerator` warn expects `vulkan` when the runner reports a `/dev/dri`
   style host truth — unchanged behaviour, not this card's surface.

## Files changed

```
.github/workflows/runtime-matrix.yml      # the header's Windows claim, the fake-OOM step, the doctor step
src/typed_gguf/runtime/capability.py      # build_number / supports_arch / require_arch / ProbeResult
src/typed_gguf/runtime/finder.py          # roles vs file names, required_files, library_glob
src/typed_gguf/runtime/ctypes_binding.py  # the E_RUNTIME_SYMBOLS text names the platform's file
src/typed_gguf/runtime/install.py         # the staging check asks finder, not the lock literally
src/typed_gguf/cli.py                     # doctor's details name the platform's files
tools/fixtures/fit_oom_bundle.c           # ggml_backend_dev_by_name + the RED-replay knob
tools/matrix_windows_doctor.py            # the closed truth (was: the pinned gap)
tools/matrix_oom_row.py                   # NEW: both fake-OOM worlds judged as data
tests/test_capability.py                  # Windows names/PE bytes (38 gates)
tests/test_matrix_windows_doctor.py       # the closed truth (13 gates)
tests/test_matrix_oom_row.py              # NEW (13 gates)
tests/test_fit_oom_fixture.py             # NEW: the fixture through production's dlopen path
tests/test_runtime_matrix.py              # the step + the header claim pinned
pyproject.toml                            # [tool.mutmut] retargeted at this card's surface
```

Commits (local only — the coordinator certifies and pushes): `519adb6` (the capability path),
`dc1ec41` (F1 fixture + the two worlds), `5b8fbe1` (F2-d doctor/workflow), plus the receipt commit.
