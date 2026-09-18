# E2 FIX — bench backend labels must follow the effective compute path (card t_603a35a0)

Branch `main` (no remote; local commits on the shared tree) · Tier **M** (the card declares none) ·
evidence schema `ggufone.evidence.backend-attribution/v1`

Box: the operator host, its container (`cgroup quota 2.0` on a 24-CPU box) and **its GPU passed
through** — `/dev/nvidia0`, `/dev/dri/renderD128`; the Vulkan ICD is not on the container's default
loader path, so every run sets `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` (without it
`llama-bench --list-devices` prints `(none)`). Bundles: the installed Vulkan
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` and a verified copy of the
pinned `linux-x64-cpu` asset (`runtime.lock`, sha256 `219cf1c7…`) at
`/work/t603-runtime/b11026-linux-x64-cpu`. Model: `Qwen3.5-4B-Q4_0.gguf` (2 583 221 408 B), the
smallest local GGUF, so the CPU-class rows finish while the E3 campaign owns the device.

Commits (this card, on top of the tree's `main`):

| commit | what |
|---|---|
| `cd42633` | `test`: the RED gates (`tests/test_bench_attribution.py`, 14) + the `device_log` seam in `tests/fake_engine.py` |
| `5fdff59` | `fix`: `runtime/devices.py`, the sink plumbing in `engine/session.py`, the attribution in `bench/harness.py` + `bench/suites.py`, the rendered `effective` column |
| `d901221` | `fix`: an accelerator claim the engine log cannot corroborate is flagged too (the mixed-bundle row captures *nothing*) |
| `52fa6bd` | `test`: the sink plumbing gate, the layer counts, ruff fixes |
| `d54485d` | `chore`: the Tier-M mutmut pair retargeted at the new parser (card convention) |
| *(the commit that carries this document)* | `fix` + `evidence`: the soft cap / unverifiable-claim interaction, this document and `.e2e/t_603a35a0-backend-attribution/` |

(SHAs as landed on the tree's `main`, which already carried the E1c FIX `t_e29734e6` and the E2
`t_f46cec41` cards when this card's commits were rebased onto it.)

Raw material for every claim below: `.e2e/t_603a35a0-backend-attribution/` (index in its
`README.md`).

## 0. What was wrong, in two call paths

`harness.backend_runtimes` registers every bundle under the backend it advertises **and under
`cpu`** (`bench/harness.py`), and `suites._throughput_row` / `_determinism_row` label the row with
the backend *it asked for*, while the report's only device-shaped field was
`placement_used.n_gpu_layers` — a statement about the **weights**. Two measured consequences on a
host with more than one bundle:

1. **`--backend all`, CPU bundle first, Vulkan bundle also visible.** The `vulkan` row
   (`n_gpu_layers=-1`, `runtime_dir` = the Vulkan bundle) ran its graph on the **host CPU**:
   12.29 tok/s prefill, `CPU_Mapped`/`CPU_REPACK` weights, 33 layers assigned to device CPU, nine
   `CPU compute buffer size` lines and **not one `Vulkan*` line in the whole process log** — report
   `ok: true`, no warning. The process then died with `double free or corruption (!prev)` after
   writing the table (exit 134).
2. **Only the Vulkan bundle installed.** `cpu` resolves to *that* bundle's directory (every bundle
   answers for `cpu`), `n_gpu_layers=0` keeps the weights on the host — and llama.cpp's **op
   offload** runs the graph on the device anyway: `sched_reserve: Vulkan0 compute buffer size` ×3 in
   the engine log, under a row labelled `cpu` whose placement note read *"CPU only: the fit plan
   offloads nothing"*. Report `ok: true`, exit 0; the auditor measured the same shape at 587.9
   tok/s (GPU-class) on the 4B Q8 model.

So the published tables could show a `vulkan` row that never touched the device and a `cpu` row
that only ran on it. E3's tables (running then) were exposed to both.

## 1. Requirement 1 — the row carries what the engine *did*, not what it was asked

`runtime/devices.py` (new, stdlib-only) reads llama.cpp's own lines back:

```
sched_reserve:        CPU compute buffer size =   166.26 MiB     -> compute_buffers {"CPU": 2}
~llama_context:    Vulkan0 compute buffer size is 545.3125 MiB   -> compute_buffers {"Vulkan0": 1}
load_tensors:   CPU_Mapped model buffer size =  4167.21 MiB     -> model_buffers {"CPU_Mapped": 1}
load_tensors: layer   3 assigned to device CPU, is_swa = 0       -> layers {"CPU": 1}
```

* **Compute buffers are the evidence** for the compute path (`effective`): `Vulkan0`/`Vulkan_Host`
  → `vulkan`, `CPU_Mapped`/`CPU_REPACK` → `cpu`; several devices are joined (`cpu+vulkan`).
* **`load_tensors: offloaded N/M layers to GPU` is a request, not a measurement** — the mixed run
  printed `offloaded 37/37 layers to GPU` immediately before walking every layer onto the CPU, so
  the parser deliberately ignores that line (pinned by
  `test_the_request_side_offload_line_is_never_evidence`).

The engine side appends its own lines to a caller-owned sink: `session.open_model(..., log=)` for
the load that succeeded and `session.ModelSession(..., log=)` for every successful context creation
(failed ladder rungs stay out of it). `bench.harness.LiveModel` owns that sink and exposes
`device_log`; `harness.device_usage(model, claimed)` turns it into the row fields
`devices`, `device_buffers`, `effective_backend`, `warnings`. Every throughput/determinism row
carries them next to `runtime_dir`, and the single-backend suites (latency/quality/calibration)
carry the same four fields on the report. The rendered table gained an `effective` and a
`device buffers` column, and every report prints
`- engine devices: CPU=3 (compute buffers) · effective backend: cpu`.

## 2. Requirement 2 — `--backend all` verifies per backend, and a lie fails the report

`devices.contradicts(claimed, usage)`:

* a claim of **`cpu`** is refuted by any accelerator in the compute buffers (op offload);
* a claim of an **accelerator** is refuted unless that backend's own device appears in the compute
  buffers — including the case where the log carries **no** compute-buffer line at all, because the
  mixed-bundle row's second bundle produces *nothing* to verify with (that is exactly the case the
  first version of this fix reported as `effective_backend: null` with no warning — found by the
  operator-box re-run, fixed in `9a26020`).

A contradicted row keeps its numbers, gains `W_BACKEND_MISMATCH` in `warnings`, gets a report note
naming the effective path and the fix ("re-run one backend per process"), and the report is
`ok: false` — `ggufone bench` then exits **1** instead of publishing the row. `W_BACKEND_MISMATCH`
joins `errors.WARNING_CODES`. An empty device set does not refute a `cpu` claim (nothing computed
elsewhere), so a genuine CPU-only box stays clean.

## 3. Requirement 3 — the regression test, RED on the parent tree

`tests/test_bench_attribution.py` builds **two real bundle directories** (a CPU one and one carrying
`libggml-vulkan.so`), drives `suites.run_suite` through the model-free seam
(`tests.fake_engine.BenchModel`, whose `device_log` is the operators' own log lines verbatim) and
pins: the parser, the effective-backend rule, both lying directions, the honest directions, the
unverifiable direction, determinism rows, the single-backend suites, the rendered table, the note
wording, and the sink plumbing (through `test_fit_oom_recovery`'s real `llama_log_set` fake).

* RED on the parent tree (`1205c0b` + the new parser module, no row/report change):
  **8 failed, 4 passed** — `logs/red_pretest.txt`.
* GREEN: `pytest -q tests/test_bench_attribution.py` → **14 passed** (`logs/green_attribution_tests.txt`).

One interaction worth naming: card `t_f46cec41` (`--quick`, the soft cap) landed on `main` while
this card was in flight. Its contract — a cap that expired before a measurement is *incompleteness,
never a gate failure* — and this card's verification rule meet in one place: a run whose budget
expired before the model load attributes **nothing** (no device evidence, no
`W_BACKEND_MISMATCH`), because no engine ran at all. `test_a_cap_truncated_run_is_incomplete_never_unverifiable`
pins it, and the rebase merged both card's report shapes (wall/truncated/quick lines plus the
`engine devices` line).

## 4. Requirement 4 — the operator-box re-run, before/after, raw

`logs/rows_before_after.txt` is the row-level diff straight from the JSON; `logs/runs.md` has the
commands, bundles and exit codes. The two configurations, `--suite throughput --runs 1 --threads 4
--sizes 64 --json` (VRAM free per run in the logs):

| row | BEFORE (parent tree) | AFTER (this card) |
|---|---|---|
| `vulkan` (mixed install) | `effective` absent, `warnings` absent, `ok: true`, exit 134 after the table | `effective_backend: null`, `devices: []`, `device_buffers: {}`, `warnings: ["W_BACKEND_MISMATCH"]`, report `ok: false` |
| `cpu` (Vulkan-only install) | note "CPU only: the fit plan offloads nothing", `ok: true`, exit 0 — while three `Vulkan0 compute buffer size` lines show the device computed | `effective_backend: "vulkan"`, `device_buffers {"Vulkan0": 3, "Vulkan_Host": 3}`, `W_BACKEND_MISMATCH`, report `ok: false`, **exit 1** |
| honest `cpu` row (mixed install) | `ok: true` | `effective_backend: "cpu"`, `device_buffers {"CPU": 3}`, no warning |

Reproduce (from the card's tree, box caveats in `logs/runs.md`):

```bash
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json            # GPU ICD inside the container
export GGUFONE_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu    # the pinned CPU asset
export GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime
uv run --frozen ggufone bench --suite throughput \
  --model /var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf --backend all \
  --runs 1 --threads 4 --sizes 64 --json

unset GGUFONE_RUNTIME_DIR          # only the Vulkan bundle: `cpu` resolves to its directory
uv run --frozen ggufone bench --suite throughput \
  --model /var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf --backend cpu \
  --runs 1 --threads 4 --sizes 64 --json
```

## 5. What was *not* changed (and why)

* **The loader.** One process still loads the second bundle's libraries into a process that already
  has the first bundle's (`_LOADED` caches per directory; `RTLD_GLOBAL` + identical SONAMEs). The
  measured symptom is that the second row's engine produces *no* log line at all and its model runs
  on the host. Root cause (dlopen/SONAME interposition across two copies of `libllama`/`libggml`) is
  a **hypothesis**, not proven here; the remedy the evidence supports is the one the auditor's F4
  already named — **one bundle per process** — and the report now says so instead of lying.
* **`n_gpu_layers` semantics.** Op offload is llama.cpp's own default (`op_offload`); the fix
  reports it rather than disabling it, so a `--backend cpu` row on an accelerator host still
  measures what the engine does — but it can no longer be *published* as a CPU row silently.

## 6. Gates

| gate | result |
|---|---|
| new file, parent tree (RED) | 8 failed, 4 passed (`logs/red_pretest.txt`) |
| new file, this tree (GREEN) | **14 passed** |
| `uv run --frozen --extra dev pytest -q` (offline, full) | **923 passed, 40 skipped** — `logs/green_full_suite.txt` |
| ruff | `All checks passed!` (src + tests) — `logs/ruff.txt` |
| coverage (changed modules) | `devices.py` **100 %** · `bench/harness.py` **93 %** · `bench/suites.py` **99 %** · `errors.py` **100 %** — `logs/coverage_changed.txt` |
| Tier-M mutation (soft), `runtime/devices.py` | 81 mutants, 73 killed → **90.1 %**; the 8 survivors are equivalent mutants (`rstrip(None)` vs `rstrip("_")`, `maxsplit=2` vs 1 on the same `[0]`, and an empty-log placeholder that changes nothing) — `logs/mutation_score.txt`, `logs/devices.py.meta` |
| live (GPU) | the four operator-box runs in §4, plus the two RED/GREEN runs quoted there |

One caveat, honestly: on a **VRAM-starved** box (`memory.free` 174 MiB) the full offline suite fails
`tests/test_fit.py::test_the_cli_fit_command_returns_the_documented_json` (a fit plan estimated on
real host facts degrades and its `warnings` list grows). That failure reproduces on the **parent
tree** in the same conditions — `logs/prefix_environmental_failure_control.txt` — and disappears
when the GPU is free; it is environment-dependent and not this card's change.

## 7. Findings for the coordinator

* **F1 (new, filed as its own card).** The mixed-bundle run aborts at teardown:
  `double free or corruption (!prev)`, exit **134**, *after* the report was written. The fix under
  test does not touch it; the report is complete but a CI step that trusts the exit code would read
  the run as failed. One bundle per process avoids it.
* **F2.** The second bundle's engine lines are captured by nothing (its libraries are shadowed);
  this is why `effective_backend` is `null` for that row rather than `cpu`. A future fix that makes
  two bundles coexist should re-run `logs/after_mixed.raw`'s command and expect a real device set.
* **F3.** `commands.reproduce` does not echo `--sizes` (nor `--kv-type`), so the recorded command
  in these evidence reports is not the exact one run; pre-existing, and the F2/provenance card's
  ground if it wants it.
* **F4.** **E3's tables must be regenerated from this commit** — the renderer gained the `effective`
  and `device buffers` columns, and a row whose label the engine log refutes now fails the report
  instead of publishing numbers under a wrong backend.

## 8. Files

`src/ggufone/runtime/devices.py` (new) · `src/ggufone/bench/harness.py` ·
`src/ggufone/bench/suites.py` · `src/ggufone/engine/session.py` · `src/ggufone/errors.py` ·
`tests/test_bench_attribution.py` (new) · `tests/fake_engine.py` ·
`tests/test_fit_oom_recovery.py`, `tests/test_fit_live.py` (note wording) · `pyproject.toml`
(mutmut pair) · `.e2e/t_603a35a0-backend-attribution/` (this card's raw material).
