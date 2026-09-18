# E1c FIX — `--fit` on a busy desktop (card t_8cb0a05e)

Card `t_8cb0a05e` · branch `main` (this repo has no remote; commits are local on the shared tree)
· Tier **M** (the card declares none) · evidence schema `ggufone.evidence.e1c.fit.fix/v1`

Every claim below is a command plus a real output tail. Raw logs live in
`.e2e/t_8cb0a05e-fit-oom/`; the host-gate log directory of the rehearsal is quoted inline and its
folded JSON (`host_gate_e1c_fit.json`) is committed next to this file.

## 0. What the card asked for, and what landed

| # | requirement | implementation |
|---|---|---|
| 1 | plan against FREE device memory at load time; re-plan when a cached plan no longer fits | `recommend.device_memory()` (one `nvidia-smi --query-gpu=memory.total,memory.free` call, amdgpu-sysfs fallback) → `fit.HostFacts.vram_free_bytes` → `HostFacts.budget_bytes` uses it → `fit.replan_for_host()` re-validates every plan (cache hits included) and rewrites the cache |
| 2 | `--fit-target MiB` must bound the plan | `fit.fit_budget(host, fit_target_mb)` is now the single budget rule, threaded through `estimate_plan`, `plan_from_binary`, `run_llama_fit_params` (the `-ngl` it asks the tool about) and the re-plan |
| 3 | automatic degradation on allocation failure, `W_FIT_DOWNGRADE`/`W_BACKEND_OOM`, never a hard failure when CPU is possible | `session.capture_llama_logs` (real `llama_log_set` ABI) + `fit.classify_load_failure` + the ladder `fewer layers → smaller kv_type → CPU` at *model load* (`fit.degrade_ladder`, layer rungs) and at *context init* (`session._kv_ladder`, kv rungs); `engine.placement` names the outcome, `--no-fit` says "CPU only" out loud |
| 4 | allocation/OOM ≠ `E_MODEL_ARCH_UNSUPPORTED` | new `errors.BackendOomError` (`E_BACKEND_OOM`, exit 3) carrying plan + free/needed bytes + the backend's own line + `--no-fit`/`--fit-target` hints; a genuine arch failure keeps `E_MODEL_ARCH_UNSUPPORTED` (now also carrying its log tail) |
| 5 | close the host-gate gap (busy desktop / low free VRAM scenario + the free-VRAM reading in every fit-touching evidence) | `tools/host_gate_e1c_fit.sh` (7 steps, free VRAM before/after, fake-allocation-failure world) + `tools/host_gate_e1c_fit_summary.py` (folds the tails into boolean checks) + `tools/fixtures/fit_oom_bundle.c` (a real `.so` that fails like ggml_vulkan) + `tools/fixtures/fake_busy_nvidia_smi.sh` |

## 1. The three bugs found while landing this (all by running the real things)

1. **`llama_log_get` is not a no-argument getter at b11026.** It is a tail jump to
   `ggml_log_get(callback *, void **)` — two OUT parameters. The first version restored the previous
   handler with `llama_log_get()`; `uv run ggufone ask …` against the pinned CPU bundle died with
   `SIGSEGV` (exit 139) and `PYTHONFAULTHANDLER=1` pointed straight at the call:
   `File "src/ggufone/engine/session.py", line 79 in capture_llama_logs`. Fix: the capture resets
   with a NULL callback (`ggml_log_set(NULL, …)` is the documented reset) and every installed
   callback is kept referenced for the process (`ctypes_binding.register_log_callback`). The
   disassembly that proves the ABI: `objdump -d libggml-base.so` at `ggml_log_get` shows
   `mov %rax,(%rdi)` / `mov %rax,(%rsi)`; `llama_log_get` in `libllama.so` is `jmp ggml_log_get@plt`.
2. **An empty KV ladder.** `_kv_ladder("auto")` returned `[]`, so the context-init loop never ran
   and every default (kv_type `auto`) run died with the generic "the runtime refused these context
   parameters". Fix: an unknown/`auto` value starts at f16 — which is what
   `GGML_TYPE_IDS.get(kv_type, f16)` resolves it to anyway.
3. **`auto` was compared against the resolved rung.** `rung != plan.kv_type` (`f16 != auto`) warned
   `W_KV_TYPE_DOWNGRADE` + `W_FIT_DOWNGRADE` on *every* default run even though nothing was
   downgraded. Fix: compare against the first rung of the ladder; `engine.kv_type` keeps reporting
   what the request asked for (HEAD's contract, `test_the_plan_is_applied_on_load_unless_no_fit`)
   while `engine.placement.kv_type` reports the rung the context really used.

## 2. Evidence — the operator's numbers, before and after

`tools/fit_oom_red_probe.py` uses only APIs that exist on the pre-fix tree, so the same script
runs against both. Pre-fix tree = a byte copy of `git archive HEAD src` (HEAD `c095c51`).

```
$ PATH=<busy driver: 8192 MiB total / 1112 MiB free> python3 tools/fit_oom_red_probe.py \
      --model Spark-X2.5-4B-Q8_0.gguf --fake-runtime <bundle>
```

| world | pre-fix (`c095c51`) | fixed |
|---|---|---|
| host budget the plan is built from | `8589934592` (nominal 8 GiB) | `1166016512` (1112 MiB free) |
| `host_plan.budget_bytes` | `7516192768` (**7168 MiB = 90 % of nominal**) | `92274688` (**88 MiB = free − target**) |
| `host_plan.n_gpu_layers` | `36` (full offload → the 1.06 GB allocation) | `0` |
| `host_plan.warnings` | `["W_FIT_ESTIMATED"]` | `["W_KV_TYPE_DOWNGRADE", "W_FIT_DOWNGRADE", "W_FIT_ESTIMATED"]` |
| host_plan note | *(none)* | `planned against free device memory: 1112 MiB free of 8192 MiB, so 0 instead of 36 layer(s) are offloaded` |
| fake allocation failure | `raised: {code: E_RUNTIME_MISSING, message: "E_MODEL_ARCH_UNSUPPORTED: llama.cpp could not load … (arch spark2_5)"}` | `loaded: {n_gpu_layers: 0, placement: "degraded after a backend allocation failure: 0 layer(s) offloaded, kv_type=f16"}` |
| verdict / exit | `{busy_desktop_plan_is_bounded: false, oom_is_classified_as_backend_oom: false}` → **3** | `{…: true, …: true, fixed: true}` → **0** |

The pre-fix row reproduces the card's own tail digit for digit: `n_gpu_layers: 36`, `n_ctx: 4096`,
`kv_type: f16`, `est_total_bytes ≈ 5.29 GB`, `budget_bytes = 7 516 192 768` — and the same
`E_MODEL_ARCH_UNSUPPORTED` sentence for an allocation failure.

`--fit-target` sensitivity (same probe): on the fixed tree `fit --fit-target 5200` narrows the plan
(`budget_bytes` `92274688 → 0`, note `--fit-target 5200 MiB`); the *test* pin for the binary path
asserts the exact layer count at the boundary (`tests/test_fit_free_vram.py::test_the_binary_plan_is_bounded_by_the_fit_target`
→ `budget = 8 GiB − 5200 MiB`, `n_gpu_layers == 23`).

## 3. Host-gate rehearsal (sandbox vehicle; the host run is PENDING)

Command (the sandbox rehearsal of `tools/host_gate_e1c_fit.sh`; the GPU world is simulated by the
fake driver, the bundle compiles for real, the model and the runtime are the pinned ones):

```
GGUFONE_GATE_FAKE_DRIVER=<fakebin> GGUFONE_GATE_MODEL=<pinned model> \
GGUFONE_RUNTIME_DIR=<b11026-linux-x64-cpu> tools/host_gate_e1c_fit.sh <logdir>
```

| step | exit | tail |
|---|---|---|
| host facts | 0 | `free_vram_before: 8192, 1112 nvidia-smi` · `fake_driver: /work/e1cfit/fakebin` · `model: … (4375021152 bytes)` |
| pytest_before | 1 | `714 passed, 28 failed` — the 20 `test_bench.py` failures are the sibling E2 card's in-flight suite (untracked file, different card); the remaining 8 are environment/contention flakes that pass in isolation (see §5) |
| fit_plan | 0 | `{"n_gpu_layers": 0, "n_ctx": 4096, "kv_type": "q4_0", "budget_bytes": 92274688, "source": "llama-fit-params"}` · `"host": {"vram_bytes": 8589934592, "vram_free_bytes": 1166016512, "vram_source": "nvidia-smi"}` · note `device memory bound: offloading 0/36 layers within 88 MiB (--fit-target 1024 MiB, 1112 MiB free of 8192 MiB)` |
| fit_target_bounded | 0 | same call with `--fit-target 5200` → `"budget_bytes": 0` · note `… within 0 MiB (--fit-target 5200 MiB, 1112 MiB free of 8192 MiB)` — the override now changes the plan |
| run_busy_desktop | 0 | a real engine run (pinned model, pinned bundle) on the busy-desktop plan: `engine.n_gpu_layers: 0`, `engine.kv_type: q4_0 == engine.fit.kv_type`, `engine.placement.note: "CPU only: the fit plan offloads nothing (kv_type=q4_0)"`, `engine.fit.budget_bytes: 92274688`, `prefill_ms 99498`, `questions_ms 137308` (**no OOM, no arch error**) |
| run_no_fit | 0 | `engine.placement.note: "fit disabled: CPU only (--no-fit sets n_gpu_layers=0)"` |
| fake_oom_degrade | 0 | `result.n_gpu_layers: 0`, `attempts: ["n_gpu_layers=36 -> oom", "n_gpu_layers=18 -> oom"]`, `warnings: ["W_BACKEND_OOM", "W_FIT_DOWNGRADE"]`, plan note `CPU only: no weights are offloaded to the device` |
| fake_oom_all_rungs | 0 | `error.code: E_BACKEND_OOM` — message carries `needed ~1010 MiB`, `1112 MiB free`, `tried 3 placement(s) down to CPU-only`, the backend line `Device memory allocation of size 1058982400 failed.` and both hints |
| free_vram_after | — | `8192, 1112 nvidia-smi` (the gate reads the driver again after the runs) |

**PENDING (operator / host run).** The three GPU-dependent readings cannot be produced in this
container (no `/dev/dri`, no `/dev/nvidia*`, no driver): the real `llama-fit-params` Vulkan row, the
real Vulkan allocation failure, and `nvidia-smi` on the actual busy desktop. Everything else above
is executed. The host run is the same script without `GGUFONE_GATE_FAKE_DRIVER`:

```
tools/host_gate_e1c_fit.sh                 # on the RTX 3060 Ti box, desktop loaded as usual
```

and its `host_gate_e1c_fit.json` will carry `is_host_run: true` (the summary derives that from the
absence of a fake driver, so a rehearsal can never masquerade as a host run).

## 4. Gates, suite, lint

- `uv run pytest -q tests/test_fit_free_vram.py tests/test_fit_oom_recovery.py` → **40 passed**
  (the new offline gates: free-VRAM probing/parsing, cache re-planning, the target bound, the
  classification table, the loader ladder, the placement surface, the `--no-fit` note).
- `uv run pytest -q tests/test_fit.py tests/test_e1c_mutation_pins.py tests/test_engine_fork.py
  tests/test_cli.py tests/test_cli_e1c.py tests/test_templates.py` → green (the E1c pins keep
  holding **unchanged**: no pin had to be weakened or rewritten for this fix, and the bounded case
  is pinned by a new test in `tests/test_fit_free_vram.py`).
- Live (`--run-network`, pinned bundle + model, CPU world):
  `tests/test_fit_live.py::test_the_log_capture_swaps_and_restores_the_real_handler` and
  `::test_a_busy_desktop_plan_loads_on_the_free_reading` pass — the latter loads the real model with
  a plan built from the operator's numbers.
- `uv run ruff check <changed files>` → `All checks passed!`
- Coverage (changed modules, E1c selection — `uv run --extra dev --with pytest-cov pytest -q
  --cov=…`, 214 passed / 8 skipped, raw output in `.e2e/t_8cb0a05e-fit-oom/logs/coverage.txt`):
  **`fit.py` 96 %** (the file the card is about, 16 missed statements), `decide.py` 95 %,
  `errors.py` 100 %, `session.py` 69 % of the whole file (the uncovered part is E1b's state
  save/load and the live-only paths), `recommend.py` 52 % / `ctypes_binding.py` 36 % of the whole
  files (E1a/E1b/RPC/`llm_*` surfaces outside this card's selection). Total over the six: 79 %.
- Mutation (Tier M, one run, soft threshold): see §6.

## 5. Suite failures at the final head (honest list)

`uv run pytest -q` at the final head: **714 passed, 28 failed, 34 skipped**. None of the 28 is in
this card's surface:

- 20 × `tests/test_bench.py` (+1 `test_cli_e1a.py::test_frozen_commands_still_exit_3[bench]`):
  the sibling E2 card (`t_858c54d1`) is writing the benchmark suite into this shared worktree right
  now; its test file is untracked and its `cli.py` hunks are not part of this card's commit.
- 7 × `test_probe_isolation` / `test_runtime_fallback` / `test_runtime_install` /
  `test_runtime_contract`: they pass individually and in their own file groups; in the full run
  they depend on a runtime bundle and a model under `$HOME`, which the container's other cards are
  rewriting concurrently. `test_runtime_contract::test_oracle_live_section_is_green_without_skips`
  additionally needs `$HOME/.hermes/models/Spark-X2.5-4B-Q8_0.gguf`, and the documented way to run
  the suite live is with the operator home (`HOME=/var/home/rybens`), not the scratch home used for
  the mutation run.
- One self-inflicted false alarm is worth recording because it is the skill's own pitfall: an
  earlier `pytest -q tests/test_fit.py` failed because this shell had exported
  `PATH=<fake busy driver>:$PATH` for the gate rehearsal; the fake driver answered
  `memory.free` for a test that assumes a GPU-less box. Re-run with a clean `PATH`: **32 passed**.
  Every claimed world in this file states its environment for exactly that reason.

## 6. Mutation run (Tier M, single owner, soft threshold)

Config (temporary, `[tool.mutmut]`): `source_paths = [src/ggufone/runtime/fit.py]`, test selection =
`test_fit.py test_fit_free_vram.py test_fit_oom_recovery.py` (the fast offline set — see the note
below on why the scope shrank), `max_children = 2`. Raw output:
`.e2e/t_8cb0a05e-fit-oom/logs/mutmut.out`. Run in a byte-copied scratch tree under `/tmp` (mutmut's
`copytree` dies `EACCES` on the shared checkout's mixed SELinux labels — the same trap the E1a
cards recorded). **Scores are read from the per-module `.meta` artifacts**
(`exit_code_by_key`), never from the tool's last screen; the status table is mutmut 3.8's own.

**`fit.py`: 1639 mutants — 1136 killed, 483 survived, 20 no-tests, 0 timeout → 69.3 % killed/ran**
(artifact: `mutants/src/ggufone/runtime/fit.py.meta`). Survivor clusters, as function-level kill
rates — which is the honest shape of the result, not a single number:

| killed | mutants | function | reading |
|---|---|---|---|
| 31 % | 16 | `_measured_rss_bytes` | it only reads `resource.getrusage`; nothing asserts the value |
| 35 % | 78 | `replan_for_host` | **this card's new cache-replan logic — the weakest spot of the new code**; the tests cover the cache-is-stale decision, not every arithmetic detail of the rewrite |
| 38 % | 55 | `backend_oom_error` | message construction: the tests assert substrings (`--no-fit`, the MiB numbers), so wording mutations survive |
| 43 % | 7 | `oom_log_line` | same: substring assertions on the parsed backend line |
| 50 % | 56 | `to_dict` | the JSON surface (hidden behind every plan field the tests read through) |
| 61 % | 123 | `read` | the GGUF reader inherited from E1b, only lightly touched here |
| 62 % | 21 | `plan_device_bytes` / `_gpu_layers` / `_kv_from_budget` | the boundary arithmetic *is* pinned (`test_the_binary_plan_is_bounded_by_the_fit_target` asserts 23 layers at the exact budget); the survivors are the non-boundary branches |

**`session.py`: attempted, not scored — recorded as a gap.** The first sweep covered
`session.py + fit.py` and was killed by this container's shared pid cap (`BlockingIOError: ...
Resource temporarily unavailable` in `os.fork()`, at 492/2583 — two sibling cards were running
node/pytest work in the same cgroup; the earlier E1a cards hit the same wall). Its partial artifact
(`.../engine/session.py.meta`, 944 mutants) is the evidence for why a re-run would not help: **400
of its mutants have no test at all** and the survivors sit in `open_model` / `_load_state` /
`prefill` / `decode` — the paths that need a live bundle, which a mutation sweep cannot use. A
score computed from the offline selection would measure the selection, not the code, so the number
is deliberately NOT folded into the one above. The load-ladder behaviour is instead pinned by the
22 offline `test_fit_oom_recovery.py` cases plus the two live tests in §4.

Tier M is a soft threshold: the run is recorded, the survivor clusters are handed to the reviewer
as findings, and no re-run loop was entered.

## 7. Files touched by this card

| area | files |
|---|---|
| free-VRAM probing | `src/ggufone/registry/recommend.py` (`DeviceMemory`, `device_memory`, the two driver readers) |
| planning + re-planning + classification + the ladder | `src/ggufone/runtime/fit.py` (`fit_budget`, `plan_device_bytes`, `degrade_ladder`, `replan_for_host`, `classify_load_failure`, `allocation_bytes_from_log`, `backend_oom_error`, `plan_from_binary`/`run_llama_fit_params` target threading) |
| loader | `src/ggufone/engine/session.py` (`capture_llama_logs`, the OOM ladder, `Placement`, `_kv_ladder`, the context-init kv ladder) |
| surface | `src/ggufone/engine/decide.py` (`engine.placement`, placement warnings), `src/ggufone/cli.py` (`fit --json` host block, `fit_disabled`, the effective plan in the response) |
| codes + ABI | `src/ggufone/errors.py` (`E_BACKEND_OOM`, `W_FIT_DOWNGRADE`, `W_BACKEND_OOM`), `src/ggufone/runtime/ctypes_binding.py` (`LLAMA_LOG_CALLBACK`, `_bind_optional`, the callback keep-alive) |
| gates | `tests/test_fit_free_vram.py`, `tests/test_fit_oom_recovery.py`, `tests/test_fit_live.py` (two live tests); the E1c pin `test_the_binary_plan_flags_a_downgrade_when_the_ladder_moved` kept passing **unmodified** |
| tools | `tools/host_gate_e1c_fit.sh`, `tools/host_gate_e1c_fit_summary.py`, `tools/fit_oom_probe.py`, `tools/fit_oom_red_probe.py`, `tools/fixtures/fit_oom_bundle.c`, `tools/fixtures/fake_busy_nvidia_smi.sh` |
