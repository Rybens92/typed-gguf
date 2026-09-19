# A `cpu` bench row must compute on the CPU — on a GPU box too (card `t_55de5779`)

Filed by the coordinator running the live determinism gate on the operator host right after the
parity fix landed (`00265ea`). One sentence: **`--backend cpu` pinned the bundle, not the compute
path.** `n_gpu_layers=0` keeps the *weights* on the host, but llama.cpp's op offload still ran the
graph on the Vulkan device the same bundle registers, so the row said `cpu` while `Vulkan0=3 ·
Vulkan_Host=3` compute buffers were built — and the attribution guard (card `t_603a35a0`), doing
exactly its job, refused to certify it: `W_BACKEND_MISMATCH`, live gate RED. This document is the
route decision, the fix, both gates (RED → GREEN), the sweep of the sibling configs, and the Tier-M
sweep with its hand table.

**Scope of every number below**: operator host = this container (`cpu.max` 2 CPU-s/s, `memory.max`
8 GiB, GPU passed through — `ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 Ti`), Vulkan bundle
`b11026-linux-x64-vulkan`, model `Qwen3.5-0.8B-UD-Q4_K_XL.gguf`. The card's exact command, with one
delta named where it appears: the container's `HOME` has no `~/.cache/llama.cpp`, so
`GGUFONE_BENCH_MODEL` names the same model the host's `_model()` picks first.

## 1. The defect, reproduced (RED)

```
GGUFONE_RUNTIME_DIR=~/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \
  uv run pytest -q --run-network tests/test_bench_live.py::test_determinism_holds_on_this_box
```

`1 failed in 53.97s` on `00265ea` (raw: `cpu_force/red_live_determinism.txt`), the gate's own message
verbatim:

```
E       AssertionError: ["W_BACKEND_MISMATCH: the row claims backend `cpu` but the engine's own log
        shows the compute on vulkan (compute buffers: Vulkan0=3 · Vulkan_Host=3); read this row as a
        vulkan measurement — re-run one backend per process (`--backend vulkan`) for a clean
        attribution."]
E       assert False is True
```

The same run's row, printed as JSON (`row_red.json`) — the claim, and the engine's own answer:

```json
{"backend": "cpu", "placement": "n_gpu_layers=0", "effective_backend": "vulkan",
 "devices": ["CPU", "CPU_Mapped", "Vulkan0", "Vulkan_Host"],
 "device_buffers": {"Vulkan0": 3, "Vulkan_Host": 3},
 "warnings": ["W_BACKEND_MISMATCH"], "identical": true,
 "digests": ["sha256:ae9260ad…", "sha256:ae9260ad…", "sha256:ae9260ad…"]}
```

Why this box hits it and CI never did: `harness.backend_runtimes` registers **every** bundle under
its accelerator *and* under `cpu` (a completed llama.cpp bundle always carries `libggml-cpu.so`,
docstring in `harness.py`), so a Vulkan-only box still answers `--backend cpu`; `DEFAULT_BACKENDS`
puts `cpu` first, so `--backend auto` (and the single-backend suites) measure the cpu row. On a
GPU-less box the same row really is CPU compute and the labels agree — the test is `@pytest.mark.model`
(live-only), which is why the defect shipped.

## 2. The route — decided here, because the card asks for it

Two routes were possible. **A: install a CPU-only bundle** and point the test at it.
**B: make the row's *load* CPU-only** — offer the loader the bundle's CPU device and nothing else.

**Taken: B.** Why:

* No CPU-only bundle exists on this box — the only installed runtime is the Vulkan build — and
  route A means a second download, a second bundle to keep in sync, and a test that only passes
  where that bundle was installed. The *label* would still be a property of the box, not of the row.
* The multi-backend bundle **is** the normal case: `load_backend: loaded Vulkan backend …` *and*
  `loaded CPU backend …` in the same process (both lines in `green_live_determinism.txt`). A
  `cpu` row has to mean "this load computes on the host" for *any* bundle.
* B makes the guarantee hold on a GPU-less box and a GPU box alike (no test-side special-casing),
  and it is one engine parameter — `open_model(cpu_only=…)` — with a typed refusal when the bundle
  cannot name its CPU device.
* The mechanism is measured, not assumed: `llama_model_params.devices` is the one lever that takes
  a device out of the **compute** path. Probes on this box: a `cpu` load with `op_offload=False`
  still logs `Vulkan_Host compute buffer size`; a load whose `devices` names the CPU device logs
  `CPU compute buffer size` only. The guard stays untouched (card requirement 1) — it is what
  caught this.

## 3. The fix

| where | what |
|---|---|
| `runtime/ctypes_binding.cpu_device(runtime)` | `ggml_backend_dev_by_name("CPU")` → the device handle, `restype = c_void_p` (**not** ctypes' default `c_int`: a 64-bit handle read as an int arrives truncated and the next call segfaults — measured). `None` = "this bundle cannot name its CPU device". |
| `engine/session._load_model(..., device=…)` | the NULL-terminated `llama_model_params.devices` list (`(c_void_p * 2)(device, None)`), kept alive across the call; `None` leaves llama.cpp's own list untouched. |
| `engine/session.open_model(cpu_only=…)` | resolves the device **before** the load; refuses a bundle that cannot name one (`E_RUNTIME_SYMBOLS`, and *no* unpinned load is attempted); asks for **zero** offload layers whatever the plan requested; skips the degradation ladder (every rung would be the same CPU-only load); records the pin on the `Placement` (`cpu_only`, and the note says it — naming the request it overrode when there was one). |
| `bench/harness.spec_for` | `cpu_only=backend == CPU_BACKEND` — the pin is derived from the row's own label, so every suite gets it. `ModelSpec.cpu_only` carries it; `LiveModel.load()` passes it to the loader. |
| `bench/harness.placement_request` / `render_report` | a pinned row prints `n_gpu_layers=N (cpu compute pinned)`, the rendered placement line ends `(cpu compute pinned)`; `placement.used` carries `cpu_only`. The *request* is untouched (`--gpu-layers` still states what was asked; the executed plan reports zero). |

## 4. The gates (RED → GREEN)

### 4.1 Offline structural gate — `tests/test_bench_cpu_force.py`

A fake runtime object behind a real bundle directory (`tests/test_fit_oom_recovery`'s rig shape):
production's dlopen path, the params call, the arch pre-flight and the placement code all run; only
libllama is a Python object. Pins: the spec's pin, the load seam, the device list and zero layers,
the unpinned row's untouched `NULL` list, the typed refusal (missing symbol *and* `NULL` answer), no
ladder walk, and the note/row/report text.

```
pre-fix tree (the rebased base):  9 failed, 1 passed   (red_offline_pin_pre_fix.txt)
landed tree:                     11 passed            (green_offline_pin.txt)
```

### 4.2 Live gate — the card's command, on the operator's tree

`1 passed in 32.40s` on the landed tree (`green_live_determinism.txt`), with the row printed by the
gate itself:

```
cpu determinism row: placement 'n_gpu_layers=0 (cpu compute pinned)' · effective backend 'cpu' ·
compute buffers {'CPU': 3} · 1 digest(s)
```

The post-fix row, verbatim (`row_green.json`, `ok: true`, `notes: []`):

```json
{"backend": "cpu", "placement": "n_gpu_layers=0 (cpu compute pinned)", "effective_backend": "cpu",
 "devices": ["CPU", "CPU_Mapped", "CPU_REPACK"], "device_buffers": {"CPU": 3},
 "warnings": [], "identical": true,
 "digests": ["sha256:3ce89618…", "sha256:3ce89618…", "sha256:3ce89618…"]}
```

The Vulkan device is still **there** (`row_green.engine_log.txt`): `ggml_vulkan: Found 1 Vulkan
devices`, `ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 Ti`, `loaded Vulkan backend …` — and no
`Vulkan0`/`Vulkan_Host` compute buffer is ever built. That is the whole claim: the row no longer
depends on the device being absent, only on the load being pinned.

### 4.3 The live gate now asserts the compute path (not just `ok`)

`tests/test_bench_live.py::test_determinism_holds_on_this_box` gained three assertions, each one a
field the pre-fix row already disagreed with (`row_red.json`): `effective_backend == "cpu"`,
`warnings == []`, and the placement string ends `(cpu compute pinned)`. A GPU-less box reports the
same values, so the strengthened gate stays a gate in CI.

## 5. The sweep (card requirement 5) — every backend-labelled path, fixed or declared

| where | what it is | verdict |
|---|---|---|
| `suites._spec` → `harness.spec_for` (`suites.py` 221/526/616/779) | the **single funnel** for every row of every suite (latency, throughput, quality, calibration, determinism) | **fixed** — `spec_for` derives the pin from the label, so all of them are pinned now |
| `--backend auto` / `--quick`'s one-backend preset (`harness.DEFAULT_BACKENDS`, `config.backend_limit`) | `auto` → the first local bundle = `cpu` on a box with an accelerator bundle; the report already names the selection (`backend_selection`, card `t_31b3943a`) | **now true** — the cpu row it selects is CPU compute; nothing to change, the label matches |
| `--backend vulkan` rows | `n_gpu_layers=-1` (or the flag) | **untouched**, still attributed per row by the guard |
| the guard (`device_usage_of`, `_mismatch_note`, `W_BACKEND_MISMATCH`, `effective_backend`) | the attribution rule | **untouched** — the card says keep it, and it is what found this |
| `bench/isolation.py` retry children | a `backend="cpu"` child is built by `spec_for` in the child process, and on success the parent publishes the **child's own row** (pin included) | **declared** — no change needed; the parent's retry metadata describes the retry, not the row |
| `tools/host_gate_e1c_fit.sh` step 4b (`bench --suite latency --gpu-layers 36`, no `--backend`) | "the E2 bench path through the same ladder … where a busy desktop really walks the ladder" | **declared, flagged** — post-fix it selects the pinned cpu row (zero offload, no walk). Out of this card's scope (`tools/`); the step wants `--backend vulkan` to keep measuring what it documents. For the coordinator to route. |
| `docs/BENCHMARKS.md` §3.2/§3.3's cpu rows | `[container]` rows from a box with no accelerator bundle | **unaffected** — no accelerator registered → genuine CPU compute before and after |
| `docs/BENCHMARKS.md` §8's E3d tables + prose | `[host]` rows that carry `effective_backend: vulkan` and `W_BACKEND_MISMATCH` while the request claims `cpu` — labeled **pre-fix** by that section, and its prose explains the mismatch | **declared** — the numbers stand for what they measure; a post-fix re-run of those 60-item tables (now a genuine cpu measurement) is a coordinator call, not a silent rewrite |

## 6. The Tier-M sweep (mutation testing)

`[tool.mutmut]`'s pair was retargeted at the module the fix's rule lives in —
`src/ggufone/engine/session.py` — with the card's own gate file plus the three fast bench/loader
gates that build real placement dicts (`test_bench_placement.py`, `test_fit_oom_recovery.py`,
`test_bench_attribution.py`), and swept by `.e2e/t_55de5779-cpu-force/mutmut_sweep_cpu_force.sh`
(mutmut 3.8, `tools/mutmut_driver.py`, `--max-children 2`, one attempt, `exit 0`, `pending 0`).
**The `[tool.mutmut]` block in `pyproject.toml` is left as the E3e card's uncommitted retarget
holds it** (a live sibling owns that file — the documented hotspot); the pair this sweep used is
recorded above and in `mutmut_cpu_force_summary.txt`.

Whole file: 1124 mutants → **389 killed · 290 survived · 441 no tests · 4 timeout**. The 441 are
`no tests` because mutmut 3.8 runs each mutant against the tests that cover its *function*, and this
selection never reaches session.py's serving internals (`_save_state` 111, `prefill` 76, `decode`
76, …) — the selection's boundary, quoted rather than folded in. The changed surface
(`open_model` 68.8 %, `_load_model` 78.3 %, `_placement_note` 63.2 %, `ModelHandle.__init__` 40.0 %,
`Placement.to_dict` 100 %) is **67.2 % killed / 412 mutants / 0 not run**. Tier M is a soft
threshold: the score is reported; the survivors are named by class (equivalent-on-the-fixture,
parameter defaults no caller leaves to the default, note/row wording, and pre-existing branches
outside the pin's path) in `mutmut_cpu_force_summary.txt`, and none of them sits on a behavioural
claim of the card.

Two facts make the sweep honest on this box: the gate strengthening between the first and the second
sweep is visible (`_placement_note` 52.6 % → 63.2 %; the two mutants that survived the first run are
killed by the added assertions), and a **hand table** over the card's own lines
(`hand_mutations.txt`) applies six behavioural mutations alone — each killed by the pin gates — plus
two controls that must survive (a docstring change, an equivalent rewrite), restoring every file
byte-identically with a printed `sha256`. mutmut's verdicts are cross-checked, not trusted.

## 7. What is not claimed

* **The published `[host]` quality tables are not re-measured here.** §5 declares them; a post-fix
  re-run changes their *compute path* (and would be a genuine cpu row), so re-publishing them is a
  separate call.
* **`tools/host_gate_e1c_fit.sh` is not edited** — the step that changes meaning is named in §5 so
  the operator is not surprised by a silent change of what step 4b measures.
* **The pin is per load, not per process.** A process that loads twice — once pinned, once not —
  gets llama.cpp's own device list the second time (`test_the_refusal_does_not_touch_unpinned_loads`
  pins exactly that). The bench loads one model per row, so the row is what is pinned.
* **`n_gpu_layers` is still what the flags asked.** A pinned row's *request* can read
  `n_gpu_layers=36` while the executed plan is zero; the placement block says both, and the rendered
  line names the pin. A reader who wants the request alone reads `placement.requested`.
* **Mutation is Tier M**: one sweep, soft threshold, the not-run/no-test boundary quoted — not a
  hardening pass.

## 8. Receipts

* gates: `docs/evidence/cpu_force/{red,green}_live_determinism*`, `row_red.json`, `row_green.json`,
  `row_green.engine_log.txt`, `red_offline_pin_pre_fix.txt`, `green_offline_pin.txt`
* suite / coverage / lint: `cpu_force/full_offline_suite.txt` (**1315 passed, 47 skipped**, exit 0,
  the clean tree = the landed commits), `cpu_force/coverage_added_lines.txt` (**33/33 added lines =
  100 %**; whole-file under the gate selection: `harness.py` 91.9 %, `suites.py` 96.2 %,
  `session.py` 72.5 %, `ctypes_binding.py` 51.0 %), `cpu_force/ruff.txt` (`ruff check src tests
  tools docs`; every file this card touches is clean — the 13 remaining errors are in sibling-owned
  or pre-existing lines, named in the QA note). The same run **in the shared tree** (which carries a
  live sibling's uncommitted `docs/BENCHMARKS.md` + `tests/test_e3e_docs.py`) reads
  `1319 passed, 47 skipped, 1 failed`; the one failure is that sibling's own in-flight E3e doc
  section — the file does not exist in the clean tree, and every suite file this card touches is
  green in both runs.
* the Tier-M sweep: `.e2e/t_55de5779-cpu-force/mutmut_sweep_cpu_force.sh` (driver, append-logging),
  `cpu_force/mutmut_cpu_force_summary.txt` (scope, score, survivor classes),
  `cpu_force/mutmut_cpu_force_report.txt`, `cpu_force/mutmut_survivors_sample.txt`,
  `cpu_force/hand_mutations.txt`
* probes: `.e2e/t_55de5779-cpu-force/{row_probe.py,hand_mutations.py,cov_added.py,sweep_report.py,
  pick_mutants.py,show_sample.sh}`, logs in `.e2e/t_55de5779-cpu-force/logs/`
* the code: commits `4797146` (RED), `b41884f` (GREEN), `1cbf948` (live gate), `430621a` (gate second
  pass + lint), `089323c` (this evidence) — landed by fast-forward on `adcb7de`
* the QA note: `.gauntlet/e2-fix-cpu-force.qa.md`
