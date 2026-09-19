bash: fork: retry: Resource temporarily unavailable
# E3 FIX — the serving path's `engine.backend` said `cpu` while the Vulkan device computed
(card `t_80f1a4c6`)

Branch `main` (no remote; local commits on the shared tree) · Tier **M** (the card declares none) ·
found by E3 (`t_a431be85`, evidence §3 of `docs/evidence/e3_t_a431be85_occamy.md`)

Box: the operator host and its container (cgroup `cpu.max` **2 CPU-s/s**, `memory.max` **8 GiB**,
`pids.max` **256**, shared with sibling workers) with its GPU passed through — `/dev/nvidia0`,
`/dev/dri/renderD128`; the Vulkan ICD is not on the container's default loader path, so every GPU
run sets `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` (without it llama.cpp reports
`No devices found`, see the E3 evidence §1.1). Bundle: the pinned
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan`. Models: Occamy 1.0
(`Accio-Lab_occamy-1.0-Q4_K_L.gguf`, 23 GB, the card's model) and the smallest local GGUF
(`Qwen3.5-4B-Q4_0.gguf`, 2.5 GB) for the cheap rows.

Commits (this card — the private clone was rebased twice onto the shared tree's moving `main`
(`15e89a9` → `a475090` → `f9d9a08`) and then landed there as a fast-forward, so these hashes are the
shared tree's):

| commit | what |
|---|---|
| `8f32620` | `test`: the RED gates (`tests/test_serving_attribution.py` — 16 offline + 1 live at that point) |
| `dfac21d` | `fix`: `session.BackendClaim`/`backend_claim`/`device_log`, `decide.device_evidence`, the response fields, `finder.backend_of_bundle`, the CLI claim |
| `00ecc46` | `chore`: the Tier-M mutmut pair retargeted at the two engine modules (card convention) |
| `3c9ff1a` | `docs`: README + the `W_BACKEND_MISMATCH` comment |
| `2812f34` | `test`: the unknown-platform gate, and drop the unused `BackendClaim.to_dict` |
| `a644ed0` | `test`: two survivor pins from the first sweep pass (log-as-lines, log-less session) + the replay tools |
| `4f30482` | `test`: two survivor pins from the second pass (explicit-backend source, caller-owned sink) |
| `f1b9272` | `evidence`: this document, the raw material under `.e2e/t_80f1a4c6-serving-backend/`, and the pyproject sweep-comment count |

Raw material for every claim below: `.e2e/t_80f1a4c6-serving-backend/` (index in its `README.md`).

## 0. What was wrong

`docs/evidence/e3_batch.json` (one `ggufone run`, 20 dev-set questions, `--backend vulkan
--threads 4 --n-seq-max 4`, Occamy 1.0, the Vulkan bundle) answered:

```json
"engine": {"runtime": "llama.cpp b11026", "backend": "cpu", ...,
           "n_gpu_layers": 3, "placement": {"note": "fit plan: 3 layer(s) offloaded, ..."},
           "fit": {"backend": "cpu", "source": "llama-fit-params", ...}}
```

while the *same process'* stderr carried

```
ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 Ti (NVIDIA) | … | matrix cores: NV_coopmat2
~llama_context:    Vulkan0 compute buffer size is 363.5412 MiB, matches expectation of 363.5412 MiB
```

Three separate statements were conflated in one field:

1. `engine.backend` came from `session.runtime_backend()` — the **install record's**
   `backend_working`/`backend_requested`, silently `cpu` when no record is visible (the E3 run
   had a scratch `$HOME`, so `runtime.json` was not read at all). Nothing checked it against the
   engine;
2. the serving path (`run`/`ask`/`calibrate` → `cli.decide_payload` → `DecisionEngine`) never
   read the engine's own log, while the **bench** path had been fixed for exactly this by card
   `t_603a35a0` (`harness.device_usage`: `devices`, `device_buffers`, `effective_backend`,
   `W_BACKEND_MISMATCH`);
3. the request's own `--backend vulkan` was validated by `schema` and then *read by nothing* — the
   flag could not influence any published field.

## 1. Requirement 1 — the serving path records the engine's own evidence

`engine/session.py`:

* `ModelHandle.load_log` keeps the lines of the load that **succeeded** (the `llama_log_set`
  capture `open_model` already took, failed ladder rungs excluded);
* `ModelSession.device_log` = those lines + the context's own (the `sched_reserve` /
  `~llama_context` compute buffers) — the same capture the bench's `LiveModel` owns, now owned by
  the session itself, so no caller has to wire a sink.

`engine/decide.py` turns that into the bench's shape on every response
(`device_evidence(session, claimed)`, the parser is `runtime/devices.py` — untouched, so both
paths read the same rule):

```json
"engine": {
  "backend": "vulkan",                  // the claim (see §2)
  "backend_source": "bundle",           // where the claim came from (request|bundle|record|default)
  "devices": ["CPU", "CPU_Mapped", "Vulkan0", "Vulkan_Host"],
  "device_buffers": {"Vulkan0": 1, "Vulkan_Host": 1},   // *compute* buffers per device
  "effective_backend": "vulkan",        // from the compute buffers; null = unverified
  ...
}
```

The fit plan's own `backend` field still exists inside `engine.fit` (it is that artifact's
provenance, see §5) but nothing in the attribution reads it.

## 2. Requirement 2 — a claim, its source, and a named warning

`session.BackendClaim` / `session.backend_claim(...)`, most specific first:

| source | what it means |
|---|---|
| `request` | the request's own `--backend` (`auto` = "ask the box") |
| `bundle` | the backend the **bundle this run loaded** carries (`finder.backend_of_bundle`; the bench's `harness.classify_runtime` now delegates to the same rule instead of keeping a second copy) |
| `record` | the install record's **proved** backend (`recorded_backend`: `backend_working` only — the E3 record's `backend_requested: cuda` names a variant that was skipped before its download, and claiming it would be the same class of lie) |
| `default` | nothing named a backend: `cpu`, reported **as a default** (`backend_source`), not as a measurement |

`decide.device_evidence` raises `W_BACKEND_MISMATCH` when `runtime/devices.contradicts` refutes the
claim (a `cpu` label with an accelerator in the compute buffers — op offload) or cannot
corroborate it (an accelerator claim with no compute-buffer line at all). A silent log with a
`cpu` claim is *unverified*: `effective_backend: null`, no warning (nothing computed anywhere
else). The response is still delivered — unlike a bench row, a served answer is not withheld —
but the label can no longer be published silently.

## 3. Requirement 3 — gates

`tests/test_serving_attribution.py` — 20 offline gates + 1 `model`-marked live gate, **RED first**
(20 failed, 1 skipped on the parent tree `f9d9a08` with the new file copied into a detached
worktree, `.e2e/t_80f1a4c6-serving-backend/red_gates_landed.txt`):

| gate | pins |
|---|---|
| `..._carries_the_device_evidence_the_engine_logged` | the E3 log's own lines → `devices`, `device_buffers`, `effective_backend` |
| `..._a_cpu_only_box_reads_as_cpu` | the honest CPU direction |
| `..._the_request_side_offload_line_is_never_evidence` | `offloaded N/M layers to GPU` + CPU compute buffers is a CPU row (t_603a35a0's rule, serving side) |
| `..._the_device_evidence_is_not_read_from_the_fit_plan_or_the_record` | the fit plan says `cpu`, the log says Vulkan → the response follows the log |
| `..._a_claim_the_log_refutes_is_a_named_warning` | the E3 shape: `backend: cpu` + Vulkan buffers → `W_BACKEND_MISMATCH`, claim not silently rewritten |
| `..._an_accelerator_claim_the_log_cannot_corroborate_is_flagged` | `vulkan` label + silent log → flagged, `effective_backend: null` |
| `..._a_silent_log_reads_as_unverified_never_as_a_claim` | silent + `cpu` claim → `null` + empty device set + source named, no warning |
| `..._a_verified_vulkan_run_is_not_flagged` | no false positive |
| `..._claim_is_what_the_run_asked_for` | source `request` |
| `..._claim_falls_back_to_the_bundle_that_loaded` | source `bundle` (accelerator and CPU bundle), and a non-bundle directory cannot name one |
| `..._claim_falls_back_to_the_install_record_then_to_cpu` | sources `record` then `default` |
| `..._a_record_without_a_working_backend_is_not_a_claim` | `backend_requested: cuda` with `backend_working: null` never becomes a claim |
| `..._the_bundle_classifier_refuses_a_platform_it_has_no_rule_for` | the classifier's `E_RUNTIME_MISSING`, same rule as `library_names` |
| `..._live_session_records_the_load_and_the_context_lines` | the live sink: `ModelHandle.load_log` + the context's lines, over the real `llama_log_set` ABI (`test_fit_oom_recovery`'s fake runtime) |
| `..._live_session_names_the_bundle_it_loaded` | a session with no flag claims the bundle it loaded |
| `..._serving_payload_hands_the_claim_and_the_log_to_the_session` | the CLI glue: `decide_payload` hands the request's claim to the session and publishes the evidence |
| `..._live_serving_run_reports_the_bundle_backend_it_computed_on` | `model`-marked: one live row on the pinned bundle (`-k live`: 3 passed, `live_gate_rebased.txt`) |
| `..._a_log_kept_as_lines_reads_like_the_joined_text` | a `device_log` that is a *sequence of lines* (the loader's `load_log` shape) reads like the joined text — the branch sweep mutants 16/17 sit on (added in `a644ed0`) |
| `..._a_session_without_a_log_reads_as_unverified` | a session object with no `device_log` at all is an empty log (unverified), not an `AttributeError` — sweep mutant 8 (added in `a644ed0`) |
| `..._an_explicit_backend_with_no_source_is_named_explicit` | `backend=` without `backend_source=`: the claim is not resolved even though the loaded bundle carries Vulkan — `backend_source: "explicit"` (sweep mutants 7/18/20/21, added in `4f30482`) |
| `..._a_caller_owned_log_sink_receives_the_context_lines` | a `log=[...]` handed to the session receives the context's compute-buffer line (sweep mutant 47, added in `4f30482`) |

## 4. Requirement 4 — Occamy 1.0, before/after

Both sides are the **same model, the same pinned bundle, the same question** (`c01` of the dev set,
state `At 09:12 the checkout page started returning HTTP 500 …`); BEFORE is the published E3 batch
response (`docs/evidence/e3_batch.json`, commit `15e89a9`), AFTER is this card's re-run
(`-questions` one item, `--state` the batch state, `--threads 4 --n-seq-max 4 --backend vulkan`,
plus `--n-ctx 256`/`--fit-target 2000`/`--threads 1` — the container was shared, see §7). Generated
by `.e2e/t_80f1a4c6-serving-backend/before_after.py`, which reads the two response JSONs and the
two stderr captures:

```
field              | BEFORE (E3 batch, 15e89a9)               | AFTER (this card)
-------------------+------------------------------------------+-----------------------------------------
backend            | "cpu"                                    | "vulkan"
backend_source     | "<absent>"                               | "request"
devices            | "<absent>"                               | ["CPU", "CPU_Mapped", "Vulkan0", "Vulkan_Host"]
device_buffers     | "<absent>"                               | {"Vulkan0": 1, "Vulkan_Host": 1}
effective_backend  | "<absent>"                               | "vulkan"
n_gpu_layers       | 3                                        | 0

before stderr:
  ~llama_context:    Vulkan0 compute buffer size is 363.5412 MiB, matches expectation of 363.5412 MiB
  ~llama_context: Vulkan_Host compute buffer size is  17.4555 MiB, matches expectation of  17.4555 MiB
after stderr:
  ~llama_context:    Vulkan0 compute buffer size is 763.8125 MiB, matches expectation of 763.8125 MiB
  ~llama_context: Vulkan_Host compute buffer size is  17.4555 MiB, matches expectation of  17.4555 MiB
```

The line a reader can check without any of this code — the *same* `Vulkan0 compute buffer size` in
both processes, while the BEFORE response says `backend: "cpu"`:

```
$ grep -m1 'Vulkan0 compute buffer size' .e2e/t_80f1a4c6-serving-backend/before_e3_batch_stderr.log
~llama_context:    Vulkan0 compute buffer size is 363.5412 MiB, matches expectation of 363.5412 MiB
$ grep -m1 'Vulkan0 compute buffer size' .e2e/t_80f1a4c6-serving-backend/after_occamy_stderr.log
~llama_context:    Vulkan0 compute buffer size is 763.8125 MiB, matches expectation of 763.8125 MiB
```

Worth naming: the AFTER row has `n_gpu_layers: 0` — the fit plan degraded to *no offloaded weights*
under the shared box's VRAM pressure, and llama.cpp's **op offload** still ran the graph on the
device. That is exactly the case the old field could not express (weights on the host, compute on
the GPU) and the reason the attribution reads the *compute* buffers rather than the placement.
`W_LOW_MASS` in the AFTER warnings is the engine's own readout warning for that answer
(`c01` under a degraded placement); no `W_BACKEND_MISMATCH` is raised — the claim and the evidence
agree.

Three cheap live rows on the smallest local GGUF (`Qwen3.5-4B-Q4_0.gguf`, ~90 s each) pin the three
claim sources end to end (`smoke_4b_*.json` / `.err`):

| run | `backend` | `backend_source` | `effective_backend` | warnings |
|---|---|---|---|---|
| `--backend vulkan` | `vulkan` | `request` | `vulkan` | — |
| no flag (the Vulkan bundle) | `vulkan` | `bundle` | `vulkan` | — |
| `--backend cpu` (a refuted claim) | `cpu` | `request` | `vulkan` | `W_BACKEND_MISMATCH` |

The last row is the E3 shape with the claim forced by hand: the label is *kept* (`cpu`) and the
refutation is *named*, instead of the reader being told nothing.

## 5. What was *not* changed (and why)

* **`engine.fit.backend`.** The fit plan's own `backend` stays as it is: the plan is an artifact of
  the host it was computed for (and is cached per `(model sha256, host fingerprint)`), and its
  `backend` field follows `fit.host_facts()` → `runtime_backend()`, which still falls back to
  `backend_requested`. The serving attribution no longer reads it. A reader who wants to know what
  computed reads `engine.effective_backend`; `engine.fit.backend` says which host world produced
  the plan.
* **Exit codes.** A refuted claim warns; it does not fail `run`/`ask` (the response is still a
  valid answer, unlike a bench row, which is withheld and fails the report).
* **`--backend` still does not select a bundle** on the serving path (it selects nothing today;
  `$GGUFONE_RUNTIME_DIR` / the installed bundle does). The fix makes the flag *honest* (it is the
  claim, and the log checks it) instead of changing what it does.

## 6. Gates

Measured on the **rebased** tree (the private clone, `a644ed0` on `f9d9a08` — the tree that landed),
with the box shared with sibling cards; the raw text of each row is in
`.e2e/t_80f1a4c6-serving-backend/`.

| gate | result | raw |
|---|---|---|
| new file, parent tree (RED, detached worktree at `f9d9a08` + this test file) | **20 failed**, 1 skipped | `red_gates_landed.txt` |
| new file, this tree (GREEN) | **20 passed**, 1 skipped | `green_gates_rebased.txt` |
| offline suite, parent tree (baseline, measured for the E3 card) | 993 passed, 41 skipped | `baseline_full_suite.txt` |
| `pytest -q -p no:randomly` (offline, full, clean env, rebased tree) | **1093 passed, 42 skipped** in 52.6 s | `green_full_suite_rebased.txt` |
| ruff (`src` + `tests` + `tools`) | `All checks passed!` | `ruff.txt` |
| coverage (full offline suite, whole tree) | 91 % total · `decide.py` 96 % · `finder.py` 97 % · `harness.py` 93 % · `cli.py` 88 % · `errors.py` 100 % · `session.py` 73 % (its libllama lifecycle is GPU-only) | `coverage_modules_rebased.txt` |
| coverage of the **changed statements** only | **60/60 statements = 100 %** (`harness.py` 1/1, `cli.py` 1/1, `decide.py` 12/12, `session.py` 30/30, `finder.py` 16/16) | `coverage_changed_statements_rebased.txt` |
| live (`VK_DRIVER_FILES` + pinned bundle, `-k live --run-network`) | **3 passed** in 31.2 s | `live_gate_rebased.txt` |
| live rows on the smallest local GGUF (three claim sources, ~90 s each) | see §4 | `smoke_4b_*.json` / `.err` |
| Tier-M mutation (soft) | **41.8 %** killed/scored (631/1508; `decide.py` 45.2 %, `session.py` 36.1 %; 488 mutant slots have no test in the selection) — **0 survivors on the new claim symbols**, 35/39 killed in `device_evidence`, and every survivor that touches a card line is classified below | `mutation_score.txt`, `mutation_survivors.txt`, `replay_survivors.txt`, `replay_session_init.txt` |

The 42 skips are the 41 the parent tree already had plus this card's `model`-marked live gate
(a live row runs with `--run-network`; see §4).

One caveat, honestly: the repo's default `pytest -q` runs under **pytest-randomly**, and in this
container the random order occasionally trips the pre-existing runtime-probe tests
(`test_runtime_fallback` / `test_runtime_install` / `test_runtime_contract`: "does not load on this
host" reasons that turn into other strings, 1–36 failures per run). It is the container, not this
change: the same test files flake the same way on the **parent tree** with this card's gate file
ignored (`parent_flake_control.txt`: one run 36 failed, the next three could not even fork —
`timeout: fork system call failed` — while sibling cards held the 256-pid cap), and they pass 40/40
on both trees when the box has headroom. The gate quoted above is therefore the **deterministic**
`-p no:randomly` run on this tree, and the flaky pair is reported as a pre-existing environment
finding (§7 F1), not as a regression. The same pressure killed the first two mutation sweeps and
the first Occamy attempt (see §7); the sweep driver in this card's raw material waits for pid
headroom and retries.

### 6b. The Tier-M sweep and its survivors

`[tool.mutmut]` (committed in `00ecc46`) mutates the two engine modules the fix moves —
`src/ggufone/engine/session.py` + `src/ggufone/engine/decide.py` — against
`tests/test_serving_attribution.py`. The sweep is driven by
`.e2e/t_80f1a4c6-serving-backend/mutmut_sweep.sh` (fresh `mutants/`, `--max-children 2`, pid-cap
waiting, retries) because the container's shared pid cgroup kills a forking mutmut run with
`BlockingIOError` (§7, F1).

The sweep ran three times against frozen trees (a score is only valid for the tree it measured):

| sweep | tree | what it measured / found | raw |
|---|---|---|---|
| A (pre-rebase run, aborted and resumed) | the fix *before* the survivor pins | 7 `device_evidence` survivors → 3 pinned by `a644ed0`, 4 classified equivalent | `replay_survivors.txt`, `mutmut.pre-rebase-1359/` |
| B | + the two `device_evidence` pins (`a644ed0`) | **41.4 %** 624/1508; its per-line analysis found 5 survivors on card lines inside `ModelSession.__init__` | `mutmut.out.round1`, `replay_session_init.txt`, `mutants.pass2-1417/` |
| C (final) | + the two `ModelSession.__init__` pins (`4f30482`) | **41.8 %** 631/1508 (`decide.py` 45.2 %, `session.py` 36.1 %; 488 slots have no test in the selection) | `mutation_score.txt`, `mutants/` |

Sweep A — the seven `device_evidence` survivors, each **replayed** against the gates
(`tools/t80_replay.py` refuses to report kills when the control row fails; the control is part of
the table):

```
# replaying 7 device_evidence survivors: ['x_device_evidence__mutmut_11', 'x_device_evidence__mutmut_12', 'x_device_evidence__mutmut_16', 'x_device_evidence__mutmut_17', 'x_device_evidence__mutmut_18', 'x_device_evidence__mutmut_5', 'x_device_evidence__mutmut_8']
# src sha256 before: ce9944a40cde0aa6bd29d23ef897a088df2cc2a1c082a6b82a76999e514acbc4
# CONTROL (unmutated tree): PASS
SURVIVED x_device_evidence__mutmut_11  
SURVIVED x_device_evidence__mutmut_12  
KILLED   x_device_evidence__mutmut_16  FAILED tests/test_serving_attribution.py::test_a_log_kept_as_lines_reads_like_the_joined_text
KILLED   x_device_evidence__mutmut_17  FAILED tests/test_serving_attribution.py::test_a_log_kept_as_lines_reads_like_the_joined_text
SURVIVED x_device_evidence__mutmut_18  
SURVIVED x_device_evidence__mutmut_5  
KILLED   x_device_evidence__mutmut_8  FAILED tests/test_serving_attribution.py::test_a_session_without_a_log_reads_as_unverified
# src sha256 after:  ce9944a40cde0aa6bd29d23ef897a088df2cc2a1c082a6b82a76999e514acbc4   restored=True
```

The control row passes, so the kills are real (a test that fails on the unmutated tree fails for
every mutant — §7 F5). 16/17 are the *sequence* side of the `isinstance(text, str)` normalization
and 8 is the `getattr` with no default; the two gates added in `a644ed0` pin them, which is why
sweep C's `device_evidence` survivors are only 5/11/12/18.

Sweep B's per-line analysis — five survivors on lines this card wrote inside `ModelSession.__init__`:
the claim/source resolution (a claim resolved even when the caller named a backend, a `None.source`
crash, two literal-source mutants) and the caller-owned log sink. Two gates added in `4f30482` pin
them; this replay names the failing test per mutant:

```
# replaying 22 survivors of src/ggufone/engine/session.py matching ['ModelSessionǁ__init__']
# sha256 before: 7e295bb07adcd6fb7be62f28c4dc6b71edaff44e424e93871f25b8499c4aa3e6
# CONTROL (unmutated tree): PASS
KILLED   xǁModelSessionǁ__init____mutmut_18  tests/test_serving_attribution.py::test_an_explicit_backend_with_no_source_is_named_explicit
SURVIVED xǁModelSessionǁ__init____mutmut_2
KILLED   xǁModelSessionǁ__init____mutmut_20  tests/test_serving_attribution.py::test_an_explicit_backend_with_no_source_is_named_explicit
KILLED   xǁModelSessionǁ__init____mutmut_21  tests/test_serving_attribution.py::test_an_explicit_backend_with_no_source_is_named_explicit
…
KILLED   xǁModelSessionǁ__init____mutmut_47  tests/test_serving_attribution.py::test_a_caller_owned_log_sink_receives_the_context_lines
KILLED   xǁModelSessionǁ__init____mutmut_7  tests/test_serving_attribution.py::test_an_explicit_backend_with_no_source_is_named_explicit
# sha256 after:  7e295bb07adcd6fb7be62f28c4dc6b71edaff44e424e93871f25b8499c4aa3e6   restored=True
```
(the elided `SURVIVED` rows are the 15 survivors of that function's *pre-existing* lines; the full
table is `replay_session_init.txt`).

Sweep C is the number in §6: a **lower bound for the modules** and a **complete measurement of the
card's new code**. The selection is the card's own gate file (`pyproject.toml`, `00ecc46`), so
the 877 survivors sit overwhelmingly in pre-existing internals the selection does not drive
(`open_model` 192, `_answer_question` 103, `decide` 98, `_score_group` 77 …). Two checks tie the
survivors to the card's own lines, both in the raw material:

* per symbol — **0 survivors in `backend_claim` / `BackendClaim` / `recorded_backend` /
  `device_log`**, 35/39 killed in `device_evidence`;
* per line — `tools/t80_survivor_lines.py` (every survivor diffed against the original) finds
  **4 survivors in `decide.py` that mutate a card line** — `device_evidence` 5/11/12 (the `getattr`
  default the `or ""` makes unobservable) and 18 (the join separator, which
  `parse_device_usage`'s line split + search cannot see) — and **3 in `session.py`**: mutants
  161/169/170 of `open_model`, all three mutating the *pre-existing* `placement=` keyword that
  shares the line with the card's new `load_log=captured`. That line's own mutant
  (`open_model__mutmut_162`, `load_log=None`) is **killed** by
  `..._the_live_session_records_the_load_and_the_context_lines`; the `placement=` mutants are the
  wider engine/bench gates' contract (`tests/test_engine_fork.py`, 2.3 s per mutant on this box),
  out of the Tier-M selection by design. No `ModelSession.__init__` survivor remains on a card
  line.

## 7. Findings for the coordinator

* **F1 (environment, already known).** The container's `pids.max = 256` is the binding constraint
  for sibling work, not just this card: `uv` itself aborted with `Os { code: 11 }` (EAGAIN) while
  resolving `--with mutmut`, `mutmut` died twice with `BlockingIOError` at `os.fork()`, llama.cpp
  died with `libgomp: Thread creation failed`, and the first Occamy after-run failed with
  `E_BACKEND_OOM` (a 705 MiB `Vulkan0` allocation refused while a sibling held 3.1 GiB of the
  8 GiB GPU) — all while the *response path under test* stayed healthy. The successful Occamy row
  needed `--n-ctx 256 --threads 1 --fit-target 2000`.
* **F2 (measured, worth a card).** On that successful row the plan degraded to
  `n_gpu_layers = 0` — *no* offloaded weights — and the graph still ran on the GPU via llama.cpp's
  op offload (`Vulkan0 compute buffer size is 763.8125 MiB`). Before this fix that run would have
  published `backend: "cpu"` with `n_gpu_layers: 0`: the placement said "host" and the compute said
  "device". `engine.effective_backend` now says `vulkan`; a reader (or a routing decision) that
  still keys on `n_gpu_layers` will keep mis-reading this shape.
* **F3 (scope note).** `engine.fit.backend` (and the cached fit plan files) still carries the fit
  host's own `backend` field; §5 says why it stays. If the coordinator wants `fit.host_facts()`'s
  `backend` to stop falling back to `backend_requested` (a request that may have failed), that is
  a one-line change with a fit-cache invalidation behind it — a separate card.
* **F4 (card follow-up).** The `serve`/`mcp` commands are still milestone stubs
  (`cli.main` prints "not implemented yet" and exits 3), so this fix covers the whole live serving
  surface (`run`/`ask`/`calibrate` → `cli.decide_payload` → `DecisionEngine`). Whatever
  implements `serve` must build its responses through the same `decide_payload` seam (or the same
  `decide.device_evidence` call), or the E3 lie comes back through a new door.
* **F5 (method, worth reusing).** The first survivor replay produced *seven false kills*: the new
  pinning test had a wrong expected count, so it failed on the unmutated tree too — and a test that
  fails on HEAD fails for every mutant. The fixed tool runs a **control row** (the pin on the clean
  tree) first and refuses to report kills if it fails; the honest table is then 3 killed
  (16/17 — the log-as-lines branch — and 8 — the log-less session) and 4 survivors classified as
  equivalent through `runtime/devices.py` (5/11/12 move the `getattr` default that the `or ""`
  makes unobservable; 18 changes the join separator, which the line-splitting parser cannot see).
  Any card that replays mutants should ship the control row with the table.
* **F6 (coordination).** This card's clone was rebased twice (`15e89a9` → `a475090` → `f9d9a08`,
  twelve sibling commits in between: E3b's `bench/labels.py` + tools, the E2 FIX teardown work,
  t_97f1bc93's `cli.run` process entry point, their `.e2e` material) and landed as a fast-forward on
  top of `f9d9a08`. The only textual conflict, both times, was `pyproject.toml`'s `[tool.mutmut]`
  pair — every card retargets the same two lines, so **that block is a permanent hotspot**: whoever
  lands second must resolve it by hand. At landing time the shared tree's working copy also carried
  sibling `t_6952f0dd`'s *uncommitted* retarget (labels pair); the landing stored that file, ran the
  fast-forward, and restored it byte-for-byte (sha256 `0ccb9bd8…`), so the sibling's WIP stayed
  theirs — see the completion handoff.

* **F7 (pre-existing, worth a card).** The repo's default `pytest -q` (pytest-randomly ordering)
  is **not reliably green in this container**: the runtime-probe files
  (`test_runtime_fallback`, `test_runtime_install`, `test_runtime_contract`) fail 1–36 assertions
  per run depending on the order/the moment, *identically on the parent tree with this card's gate
  file ignored* (`.e2e/t_80f1a4c6-serving-backend/parent_flake_control.txt`; two of those runs
  could not even start: `fork system call failed` at the 256-pid cap). Under `-p no:randomly` both
  trees are green (parent 1073/41, this tree 1093/42). The failure shape is the install probe's
  reason string ("… does not load on this host") turning into another string, i.e. the probes'
  behaviour under pid pressure — a suite-level determinism problem (pin an order or make the probes
  pressure-proof), not a product bug in this change. The QA note quotes the deterministic run for
  that reason.

## 8. Files

`src/ggufone/engine/session.py` · `src/ggufone/engine/decide.py` · `src/ggufone/runtime/finder.py` ·
`src/ggufone/bench/harness.py` (the classifier now delegates) · `src/ggufone/cli.py` ·
`src/ggufone/errors.py` · `tests/test_serving_attribution.py` (new) · `tests/fake_engine.py` ·
`README.md` · `pyproject.toml` (mutmut pair) · `.e2e/t_80f1a4c6-serving-backend/` (raw material).
