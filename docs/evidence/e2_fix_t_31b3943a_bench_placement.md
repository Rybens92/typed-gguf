# E2 FIX — `ggufone bench` and the loader's degrade ladder (card t_31b3943a)

Branch `main` (this repo has no remote; the commits are local on the shared tree) · Tier **M**
(the card declares none) · evidence schema `ggufone.evidence.bench-placement/v1`
Box: the shared container (24 CPUs seen, **2 CPU-seconds/s cgroup quota**, no `/dev/dri`), the pinned
CPU bundle `/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu`, the pinned models
`Spark-X2.5-4B-Q8_0.gguf` (4 375 021 152 B) and `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (558 772 480 B).

Commits: `8d4fc9f` (fix + the 8 new gates), `d459601` (CI + host gate + rehearsal script),
`<evidence commit>` (this document, the raw logs in `.e2e/t_31b3943a-bench-placement/`).

## 0. What was wrong, in one call path

```
suites._run_latency -> harness.LiveModel.load
  -> session.open_model(fit_plan=harness.Placement(spec.n_gpu_layers))
  -> fit.degrade_ladder(fit_plan, facts)          # built before the FIRST load attempt
  -> plan.kv_type                                 # AttributeError: 'Placement' has no attribute
```

`bench` names its placement with a minimal object on purpose (`--gpu-layers`; a published row must be
reproducible from its flags, not from a fit plan written on another box). `fit.degrade_ladder`
documented a `FitPlan` and read `kv_type`, `n_ctx`, `warnings`, `notes`, `est_*` off it. The mismatch
raised inside `open_model`, i.e. *before* any device was touched, and the CLI's catch-all turned it
into `E_INTERNAL` (exit 4).

Two corrections to the card's framing, both measured on this box:

1. **The bug is not GPU-specific.** It reproduces in this container with the pinned **CPU** bundle
   and the pinned model, through the exact command the published E2 evidence records (§1). Only the
   header read has to succeed; the ladder is built before the first load attempt.
2. **`--gpu-layers -1` was a second hole in the same contract.** `spec_for` gives a non-CPU backend
   `-1` when no `--gpu-layers` is passed, and llama.cpp reads `n_gpu_layers < 0` as "offload every
   layer". The ladder read `layers <= 0` as "nothing to reduce", so the one host class the card is
   about (a Vulkan box with the default flags) had **no placement retry at all**, and
   `plan_device_bytes` counted a full offload as 0 bytes of device memory.

Consequence for E2's published tables (reported to @bots-coordinator in the groupchat): the
`reproduce:` line inside `docs/evidence/e2_latency.json` crashed on the very tree that committed it
(`4e1d549`), so those JSONs cannot have been produced by that tree state — the numbers may be real,
but the recorded command was broken until this fix. §6 runs the same command shape after the fix.

## 1. Raw repro on this box (before the fix)

`docs/evidence/e2_latency.json` names this command; it is the card's repro too:

```
$ uv run ggufone bench --suite latency --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
      --backend auto --runs 5 --threads 4 --json
load_backend: loaded RPC backend from .../b11026-linux-x64-cpu/libggml-rpc.so
load_backend: loaded CPU backend from .../b11026-linux-x64-cpu/libggml-cpu-haswell.so
error: E_INTERNAL: AttributeError: 'Placement' object has no attribute 'kv_type'
EXIT=4
```

raw: `.e2e/t_31b3943a-bench-placement/repro_published_cmd.err` (and the `.exit`-equivalent line above
is the shell's own `EXIT=` capture, stdout empty).

## 2. The fix

One normalization at the loader boundary plus a total ladder (`8d4fc9f`):

| what | where | why |
|---|---|---|
| `fit.coerce_plan(plan)` | `runtime/fit.py` | a `FitPlan` is returned unchanged (identity preserved — `handle.fit_plan is plan` still holds for real plans); a minimal placement gains the honest defaults: `kv_type="auto"` (nothing pinned; the context init resolves it), `n_ctx=0`, `est_*=0`, `budget_bytes=0` (a load sizes no cache) |
| `fit.kv_start(kv_type)` | `runtime/fit.py` | `auto`/an unknown word starts the KV ladder at the top rung — the rule `estimate_plan` and `session._kv_ladder` already use. No `KeyError`, and `auto -> f16` no longer emits a bogus `W_KV_TYPE_DOWNGRADE` |
| `fit.planned_layers(plan, model)` | `runtime/fit.py` | `n_gpu_layers < 0` = every layer (llama.cpp's reading), so the ladder can reduce from it; `plan_device_bytes` charges it the full weight footprint instead of 0 |
| `session.open_model` | `engine/session.py` | normalizes ONCE (`plan = fit.coerce_plan(fit_plan)`), builds the load-time walk from `fit.kv_start(plan.kv_type)`, and gives a negative placement the same one-CPU retry a positive one gets |
| `harness.placement_of` + `LiveModel.placement` | `bench/harness.py`, `bench/suites.py` | a row now carries `placement.requested` **and** `placement.used` (`n_gpu_layers`, `kv_type`, `degraded`, `attempts`): a degraded retry offloads fewer layers than the flags asked, and the table must say so instead of printing the request |

`docs/` is untouched by the fix: `degrade_ladder`'s docstring is the contract that changed (it now
documents "a `FitPlan`, or any placement-like object" and the negative-layer reading).

## 3. RED -> GREEN (requirement 2)

`tests/test_bench_placement.py` — 10 gates over the bench load seam, the ladder contract, the typed
`E_BACKEND_OOM`, the CLI row, the rendered placement line and the negative-placement note. Vehicle:
the fake runtime of `tests/test_fit_oom_recovery.py` (a real bundle directory + the operator's own
OOM tail through the real `llama_log_set` ABI) — model-free, no GPU.

* RED on the parent commit `4e1d549`, in a separate worktree (`git worktree add --detach`, the
  final test file copied in unchanged, pytest's `pythonpath = ["src"]` binding the pre-fix package):
  **`10 failed in 1.15s`** — including the raw
  `assert 'E_BACKEND_OOM' in "AttributeError: 'Placement' object has no attribute 'kv_type'"`.
  raw: `.e2e/t_31b3943a-bench-placement/red_pretree_final.txt`
* GREEN on this tree: **`10 passed in 1.80s`**.
  raw: `.e2e/t_31b3943a-bench-placement/green_fixed_tree_final.txt`
  (an earlier 8-gate RED/GREEN pair is kept as `red_pretree.txt` / `green_fixed_tree.txt`.)

## 4. Coverage and static gates

`uv run --with pytest-cov pytest -q --cov=ggufone.runtime.fit --cov=ggufone.engine.session
--cov=ggufone.bench.harness --cov=ggufone.bench.suites` → `757 passed, 38 skipped` (raw:
`.e2e/t_31b3943a-bench-placement/coverage_offline.txt`; the final full-suite run after the last two
gates is `759 passed, 38 skipped`, exit 0 — `final_full_suite.txt`).

| module | stmts | miss | cover |
|---|---|---|---|
| `bench/harness.py` | 392 | 35 | 91 % |
| `bench/suites.py` | 321 | 4 | 99 % |
| `engine/session.py` | 360 | 110 | 69 % |
| `runtime/fit.py` | 461 | 16 | 97 % |

Every line this fix added or changed is covered by the offline gate (the missing `session.py` lines
are the pre-existing live-only paths, e.g. `ModelSession.prefill`/fork 440-472, which the
`--run-network` suite exercises). `ruff check src tests tools docs .github` → clean.

## 5. The scenario that runs without a GPU (requirement 3)

`runtime-matrix.yml` (job `linux-cpu`, which already downloads the pinned bundle) gained two steps:

1. `bench --suite latency --gpu-layers 4 --runs 1 --sizes 256` on the pinned bundle + a downloaded
   0.5B GGUF; the assertion requires a `model_load` row **and** `placement.used` in the report.
2. the fake-alloc bundle (`cc … tools/fixtures/fit_oom_bundle.c`) with `GGUFONE_FAKE_OOM_ALL=1` and a
   synthetic GGUF (`tools/fit_oom_probe.py --make-gguf`): the run must exit 1 with a row whose reason
   names `E_BACKEND_OOM` and `3 placement(s)`, and must NOT contain `AttributeError`/`E_INTERNAL`.

`tools/host_gate_e1c_fit.sh` gained the same two worlds on the operator's own bundle (step 8:
`bench --gpu-layers 36`; step 9: bench + the fake-alloc bundle), and
`host_gate_e1c_fit_summary.py` folds them into `bench_placement` / `bench_placement_oom` plus four
boolean checks (`bench_run_loaded_a_model`, `bench_row_prints_the_placement_used`,
`bench_row_keeps_the_request_next_to_it`, `bench_retry_row_is_a_typed_error`).

Sandbox rehearsal of the two CI commands (`tools/rehearse_bench_placement_ci.sh`; substitutions are
stated in the script header: the local pinned 0.8B GGUF for the downloaded 0.5B, the local pinned CPU
bundle for `/tmp/ggufone-rt`): **both steps pass on this tree** —

* step A `bench --suite latency --gpu-layers 4` → `exit=0 elapsed=503s`, and the row carries
  `requested: n_gpu_layers=4` / `used: {n_gpu_layers: 4, kv_type: "auto", degraded: false}`;
* step B `bench --suite throughput` on the fake-alloc bundle (`GGUFONE_FAKE_OOM_ALL=1`) → `exit=1`,
  and the row reason is
  `BackendOomError: E_BACKEND_OOM: … tried 3 placement(s) down to CPU-only, none fit:
  n_gpu_layers=4 -> oom; n_gpu_layers=2 -> oom; n_gpu_layers=0 -> oom` — no `AttributeError`.

Raw: `.e2e/t_31b3943a-bench-placement/ci_rehearsal/` (`bench_placement.{out,err,exit}`,
`placement.json`, `bench_placement_oom.*`, `placement-oom.json`).

Parent-rev control (`tools/rehearse_bench_placement_ci_red.sh`: the *same* two commands in a
`git worktree` of `4e1d549`) — **both CI steps fail there, i.e. the automation catches this class**:

* step A → `exit=4`, stderr tail `error: E_INTERNAL: AttributeError: 'Placement' object has no
  attribute 'kv_type'` (`ci_rehearsal_red/ci_step1.exit`, `ci_step1.err`);
* step B → `exit=1` and the row reason is exactly the AttributeError
  (`ci_rehearsal_red/placement-oom.json` → `backends[0].reason`).

Raw: `.e2e/t_31b3943a-bench-placement/ci_rehearsal_red/`.

## 6. Post-fix live run (CPU placement; the Vulkan twin is PENDING on the operator host)

**The ask control (the card's own control: the engine path kept working while bench crashed).**
`ggufone ask --model Qwen3.5-0.8B-UD-Q4_K_XL.gguf --no-fit-cache --state @state.txt --choice ...` on
the same pinned CPU bundle: `exit=0`, and the response carries the placement surface —

```
engine.placement = {"note": "CPU only: the fit plan offloads nothing (kv_type=f16)",
                    "n_gpu_layers": 0, "kv_type": "f16", "degraded": false, "attempts": [], "warnings": []}
engine.fit       = plan (n_ctx 4096, n_seq_max 8, source estimate, budget 32474873856)
```

raw: `.e2e/t_31b3943a-bench-placement/ask_control.json`, `ask_control.exit`, `ask_control.err`.
This is the container twin of the coordinator's host control (where `ask` returned a decision with
`n_gpu_layers=36` while `bench` died).

**The published-command twin (4B, pinned CPU bundle, post-fix).** The card's repro flags on the
pinned 4B model — `bench --suite latency --runs 1 --sizes 256 --threads 4` — now run to completion:
`EXIT=0`, 43 s of `llama_context` work, a full latency table (model load, prefill, per-question,
wave scaling, warm cache, amortisation). Raw: `post_fix_spark_latency.{out,err,exit}` +
`post_fix_spark_latency.json`; rendered head:

```
### latency — Spark-X2.5-4B-Q8_0.gguf
- generated: 2026-09-18T11:08:40Z
- host: Linux-...x86_64 · cpus 24 · cgroup quota 2.0
- config: backend=auto runs=1 threads=4
- reproduce: `uv run ggufone bench --suite latency --model .../Spark-X2.5-4B-Q8_0.gguf --backend auto --runs 1 --threads 4 --json`

**model load (ms)**            | row           | n | p50      |
                               | model_load_ms | 1 | 738.265  |
**prefill**                    | 256 tokens    | 1 | 23,222.646 ms  (11.024 tok/s)
**per question**               | 2 / 4 / 10 candidates -> 4,244.292 / 5,786.384 / 9,685.072 ms p50
                                 (waves 2, forks 2/4/10)
**wave scaling (N questions)** | N=1..16 rows, p50 from 5,689 ms (N=1) to ~156,615 ms (N=12)
```

Those numbers are the same order as the published E2 table (the 4B is ~10 tok/s prefill on this
2-CPU-second box; the box is shared, so `p50` here is "this box, this minute"). What matters for
this card is that the *command* completes and the table exists — on the parent rev it died at
4 seconds.

*Placement caveat, stated instead of hidden:* this 4B process was already running when the
`placement.requested` / `placement.used` reporting landed in the same session, so its JSON does not
carry that key; the 4B **throughput** run started afterwards does (below), as do the rehearsal
(`ci_rehearsal/placement.json`) and the CI steps.

## 7. Mutation (Tier M)

Scope (`pyproject.toml`): `source_paths = ["src/ggufone/bench"]` — `harness.py`, `suites.py`,
`devset.py`; test selection `tests/test_bench.py` **+ `tests/test_bench_placement.py`** (widened by
this card; the new gate file kills the bench↔loader seam mutants the CLI file cannot).

```
uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2
python tools/mutation_score.py <mutants-dir>          # scores the .meta artifacts
```

Result (`mutmut_round2_snapshot/` + `mutmut_round2_score.txt`, the run's own artifacts — the score
convention is card t_c8e36cad's triage: `killed / (killed + survived + timeout + segfault +
suspicious)`):

```
devset.py.meta    total  234  killed  143  survived   75  score 65.6 %  not-run  0  no-tests 16
harness.py.meta   total 1586  killed  738  survived  716  score 50.8 %  not-run 85  no-tests 47
suites.py.meta    total 1416  killed  948  survived  427  score 68.9 %  not-run 41  no-tests  0
TOTAL             total 3236  killed 1829  survived 1218  no-tests 63  not-run 126
SCORE killed/scored = 60.0 %   (scored 3047)
```

Honest reading of that number (Tier M is a soft threshold: run once, report, survivors are
findings):

* **126 mutants (3.9 %) were never run** — the sweep is time-boxed (3300 s; this box gives the
  container 2 CPU-seconds/s and a 256-pid cap, and `max_children` was 2). They are reported as
  `not-run`, never folded into the score. `no-tests` (63) is excluded from the denominator too.
* **60.0 % matches the E2 card's own Tier-M number for the same package** (killed/3071 = 60.0 %) —
  this card neither inflated nor repaired the bench's test-quality picture; the new gate file was
  added to the same selection rather than replacing it.
* **The survivors live where E2 already said they do**: the renderer (`x_render_report`), the
  quality/determinism suite internals, `devset.validate/load`, `harness` statistics. That is a
  pre-existing property of the bench gates, not of this fix.
* **No survivor sits on a line this fix added.** `x_placement_of` has zero survivors, and
  `tools/mutation_span_check.py` over the touched lines in the mutants copy
  (`self.placement` in `LiveModel.__init__`) reports `0 survived` for every line that carries it.
* The surviving `x.LiveModel.__init__` mutants (13) are on the pre-existing lines of that
  constructor (alias/tempdir), not on the `self.placement` line the fix added.

**Second pass over the same scope with the *current* gate file** (`mutmut_round3_render_retry.log`,
started because the first pass copied the test selection 3 minutes before the render gate was added
to it; the pattern argument is ignored by mutmut 3.8 — it re-ran the whole scope):

```
harness.py.meta   killed 782  survived 671  score 53.8 %   (was 50.8 % — +44 kills)
TOTAL             killed 1873  survived 1173  no-tests 63  not-run 126  timeout 1
SCORE killed/scored = 61.5 %   (scored 3047)
```

The two passes bracket the number: **60.0 % -> 61.5 %** on the same scope, the delta being the
render/negative-placement gates this card added. The 126 unrun mutants are the same tail in both
passes (the sweep is time-boxed: 3300 s and 1500 s boxes, `max_children` 2 then 1 after a
`BlockingIOError` from a sibling card on this shared box); they are reported, never counted.

## 8. What is PENDING and the exact command for the operator host

Requirement 4 asks for the re-run on the operator host (RTX 3060 Ti, Vulkan `b11026`). This sandbox
has no `/dev/dri` and no GPU, so that slot is **PENDING — operator run required**; everything above is
the container's CPU-side equivalent.

```
tools/host_gate_e1c_fit.sh                       # steps 8-9 are the new bench scenarios
# or the card's own command:
GGUFONE_RUNTIME_DIR=<vulkan b11026 bundle> uv run ggufone bench --suite latency \
    --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --runs 3 --threads 4 \
    --json --out /tmp/host_latency.json
```

Expected shape (what the reviewer should see): the table head, and
`placement.requested == "n_gpu_layers=-1"` (or `36` with `--gpu-layers 36`) next to
`placement.used.n_gpu_layers` + `placement.used.degraded` + `placement.used.attempts` — a busy desktop
that cannot hold the offload must show `degraded: true` with the walked rungs, not an `E_INTERNAL`.
