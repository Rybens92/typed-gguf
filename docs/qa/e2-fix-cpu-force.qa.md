# FIX — a `cpu` bench row must compute on the CPU, on a GPU box too (card t_55de5779) — QA Report

Date: 2026-09-19 · Tier **M** (default; the card declares none) · Decision: **ship** — all five card
requirements are measured, including requirement 3 (the live gate on the operator's tree, this box,
the Vulkan bundle).

Commits (landed by fast-forward on `adcb7de`): `4797146` (RED — the pin's gate, 9 failing),
`b41884f` (GREEN — `cpu_only` pins the loader's device list, zero offload layers), `1cbf948` (the
live gate asserts the compute path), `430621a` (the note's contract + the placement string + lint),
`089323c` (evidence). Evidence doc: `docs/evidence/e2_fix_t_55de5779_cpu_force.md`.

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none open.**

🟡 WORTH CONSIDERING (3, each recorded, none blocks)

* **`tools/host_gate_e1c_fit.sh` step 4b changes meaning** (`bench --suite latency --gpu-layers 36`,
  no `--backend`): `auto` selects the `cpu` row, which is now pinned to zero offload layers, so the
  step no longer measures "the host bundle's ladder walk" that its own comment describes. The
  tool is outside this card's scope (`bench/suites.py` + `bench/harness.py`); the fix is
  `--backend vulkan`, and the evidence doc §5 declares it for the coordinator to route. Named here
  rather than patched silently — this is exactly the class of silent label change the card is about.
* **`docs/BENCHMARKS.md` §8's `[host]` E3d tables** carry `effective_backend: vulkan` +
  `W_BACKEND_MISMATCH` under a `cpu` request (the section labels them pre-fix and explains the
  mismatch). A post-fix re-run of those 60 items would be a genuine cpu measurement; re-publishing
  is a separate call, not a silent rewrite.
* **Mutation (Tier M, soft threshold): 67.2 %** over the changed block (277 killed / 131 survived /
  412 mutants / 0 not run), whole file 34.6 % over 1124 mutants with **441 `no tests`** and 4
  timeouts (`no tests` = mutmut runs each mutant against the functions the selection covers;
  session.py's serving internals are outside this selection — quoted, never folded in). Survivor
  classes are named in the evidence doc §6 (equivalent-on-the-fixture, caller-set defaults,
  note/row wording, pre-existing branches); **no survivor sits on a behavioural claim of the card**,
  and the only class touching the added lines is text. The verdicts are cross-checked by a hand
  table (6 behavioural mutations on the card's own lines, each killed by the pin gates; 2 controls
  survive; every file restored byte-identically, `sha256` printed).

🟢 ACCEPTABLE

* **The gates the card asks for, RED before / GREEN after**: offline structural gate
  `tests/test_bench_cpu_force.py` — 9 failed / 1 passed on the pre-fix tree, 11 passed on the landed
  tree; the live gate (the card's exact command) — 1 failed in 53.97s on `00265ea` with
  `W_BACKEND_MISMATCH` (`Vulkan0=3 · Vulkan_Host=3`), **1 passed in 32.40s** on the landed tree with
  `effective_backend: cpu`, `device_buffers {CPU: 3}`, `warnings: []`, `identical: true` — while the
  engine log still shows the Vulkan device and both backends loaded.
* **The guard is untouched** (card requirement 1): `device_usage_of` / `_mismatch_note` /
  `W_BACKEND_MISMATCH` / `effective_backend` unchanged; the live test's three new assertions use its
  fields. The fix makes the label true; the rule that checks labels stayed as it was.
* **Route B chosen and recorded** (requirement 2): pin the *load*, not the bundle. No CPU-only
  bundle is installed here, the multi-backend bundle is the normal case (a completed bundle always
  carries `libggml-cpu.so`, so a Vulkan-only box answers `--backend cpu` at all), and the pin makes
  the row true on both box classes with one engine parameter and a typed refusal.
* **The sweep (requirement 5)** is a table in the evidence doc §5: the single funnel
  (`suites._spec` → `harness.spec_for`) is fixed for all suites; `--backend auto` / `--quick`'s
  preset are now true; accelerator rows and the guard are untouched; `bench/isolation.py`'s children
  are declared (the child's own row is published); `docs/BENCHMARKS.md`'s `[container]` cpu rows are
  unaffected; the `[host]` rows and the host-gate step are declared above.
* Full offline suite **1315 passed, 47 skipped** (exit 0) on the clean tree; the shared tree reads
  `1319 passed, 47 skipped, 1 failed` — the one failure is a **live sibling's uncommitted WIP**
  (`tests/test_e3e_docs.py`, absent from the clean tree, expecting an E3e section in their dirty
  `docs/BENCHMARKS.md`). No file this card touches is red in either run.
* Coverage of the **added lines: 33/33 = 100 %** (`harness.py`, `suites.py`, `session.py`,
  `ctypes_binding.py`); whole-file under the gate selection 91.9 / 96.2 / 72.5 / 51.0 %.
* `ruff check src tests tools docs`: every file this card touches is clean. The 13 remaining errors
  are pre-existing or sibling-owned and none is on a line this diff adds or changes:
  `tests/test_e3e_roles.py` (6, committed E3e), `src/ggufone/engine/prompt.py` (4 × F541, E3e),
  `src/ggufone/bench/harness.py:966` (E501, an E3e line my hunks merely shifted),
  `src/ggufone/engine/decide.py:748` (E501), `tools/e2_reproduce.py:58` (E501).
* Additive interface: `open_model` gains `cpu_only` (default `False` = today's behaviour),
  `ModelSpec` gains `cpu_only`, `Placement` gains `cpu_only` in `to_dict`, the row/report text gains
  the pin marker. Two test files' expectations moved with the surface (`used.cpu_only`,
  `"n_gpu_layers=0 (cpu compute pinned)"`), each with the reason in place.

## Decision matrix

**Option A — ship now** (the card's five requirements all measured)
  ✅ the defect reproduced first-hand, both gates RED → GREEN, the live run on the operator's tree
  ✅ the route decided and recorded; the siblings swept, each fixed or declared
  ✅ the sweep's verdicts cross-checked by hand on this box; coverage 100 % on the added lines
  ⚠️ the host-gate step and the `[host]` tables are declared, not fixed/re-published (out of scope)
  → overall risk: **LOW**

**Option B — hold for a host-gate patch**
  ✅ step 4b would keep measuring the ladder walk it documents
  ⚠️ `tools/` is explicitly outside the card; the operator's gate is theirs to re-point, and the
     declaration above is what makes it unsilent
  → cost without a new claim about *this* card's contract

💡 **Recommendation: Option A**, with the two declared items routed by the coordinator (the
`tools/host_gate_e1c_fit.sh` one-liner, and the question of re-publishing the `[host]` tables).

## What could go wrong (only where it changes the decision)

⚡ Scenario: a box whose bundle cannot name its CPU device (`ggml_backend_dev_by_name` missing, or
  `NULL` for `"CPU"`).
  Behaviour now: `E_RUNTIME_SYMBOLS` — a typed refusal, **no** unpinned fallback load (pinned by a
  parametrized gate). Before: an unhonoured pin would have been the same lie in a new place.
  Detectability: HIGH (typed error, row carries `reason`). Blast radius: one bench row.

⚡ Scenario: a `cpu` row is requested together with `--gpu-layers 36`.
  Behaviour now: the request is preserved (`placement.requested`), the executed plan is zero layers,
  the note names the overridden request, the rendered line says "(cpu compute pinned)".
  Detectability: HIGH. Blast radius: none (the row is honest about both numbers).

⚡ Scenario: a `cpu` row on a **GPU-less** box (CI).
  Behaviour: unchanged numbers, the pin resolves the only device there is; the strengthened live
  gate reports `effective_backend: cpu` there too, so it stays a gate rather than a host-only check.
  Detectability: HIGH.

## Confidence

CONFIDENCE: **9/10**

  + the defect is reproduced first-hand on this box (raw RED row + engine message), and both gates
    are RED → GREEN with the pre-fix arms re-run against the rebased base;
  + the GREEN live row is from the *landed* tree, by the card's own command, with the engine log
    showing the Vulkan device still present — the claim is about the compute path, not the device;
  + the sweep's verdicts are cross-checked (hand table: 6 killed, 2 controls survived, byte-identical
    restores) and the not-run/no-test boundary is quoted;
  + coverage 100 % on the added lines; lint clean in every touched file; the full suite green on the
    clean tree;
  − the shared tree's suite shows one failure from a live sibling's uncommitted WIP (named above,
    outside this card's files);
  − the Tier-M sweep is one run with a soft threshold by design — the score is reported, not argued.
