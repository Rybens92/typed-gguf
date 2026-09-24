# E2 FIX — `bench` placement through the loader's ladder (card t_31b3943a) — QA Report

Date: 2026-09-18 · Tier **M** (default; the card declares none) · Decision: **ship** — every
requirement of the card is now measured, including requirement 4 (the operator-host run, executed
and reported by @bots-coordinator; raw report in
`.e2e/t_31b3943a-bench-placement/host_run_vulkan_reported.md`).

Commits: `8d4fc9f` (fix + the first gates), `d459601` (CI + host gate + rehearsal scripts),
`cb2022f`/`c614b47` (render + negative-placement gates), `a42aa13` (container evidence),
this report's commit (host-run record + the `--backend auto` selection record + docs).

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none open.** The one that was here (requirement 4: the accelerated
placement unmeasured in a GPU-less sandbox) is **closed by the coordinator's host run**: Vulkan
`b11026`, `bench --suite latency --backend vulkan --gpu-layers -1 --runs 3 --threads 4` →
`exit 0`, `placement.used {n_gpu_layers: -1, degraded: false, attempts: [], kv_type: auto}`
(no degradation), table head + determinism `sha256:d9978816…` ×3 in evidence §8. Provenance limit,
stated: the raw JSONs stayed on the operator host (`~/.ggufone-host-gate-2026-09-18/`) — this sandbox
has no bind mount of the host home — so that row is a *reported* measurement, tagged **[host]**, never
re-derived here.

🟡 WORTH CONSIDERING (2, both recorded, neither blocks)

* **The E2 published tables are not reproducible from the commit that carries them** — measured:
  `docs/evidence/e2_latency.json`'s own `reproduce:` line dies with `E_INTERNAL` on `4e1d549` (and
  the coordinator confirmed it independently on `fff127e`, also with `--backend cpu`). This fix makes
  that command shape work again (evidence §6/§8.1); the *verdict* on the E2 tables belongs to AUDIT
  `t_78f5ea7a` (auditor), which the coordinator opened for exactly that, not to this fix card.
* **Mutation (Tier M, soft threshold): 61.5 %** over 3047 scored mutants (killed 1873, survived 1173,
  `no-tests` 63, **126 never run** — the sweep is time-boxed on this 2-CPU-second shared box; the unrun
  tail is reported, never folded in). The first pass over the same scope scored 60.0 %, which equals
  the E2 card's own bench-package number; the +1.5 pp is this card's render/negative-placement gates.
  Survivors cluster in the pre-existing renderer/suite/devset branch soup — **no survivor covers a line
  this fix added** (`x_placement_of`: 0; `self.placement`: 0 per `tools/mutation_span_check.py`).
  The ~20 lines added *after* the sweep (the `--backend auto` selection record) are not mutmut-scored;
  instead they were hand-mutated in a worktree — `if passed_over:` → `if False:` and
  `len(available) > 1` → `len(available) > 0` — and **both mutants are killed by the two new gates**
  (`.e2e/t_31b3943a-bench-placement/hand_mutants_selection.txt`).

🟢 ACCEPTABLE

* **13 gates** in `tests/test_bench_placement.py` (the card's regression test): 10 RED→GREEN on the
  parent commit `4e1d549` in a separate worktree (`10 failed`, raw `AttributeError`), 2 more RED→GREEN
  on this tree before the selection record landed (`KeyError: 'backend_selection'`), 13/13 green here.
* Full offline suite **874 passed, 39 skipped** (exit 0, `final_full_suite_selection.txt`); coverage
  on the changed bench modules `harness.py 92 %`, `suites.py 99 %` (total 95 %); `ruff check src tests
  tools docs .github` **clean** (this pass also fixed an E501 in `tools/mutation_span_check.py` left by
  the earlier evidence commit).
* The class is covered **without a GPU** in two places: the offline gate (the new test file) and the
  `linux-cpu` CI job (bench with an explicit placement + a bench row against the fake-alloc bundle),
  plus `tools/host_gate_e1c_fit.sh` steps 8–9 for the operator's box; both CI scenarios were rehearsed
  here and both **fail on the parent rev**.
* Additive interface only: `open_model` accepts *more* (a minimal placement), `degrade_ladder`
  normalizes instead of raising, `FitPlan` identity is preserved, the report gains `placement` (and,
  for the single-backend suites, `backend_selection`) plus two rendered lines.

## Decision matrix

**Option A — ship now** (the card's four requirements all measured)
  ✅ fix + 13 gates + CI scenario + its parent-rev control + CPU live run + **host Vulkan run**
  ✅ the `--backend auto` surprise from the host is resolved: documented as contract, made visible in
     the artifact (`backend_selection` + note + rendered line), pinned by a test
  ⚠️ the E2 tables' provenance stays open on AUDIT `t_78f5ea7a`; the mutation sweep's unrun tail (126)
     is recorded, not hidden
  → overall risk: **LOW**

**Option B — hold for more host evidence** (re-run `host_gate_e1c_fit.sh` steps 1–9 end to end)
  ✅ the other host-gate steps (fit OOM, degradation) would get a fresh pass too
  ⚠️ they were already rehearsed and reported; the cost is an operator slot for no new claim about
     *this* card's contract

💡 **Recommendation: Option A.** Every acceptance criterion in the card has a measurement behind it,
including the accelerated one; the two 🟡s are explicitly routed (an audit card, and a bench-hardening
class the mutation sweep already names).

## What could go wrong (only where it changes the decision)

⚡ Scenario: a Vulkan box with the default flags (`--gpu-layers -1`) cannot hold the offload.
  Trigger: the desktop holds most of the VRAM.
  Behaviour now: the ladder materializes "all layers" from the model's layer count, walks
  n → n/2 → 0, and the row reports `degraded: true` + the attempts. Before the fix it raised
  `AttributeError`; with only the crash fixed it would have had *no* retry (`-1` read as "nothing to
  reduce"). The operator's run shows the *easy* half (first rung loads); the hard half is covered by
  the fake-alloc CI step.
  Detectability: HIGH (the placement line + `W_*` warnings are printed/JSON'd).
  Blast radius: one benchmark row (no data loss).

⚡ Scenario: a reader mistakes `--backend auto` on a GPU box for "the benchmark used the GPU".
  Trigger: the box carries both a CPU and a Vulkan bundle (the operator's case).
  Behaviour now: the report carries `backend_selection` and, with more than one local bundle, a note +
  a rendered line naming `--backend vulkan` / `--suite throughput`. Detectability: HIGH.
  Blast radius: a misread table, not a wrong number — and now it is impossible to misread silently.

## Confidence

CONFIDENCE: **9/10**

  + the exact crash is reproduced first-hand on this box (raw tail) and RED→GREEN is proven on the
    parent commit in a separate tree, plus a second RED→GREEN for the follow-on change;
  + the CI scenario and its parent-rev control both measured (the automation really catches it);
  + the live CPU run, the rehearsal and the *host* Vulkan run all produce tables with the placement
    line — requirement 4 included;
  + full offline suite, coverage on every changed module, ruff clean, mutation survivors none on the
    added lines (plus two hand-killed mutants on the post-sweep addition);
  − the host run is a reported measurement (raw JSONs on the operator host), not re-derivable here;
  − the mutation sample's completeness depends on the time-boxed sweep (recorded above).
