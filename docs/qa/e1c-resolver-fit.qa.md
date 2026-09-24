# E1c — reasoning resolver + fit — QA Report (Tier M)

Date: 2026-09-17 | Card `t_c8e36cad` | Head: the E1c commits on `main` (no remote; local commits)
Report tier: **M** (the card declares none; default M) — Risk-Weighted Summary + Decision Matrix +
Confidence. Full gate table: `docs/evidence/e1c_t_c8e36cad_resolver_fit.md`.

## Gate results (what a reviewer can re-run)

| gate | result |
|---|---|
| offline unit gate | `uv run pytest -q` → **666 passed, 33 skipped** (clean env) |
| offline unit gate, pre-pin tree | `uv run pytest -q` → 590 passed, 33 skipped (the tree the first sweep measured) |
| live suite (fit + templates + engine + CLI, `--run-network`) | **105 passed in 302.86 s** (at `324324c`); re-run at the final head → see §"Phase-5 verify" below |
| network-disabled gate (A-E1c-10) | `tools/e1c_offline_gate.py` → **218 passed, 21 skipped** (exit 0); the pre-pin tree ran 142/21 |
| linter | `uv run ruff check src tests tools` → **All checks passed** |
| oracle | unchanged by this card (E1a/E1b sections still green; run in the live suite) |
| A-E1c-1 … A-E1c-10 | every criterion has a named, re-runnable gate — see evidence §1 |
| coverage of the two new modules (E1c test selection + pins, offline) | `engine/template.py` **84 %**, `runtime/fit.py` **97 %** line (pre-pin: 81 % / 90 %); `engine/decide.py` 95 %, `engine/prompt.py` 95 %, `schema.py` 84 % |
| mutation testing (Tier M, soft) | two scoped sweeps (r1 60.9 % → r2 69.4 % killed/ran) with a per-key replay proof; result block below |

## Risk-Weighted Summary

🔴 **REQUIRES ATTENTION (blocks ship):**

*None.* Every acceptance criterion has a gate that passed on this box; no criterion is
unverified.

🟡 **WORTH CONSIDERING (your call):**

1. **`coverage` reports `low_mass` on every realistic run of the pinned model.**
   Risk: a caller who treats `reliability` as "the engine is unsure" will under-use the engine;
   the distributions themselves are decisive (e.g. `queue: billing 0.92`, `decision:
   ship-migration 0.98`).
   Cause (measured, `docs/TEMPLATES.md` §4): the model puts ~94 % of its next-token mass on a
   newline after any instruction line, so the candidate mass at the cue is 0.3–3 %.
   Cost to fix: a prompt/readout change (blank-line cue measured at ~1.4× the mass, or a two-step
   readout) plus re-pinning the E2E numbers — a half-day, and it needs the labeled set to prove it
   helps.
   Recommendation: **defer to E2** (quality suite) — it is a quality question, not a correctness
   one; the numbers are recorded so E2 can start from them.
2. **The `k2-horizon` policy row is documented from published templates, not measured here**
   (no GGUF of that family on this box), and its `/no_think` marker is *advisory* (that family's
   thinking is controlled by the serving stack).
   Risk: a user running a K2-family GGUF gets the strip guarantee but no model-endorsed marker.
   Cost to fix: needs the model file (multi-GB pull) — out of scope for E1c.
   Recommendation: acceptable, tagged `[recon]`/`[UNVERIFIED]` in `docs/TEMPLATES.md`; the chain
   still resolves it (llama.cpp also ships a `kimi-k2` built-in for step 2).
3. **One pre-existing gate is environment-sensitive** — `test_runtime_contract.py::test_oracle_live_section_is_green_without_skips`
   runs the oracle with the runtime forced, and the oracle's live section *skips* two model probes
   when the pinned Spark GGUF is not under `$HOME`, so the "no SKIP" assertion fails.
   Measured matrix at the final head:
   `$HOME` with the model + `GGUFONE_RUNTIME_DIR` set → **green**;
   `$HOME` without the model + the var set → this one test red (2 documented SKIPs, the pin
   download-verify and the header-parse re-run);
   `$HOME` without the model + no var → green (666 passed / 33 skipped, the canonical gate).
   Risk: a sandbox run whose `$HOME` has no models shows one red test unrelated to any card.
   Cost to fix: the test would have to skip when the model is absent (it is the *runtime* gate, so
   weakening it is a call for its owner) — recorded, not touched by E1c.
   **Fixed here instead:** the two `test_runtime_fallback.py` cases (`test_find_runtime_prefers_the_recorded_variant`,
   `test_find_runtime_falls_back_to_a_scan_when_the_record_is_stale`) *were* an env-hygiene bug —
   they exercise `home=`-based discovery while an ambient `GGUFONE_RUNTIME_DIR` legitimately wins
   over it, so they now clear the variable with the file's own `monkeypatch.delenv` idiom (same
   lines already used by `test_doctor_reports_the_working_backend_and_the_recorded_fallback`).
   Verified: with the variable set and the models present the suite is **667 passed, 32 skipped,
   0 failed** (was 2 failed). No production code touched.

🟢 **ACCEPTABLE (no action needed):**

* The chain resolves the pinned models through **step 1** (their own GGUF template) with no
  fallback warning; the fallback path is test-exercised, not production-exercised.
* Fork equivalence is `0.000e+00` on **both** families at this head — with the chat template and
  the fit plan in the path.
* The fit estimate cross-checks against measured load RSS at **−9.3 % / +2.3 %** (target ±20 %).
* `llama-fit-params` agrees with the GGUF tensor index to 212 KiB on the pinned model.
* No new error/warning codes were invented: the response uses only the frozen catalog.
* The `llama_chat_apply_template` signature bug found here was fixed **and pinned live**.

## DECISION: ship or fix?

**Option A — ship E1c now** (recommended)
  ✅ all 10 acceptance criteria gated and green; offline + live + network-disabled gates green;
  ✅ the two new modules are 81 % / 90 % line-covered by the offline E1c gates (the runtime
  branches are covered by the `model`-marked half, which the live suite runs);
  ✅ the template/fit surface is observable in every response (`engine.template`, `engine.fit`,
  `engine.kv_type`), so a caller can tell which path ran;
  ⚠️ accepted: `low_mass` on 10 of 11 E2E questions (diagnostic, not an error — E2's call);
  ⚠️ accepted: `qwen35moe` / `k2-horizon` rows are documented, not measured here.
  → Overall risk: **LOW**

**Option B — fix the E2E readout policy first** (+0.5–1 day)
  ✅ could lift the candidate mass ~1.4× (measured) and re-sharpen the distributions;
  ⚠️ re-pins every E2E number without a labeled set proving agreement improves;
  ⚠️ delays E2, which is the milestone that can actually measure it.
  → Overall risk: LOW, cost: one day of re-pinning.

**Option C — pull a MoE/K2 GGUF and measure those rows too** (+hours of download, no GPU)
  ✅ removes the last `[UNVERIFIED]` doc rows; ⚠️ nothing in the E1c gate set depends on it.
  → Overall risk: LOW, cost: high.

💡 **Recommendation: Option A.** The card's job was the resolver + the fit plan, both are gated
with executed evidence, and the one open quality question is explicitly assigned to E2 by the
SPEC — carrying it forward with measurements is better than guessing at it now.

## Mutation testing (Tier M — one run, soft threshold)

Runner: mutmut 3.8 via `tools/mutmut_driver.py` (container xattr workaround). Scope: the two new
modules, with the E1c gates as the test selection (now including `tests/test_e1c_mutation_pins.py`).

| round | tree | module | mutants | killed | survived | no-tests | score (killed/ran) |
|---|---|---|---|---|---|---|---|
| r1 | E1c gates | `engine/template.py` | 2465 | 1219 | 961 | 276 | 55.7 % |
| r1 | | `runtime/fit.py` | 1209 | 831 | 348 | 30 | 70.5 % |
| r2 | + 76 pins | `engine/template.py` | 2465 | 1409 | 840 | 208 | **62.4 %** |
| r2 | | `runtime/fit.py` | 1209 | 974 | 205 | 30 | **82.6 %** |

Combined killed/ran: **60.9 % → 69.4 %**. The pin round was driven by the r1 survivor list:
`Resolution.to_dict()` provenance, the fit plan's exact-budget boundary and KV ladder, the cache
guards, the GGUF array walk, the suppression mode strings.

**Harness proof (three independent checks, evidence §6.2):** a killed key reproduces through the
mutant tree (`xǁModelFactsǁread__mutmut_1` → the named test fails; no-mutant control green);
32 r1-survivor keys replayed one by one with `MUTANT_UNDER_TEST` → 27 fail (killed), 5 survive;
and those 32 verdicts agree with the untouched r2 `.meta` on every key.

**Residual survivors (r2):** dominated by the internal Jinja-subset parser/renderer
(`ExprParser.parse_comparison` 87, `_scan` 54, `LoopState.attr` 51) and the GGUF header reader
(`ModelFacts.read` 42, `read_tensor_index` 31). Class table + the argument for leaving them
(default-argument mutants, initialisers, identical-output branch flips, message text, metadata
types no fixture contains) is evidence §6.3; the 238 `no-tests` mutants are reported as the
coverage gap they are.

The first replay pass produced **two false kills** from a flaky assertion in the pin file (two
RSS reads compared for equality); it was found by running the pins under `coverage run`, fixed
(bounded Δ), and the affected keys re-verified — they survive, matching the sweep. Recorded
because a mutation number is only worth what its harness proof is worth.

## Confidence

📈 **CONFIDENCE: 8/10**

Increasing:
  + every A-E1c-* criterion is a named test, and the live ones ran against the pinned model;
  + the two riskiest claims (thinking suppression, RSS estimate) are measured on real bytes/rows;
  + the response tells the caller which chain step and which fit source ran;
  + the ABI bug class that bit E1b (wrong ctypes signature) was found again and now has a live pin;
  + the mutation round added 76 pins that kill 27 of the replayed r1 survivors, and the harness
    itself was proven by replay in both directions (a kill reproduces, a pin kills).

Decreasing:
  - mutation survivors in the renderer's *internal* implementation (a parser is a large surface
    with a small behavioural contract) — the E1c gates pin the contract, not every branch, so
    `template.py` stays at 62 % killed/ran by construction (the card's contract is the chain +
    suppression + provenance, all pinned);
  - `qwen35moe` / `k2-horizon` have no live run on this box;
  - no GPU on this box: placement (`n_gpu_layers`) is unit-tested, not measured.

## Phase-5 verify (fresh run at the final head)

Run on the committed tree, in this order, with the pinned runtime exported:

```
$ uv run pytest -q                      → 666 passed, 33 skipped in 24.09s
$ uv run ruff check src tests tools     → All checks passed!
$ uv run python tools/e1c_offline_gate.py  (network disabled)
                                        → 218 passed, 21 skipped, exit 0
$ uv run pytest -q --run-network -s tests/test_fit_live.py tests/test_templates.py \
      tests/test_engine_fork.py tests/test_ctypes_binding.py tests/test_cli.py
                                        → 106 passed in 383.60s
                                          (.e2e/t_c8e36cad-e1c/logs/live_all_2.log)
$ uv run pytest -q tests/test_e1c_mutation_pins.py    # 3x, flake check: 76 passed each time
```

Environment matrix of the full offline gate (why the two run modes above are the honest ones):

| `$HOME` has the pinned model | `GGUFONE_RUNTIME_DIR` | result |
|---|---|---|
| no | unset | **666 passed, 33 skipped** (canonical) |
| yes | set | **667 passed, 32 skipped, 0 failed** (`HOME=/var/home/rybens`) |
| no | set | 1 failed — the oracle's *model* probes skip, see 🟡-3 |

No flaky test remains in the E1c surface: the one found (two RSS reads compared for equality)
was fixed and re-run three times.
