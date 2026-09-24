# E3 FIX — the serving path's `engine.backend` (card `t_80f1a4c6`) — QA Report

Date: 2026-09-19 · Tier **M** (the card declares none) · Decision: **ship** — the field that lied
(`engine.backend` on `run`/`ask`/`calibrate`) now carries both the claim *and* the device the
engine's own log proves, on a fake session and on a live row of the pinned bundle; the serving
path's own gates pin the new lines, and no unclassified mutant survives on one.

Commits (private clone `/work/t80serve`, rebased onto the shared tree's `main` at `a475090`):
`8f32620` (RED gates) · `dfac21d` (fix) · `00ecc46` (mutmut pair) · `3c9ff1a` (docs) ·
`2812f34` (unknown-platform gate) · `a644ed0` (survivor pins 8/16/17) · `4f30482` (survivor pins
7/18/20/21/47) · *(the commit that carries the document)* (evidence + raw material).
Published document: `docs/evidence/e3_fix_t_80f1a4c6_serving_backend.md`; raw material:
`.e2e/t_80f1a4c6-serving-backend/`.

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none open.**

🟡 WORTH CONSIDERING (3, all recorded, none blocks)

* **The Tier-M score is a module number, not a change number: 41.8 % killed/scored (631/1508)**
  over `engine/session.py` + `engine/decide.py`. It is dominated by pre-existing internals the
  narrow selection does not drive (`open_model` 192 survivors, `_answer_question` 103, `decide` 98,
  `_score_group` 77 …). The card's own lines measure separately: **0 survivors on the new claim
  symbols** (`backend_claim`, `BackendClaim`, `recorded_backend`, `device_log`), **35/39 killed in
  `device_evidence`**, and every survivor that touches a card line is classified (§6b of the
  document: 4 equivalent in `device_evidence`, 3 pre-existing `placement=` mutants in `open_model`).
  Cost to lift the module number: add the wider engine/bench gate files to the selection
  (`tests/test_engine_fork.py` alone is 2.3 s per mutant on this shared box). Recommendation:
  **defer**; the per-line closure is the honest measure for this change.
* **`engine.fit.backend` still names the fit host's world** (it may fall back to
  `backend_requested`); the fix deliberately does not read it (§5). A reader who wants what computed
  must use `engine.effective_backend`. Cost to change: a fit-cache invalidation. Recommendation:
  separate card (already §7 F3).
* **`serve`/`mcp` are still stubs**; whoever implements them must build responses through
  `cli.decide_payload` (or call `decide.device_evidence`), or the E3 lie returns through a new door
  (§7 F4). Recommendation: note for the E4/`serve` card.

🟢 ACCEPTABLE (no action)

* 20 offline gates + 1 live row; **RED first** on the parent tree (`a475090`: 20 failed, 1 skipped).
* Full offline suite on the rebased tree: **1089 passed, 42 skipped**; ruff clean.
* Changed statements: **60/60 = 100 %** line coverage (whole tree 91 %).
* Live: the `model`-marked gate on the pinned `b11026` bundle (3 passed) **and** the Occamy 1.0
  before/after pair, quoting `Vulkan0 compute buffer size is 763.8125 MiB` next to
  `"backend": "vulkan"`, `"effective_backend": "vulkan"` (the BEFORE row: the same line at
  363.5412 MiB beside `"backend": "cpu"`).
* The claim contract is exercised live in all three source directions (request / bundle / refuted
  `cpu` claim → `W_BACKEND_MISMATCH`).

## Decision matrix

Option A — **ship now** (recommended)
  ✅ every card acceptance criterion is verified with raw material; the fix is additive (new
     fields + one named warning), the response contract for existing readers is unchanged
     (`backend` keeps its value and now carries `backend_source` beside it)
  ⚠️ accepted: the module-level mutation number stays low because the Tier-M selection is the
     card's own gate file; `serve`/`mcp` are not yet covered because they are stubs
  → overall risk: **LOW**

Option B — fix the 🟡 items first (+1–2 h)
  ✅ module mutation number rises; `fit.backend` stops misleading
  ⚠️ touches a cached artifact (fit plans) and unrelated cards' surfaces — a bigger blast radius
     than the change it hardens
  → recommendation: route as separate cards (§7 F3 + the `serve` card)

Option C — fix everything
  ✅ all green on every metric
  → not warranted: none of the 🟡 items is a defect in this change.

💡 **Recommendation: Option A** — the card's acceptance is measured end to end (offline fixtures +
live row + before/after on the card's own model), and the survivors touching the change are either
pinned or classified equivalent with a named reason.

## Confidence: 9/10

Increasing: 20 offline gates RED-first on the parent tree; the live row and the Occamy before/after
on the real bundle; 0 survivors on the new claim symbols; changed-statement coverage 100 %; the
replay tool ships its own control row (the first attempt's seven "kills" were fake and were caught
by exactly that control).

Decreasing: the module-wide Tier-M score is low for reasons outside this change (narrow selection);
two live "gate" rows (the 4B claim-source trio) were measured on the pre-rebase tree and are quoted
as such; the pid-capped box forced three sweep passes instead of one (each one is frozen, documented
and named).

## Accepted risks

* The response still *delivers* an answer when the claim is refuted (by design: `W_BACKEND_MISMATCH`
  warns, exit code stays 0 — unlike a bench row, which is withheld). A consumer that ignores
  `warnings` sees the old behaviour plus a correct `effective_backend`.
* `engine.fit.backend` remains a fit-host field; nothing in the attribution reads it.
