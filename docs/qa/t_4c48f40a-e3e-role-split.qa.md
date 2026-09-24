# E3e — the question's placement + the instructed JSON — QA Report

Card `t_4c48f40a` · Tier **M** (the card carries no `Tier:` line; the sweep is the M-shaped one-run
class, so M is the honest read — and the tier is *not* downgraded: three sweep rounds ran, see below).
Date: 2026-09-20. Decision: **pending the reviewer** (this report is the reviewer's input, not a
self-approval).

## Summary

🟢 **ACCEPTABLE (the card's evidence, defaults unmoved)**

* the card's question is answered with live, paired numbers on the committed 60 items: the **full
  3 × 2 policy table** (`{shipped, two_step, json_instructed} × {answer_sheet, role_split}` plus the
  amendment's two `json_contract=system` arms), one instrument (`--backend vulkan`, `--threads 4`,
  temperature 0, one SHA-checked 4B), every cell paired by item. `json_instructed`/`answer_sheet`
  (+0.150, CI +0.021…+0.279, McNemar p = 0.049) is the only cell that clears the card's own rule, and
  the placement alone moves the collapse (42 → 50/60, `low_mass` 45/60 → 0/60, median coverage
  3.9 % → 99.9 %).
* **no default moved**, and that is checked, not asserted: the six-item freeze probe (the baseline's
  own `--backend auto` recipe) reproduces the committed baseline's **prompt bytes and decisions**,
  and a second probe re-measures the default *cell* under the table's own instrument: **60/60
  decisions identical, `prefix_tokens` identical on every item**, max |Δp| = 1.14e-02. The tool exits
  non-zero if either ever stops holding.
* the refusal story is intact and extended: the value-row verdicts (`refused` / `empty_value` /
  `wrong_field` / `answered`) are gated, and the table reads them per cell (every `json_instructed`
  row is `answered` on 60/60).
* gates: **full suite 1326 passed, 47 skipped, 0 failed**; the four E3e gate files **70 passed,
  3 skipped** (the 3 are the model-marked family checks, `--run-network`) — and **70 failed, 3
  skipped** on the pre-E3e commit `00265ea`, every failure a missing piece of this card
  (`.e3e/logs/red_pre_e3e.txt`). `ruff check src tests tools` **clean**.
* mutation (Tier M, mutmut 3.8, scope `engine/prompt.py` + `engine/cue.py` against the card's three
  gate files): **407/567 killed = 71.8 %**, up from 394 (69.5 %) — and the +13 is *not* score
  chasing: each round followed a named finding from the previous sweep (the two unreached acceptance
  guards, then the inverted `chat_format` guard whose plan-prefix assertion was missing). The card's
  changed function `role_split_render` is **125/155 = 80.6 %**. Every surviving class is named in
  `.e3e/mutmut_summary.txt` (message strings 33, caller-supplied arguments 46, three equivalent
  `role` arms, one unpinned optimisation, `empty_candidate_code` 12 `no tests`, cue.py's
  special-token catalogue 17) — none of them a guard this card added.
* selection coverage of the two swept modules under the card's own gates: **prompt.py 97.8 %**
  (3 missing: the `json.dumps`-object branch, the list branch of `render_instructions`, the
  contract-code map) and **cue.py 98.1 %** (1 missing: the `setdefault` fallback in the special
  catalogue).

🟡 **WORTH CONSIDERING (documented, not blocking)**

* **A table cell that collapses is a real reading, not a bug**: `two_step` × `role_split` measures
  28/60 with 60/60 `low_mass`. The mechanism is in the reports (the label sits *at* the cue row under
  the role split, so `two_step`'s one-token advance reads a non-label row). Consequence: the two
  switches are **not composable**, and any future default must not pair them. Recorded in §9 and in
  the evidence doc; the [host] probe card should re-check this cell first.
* **One item of instrument noise is quantified, not hidden**: E3d's `two_step` cell publishes 47/60
  (old instrument) and measures 46/60 under `--backend vulkan` — so a single decision in sixty can
  sit inside the placement's own noise. Every claim in the table is paired, and the only **wins**
  verdict has a 13-vs-4 discordant split. Cost to fix: none (the drift is measurement).
* **Mutation score is 71.8 % overall, below the 80 % soft reference** — the survivors are classified
  and none is an unpinned behaviour of this card (see the summary; the two files also carry
  pre-E3e code whose callers live in suites outside the selection). Cost to close further: a wider
  test selection (adding `tests/test_engine_fork.py` + the CLI suites) would lift it, at the price of
  a much slower sweep on this shared box — deferred, named in the summary as the widening a later
  card can take.
* **The [host] half of the card's item 2 is routed, not claimed**: Tiel/Occamy (≥21 GB) against this
  box's 8 GiB cgroup. The Tiel reading from `t_7c926398` (22/60, 57/60 `low_mass`, 59/60 refusals;
  `--cue two_step` inert) is what makes the placement + instructed-JSON levers the interesting ones,
  and the evidence doc says what that run needs first (Tiel's template is outside the internal
  renderer's subset → the built-in bridge, or `--template plain` as the documented fallback).

## Decision Matrix

```
DECISION: ship the table as evidence, or move a default?

Option A: Ship the table, keep both switches off (this card's scope).  ← recommended
  ✅ every published row stays byte-frozen and provably so (two probes + the pin files)
  ✅ the card's answer is a documented recommendation, not a silent policy change
  ⚠️ the recommendation needs a follow-up card to become a default (it costs new instrumentation
     numbers and invalidates the tables the evidence doc lists — that is the price, stated)
  → Overall risk: LOW

Option B: Adopt json_instructed now, re-publish in the same card
  ✅ the strongest cell becomes the default
  ⚠️ the card's item 4 forbids it; every published quality row would need re-measurement under the
     new prompt while this box holds only the 4B; and the [host] families are unmeasured
  → Overall risk: MEDIUM (unmeasured re-publication on two of five families)

Option C: Hold the table until the [host] Tiel/Occamy probe lands
  ✅ the recommendation gains the family it was aimed at
  ⚠️ blocks a card whose container half is complete and whose acceptance item 2 is explicitly
     two-stage ("4B first; then Tiel on the host")
  → Overall risk: LOW, but it stalls the queue; the probe is routed as its own card instead

Recommendation: Option A — the table, the freeze and the recommendation are the card's deliverables,
and the follow-ups (a default-change card; the [host] probe) are cards, not this card's claims.
```

## Confidence

```
CONFIDENCE: 8/10

Increasing:
  + all four gate files RED on the pre-E3e commit, GREEN here (70/70, same collected count)
  + the default's freeze is checked twice (bytes + decisions, and a full 60-item re-score)
  + the table's cells are one flag apart from each other, same items, same box, one instrument
  + the two real mutation gaps the sweep exposed were found, pinned and verified by re-running
  + full suite green (1326), lint clean, every number in the docs spliced from the tool's report

Decreasing:
  - the [host] families are unmeasured here (routed), so the recommendation's reach is 4B-evidence
  - the 60-item unit makes a single cell's interval ~22 points wide: per-type cells are leads only
  - two survivors are an internal optimisation the gates do not discriminate (named, unpatched)
```

## Artifacts

* `docs/evidence/e3e_roles_decision.md` / `.json` — the tool's own report (table, freeze, re-score)
* `docs/evidence/e3e_role_split_t_4c48f40a.md` — the narrative evidence (spliced numbers)
* `.e3e/mutmut_summary.txt` — the three-round sweep record with the survivor classes
* `.e3e/logs/red_pre_e3e.txt` — the RED gate log on the pre-E3e tree
* `.e3e/bench_*.json` + `.e3e/logs/bench_*.log` — the eight arms
