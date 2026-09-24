# E3e audit F1-F4 (card `t_c8c76cc8`) — QA Report

Date: 2026-09-20 | Tier: **M** (the card declares none; the profile default). Report format: M
(Risk-Weighted Summary + Decision Matrix + Confidence).

## What changed

Four findings from the E3e audit (`t_57bd3db2`) applied to the E3e instrument and its documents:
the freeze check's compared field set (+`cue.refused`, F1), `decide()`'s fallback verdict wording
(F2), the two-Wilson-conventions note (F3), the instrument line's real command (F4). No default
moved, no arm re-measured, no number recomputed: the regenerated docs are the same measurements.

## 🟢 ACCEPTABLE (no action needed)

* **F1 ships with its expected consequence, and the consequence is the point.** The placement line
  now reads **59/60 decisions identical, `n16` named** (`refused` true in the committed baseline,
  false in the re-score; the answer `no`/`low_mass` did not move). `bash .e3e/report.sh` therefore
  exits **4** — the tool's own "the table's own baseline is not the committed one" condition, which
  §8.1 asked for. The report and JSON are written before the exit; the splice exits 0. A reader
  following the doc's Reproduce line sees a non-zero exit *by design* and the document says so.
* **Zero numbers move**: the regenerated JSON differs from the committed one in 14 fields, all of
  them the four intended classes (caveats +1 line; 4 verdict strings; the placement block; the
  `numeric_only` list losing the item that became a difference). Nothing under `cells`/`pairs`.
  The ranking and the single `wins` cell (`json_instructed/answer_sheet`, 51/60, p = 0.049) are
  identical — the decision-rule outcome is unchanged, explicitly.
* **Regeneration is a fixed point**: two consecutive `report.sh` + `splice_docs.py` runs leave the
  four documents byte-identical (`e6eab840…d8a4`), and the F4/F1 prose edits sit outside the
  `E3E-TABLE` markers, so the splice cannot eat them.
* **Gates**: the four E3e gate files **76 passed, 3 skipped** (+6 tests); the full offline suite
  **1332 passed, 47 skipped**; `ruff check src tests tools` clean; tool coverage under its own gate
  file **82 %** (58 missed: `main()`'s CLI wiring and render branches the gate file never drives).
* **Tier-M sweep, two honest rounds** on the file the findings live in (mutmut 3.8, 1941 mutants,
  ~3.3 min/run, verdicts from `mutants/tools/e3e_roles_decision.py.meta`):
  round 1 **1404/1941 = 72.3 %**, round 2 (after the pin round 1 named) **1412/1941 = 72.7 %**.
  The changed surface is **fully killed** in both rounds' relevant hunks (F1 22/22; F2 12/12 → 14/14).
* **Hand-mutation replay** of the round-1 survivor: the `label`-key rename makes exactly the new
  render test fail, then the file is restored byte-identical. The pin is load-bearing, not decorative.

## 🟡 WORTH CONSIDERING (your call)

* **The sweep's 72.7 % is a selection number, not the tool's.** 1941 mutants cover `render`,
  `main`, the per-type cells and the report tables, which this card did not touch and whose gates
  live in other files (`tests/test_e3e_docs.py`, the CLI suites). Everything on the changed surface
  is killed; the untouched surface is out of scope for this card's sweep. Recommendation: accept.
* **F4's residual in `.e3e/run_arms.sh`.** The script's header comment and the `campaign.log` echo
  it writes still say `runs: 1`; §8.3 scoped the fix to the documentation line, and `campaign.log`
  is a historical receipt, so both were left. Recommendation: accept, or let the auditor widen F4
  to the echo in a follow-up (one line, changes what the *next* campaign log claims).
* **The rule's *name* still says "everything else is not by more than the CI noise"** (the tool's
  `rule` string + `decide()`'s docstring). §8.2 explicitly allows keeping it ("the label is what the
  reader meets next to the numbers"), and after the patch the phrase is only printed on the pairs it
  is true for. Recommendation: accept as-is; a re-word would change the record's `rule` field, which
  is a card-level policy statement, not a verdict.
* **`tools/e3d_cue_decision.py` carries the same phrase** in its docstring (line 22) for the E3d
  rule. Out of this card's scope (different instrument, its own documents). Recommendation: a note
  for the auditor, not a change here.

## 🤔 DECISION

* **Option A — ship as one change-set**: patches verbatim (§8.1/§8.2/§8.3 token-for-token, one
  100-char re-wrap of one literal, named), docs regenerated and byte-reproducible, rule outcome
  unchanged, six new pins, sweep rounds recorded. Accepted risks: the 🟡 items above.
  → Overall risk: **LOW**.
* **Option B — widen F4 to the shell echo / re-word the rule name / sweep the wider surface**: more
  surface, more risk of drifting from "apply the auditor's patches verbatim", for no number or
  claim that is wrong today. → Overall risk: lower on F4's residual, higher on scope discipline.
* 💡 **Recommendation: Option A.** Every finding the audit raised is applied or explicitly recorded,
  the one behavioural change (F1) is the audit's intended consequence, and the only numbers that
  moved did so inside the generated text, not in any measurement.

## 📈 CONFIDENCE: 9/10

Increasing: patches verified verbatim (token-level check against the scorecard); the F1 outcome
recomputed from the raw rows of both reports; the JSON diff is a field-level enumeration, not a
spot check; regeneration is idempotent (hash-compared); the changed surface is fully killed by the
sweep and a hand-mutation replay proves the pin; the full suite is green on the final tree.

Decreasing: the sweep's 72.7 % carries the untouched surface (named above, not this card's gates);
the F3 caveat literal's casing survives mutation (prose — accepted); `report.sh`'s new exit 4 will
surprise a literal reader of a `&&`-chained recipe until they read the paragraph that says so.
