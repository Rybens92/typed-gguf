# E3e CI reading — auditor scorecard (card `t_57bd3db2`, audit of `t_4c48f40a`)

**Verdict (TL;DR).**
* **The table: CONFIRMED.** All eight cells, all 28 paired comparisons (discordant counts, the
  Wald risk-difference intervals, the exact McNemar p-values, the win flags), the ranking, the
  decision verdicts and both freeze probes reproduce *exactly* from the raw arm reports under an
  independent implementation I wrote for this card — **655 checks, zero disagreements**, down to
  the last digit of every probability and coverage value. The record regenerates **byte-for-byte**
  (`docs/evidence/e3e/report.sh` + `docs/evidence/e3e/splice_docs.py` into a fresh copy of `4ef74f6`), so the §9 splice in
  `docs/BENCHMARKS.md` and the evidence document carry the tool's own output, not a transcription.
  The RED claim is confirmed **by my own execution**: the four gate files on a fresh copy of the
  pre-E3e commit `00265ea` → **70 failed, 3 skipped**; on HEAD → **70 passed, 3 skipped**.
* **The freeze claim ("defaults provably frozen"): CONFIRMED-WITH-NOTES.** Prompt bytes
  (`prefix_tokens`) identical item by item, answers identical 60/60 (and 6/6 for the probe), no
  value in `schema.OPTION_DEFAULTS` moved (the diff is two *added* keys with the frozen defaults).
  One note: the compared "decision" is `got` + `reliability` only; the **cue-refusal verdict is
  outside that field set and it moved on one item** — `n16`, `refused` true in the committed
  baseline, false in the re-score (a 0.4476/0.4536 near-tie at the cue row). See F1.
* **The recommendation: CONFIRMED-WITH-NOTES.** Every number it quotes is exact, and the
  conclusion ("the defensible change is the instructed contract with the answer-sheet placement")
  follows from the table as read. The notes are about evidence *strength*, not correctness: the
  winner clears the 0.05 gate by **0.0010 of p** and one item moved flips it (13/4 → 12/5 → p =
  0.1435, CI includes zero); the "lands on the number E3d measured (51/60)" parallel is a
  *marginal* parallel (E3d's paired split was 12 vs 3, p = 0.035 — the E3e cell's is 13 vs 4,
  p = 0.049); and the "two 35B-A3B probes predict this" pointer is outside this card's artifacts
  (not re-run here). Every number the recommendation states is verified in §7 below.

Card that settles it, verbatim: *"a challenger wins only when the paired difference's 95 % interval
excludes zero **and** the exact two-sided McNemar p < 0.05; ties are called identical, everything
else is not by more than the CI noise"* — and *"Do NOT change the product's defaults; this card is
a reading, not a policy change."* Defaults unmoved: confirmed (§3). No policy change proposed.

**Per-item verdicts on the card's six asks:**

| # | ask | verdict |
|---|---|---|
| 1 | the eight cells' counts, with my own implementation | **CONFIRMED** — every field of all eight cells matched to full precision; 655 checks, 0 disagreements (§1) |
| 2 | the paired statistics, recomputed seed-free | **CONFIRMED** — all 28 pairs exact; the exact-DP bootstrap (E3d's convention, no seed) re-derived and calibrated on E3d's own cell (§2). One convention note, no number wrong (F3) |
| 3 | the freeze claim (bytes / decisions / defaults) | **CONFIRMED-WITH-NOTES** — `prefix_tokens` identical item by item, answers 60/60 and 6/6, no default moved; the compared "decision" excludes the cue **refusal verdict**, which moved on one item (`n16`) — F1 (§3) |
| 4 | the interaction finding, from the raw rows | **CONFIRMED** — the label sits at the cue row; the one-token advance reads past it; all receipts from raw rows + offline per-family renders (§4) |
| 5 | the RED proof, by execution | **CONFIRMED by execution** — 70 failed, 3 skipped at `00265ea`; 70 passed, 3 skipped on HEAD (§5) |
| 6 | the recommendation, incl. its comparability caution | **CONFIRMED-WITH-NOTES** — every number it quotes is exact; the notes are strength-of-evidence (one-item fragility; the E3d parallel is marginal-level; the 35B pointer is outside this card) — §7 |
| — | **the table (overall)** | **CONFIRMED** |
| — | **the recommendation (overall)** | **CONFIRMED-WITH-NOTES** |

---

## 0. Scope and method

* Sources read (read-only): `docs/evidence/e3e/bench_*.json` (8 arms, 60 items each), `docs/evidence/e3e/probe_default.json`,
  `docs/evidence/e3d/bench_templated_shipped.json`, `docs/evidence/e3e_roles_decision.{json,md}`,
  `docs/evidence/e3e_role_split_t_4c48f40a.md`, `docs/BENCHMARKS.md` §9, `src/ggufone/bench/devset.jsonl`,
  `tools/e3e_roles_decision.py` (read for semantics, not imported), `src/ggufone/engine/{cue,decide,prompt}.py`,
  `src/ggufone/bench/{harness,suites}.py`, `docs/evidence/e3e/logs/*`.
* My implementation: `scripts/recompute_e3e.py` (pure stdlib, no repo imports; Wilson, exact McNemar
  via both a comb sum and a regularized incomplete beta for cross-check, Wald interval, an exact-DP
  bootstrap of the paired difference, the freeze comparison, the dev-set identity check).
* Re-runs executed for this card: the RED/GREEN gate runs (§5), the full regeneration chain (§6),
  the offline per-family role render for three cue shapes (§4). The 60-item arms themselves were
  **not** re-run: they are the card's inputs and cost hours on a >16 GiB host; §2 shows that every
  number in them is internally consistent and reproduces.

## 1. The eight cells — my reading beside the record's (all exact)

`correct/n`, Wilson 95 %, per type `choice/noul/score`, `low_mass`, refusals, cue verdicts, coverage
p50, prefix-token range. "my" and "record" are printed **only where they could differ** — they did
not: every field below matched to the record's full precision (`scripts/recompute_e3e.py`, 655 checks).

| cell | correct | Wilson 95 % | choice | noul | score | low_mass | refusals | cue verdicts | coverage p50 | prefix |
|---|---|---|---|---|---|---|---|---|---|---|
| `shipped/answer_sheet` | 42/60 = 0.700000 | 0.574912920531–0.801018338612 | 20/24 | 16/18 | 6/18 | 45 | 0 | — | 0.038815447429384 | 82–119 |
| `shipped/role_split` | 50/60 = 0.833333 | 0.719683868364–0.906868230208 | 22/24 | 17/18 | 11/18 | 0 | 0 | — | 0.998837653709035 | 78–115 |
| `two_step/answer_sheet` | 46/60 = 0.766667 | 0.645637296224–0.855604382634 | 21/24 | 14/18 | 11/18 | 2 | 2 | — | 0.972770011355850 | 82–119 |
| `two_step/role_split` | 28/60 = 0.466667 | 0.346279060414–0.591065729728 | 9/24 | 14/18 | 5/18 | 60 | 40 | — | 1.434034412107113e-09 | 78–115 |
| `json_instructed/answer_sheet` | 51/60 = 0.850000 | 0.738854109302–0.919025594198 | 21/24 | 17/18 | 13/18 | 0 | 0 | answered 60 | 0.999812585443008 | 86–123 |
| `json_instructed/role_split` | 50/60 = 0.833333 | 0.719683868364–0.906868230208 | 21/24 | 18/18 | 11/18 | 0 | 0 | answered 60 | 0.999982301865968 | 82–119 |
| `json_instructed/answer_sheet/system` | 48/60 = 0.800000 | 0.682181941944–0.881714946771 | 21/24 | 17/18 | 10/18 | 0 | 0 | answered 60 | 0.999774864542471 | 131–168 |
| `json_instructed/role_split/system` | 50/60 = 0.833333 | 0.719683868364–0.906868230208 | 21/24 | 17/18 | 12/18 | 0 | 0 | answered 60 | 0.999982324468974 | 127–164 |

Also verified against the rows (not just the record): each arm's own `overall` and `per_type` blocks
are exactly the count/agreement its rows imply, item order matches the committed dev set (60 ids,
`choice` 24 / `noul` 18 / `score` 18), and every row's `expected`/`correct` follows the dev set's gold
under the `True/False → yes/no` wire mapping. The record's `devset` block (60 items, 60 measured, the
committed path) is accurate.

**Observation (F3).** The arms' *own* stored `overall`/`per_type` CI blocks were computed by
`harness.wilson_interval` with `z = 1.96` exactly, while the decision tool's cells use
`Z = 1.959963984540054`; the two conventions differ in the 6th decimal at n = 60 (e.g.
`shipped/answer_sheet` ci[0] = **0.574910530336** in the arm file vs **0.574912920531** in the
record — both correct under their own convention, and the record recomputes from the rows, so no
published number depends on it). Receipts: `logs/wilson_conventions.txt`.

## 2. The paired statistics — all 28 pairs, recomputed seed-free

Every one of the 28 pairs in the record reproduces exactly: discordant counts (`challenger_only`,
`baseline_only`, `both`, `neither`), `difference`, the Wald interval on the discordant counts, the
exact two-sided McNemar p, and the `challenger_wins` flag (1e-15 tolerance; no disagreement).
The card's seven verdicts against the baseline and the ranking also reproduce.

Per the card's item 2 I also computed the **exact-DP bootstrap** of the paired difference (E3d's
convention: the resampling distribution of `(B*−C*)/n` computed analytically by DP over the four
cells of the 2×2 — **no seed at all**), and it agrees with E3d's published number on E3d's own cell
(`two_step`: `P(diff ≤ 0) = 0.1284`, exactly the E3d scorecard's value), which calibrates the
convention. The record uses the closed-form Wald interval instead — declared in the tool and a
legitimate choice, since every number is then read directly from the reports.

| vs `shipped/answer_sheet` | only challenger | only baseline | diff | Wald 95 % | exact McNemar p | exact-DP 95 % (my run) | P(diff ≤ 0) | verdict |
|---|---|---|---|---|---|---|---|---|
| `shipped/role_split` | 12 | 4 | +0.133333 | +0.007..+0.260 | 0.076812744 | +0.017..+0.267 | 0.0249 | not a win |
| `two_step/answer_sheet` | 7 | 3 | +0.066667 | −0.035..+0.169 | 0.343750000 | −0.033..+0.167 | 0.1284 | not a win |
| `two_step/role_split` | 5 | 19 | −0.233333 | −0.382..−0.085 | 0.006610751 | −0.383..−0.083 | 0.9993 | not a win |
| `json_instructed/answer_sheet` | 13 | 4 | +0.150000 | +0.021..+0.279 | 0.049041748 | +0.017..+0.283 | 0.0150 | **wins** |
| `json_instructed/role_split` | 11 | 3 | +0.133333 | +0.016..+0.251 | 0.057373047 | +0.017..+0.250 | 0.0168 | not a win |
| `json_instructed/answer_sheet/system` | 10 | 4 | +0.100000 | −0.020..+0.220 | 0.179565430 | −0.017..+0.217 | 0.0646 | not a win |
| `json_instructed/role_split/system` | 11 | 3 | +0.133333 | +0.016..+0.251 | 0.057373047 | +0.017..+0.250 | 0.0168 | not a win |

Two things worth carrying forward:

* **A naming note on the audit card itself.** The card's why-section quotes this cell as
  "`two_step/role_split`: 28/60, −0.367, p = 0.007". That −0.367 is the **marginal** difference
  (28/60 − 50/60), not the table's **paired** difference (−0.233, from 5 vs 19 discordant items);
  p = 0.007 and the direction match. Both numbers are correct as their own quantity — the rule uses
  the paired one, which is why the table publishes −0.233. Named here so the two quantities are not
  read as a disagreement later.
* **The rule's outcome is interval-convention-robust.** Under the exact-DP bootstrap three more
  cells (`shipped/role_split`, and both `json_instructed/role_split` variants) have an interval that
  excludes zero, but their **exact two-sided McNemar p-values** — the convention-free leg of the rule
  — stay above 0.05 (0.077, 0.057, 0.057), so the conjunction still selects only
  `json_instructed/answer_sheet`. (For the record: those cells' one-sided bootstrap tail masses are
  0.0249 / 0.0168 / 0.0168 — if one ever read the interval's tail mass as *the* test at a one-sided
  0.025 bar, three cells would clear it. The card's rule is not that, and the table does not use it;
  recorded so nobody swaps the test silently.)
* **The winner is one item wide** (`scripts/fragility.py`): 13 vs 4 → p = 0.049042 (margin 0.0010
  below the gate); move one item → 12 vs 5 → p = 0.1435 and the CI crosses zero
  (−0.015..+0.248). Move one item *the other way* → 14 vs 3 → p = 0.0127. This is the same
  one-item width the E3d audit named for `json_field`; it does not refute the verdict, it bounds it.

## 3. The freeze claim — checked field by field

Against the committed baseline `docs/evidence/e3d/bench_templated_shipped.json` (t_6de5fc53, `--backend auto`):

* **the six-item probe** (`docs/evidence/e3e/probe_default.json`, the baseline's own `--backend auto` recipe —
  its `backend_selection` block reads `requested: auto, selected: cpu`, and the probe log carries the
  CPU compute rows): **6/6 items same `got`, same `reliability`, `prefix_tokens` identical item by
  item**; numbers re-scored (max |Δp| = 3.783e-02, max |Δcoverage| = 1.066e-01 over all six items);
  `frozen: true`, `bit_frozen: false`, tolerance 1e-9 — the record's block, recomputed exactly.
* **the placement probe** (`docs/evidence/e3e/bench_shipped_answer_sheet.json`, the same cell re-measured under
  the table's own instrument, `--backend vulkan`): **60/60 items same `got`, same `reliability`,
  `prefix_tokens` identical on every item**; max |Δp| = 1.135e-02, max |Δcoverage| = 6.246e-03 over
  the tolerance 5e-3, `numeric_only` = the 8 items the record names (`n08 n11 n14 n16 n17 s01 s05 s10`).
  This is the load-bearing check: the table's baseline cell is the committed one item by item, so
  the eight arms really are one measurement with one policy flag apart.
* **the defaults**: `git show 00265ea:src/ggufone/schema.py` vs HEAD — `OPTION_DEFAULTS` gained the
  two documented keys `chat_format: ANSWER_SHEET` and `json_contract: JSON_CONTRACT`
  (= `"answer_sheet"`, `"question"`); **no existing value changed** (`cue` still `"shipped"`).
  The options are switches; nothing moved.
* **Note (F1)** — the identity the tool checks is `got` + `reliability` (+ `prefix_tokens`), and the
  **cue-refusal verdict is not in that field set**. On this very comparison it moved on one item:

      n16: refused False != True (token 198/1, mass 0.453601/0.447562, closer None/'<|end_of_sentence|>')
           probe: got='no' rel='low_mass' cov=4.819261e-02
           base : got='no' rel='low_mass' cov=4.618079e-02

  The answer did not move (both `no`, both `low_mass`), and the two masses are a near-tie
  (0.4476 vs 0.4536) inside the drift band the document quantifies — but the *refusal
  classification* of that row is published (the report's cue-verdict table, and the `W_CUE_REFUSED`
  family of claims in §8) and it is exactly the classification E3d's constraint 4 protects. The
  evidence doc's "no decision moved" is true for answers and silent on this. Receipt:
  `logs/freeze_fields_diff.txt`.

## 4. The interaction finding — reproduced from the raw rows, not from prose

The claim (evidence doc): under the role split the **label sits at the cue row**, so `two_step`'s
one-token advance reads the row *after* the answer, "a row that holds ~1.0 of its mass on a single
non-label token (`c01`: token 79152) with 0 coverage".

My reading of the four arms, item by item (`scripts/interaction_probe.py`):

| arm | c01 cue row / decision row | top token | mass | coverage | got | verdict |
|---|---|---|---|---|---|---|
| `shipped/answer_sheet` | cue row (= decision row) | 198 | 0.961105 | 0.0107 | `billing` ✗ | low_mass |
| `shipped/role_split` | cue row (= decision row) | **97** | 0.901356 | **0.9994** | `technical` ✓ | ok |
| `two_step/answer_sheet` | decision row (one past the cue) | **97** | 0.862210 | **0.9484** | `technical` ✓ | ok |
| `two_step/role_split` | decision row (one past the cue) | **79152** | 0.999994 | **7.6e-13** | `billing` ✗ | low_mass |

* The **label sits at the cue row under the role split**: `shipped/role_split` reads the row it
  lands on — the first assistant row — and gets coverage 99.94 % with the gold label; this is what
  "the question is not inside the assistant turn any more" means in numbers (0 refusals, 50/60,
  `low_mass` 0/60 across the arm).
* **The one-token advance reads past it**: `two_step/role_split`'s decision row is a single
  non-label token at ~1.0 mass with coverage ~1e-13. For `c01` the same token id **97** is the label
  row in both placements (`shipped/role_split`'s cue row, `two_step/answer_sheet`'s decision row),
  which is what makes the derivation checkable: the cue row's argmax mass (0.90) plus the row's
  label coverage (0.9994) can only coexist if the argmax token **is** one of the candidate labels'
  first tokens — so the row two_step lands on is one row *past* the label, never the label row.
* The 40 refusals are read on that same decision row: 40 of 60 items hold the single-token closer
  `<|end_of_sentence|>` (token 1) at ~1.0 mass there, coverage ≤ 5.1e-7 on all 60 (p50 1.43e-09,
  `low_mass` 60/60). Note for the next reader: `W_CUE_REFUSED` in the arm's cue-verdict table is
  printed from the *published* (decision-row) block — the engine's cue-row refusal warning fires only
  when the cue row itself is closed, which is not what these 40 rows are; they are "the model has
  answered (the label is the row before) and now closes the turn".
* **The bytes really are the same for `shipped`/`two_step`**: per-item `prefix_tokens` identical on
  all 60 items in both placements, and the offline per-family render
  (`tools/e3e_role_render.py --cue shipped|two_step`) yields identical rendered prefix/tail
  characteristics (lengths, heads, tails, opener, prompt size) for all five GGUFs on the box — the
  render record keeps excerpts, not whole bytes; the identity claim rests on the `prefix_tokens`
  arrays plus this field-for-field agreement. `json_instructed` differs exactly by the contract text
  and the opener (`{"choice": "`) — which is the comparability cost the record carries. Receipts:
  `logs/role_render_diff.txt`, `logs/interaction_probe.txt`.

## 5. The RED proof — confirmed by execution, not by the worker's log

| run | tree | command | result |
|---|---|---|---|
| RED | fresh `git archive 00265ea` copy + the four gate files from HEAD | `uv run --frozen --extra dev python -m pytest tests/test_e3e_roles.py tests/test_e3e_role_tool.py tests/test_e3e_roles_decision.py tests/test_e3e_docs.py -q` | **70 failed, 3 skipped** (1.28 s) |
| GREEN | `4ef74f6` (HEAD) | same command | **70 passed, 3 skipped** (0.22 s) |

The failure kinds match the claim (`AttributeError: module 'ggufone.engine.prompt' has no attribute
'question_block'/'role_split_render'`, `KeyError: 'chat_format'`, `'Options' object has no attribute
'chat_format'`, the two missing tools, the doc gates) — at `00265ea` the four gate files do not exist
at all, so the RED disposition is "copy the gates onto the pre-E3e tree", exactly as the evidence doc
says. Receipts: `logs/red_pre_e3e_auditor.txt`, `logs/green_head_auditor.txt`.

## 6. Regeneration — the docs carry the tool's own output

In a fresh `git archive HEAD` copy (machine-independent of the committed worktree), with the pinned
runtime absent (no model load needed): `bash docs/evidence/e3e/report.sh` exits **0** (the freeze probe and the
placement check both pass inside the tool), and

    diff docs/evidence/e3e_roles_decision.json  → byte-identical
    diff docs/evidence/e3e_roles_decision.md    → byte-identical
    uv run python docs/evidence/e3e/splice_docs.py && diff docs/BENCHMARKS.md → byte-identical
    diff docs/evidence/e3e_role_split_t_4c48f40a.md           → byte-identical

(CPython 3.13.15; unlike E3d's JSON, this artifact has no last-ulp sensitivity.) So every number in
the table *and in the §9 splice* is the tool's output from the raw arms, which §1–§3 then verified
number by number. Receipt: `logs/regeneration.txt`.

## 7. The recommendation — every claim checked

| claim in the recommendation | my reading |
|---|---|
| `json_instructed/answer_sheet` is the only cell clearing the rule: 51/60 vs 42/60, +0.150 (CI +0.021…+0.279), p = 0.049 | **exact** (§1, §2); it is the only `wins` verdict of the seven |
| "lands on the number E3d measured for the *uninstructed* opener (51/60)" | **exact as a marginal**: `docs/evidence/e3d/bench_templated_json_field.json` = 51/60 ✓. Note: E3d's paired split was 12 vs 3 (p = 0.0352); this cell's is 13 vs 4 (p = 0.0490) — same count, weaker paired evidence |
| `json_instructed/role_split` is one item behind: 50/60, +0.133, p = 0.057, "outside the rule by 0.007" | **exact** (0.057373 − 0.05 = 0.0074) |
| …and indistinguishable from the answer-sheet variant (p = 1.000) | **exact** (1 vs 2 discordant, CI −0.073..+0.040) |
| `role_split` alone lifts the shipped cue 42/60 → 50/60, `low_mass` 45/60 → 0/60, median coverage 3.9 % → 99.9 % | **exact** (0.038815 → 0.998838; `low_mass` 45 → 0) |
| "which is the reading the two 35B-A3B probes predict" | **not verifiable from this card's artifacts** — a pointer to §7.4.1/`t_7c926398` and the E3 probes; the card itself defers the 35B question to a [host] run. Recorded as a confidence limit, not as an error |
| "the placement costs one item of agreement in sixty, inside every paired interval, and is the half the failing families need" | **first half exact** (51 vs 50, p = 1.000, CI spans zero); the second half is the same §7.4.1 pointer |
| "the cheapest prompt rewrite (no per-family role-split acceptance)" | **supported**: the offline render shows 4/5 families render the role split and Tiel is `not-renderable` (outside the internal renderer's subset), so the placement carries a per-family acceptance step the instructed answer-sheet shape does not |
| "Neither moves here … both ship as switches with frozen defaults … the re-publication … is a follow-up card, because this table's headers are a new policy generation" | **exact**: defaults unmoved (§3); the comparability text is printed with the table and the invalidation list is in the evidence doc |

**Reading (my judgment, labelled as such).** The recommendation is the reading the table supports:
one win, on the instructed contract; the placement is a coverage/row-measurement fix whose agreement
gain does not clear the rule; the two levers are demonstrably not additive (the `two_step ×
role_split` collapse is mechanical, §4). I would keep the two sentences the evidence doc already
carries (borderline p; follow-up card) and add the one-item fragility number next to them, because
0.049 is 0.001 from failure and the doc's own drift note shows a re-score can move a near-tie item.

## 8. Findings and proposals (nothing applied; AWAITING APPROVAL)

| id | sev | class | what (receipt) | proposed fix |
|---|---|---|---|---|
| F1 | IMPROVE | PROCEDURAL | the freeze check's compared "decision" is `got` + `reliability`; the cue **refusal verdict** is outside it and moved on `n16` between the committed baseline and the placement probe (`docs/evidence/e3d/bench_templated_shipped.json` `refused: true`, `docs/evidence/e3e/bench_shipped_answer_sheet.json` `refused: false`) — the published classification, not the answer, moved | add `cue.refused` to the hard field set in `tools/e3e_roles_decision.py::freeze_check` and report it as a named row; exact patch in §8.1 |
| F2 | IMPROVE | PROCEDURAL | the fallback verdict text "not by more than the CI noise" is contradicted by the interval for **4 of the 7** verdicts it is printed on: `shipped/role_split` (+0.007..+0.260), `json_instructed/role_split` and `.../role_split/system` (+0.016..+0.251) clear the *interval* leg and fail only the exact-test leg (p = 0.077 / 0.057), and `two_step/role_split` (−0.382..−0.085) is significantly *worse* — under **both** the Wald and the exact-DP convention. The phrase mis-describes them in the generated report and §9 | re-word `decide()`'s fallback verdict; exact replacement text in §8.2 (tool + regeneration; zero numbers move) |
| F3 | NICE | TOOLING | two Wilson conventions across committed artifacts: the arm reports' own `overall`/`per_type` blocks use `z = 1.96` (`harness.wilson_interval`), the record's cells use `z = 1.959963984540054`; the same cell shows 0.574910530336 vs 0.574912920531 (6th decimal) | either a one-line note in the tool's docstring/§9 ("the arm block's interval is z=1.96; the table recomputes at the precise z"), or regenerate the arms — the auditor recommends the note (regeneration costs hours for a 6th-decimal difference) |
| F4 | NICE | PROCEDURAL | the evidence doc's instrument line says "`--runs 1`" (so does `docs/evidence/e3e/run_arms.sh` in its comment and in its `campaign.log` echo); the arm reports' own `commands.reproduce` and `config` say `--runs 5` — `run_arms.sh` passes no `--runs`, so `harness.DEFAULT_RUNS = 5` applied. The quality suite measures one decode per item, so **no quality number moves** — the sentence is wrong, the table is not | name the real command in the doc line (exact text in §8.3) |
| F5 | NICE | — | the exact-DP bootstrap (seed-free) reproduces E3d's published tail mass on E3d's own cell and leaves the rule's outcome unchanged (three more cells clear the interval leg; none clears p < 0.05) | none — recorded so the next reader knows the win is convention-robust |

### 8.1 F1 — exact patch (ready to paste into `tools/e3e_roles_decision.py::freeze_check`)

Old (the loop body, `tools/e3e_roles_decision.py` lines ~364–372):

    hard: list[str] = []            # the prompt bytes and the answer: what "frozen" means
    soft: list[str] = []            # the numbers: what a re-score is allowed to move
    decision_same = (left.get("got") == right.get("got")
                     and left.get("reliability") == right.get("reliability"))
    decisions_agree += int(decision_same)
    if not decision_same:
        hard.append(f"got {left.get('got')!r} != {right.get('got')!r}"
                    f" (verdict {left.get('reliability')!r}"
                    f" != {right.get('reliability')!r})")

New:

    hard: list[str] = []            # the prompt bytes and the answer: what "frozen" means
    soft: list[str] = []            # the numbers: what a re-score is allowed to move
    left_refused = bool((left.get("cue") or {}).get("refused"))
    right_refused = bool((right.get("cue") or {}).get("refused"))
    decision_same = (left.get("got") == right.get("got")
                     and left.get("reliability") == right.get("reliability")
                     and left_refused == right_refused)
    decisions_agree += int(decision_same)
    if not decision_same:
        hard.append(f"got {left.get('got')!r} != {right.get('got')!r}"
                    f" (verdict {left.get('reliability')!r}"
                    f" != {right.get('reliability')!r}; refused"
                    f" {left_refused} != {right_refused})")

Consequence to expect on this data: the placement line becomes "**59/60 decisions identical, 1
refusal verdict moved (`n16`)**" and the tool's exit-code-4 condition
(`decisions_agree < decisions`) starts to bite — which is the point: the refusal classification is a
published field and a check that cannot see it is not the check the claim needs. The doc sentence
("60/60 decisions identical" / "no decision moved") and §9 need the same edit.

### 8.2 F2 — exact replacement text (ready to paste into `tools/e3e_roles_decision.py::decide`)

Old (lines ~283–289):

    if pair["challenger_wins"]:
        verdicts.append({"label": label, "verdict": "wins", "why": why})
    elif pair["challenger_only"] == 0 and pair["baseline_only"] == 0:
        verdicts.append({"label": label, "verdict": "identical on these 60 items", "why": why})
    else:
        verdicts.append({"label": label, "verdict": "not by more than the CI noise",
                         "why": why})

New:

    if pair["challenger_wins"]:
        verdicts.append({"label": label, "verdict": "wins", "why": why})
    elif pair["challenger_only"] == 0 and pair["baseline_only"] == 0:
        verdicts.append({"label": label, "verdict": "identical on these 60 items", "why": why})
    elif pair["ci"][0] is not None and pair["ci"][0] > 0.0:
        verdicts.append({"label": label, "verdict": "interval clears zero, exact test does not",
                         "why": why})
    elif pair["ci"][1] is not None and pair["ci"][1] < 0.0:
        verdicts.append({"label": label, "verdict": "interval clears zero below, exact test does "
                                                    "not", "why": why})
    else:
        verdicts.append({"label": label, "verdict": "not by more than the CI noise",
                         "why": why})

Then `bash docs/evidence/e3e/report.sh && uv run python docs/evidence/e3e/splice_docs.py` and
`uv run pytest tests/test_e3e_roles_decision.py -q` (the render tests pin fragments of the text;
adjust the assertions in the same commit). Zero numbers move; the regenerated docs stay
byte-reproducible. The phrase also appears once as the *name of the rule* in `decide()`'s docstring
(line 237); keep it there or re-word both together — the label is what the reader meets next to the
numbers.

### 8.3 F4 — exact replacement text (evidence doc, the instrument paragraph)

Old: "`--threads 4`, `--runs 1`, no `W_BACKEND_MISMATCH` on any row."

New: "`--threads 4` (the quality suite decodes each item once; the reports' `--runs 5` is the
harness default the reproduce line names, not a repetition of the quality rows), no
`W_BACKEND_MISMATCH` on any row."

## 9. Keep & replicate

* **the deterministic offline number generator chain** — record → tool → generated report → spliced
  docs, byte-reproducible in a fresh checkout, with the tool's own output (not a transcription) in
  every document; the strongest form of evidence hygiene seen in this repo so far;
* **the freeze probe with non-zero exit codes** (3 = the default moved, 4 = the table's own baseline
  is not the committed one) and the *six-item* size — cheap to re-run, and it caught nothing, which
  is the point of running it;
* **one arm = one named policy flag, one backend named per run** (`--backend vulkan`, no
  `W_BACKEND_MISMATCH`), with the re-score's drift quantified instead of assumed;
* **the paired-only claims**: every promotion claim is a paired reading on the same 60 items, with
  the marginal Wilson intervals printed next to them precisely as the tempting wrong reading;
* **the RED log committed next to the GREEN one**, with the same file set on both sides (70 + 3
  skipped on each), so the TDD claim is auditable by re-execution;
* **the interaction cell left in the table at full weight** (`two_step × role_split`, 28/60): the
  table publishes a collapse of one of its own levers instead of dropping the cell;
* **the comparability block printed with the table** and the invalidation list in the evidence doc.

## 10. What I did not re-run / confidence

* I did not re-run the eight 60-item arms or the probes (hours on a >16 GiB-free host, and the card's
  inputs are the committed reports). What I did instead: independent recomputation of every number
  from the raw rows (§1–§3), the full regeneration (§6) and the offline renders (§4). The
  regeneration being byte-identical means the committed docs *are* the tool's output from those arms.
* `wall_ms` and the per-item timings are the reports' self-reports (timing, not correctness).
* The 35B-A3B pointers (`t_7c926398` §7.4.1, the E3 probes) are outside this card's artifacts; I did
  not re-run any 35B probe. The recommendation's use of them is a pointer, not a measurement.
* The `two_step × role_split` mechanism is verified at the row level and via the code's own semantics
  (`decide._score_group` / `_advance_token`); I did not re-run a live decode to watch the rows move.
* Confidence: numbers — CERTAIN (replica-exact, two independent routes for the test statistic).
  The freeze claim's scope gap — HIGH. The reading of the recommendation — HIGH; the fragility
  statement — CERTAIN (arithmetic).

## 11. How to reproduce my checks

    # 1. independent recomputation (pure stdlib; 655 checks, expects "no disagreement")
    python3 state/fights/e3e-ci/scripts/recompute_e3e.py
    # 2. the paired statistics, in detail
    python3 state/fights/e3e-ci/scripts/fragility.py      # one-item fragility
    python3 state/fights/e3e-ci/scripts/wilson_conventions.py
    # 3. freeze, field by field (including the fields the tool does not compare)
    python3 state/fights/e3e-ci/scripts/freeze_fields_diff.py
    python3 state/fights/e3e-ci/scripts/receipts.py
    # 4. the interaction cell, from the raw rows
    python3 state/fights/e3e-ci/scripts/interaction_probe.py
    # 4b. the cue-shape byte comparison (offline; no model load) — needs two renders first
    uv run --frozen python tools/e3e_role_render.py --models ~/.hermes/models --cue shipped \
        --json /tmp/rr_shipped.json >/dev/null
    uv run --frozen python tools/e3e_role_render.py --models ~/.hermes/models --cue two_step \
        --json /tmp/rr_two_step.json >/dev/null
    uv run --frozen python tools/e3e_role_render.py --models ~/.hermes/models --cue json_instructed \
        --json /tmp/rr_json.json >/dev/null
    python3 state/fights/e3e-ci/scripts/role_render_diff.py \
        shipped=/tmp/rr_shipped.json two_step=/tmp/rr_two_step.json json_instructed=/tmp/rr_json.json
    # 5. regeneration byte-identity (fresh copy; no model)
    S=$(mktemp -d); git archive HEAD | tar -x -C "$S"; cd "$S"
    UV_CACHE_DIR=~/.cache/uv bash docs/evidence/e3e/report.sh && uv run --frozen python docs/evidence/e3e/splice_docs.py
    diff <repo>/docs/BENCHMARKS.md docs/BENCHMARKS.md
    # 6. RED (fresh archive of 00265ea + the four gate files from HEAD) / GREEN (HEAD)
    uv run --frozen --extra dev python -m pytest tests/test_e3e_roles.py tests/test_e3e_role_tool.py \
        tests/test_e3e_roles_decision.py tests/test_e3e_docs.py -q

Receipts (this directory): `scripts/` (all seven checkers, incl. `mcnemar_check.py` for the two
partitioned p-values quoted above), `logs/recompute_e3e.txt`,
`logs/fragility.txt`, `logs/wilson_conventions.txt`, `logs/freeze_fields_diff.txt`,
`logs/receipts_baselines.txt`, `logs/interaction_probe.txt`, `logs/role_render_diff.txt`,
`logs/regeneration.txt`, `logs/green_head_auditor.txt`, `logs/red_pre_e3e_auditor.txt`,
`logs/mcnemar_check.txt`; `scripts/inspect_render.py` is the record-shape prober used for §4.

Audited tree: `4ef74f6e8194f27d57480907e88aa19a11c7130b` (HEAD when this audit ran; the E3e file set
was clean — no working-tree modification in `docs/evidence/e3e/`, `docs/evidence/e3d/`, `docs/`, `src/`, `tests/`, `tools/`).

_Proposals status: AWAITING APPROVAL (F1/F2/F4 recommended; nothing applied by the auditor.
The default decision goes to the user via @bots-coordinator — this card is a reading.)_
