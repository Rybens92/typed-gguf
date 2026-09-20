# E3e — the question's placement and the instructed JSON (card `t_4c48f40a`)

*Instrument: the committed 60-item dev set, temperature 0, fixed seed, **`--backend vulkan`** — one
backend named per run, which is the card's rule and this card's own lesson: `--backend auto` has
selected the `cpu` bundle on this box while the engine log showed Vulkan0 computing, and an earlier
arm of this very campaign fell through to `CPU compute buffer size` rows, which would be two
placements inside one table. `--threads 4`, `--runs 1`, no `W_BACKEND_MISMATCH` on any row. One card,
one box, one model — the 4B (`Spark-X2.5-4B-Q8_0.gguf`, sha256
`5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2`), the model every other published
quality row on this box was measured with. Every cell below is that same instrument with only the
policy flags moved; the reports are `.e3e/bench_<cue>_<chat_format>[_system].json`, and
`.e3e/logs/campaign.log` carries the SHA and the placement reading taken before the first arm.*

## Why this card exists

E3d (`t_d90404ac`) ended with the label policy measured and the shape switch shipped, and with two
readings that pointed here:

* the winner (`--cue json_field`, 51/60) wins by *appending an opener* — the model never gets a row
  where it can refuse the cue. It is a prompt-level change, so the E3d card kept it a switch.
* the runner-up (`--cue two_step`) is a **readout** fix, and on the families that need help it is
  inert: `decide._advance_token` will not advance past a cue the model closed, and
  `qwen35moe` (Tiel) closes **59 of 60** cue rows under the product's own prompt
  (`docs/BENCHMARKS.md` §7.4.1, card `t_7c926398`). A readout-position fix cannot reach a model that
  refuses the cue, because there is nothing to read past.

So the two levers E3e tests are the two that change **what the model is asked**, not where the
readout sits:

1. **`options.cue = json_instructed`** — the ask line *is* a JSON contract (the key per question
   type from `JSON_FIELDS`), and the assistant turn is prefilled with the opened field. E3d's
   `json_field` appended an opener to a cue line that still said *"Answer with exactly one candidate
   name"*; the instructed shape makes the instruction and the shape agree.
2. **`options.chat_format = role_split`** — the question stops living inside the assistant turn.
   Every published row to date asks it there (a question prefilled after the template's generation
   prompt), because the shipped instrument prefills a `system + state` prefix and then the question.
   A model whose template has an assistant turn that opens with a think block — Spark, Qwen3.5 —
   sees that question where a *reply* is expected.

Both are switches with frozen defaults, in the E3d manner: a request that does not set them renders
the bytes the published tables were measured on.

The amendment added a third question — **where the contract is stated** (`options.json_contract`):
the question block names its own key (`question`, the default) or the system framing states all
three contracts up front (`system`). Both are measured below.

## The family acceptance, before any numbers

The role split is only real if the family's own `tokenizer.chat_template` can render it. That is
measured offline for every GGUF on the box (`tools/e3e_role_render.py`, no model loaded) and gated in
`tests/test_e3e_roles.py`: four of the five families render the two-user-turn conversation with all
five checks green (Spark, Ternary-Bonsai/Qwen3.5-27B, Occamy/Qwen3.5-MoE, Ling/bailingmoe3); Tiel's
own template is outside the internal renderer's subset and the tool records that with its fallback
(`--template plain`, or a live run through the built-in bridge) instead of guessing. The per-family
bytes — prefix, tail, the residual Spark's template cannot share, and the generation prompt each
family leaves in front of the opener — are in `.e3e/role_render.md` and pinned in the tests.

## The table

<!-- E3E-TABLE:START — spliced by `.e3e/splice_docs.py` from the tool's own report
     (`docs/evidence/e3e_roles_decision.md`); edit the tool, never this block. -->
- probe `.e3e/probe_default.json` vs baseline `.e3d/bench_templated_shipped.json` (6 shared items): **frozen** — the prompt bytes and the decisions are the committed baseline's: 6/6 items, prefix_tokens identical on every item; numbers re-scored, not bit-identical (max |delta p| = 3.78e-02 over 6 item(s))
- the table's own instrument, `.e3e/bench_shipped_answer_sheet.json` vs `.e3d/bench_templated_shipped.json` (60 shared items, tolerance 0.005): **decisions identical** (60/60) — the instrument moved the numbers, not the answers; max |delta p| = 1.14e-02, max |delta coverage| = 6.25e-03, prefix_tokens identical on every item

### The cells

| cell | correct | agreement (Wilson 95 %) | choice | noul | score | low_mass | refusals | cue verdicts | coverage p50 | prefix tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `shipped/answer_sheet` | 42/60 | 70.0 % (57.5–80.1 %) | 20/24 | 16/18 | 6/18 | 45 | 0 | — | 3.9 % | 82–119 |
| `shipped/role_split` | 50/60 | 83.3 % (72.0–90.7 %) | 22/24 | 17/18 | 11/18 | 0 | 0 | — | 99.9 % | 78–115 |
| `two_step/answer_sheet` | 46/60 | 76.7 % (64.6–85.6 %) | 21/24 | 14/18 | 11/18 | 2 | 2 | — | 97.3 % | 82–119 |
| `two_step/role_split` | 28/60 | 46.7 % (34.6–59.1 %) | 9/24 | 14/18 | 5/18 | 60 | 40 | — | 0.0 % | 78–115 |
| `json_instructed/answer_sheet` | 51/60 | 85.0 % (73.9–91.9 %) | 21/24 | 17/18 | 13/18 | 0 | 0 | answered 60 | 100.0 % | 86–123 |
| `json_instructed/role_split` | 50/60 | 83.3 % (72.0–90.7 %) | 21/24 | 18/18 | 11/18 | 0 | 0 | answered 60 | 100.0 % | 82–119 |
| `json_instructed/answer_sheet/system` | 48/60 | 80.0 % (68.2–88.2 %) | 21/24 | 17/18 | 10/18 | 0 | 0 | answered 60 | 100.0 % | 131–168 |
| `json_instructed/role_split/system` | 50/60 | 83.3 % (72.0–90.7 %) | 21/24 | 17/18 | 12/18 | 0 | 0 | answered 60 | 100.0 % | 127–164 |

### Paired comparisons (same items, item by item)

| baseline | challenger | only challenger | only baseline | both | neither | difference (95 % CI) | exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `shipped/answer_sheet` | `shipped/role_split` | 12 | 4 | 38 | 6 | +0.133 (+0.007..+0.260) | 0.077 |
| `shipped/answer_sheet` | `two_step/answer_sheet` | 7 | 3 | 39 | 11 | +0.067 (-0.035..+0.169) | 0.344 |
| `shipped/answer_sheet` | `two_step/role_split` | 5 | 19 | 23 | 13 | -0.233 (-0.382..-0.085) | 0.007 |
| `shipped/answer_sheet` | `json_instructed/answer_sheet` | 13 | 4 | 38 | 5 | +0.150 (+0.021..+0.279) | 0.049 |
| `shipped/answer_sheet` | `json_instructed/role_split` | 11 | 3 | 39 | 7 | +0.133 (+0.016..+0.251) | 0.057 |
| `shipped/answer_sheet` | `json_instructed/answer_sheet/system` | 10 | 4 | 38 | 8 | +0.100 (-0.020..+0.220) | 0.180 |
| `shipped/answer_sheet` | `json_instructed/role_split/system` | 11 | 3 | 39 | 7 | +0.133 (+0.016..+0.251) | 0.057 |
| `shipped/role_split` | `two_step/answer_sheet` | 3 | 7 | 43 | 7 | -0.067 (-0.169..+0.035) | 0.344 |
| `shipped/role_split` | `two_step/role_split` | 2 | 24 | 26 | 8 | -0.367 (-0.505..-0.228) | 0.000 |
| `shipped/role_split` | `json_instructed/answer_sheet` | 2 | 1 | 49 | 8 | +0.017 (-0.040..+0.073) | 1.000 |
| `shipped/role_split` | `json_instructed/role_split` | 3 | 3 | 47 | 7 | +0.000 (-0.080..+0.080) | 1.000 |
| `shipped/role_split` | `json_instructed/answer_sheet/system` | 2 | 4 | 46 | 8 | -0.033 (-0.113..+0.046) | 0.688 |
| `shipped/role_split` | `json_instructed/role_split/system` | 3 | 3 | 47 | 7 | +0.000 (-0.080..+0.080) | 1.000 |
| `two_step/answer_sheet` | `two_step/role_split` | 5 | 23 | 23 | 9 | -0.300 (-0.455..-0.145) | 0.001 |
| `two_step/answer_sheet` | `json_instructed/answer_sheet` | 8 | 3 | 43 | 6 | +0.083 (-0.023..+0.190) | 0.227 |
| `two_step/answer_sheet` | `json_instructed/role_split` | 9 | 5 | 41 | 5 | +0.067 (-0.054..+0.188) | 0.424 |
| `two_step/answer_sheet` | `json_instructed/answer_sheet/system` | 7 | 5 | 41 | 7 | +0.033 (-0.080..+0.146) | 0.774 |
| `two_step/answer_sheet` | `json_instructed/role_split/system` | 8 | 4 | 42 | 6 | +0.067 (-0.045..+0.179) | 0.388 |
| `two_step/role_split` | `json_instructed/answer_sheet` | 24 | 1 | 27 | 8 | +0.383 (+0.252..+0.515) | 0.000 |
| `two_step/role_split` | `json_instructed/role_split` | 22 | 0 | 28 | 10 | +0.367 (+0.245..+0.489) | 0.000 |
| `two_step/role_split` | `json_instructed/answer_sheet/system` | 21 | 1 | 27 | 11 | +0.333 (+0.205..+0.461) | 0.000 |
| `two_step/role_split` | `json_instructed/role_split/system` | 23 | 1 | 27 | 9 | +0.367 (+0.236..+0.497) | 0.000 |
| `json_instructed/answer_sheet` | `json_instructed/role_split` | 1 | 2 | 49 | 8 | -0.017 (-0.073..+0.040) | 1.000 |
| `json_instructed/answer_sheet` | `json_instructed/answer_sheet/system` | 0 | 3 | 48 | 9 | -0.050 (-0.105..+0.005) | 0.250 |
| `json_instructed/answer_sheet` | `json_instructed/role_split/system` | 1 | 2 | 49 | 8 | -0.017 (-0.073..+0.040) | 1.000 |
| `json_instructed/role_split` | `json_instructed/answer_sheet/system` | 1 | 3 | 47 | 9 | -0.033 (-0.098..+0.031) | 0.625 |
| `json_instructed/role_split` | `json_instructed/role_split/system` | 1 | 1 | 49 | 9 | +0.000 (-0.046..+0.046) | 1.000 |
| `json_instructed/answer_sheet/system` | `json_instructed/role_split/system` | 3 | 1 | 47 | 9 | +0.033 (-0.031..+0.098) | 0.625 |

### The decision

- best cell: `json_instructed/answer_sheet` — 51/60 correct, 0 refusals

- `json_instructed/answer_sheet` vs `shipped/answer_sheet`: **wins** — 13 items only it got right, 4 only shipped/answer_sheet did; difference +0.150 (95 % CI +0.021..+0.279), exact McNemar p=0.049
- `json_instructed/role_split` vs `shipped/answer_sheet`: **not by more than the CI noise** — 11 items only it got right, 3 only shipped/answer_sheet did; difference +0.133 (95 % CI +0.016..+0.251), exact McNemar p=0.057
- `json_instructed/role_split/system` vs `shipped/answer_sheet`: **not by more than the CI noise** — 11 items only it got right, 3 only shipped/answer_sheet did; difference +0.133 (95 % CI +0.016..+0.251), exact McNemar p=0.057
- `shipped/role_split` vs `shipped/answer_sheet`: **not by more than the CI noise** — 12 items only it got right, 4 only shipped/answer_sheet did; difference +0.133 (95 % CI +0.007..+0.260), exact McNemar p=0.077
- `json_instructed/answer_sheet/system` vs `shipped/answer_sheet`: **not by more than the CI noise** — 10 items only it got right, 4 only shipped/answer_sheet did; difference +0.100 (95 % CI -0.020..+0.220), exact McNemar p=0.180
- `two_step/answer_sheet` vs `shipped/answer_sheet`: **not by more than the CI noise** — 7 items only it got right, 3 only shipped/answer_sheet did; difference +0.067 (95 % CI -0.035..+0.169), exact McNemar p=0.344
- `two_step/role_split` vs `shipped/answer_sheet`: **not by more than the CI noise** — 5 items only it got right, 19 only shipped/answer_sheet did; difference -0.233 (95 % CI -0.382..-0.085), exact McNemar p=0.007

### The recommendation

Take the instructed contract; keep the placement a switch. `json_instructed` is the only lever this
card measures that clears the card's own rule against the shipped cell: 51/60 vs 42/60 on the 60
committed items, paired +0.150 (95 % CI +0.021…+0.279), exact McNemar p = 0.049 — and it lands on the
number E3d measured for the *uninstructed* opener (51/60), now with the instruction the shape asks
for. `json_instructed/role_split` is one item behind (50/60, +0.133, p = 0.057, outside the rule by
0.007) and indistinguishable from the answer-sheet variant (p = 1.000); `role_split` alone lifts the
shipped cue from 42/60 to 50/60 and removes the collapse this card was pointed at (`low_mass` 45/60 →
0/60, median coverage 3.9 % → 99.9 %), which is the reading the two 35B-A3B probes predict. If the
default moves later, the defensible change is the *instructed contract* with the answer-sheet
placement — the cell the rule accepts, and the cheapest prompt rewrite (no per-family role-split
acceptance). If the prompt must look like the format the model was trained on (the amendment's
rationale), the placement costs one item of agreement in sixty, inside every paired interval, and is
the half the failing families need. Neither moves here: both ship as switches with frozen defaults,
and the re-publication — if the coordinator takes this recommendation — is a follow-up card, because
this table's headers are a new policy generation.

### The caveats the numbers carry

- 60 committed dev items: a single cell's 95 % interval is up to 24.5 points wide, so single-cell differences below that are noise by construction — which is what the paired columns are for.
- `score` is the type E3c/E3d found hardest: a policy that helps choice/noul and does not help score is still a policy decision, not a quality result.

**Comparability cost.** Every cell in this table is measured under the *same* instrument (the same 60 committed dev items, temperature 0, fixed seed, `--backend vulkan`) — so the table compares itself and nothing else. What it does **not** compare against is any published row: a `role_split` placement, a `json_instructed` ask line, or a contract stated in the framing each change the bytes the model sees. A cell published as *the* quality row would carry its policy line (`- prompt policy: …`) and every other published quality row would have to be re-measured under the same policy before it could sit next to it.

The generated report this block was spliced from is `docs/evidence/e3e_roles_decision.md` (the tool's own output, re-generated by `.e3e/report.sh`); the machine-readable record is `docs/evidence/e3e_roles_decision.json`.
<!-- E3E-TABLE:END -->

## What the numbers mean (and what they do not)

### The readings the table carries

**The placement is the bigger lever on this model — and it is not an accuracy claim.** Moving the
question out of the assistant turn (`shipped`/`answer_sheet` 42/60 → `shipped`/`role_split` 50/60) is
worth +0.133 paired (CI +0.007…+0.260, McNemar p = 0.077 — outside the rule, by 0.03), and it
removes the collapse this card was pointed at: `low_mass` falls from **45/60 to 0/60** and the median
candidate coverage rises from **3.9 % to 99.9 %**. A coverage distribution is not a marginal effect:
under the shipped placement the model is handed its question where its own reply belongs, and the
mass says so; under the role split every row is a `measured` row.

**The instructed contract is the lever the rule accepts.** `json_instructed`/`answer_sheet` is 51/60
(+0.150, CI +0.021…+0.279, exact McNemar p = 0.049) — the only cell that clears both halves of the
E3d unit — and it is the *instructed* shape of the cell E3d measured with a bare opener (51/60):
the opener now continues an instruction instead of contradicting one. Its coverage p50 is 100 %, 0
`low_mass`, and every row's cue verdict is `answered` — the row the readout reads is the row the
instruction names.

**The two levers are not additive, and one pair is a collapse.** `two_step`/`role_split` measures
28/60 with `low_mass` 60/60 and 40 refusals. The mechanism is in the cue blocks: under the role split
the **label itself** sits at the cue row (Spark's row for `c01` holds `technical`, token 97, mass
0.90 in `shipped`/`role_split`), so `two_step`'s one-token advance reads the row *after* the answer —
a row that holds ~1.0 of its mass on a single non-label token (`c01`: token 79152) with 0 coverage.
E3d's `two_step` premise is that the cue line is a *formatter* and the label follows it; the
role-split shape has no such formatter, so the cue-shape lever and the placement lever are not
composable. That is why the recommendation keeps `two_step` out of any role-split default, and it is
the one cell of this table a reviewer should re-check first.

**Where the contract is stated matters less than whether it is stated.** Stating it once in the
system framing is 48/60 (answer sheet) and 50/60 (role split) against 51/60 and 50/60 inline — inline
is at least as good on this model, and the system variant costs 45–50 extra prefix tokens on every
request (131–168 vs 82–123). The amendment's second variant is measured, not recommended.

**`score` stays the hard type**: the best cell moves it from 6/18 to 13/18, and every cell that lifts
choice/noul leaves `score` behind — the reading E3c and E3d both published.

### The freeze, and the re-score

The card's "nothing moves by default" is checked by two arms against the *committed* baseline
(`.e3d/bench_templated_shipped.json`, which the E3d card measured with `--backend auto`):

* **`--freeze-probe .e3e/probe_default.json`** — the same six items run with the baseline's own
  recipe. The prompt bytes (`prefix_tokens`, item by item) and the decisions are the baseline's; the
  *numbers* are not, and cannot be: card `t_55de5779` landed after that baseline was published, and a
  `--backend auto` row that claims `cpu` now really computes on the CPU (the probe's log carries 12
  `CPU compute buffer size` rows — the fix working). The byte-level claim therefore rides on
  `prefix_tokens` plus the pin files, never on float equality, and the tool's exit code says so.
* **`--placement-probe .e3e/bench_shipped_answer_sheet.json`** — the same cell under the table's own
  instrument (`--backend vulkan`, 60 items): **60/60 decisions identical**, `prefix_tokens` identical
  on every item, max |Δp| = 1.14e-02, max |Δcoverage| = 6.25e-03.
* The drift is not free, and it is quantified rather than waved away: E3d's `two_step` cell publishes
  47/60 under the old instrument and measures 46/60 here, so a single decision in sixty can sit
  inside the placement's own noise. Every claim in this table is therefore *paired* by item, and the
  only **wins** verdict is one whose discordant split (13 vs 4) is far outside a one-item wobble.

### The gates, RED first

`tests/test_e3e_roles.py`, `tests/test_e3e_role_tool.py`, `tests/test_e3e_roles_decision.py` and
`tests/test_e3e_docs.py` are this card's gates, and they were run against the **pre-E3e commit**
(`00265ea`, the parent of the card's first commit) in a detached worktree with the same files copied
in: **70 failed, 3 skipped** (`.e3e/logs/red_pre_e3e.txt`). Every failure is a missing piece of this
card — `AttributeError: module 'ggufone.engine.prompt' has no attribute 'role_split_render'` (and
`question_block`), `'Options' object has no attribute 'chat_format'`, `options.cue must be one of
shipped, two_step, json_field (got 'json_instructed')`, `module 'ggufone.engine.cue' has no attribute
'value_verdict'`, and the two tools that do not exist yet. On the landed tree the same four files are
**70 passed, 3 skipped** (the same 73 collected, so the numbers line up; the three skips are the
model-marked family checks, which need `--run-network`).

The decision rule is in the tool and printed with the table: a challenger displaces the baseline
only when the paired difference's 95 % interval excludes zero **and** the exact two-sided McNemar
p is below 0.05 — the same unit E3d used (the same 60 items, paired by item), with the closed-form
interval instead of a bootstrap so every number in the table is read directly from the reports.

Three limits travel with the table and are printed inside it:

* **60 items.** A single cell's interval is up to ~22 points wide; only the paired columns can say
  anything finer. A per-type cell (18–24 items) is a lead, not a result.
* **`score` is the hard type** (E3c/E3d both found this): a policy that lifts `choice`/`noul` and
  moves `score` little is a policy decision, not an accuracy claim.
* **Comparability.** The cells compare with *each other* — same instrument, same items, same box,
  one policy flag apart. They do **not** compare with any published row: `role_split`,
  `json_instructed` and `json_contract=system` each change the prompt bytes, so a cell published as
  *the* quality row would carry its `- prompt policy:` line and every other published row would have
  to be re-measured under it first. That is exactly why the defaults stay frozen: the E3e options
  ship as switches, in the E3d manner.
* **The freeze is checked, not assumed.** The baseline cell
  (`shipped`/`answer_sheet`) is the committed `.e3d/bench_templated_shipped.json`; a six-item probe
  on this tree must reproduce its *prompt bytes* (`prefix_tokens`, item for item) and its
  *decisions* — the numbers move with the instrument, and the section above says by how much
  (`.e3e/probe_default.json`, `--freeze-probe` in the decision tool; a probe that moved a byte or
  flips a decision fails and the tool exits non-zero).

### The Tier-M sweep, in three rounds

`[tool.mutmut]` sweeps `engine/prompt.py` + `engine/cue.py` against this card's three gate files;
the verdicts are read from `mutants/<module>.meta` (`exit_code_by_key`) because `mutmut results`
lists survivors only, and the honest statement is always "of the mutants that ran". The full record
is `.e3e/mutmut_summary.txt` (regenerate: `uv run python .e3e/mutmut_summary.py`):

| round | what it followed | prompt.py | cue.py | total |
|---|---|---|---|---|
| 1 | the card's change | 274/423 (64.8 %) | 120/144 (83.3 %) | 394/567 (69.5 %) |
| 2 | the two acceptance pins | 286/423 (67.6 %) | 120/144 (83.3 %) | 406/567 (71.6 %) |
| 3 | the plan's `prefix_tokens` pin | 287/423 (67.8 %) | 120/144 (83.3 %) | **407/567 (71.8 %)** |

Each round followed a *named* finding, not a score: round 1 exposed the two acceptance guards inside
`role_split_render` that no test reached — the module's only uncovered lines (267, 274), which is
the reference's warning about a high score sitting above unreached code; round 2's survivor list
exposed the inverted `chat_format` guard in `build_prefix`, alive because the gates never compared
the plan's prefix *tokens* with the role-split prefix they were rendered from; round 3 pins that
(`test_the_plan_carries_the_role_split_it_rendered`) and kills it. The load-bearingness of the two
round-2 pins was checked by hand-mutation replay before the sweep (guard set to `False` alone →
exactly its own test fails; file restored from `git checkout`, byte-identical).

What remains is classified in the summary by diffing each survivor against its `_orig` twin:
message strings (33 — the gates assert codes and bytes, not prose), arguments every caller supplies
(46, including the `json.dumps` kwargs that `tests/test_engine_fork.py` pins, a file outside the
selection), three equivalent `role` arms, the shared-prefix optimisation (2), `empty_candidate_code`
(12, `no tests` — its callers are the CLI suites) and cue.py's special-token catalogue (17, pinned by
`tests/test_e3c_cue_specials.py`, also outside the selection). Selection coverage of the two modules
under the card's own gates is **97.8 % of prompt.py and 98.1 % of cue.py**.

## What this card does not do

* It does not move any default. `cue=shipped`, `chat_format=answer_sheet`, `json_contract=question`
  — a request that sets none of them renders the bytes the published rows were measured on, and the
  pin files keep it that way.
* It does not re-measure the E3d arms. Their reports are the E3d evidence, unchanged; the one place
  this card reads an E3d number again is the `two_step` drift note above.
* It does not measure the 35B-A3B families. Both are ≥21 GB against this box's 8 GiB cgroup, so
  "does `role_split` + `json_instructed` rescue Tiel" is a **[host]** question, routed in this card's
  hand-off rather than claimed: the corrected Tiel row (§7.4.1, `t_7c926398`) is what the placement
  half is aimed at, and this card says what that run needs first — Tiel's own template is outside the
  internal renderer's subset, so the [host] arm renders through the built-in bridge, with
  `--template plain` recorded as the documented fallback.

## What a default change would invalidate

The comparability note, as the list the card asks for. Each published table below was measured on the
shipped prompt bytes, so a follow-up card that publishes `json_instructed` (or the role split) as
*the* policy makes all of them a different measurement:

| published table | where | why it would have to be re-measured |
|---|---|---|
| the §7 quality rows, `[container]` and `[host]`, including the corrected Tiel row §7.4.1 | `docs/BENCHMARKS.md` | the ask line and/or the placement change the bytes every row was read at |
| the §8 cue-shape table (shipped 42 / two_step 47 / json_field 51) | `docs/BENCHMARKS.md`, `docs/evidence/e3d_cue_decision_4b.md` | `json_instructed` replaces the ask line `two_step` and `json_field` build on, and on a role split `two_step` measures 28/60 — the two tables are not additive |
| the E3/§6 and E3c/§7 probe rows (the per-family collapses) | `docs/BENCHMARKS.md`, `docs/evidence/e3_*` | those rows are the collapse the placement moves (`low_mass` 45/60 → 0/60 on the 4B); their headers name a prompt they no longer measure |
| §9 itself | `docs/BENCHMARKS.md`, `docs/evidence/e3e_roles_decision.md` | one table, one instrument: re-run `.e3e/report.sh` and `.e3e/splice_docs.py` after any cell's prompt moves |

## Reproduce

```bash
bash .e3e/run_arms.sh          # the freeze probe (--backend auto) + the eight arms (--backend vulkan)
bash .e3e/report.sh            # wraps `tools/e3e_roles_decision.py` (the decision rule lives there):
                               # the table, the freeze and the re-score → docs/evidence/e3e_roles_decision.md
uv run python .e3e/splice_docs.py   # re-splices that report into this doc and docs/BENCHMARKS.md §9
uv run python tools/e3e_role_render.py --models ~/.hermes/models \
    --json .e3e/role_render.json --report .e3e/role_render.md   # the per-family acceptance (offline)
bash .e3e/mutmut_sweep.sh      # the Tier-M sweep (engine/prompt.py + engine/cue.py, the card's gates)
```

The gates, run from the repo root:

```bash
uv run --frozen --extra dev python -m pytest tests/test_e3e_roles.py tests/test_e3e_role_tool.py \
    tests/test_e3e_roles_decision.py tests/test_e3e_docs.py -q
```
