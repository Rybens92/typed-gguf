# E3d — the cue shape on the full dev set (is the two-step readout the better default?)

- card `t_d90404ac` · model `Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes) · arch `spark2_5`
- run `.e3d/full.json` · runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend claim `cpu`
- devset `/workspace/ggufone/src/ggufone/bench/devset.jsonl` · items choice 24, noul 18, score 18 · engine floor 0.10
- policies: `shipped=bare`, `two_step_shipped=bare`, `json_field=bare`

## 1. Agreement per policy (marginal Wilson intervals — *not* the paired reading)

| policy | readout row | n | correct | agreement | 95 % Wilson | `low_mass` | `W_CUE_REFUSED` |
|---|---|---|---|---|---|---|---|
| `shipped=bare` | cue | 60 | 42 | 0.700 | 0.575–0.801 | 44/60 | 0/60 |
| `two_step_shipped=bare` | advanced | 60 | 46 | 0.767 | 0.646–0.856 | 2/60 | 0/60 |
| `json_field=bare` | cue | 60 | 51 | 0.850 | 0.739–0.919 | 0/60 | 0/60 |

## 2. The paired comparison (the decision's actual statistic)

### `two_step_shipped=bare` vs `shipped=bare`

- paired on **60** items: 3 correct only for the baseline, 7 correct only for the challenger
- exact McNemar p = **0.3438** (two-sided binomial tail on the discordant pairs)
- risk difference **+0.067**; 10000-draw paired percentile bootstrap 95 % CI **-0.033…+0.167** (seed 20260919)
- marginal Wilson intervals: baseline 0.575–0.801, challenger 0.646–0.856 (they overlap)
- coverage: baseline 0.267 above the floor / 44 `low_mass`; challenger 0.967 / 2
- **verdict: KEEP** — paired reading: the 60 items both policies measured, 3 won by shipped=bare only and 7 by two_step_shipped=bare only (exact McNemar p = 0.3438); risk difference +0.067 (95 % CI -0.033…+0.167). The marginal Wilson intervals overlap, which is why they are not the test.

### `json_field=bare` vs `shipped=bare`

- paired on **60** items: 3 correct only for the baseline, 12 correct only for the challenger
- exact McNemar p = **0.03516** (two-sided binomial tail on the discordant pairs)
- risk difference **+0.150**; 10000-draw paired percentile bootstrap 95 % CI **+0.033…+0.267** (seed 20260919)
- marginal Wilson intervals: baseline 0.575–0.801, challenger 0.739–0.919 (they overlap)
- coverage: baseline 0.267 above the floor / 44 `low_mass`; challenger 1.000 / 0
- **verdict: PROMOTE** — paired reading: the 60 items both policies measured, 3 won by shipped=bare only and 12 by json_field=bare only (exact McNemar p = 0.03516); risk difference +0.150 (95 % CI +0.033…+0.267). The marginal Wilson intervals overlap, which is why they are not the test.

## 3. Coverage distribution per policy

| policy | n | min | q1 | median | q3 | max | above floor | `low_mass` |
|---|---|---|---|---|---|---|---|---|
| `shipped=bare` | 60 | 1.15e-07 | 0.0143 | 0.0438 | 0.101 | 0.714 | 16/60 | 44/60 |
| `two_step_shipped=bare` | 60 | 4.27e-08 | 0.921 | 0.973 | 0.99 | 0.999 | 58/60 | 2/60 |
| `json_field=bare` | 60 | 0.651 | 0.994 | 1 | 1 | 1 | 60/60 | 0/60 |

Coverage is the full-vocabulary mass of the label's first token at the shape's own readout row (engine floor 0.10); for `two_step_*` that row is the **advanced** one, i.e. after the model's own first content token. `low_mass` counts the engine's verdict, which also depends on the answer's own reliability.

## 4. Agreement per question type

| policy | `choice` | `noul` | `score` |
|---|---|---|---|---|
| `shipped=bare` | 20/24 (0.833) | 16/18 (0.889) | 6/18 (0.333) |
| `two_step_shipped=bare` | 21/24 (0.875) | 14/18 (0.778) | 11/18 (0.611) |
| `json_field=bare` | 23/24 (0.958) | 17/18 (0.944) | 11/18 (0.611) |

Per question type: correct/n (agreement). The full per-item rows and the coverage quartiles per type are in the JSON next to this report.

## 5. The decision (card t_d90404ac)

The two candidates that beat the shipped shape are not the same kind of change: the
paired reading above decides the *agreement* claim, the engine's own floor decides the
*coverage* claim, and the shape decides what a published table means afterwards:

| candidate | agreement | paired risk difference | coverage | prompt bytes | at-the-cue refusal (`W_CUE_REFUSED`) |
|---|---|---|---|---|---|
| `shipped=bare` (shipped, the control) | 42/60 = 0.700 | — | 0.267 above the floor / 44 `low_mass` | unchanged | read at the cue (E3c) |
| `two_step` | 46/60 = 0.767 | -0.033…+0.167 (includes 0, p = 0.3438) | 0.967 above the floor / 2 `low_mass` | **identical** (the readout moves, the prompt does not) | preserved: the engine advances only when the cue row's argmax is not a turn-closer |
| `json_field` | 51/60 = 0.850 | +0.033…+0.267 (excludes 0, p = 0.03516) | 1.000 above the floor / 0 `low_mass` | rewritten (the cue line plus the per-type opener) | **gone**: the opener *is* the prompt, so the refusal never gets a row of its own |

**What ships.** The mechanism, not the default: `options.cue` / `--cue shipped|two_step|json_field` (the bench reads the same knob), with `shipped` still the
frozen default, so no published row moves by itself.

**What the data supports — and what it licenses.** `two_step` — the only candidate that
leaves every prompt byte, every published prompt-level table and the refusal verdict
intact, and takes the engine's own coverage verdict from a mostly-unusable row to a
measured one. Its agreement gain is *not* significant (the paired CI above includes
zero), so under the card's promotion rule (a paired agreement CI that excludes zero)
**nothing here is licensed to become the default** — this is a readout-position fix, not
an accuracy claim. The reliability axis (44 → 2 `low_mass`; exact McNemar 3.1e-11 on the
same 60 items) is a separate criterion the card does not state — promoting on it is a
coordinator/user decision, not an inference from this table (auditor, card t_fc037544).

**Why the agreement winner is not the default.** `json_field` clears the card's literal
rule (the paired CI excludes zero) and is excluded by the card's own constraint 4: the
opener is part of the prompt, so a family that refuses the cue can no longer be *seen*
refusing it — the verdict E3b/E3c built for Occamy (30/30 at-the-cue refusals) would read
as an answered row with a tiny mass instead. It stays available as `--cue json_field`,
measured and documented, for models that answer.

**What a flip would need (not licensed by this data).** One constant,
`schema.OPTION_DEFAULTS["cue"]` (plus `Options.cue`) and this document's default
column; the mechanics (the readout, the refusal stop, the bench knob, the CLI flag, the
payload key) are already gated by `tests/test_e3d_cue_switch.py`. A future flip needs
either (a) a `two_step` agreement CI that excludes zero — at the observed discordant
rates (3/60 vs 7/60) roughly a 140-160 item set — or (b) an explicit decision to promote
on the readout-availability axis instead, with the bench seam of section 6 fixed first.

**What a default change invalidates** (the card's risk note):

* `two_step` — every published *readout* column: the `coverage` / `reliability`
  values and the `W_CUE_REFUSED` counts of `docs/BENCHMARKS.md` §7 and
  `docs/evidence/e2_quality.json`
  (the row the label is read from moves one token in; the prompt tables — the E1c label
  policy, E3b's 15 cue × label cells, E3c's seven shapes — stay valid).
* `json_field` — everything printed *at the shipped cue*, because the bytes change:
  the E1c label-policy table, E3b's cue × label grid, E3c's seven shapes, and the
  refusal verdict itself. A re-run, not a footnote.

**Reproducing this document** (no model for the first two):

    bash .e3d/run_full.sh                    # the probe record -> .e3d/full.json
    python3 tools/e3d_cue_decision.py report --run .e3d/full.json \
        --out docs/evidence/e3d_cue_decision_4b.md \
        --json docs/evidence/e3d_cue_decision_4b.json
    python3 tools/e3d_engine_check.py --record .e3d/full.json --items 6
    bash .e3d/run_bench_arms.sh               # the bench arms (## 6)

Byte-identity of the JSON holds under CPython 3.11; 3.12+ changes the last ulp of
`mean` fields — diff with tolerance.

## 6. The bench arms — the switch through the instrument, before and after the fix

Same box, same model, same 60 committed items, same context; only `--cue` moves.
The `framing` column is the arm's own report marker (`engine.template`, summarized over the rows — `harness.framing_label`):

| arm | framing | agreement | Wilson | `low_mass` | refused at the cue | coverage median | above floor |
|---|---|---|---|---|---|---|---|
| `--cue shipped` | chat-template: spark2_5 / internal | 42/60 = 0.700 | 0.575–0.801 | 45/60 | 1/60 | 0.03847 | 15/60 |
| `--cue two_step` | chat-template: spark2_5 / internal | 47/60 = 0.783 | 0.664–0.869 | 3/60 | 3/60 | 0.9708 | 57/60 |
| `--cue json_field` | chat-template: spark2_5 / internal | 51/60 = 0.850 | 0.739–0.919 | 0/60 | 0/60 | 0.9997 | 60/60 |
| `--cue shipped` | plain (prompt.py E1b framing) | 36/60 = 0.600 | 0.474–0.714 | 13/60 | 0/60 | 0.2711 | 47/60 |
| `--cue two_step` | plain (prompt.py E1b framing) | 27/60 = 0.450 | 0.331–0.575 | 29/60 | 8/60 | 0.1103 | 31/60 |
| `--cue json_field` | plain (prompt.py E1b framing) | 46/60 = 0.767 | 0.646–0.856 | 0/60 | 0/60 | 0.8544 | 60/60 |

**The plain rows are the pre-fix instrument** (kept because they were published): `LiveModel.decide` planned the executed context from the live session, and a `ModelSession` carries no `.model`/`.runtime`, so `resolve_template(request, session)` returned `None` and `prompt.build_prefix(state, resolution=None)` silently fell back to the plain E1b framing — the documented escape hatch for a session with no model handle, not a default for a real one. The chat-template rows are the same command after card `t_6de5fc53` (one plan, resolved from the **handle** — the source the serving path uses); on dev item `c01` that is 102 prefix tokens against 119 and the cue row's label mass 0.03677 against 0.00696.

**`two_step` vs `shipped` on the corrected instrument** — agreement 42/60 -> 47/60, `low_mass` 45 -> 3, refusals at the cue 1 -> 3, coverage median 0.03847 -> 0.9708, above the 0.10 floor 15 -> 57.
The corrected bench's own delta is +5 correct items — the signs agree, and the plain arm's reversal (`--cue two_step` costing agreement and doubling `low_mass`) is a property of the *pre-fix framing*, not of the cue shape. (section 2's paired verdict for `two_step` is **KEEP (paired risk difference +0.067, 95 % CI -0.033…+0.167)**, so the probe's paired reading does not put `two_step` behind `shipped` on agreement.)

**`two_step` vs `shipped` on the pre-fix instrument** — agreement 36/60 -> 27/60, `low_mass` 13 -> 29, refusals at the cue 0 -> 8, coverage median 0.2711 -> 0.1103, above the 0.10 floor 47 -> 31.
**What this changes.** The bench now sends the prompt the product sends, so its quality tables describe the serving path; the pre-fix rows above are superseded (they measure the plain framing) and `docs/BENCHMARKS.md` §2/§7/§8 carry the corrected numbers next to them. The cue default is frozen by the E3d card and does not move here.

