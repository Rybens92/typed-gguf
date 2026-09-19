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

**What the numbers recommend.** `two_step` — the only candidate that leaves every prompt
byte, every published prompt-level table and the refusal verdict intact, and takes the
engine's own coverage verdict from a mostly-unusable row to a measured one. Its agreement
gain is *not* significant (the paired CI above includes zero) and the evidence document
says so: this is a readout-position fix, not an accuracy claim.

**Why the agreement winner is not the default.** `json_field` clears the card's literal
rule (the paired CI excludes zero) and is excluded by the card's own constraint 4: the
opener is part of the prompt, so a family that refuses the cue can no longer be *seen*
refusing it — the verdict E3b/E3c built for Occamy (30/30 at-the-cue refusals) would read
as an answered row with a tiny mass instead. It stays available as `--cue json_field`,
measured and documented, for models that answer.

**The flip, if the second opinion agrees.** One constant,
`schema.OPTION_DEFAULTS["cue"]`
+ `Options.cue`) and this document's default column; the mechanics (the readout, the
refusal stop, the bench knob, the CLI flag, the payload key) are already gated by
`tests/test_e3d_cue_switch.py`.

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

## 6. The bench arms — and the instrument split they expose

Same box, same model, same 60 committed items, same context; only `--cue` moves.
Both arms ran the shipped configuration (the Vulkan bundle, the device visible).

| arm | agreement | Wilson | `low_mass` | refused at the cue | coverage median | above floor |
|---|---|---|---|---|---|---|
| `--cue shipped` | 36/60 = 0.600 | 0.474–0.714 | 13/60 | 0/60 | 0.2711 | 47/60 |
| `--cue two_step` | 27/60 = 0.450 | 0.331–0.575 | 29/60 | 8/60 | 0.1103 | 31/60 |

The two arms point the *other* way from section 1: in the bench's framing the two-step readout costs agreement (36/60 -> 27/60), doubles `low_mass` (13 -> 29) and finds 8/60 refusals at the cue row that the shipped shape does not see. That is not a contradiction of section 1 — it is the framing split, measured:

* the bench executes the **plain** E1b framing. `LiveModel.decide` plans twice: from the handle to size the session, then again from the live session — and `resolve_template(request, session)` returns `None` because a `ModelSession` carries no `.model`/`.runtime`, so the plan that runs is the plain one (`prompt.build_prefix(state, resolution=None)`, the documented escape hatch);
* the serving path (`cli.py` `ask`/`run`) and the probe both plan from the **handle**, i.e. the model's chat template. On item c01 the two framings are 102 vs 119 prefix tokens and their cue rows disagree on the *magnitude* of the label mass — 0.03677 against 0.00696, a factor of 5.3 — while still picking the same winner and the same `low_mass`/`ok` word.

So the switch's measured effect is framing-dependent, so the default stays frozen here: the product runs the chat framing (section 1's numbers, reproduced through the serving path by `tools/e3d_engine_check.py`: 12/12 decision cells on six stratified items), and the bench — the instrument every published quality table comes from — cannot yet confirm it because it does not send that prompt. Fixing the bench is its own card (the `plan_context` seam), and the cue default should not move until it lands.

