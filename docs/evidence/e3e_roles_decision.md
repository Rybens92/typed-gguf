# E3e — which prompt policy beats the shipped one (2026-09-20T10:55:14Z)

- dev set: `devset.jsonl` (choice 24, noul 18, score 18) · items per cell: 60 · baseline: `shipped/answer_sheet`
- model: Spark-X2.5-4B-Q8_0.gguf · backend: vulkan · gpu-layers None · threads 4
- reports: 8 cells, 28 paired comparisons (discordant-pairs)

## The default is frozen (the probe and the table's own re-score)

- probe `.e3e/probe_default.json` vs baseline `.e3d/bench_templated_shipped.json` (6 shared items): **frozen** — the prompt bytes and the decisions are the committed baseline's: 6/6 items, prefix_tokens identical on every item; numbers re-scored, not bit-identical (max |delta p| = 3.78e-02 over 6 item(s))
- the table's own instrument, `.e3e/bench_shipped_answer_sheet.json` vs `.e3d/bench_templated_shipped.json` (60 shared items, tolerance 0.005): **1 decision(s) moved** — the re-score is not the same measurement: n16: got 'no' != 'no' (verdict 'low_mass' != 'low_mass'; refused False != True), p(no) 0.6925871631703637 != 0.6812322310342762, p(yes) 0.30741283682963627 != 0.31876776896572384; max |delta p| = 1.14e-02, max |delta coverage| = 6.25e-03, prefix_tokens identical on every item

## The cells

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

## The policy each cell ran under

| cell | cue | chat_format | json_contract | framing | renderer (thinking) | family | labels | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `shipped/answer_sheet` | shipped | answer_sheet | question | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `shipped/role_split` | shipped | role_split | question | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `two_step/answer_sheet` | two_step | answer_sheet | question | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `two_step/role_split` | two_step | role_split | question | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `json_instructed/answer_sheet` | json_instructed | answer_sheet | question | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `json_instructed/role_split` | json_instructed | role_split | question | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `json_instructed/answer_sheet/system` | json_instructed | answer_sheet | system | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |
| `json_instructed/role_split/system` | json_instructed | role_split | system | gguf-renderer | internal (suppressed) | spark2_5 | chat-template: spark2_5 / internal | — |

## Paired comparisons (same items, item by item)

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

## Decision

- rule: agreement on the 60 committed dev items, paired by item: a challenger wins only when the paired difference's 95 % interval excludes zero and the exact two-sided McNemar p < 0.05; ties are called identical, everything else is not by more than the CI noise
- best cell: `json_instructed/answer_sheet` — 51/60 correct, 0 refusals

- `json_instructed/answer_sheet` vs `shipped/answer_sheet`: **wins** — 13 items only it got right, 4 only shipped/answer_sheet did; difference +0.150 (95 % CI +0.021..+0.279), exact McNemar p=0.049
- `json_instructed/role_split` vs `shipped/answer_sheet`: **interval clears zero, exact test does not** — 11 items only it got right, 3 only shipped/answer_sheet did; difference +0.133 (95 % CI +0.016..+0.251), exact McNemar p=0.057
- `json_instructed/role_split/system` vs `shipped/answer_sheet`: **interval clears zero, exact test does not** — 11 items only it got right, 3 only shipped/answer_sheet did; difference +0.133 (95 % CI +0.016..+0.251), exact McNemar p=0.057
- `shipped/role_split` vs `shipped/answer_sheet`: **interval clears zero, exact test does not** — 12 items only it got right, 4 only shipped/answer_sheet did; difference +0.133 (95 % CI +0.007..+0.260), exact McNemar p=0.077
- `json_instructed/answer_sheet/system` vs `shipped/answer_sheet`: **not by more than the CI noise** — 10 items only it got right, 4 only shipped/answer_sheet did; difference +0.100 (95 % CI -0.020..+0.220), exact McNemar p=0.180
- `two_step/answer_sheet` vs `shipped/answer_sheet`: **not by more than the CI noise** — 7 items only it got right, 3 only shipped/answer_sheet did; difference +0.067 (95 % CI -0.035..+0.169), exact McNemar p=0.344
- `two_step/role_split` vs `shipped/answer_sheet`: **interval clears zero below, exact test does not** — 5 items only it got right, 19 only shipped/answer_sheet did; difference -0.233 (95 % CI -0.382..-0.085), exact McNemar p=0.007

## Recommendation

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

## Caveats the numbers carry

- 60 committed dev items: a single cell's 95 % interval is up to 24.5 points wide, so single-cell differences below that are noise by construction — which is what the paired columns are for.
- the arm reports' *own* `overall`/`per_type` intervals use `z = 1.96` (`harness.wilson_interval`); every interval in this table is recomputed from the rows at the precise `z` (1.959963984540054), so the two differ from the 6th decimal on (0.574910530336 vs 0.574912920531) — both correct, no number depends on the choice.
- `score` is the type E3c/E3d found hardest: a policy that helps choice/noul and does not help score is still a policy decision, not a quality result.

**Comparability cost.** Every cell in this table is measured under the *same* instrument (the same 60 committed dev items, temperature 0, fixed seed, `--backend vulkan`) — so the table compares itself and nothing else. What it does **not** compare against is any published row: a `role_split` placement, a `json_instructed` ask line, or a contract stated in the framing each change the bytes the model sees. A cell published as *the* quality row would carry its policy line (`- prompt policy: …`) and every other published quality row would have to be re-measured under the same policy before it could sit next to it.
