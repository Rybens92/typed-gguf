# The bench must run the product wiring — prompt parity (card `t_6de5fc53`)

Routed from `t_6d066e9a` (filed while running E3d, card `t_d90404ac`). One sentence: **the bench
planned the executed context twice, and the plan that ran was built from a `ModelSession`** — an
object that resolves no chat template — so every published quality row measured the plain E1b
framing while `ggufone ask`/`run` sent the model's chat template. This document is the fix, the
two parity gates (RED → GREEN), the re-measured tables with their framing markers, and the E3d
re-check. **Framing is named on every row below**: `plain` = the pre-fix instrument, `chat-template`
= the fixed one.

## 1. The defect, reproduced

`src/ggufone/bench/harness.py` → `LiveModel.decide` called `decide.plan_context` **twice**:

* `plan_context(request, handle)` — the model handle, so `decide.resolve_template` resolves the
  model's own chat template. Used only to size the session (`n_ctx`).
* `plan_context(request, live)` — the live `ModelSession`, and **this plan is the one that ran**.
  A `ModelSession` exposes no `.model`/`.runtime`, so `resolve_template(request, session)` returns
  `None` and `prompt.build_prefix(state, resolution=None)` silently falls back to the plain E1b
  framing — which `prompt.py`'s own docstring calls "the escape hatch for a session with no model
  handle, not a silent default for a real model".

The serving path (`cli.py` `ask`/`run`) plans from the handle, so the two instruments sent
different prompts. Measured on dev item `c01`, `Spark-X2.5-4B-Q8_0.gguf`, by the live parity gate
below (the pre-fix tree):

| what | bench (pre-fix, plain) | serving path (chat template) |
|---|---|---|
| prefix tokens prefilled | **102** | **119** |
| cue-row label mass | 0.0367732 (= `docs/evidence/e2_quality.json`) | 0.00696086 |

## 2. The fix

`LiveModel.decide` now plans **once**, from the **handle**, and the session is sized from that same
plan (`n_ctx=n_ctx or plan.n_ctx`); the executed plan is that plan. The silent-plain path for a
real handle is gone — `plain` is reachable only by asking for it (`--template plain`).

The reports now carry the framing they measured, so no table can be silent about its prompt again
(card requirement 5):

* every quality/calibration row: `framing` (the response's own `engine.template` surface) and
  `prefix_tokens` (`suites._devset_row`);
* the report: `framing` = `{labels, prefix_tokens, mixed}`, printed by `harness.render_report` as
  `- framing: chat-template: spark2_5 / internal · prefix tokens: …` (`mixed: true` when the rows
  disagree — a table that mixed framings says so).

## 3. The parity gates (RED → GREEN)

**Offline structural gate** (`tests/test_bench_prompt_parity.py`): `plan_context` is called once,
with the handle; the executed plan is that plan. On the pre-fix tree:

```
E       AssertionError: the executed plan is not the one resolved from the handle: the bench
        planned from the session (plain framing), which is a different prompt than the serving
        path sends
E       assert (1, 2) == (101, 102, 103)
1 failed, 1 passed in 0.27s
```

**Live byte-identity gate** (`tests/test_bench_live.py::test_the_bench_sends_the_same_prompt_as_the_serving_path`):
the same request and the same model through `cli.decide_payload` (the serving path) and through
`suites.run_suite --suite quality` (the bench), capturing what each path *prefills*. On the
pre-fix tree, with `GGUFONE_BENCH_MODEL=Spark-X2.5-4B-Q8_0.gguf`:

```
E       AssertionError: the bench sends 102 prefix tokens and the serving path 119: the executed
        plan is not the one the product resolves
E       assert [3683, 599, 2...8067, 27, ...] == [0, 130972, 1...599, 259, ...]
1 failed in 23.69s
```

After the fix (same command, `-s`):

```
item c01: serving prefix 119 tokens, bench prefix 119 tokens, framing chat-template: spark2_5 / internal
1 passed
```

Raw logs: `docs/evidence/framing/red_live_parity.txt`, `red_offline_parity.txt`,
`green_live_parity.txt`, `green_offline_parity.txt`.

## 4. The re-measured tables

**Scope of every row below.** Same box (the worker container: `cpu.max 200000 100000` → 2 CPU-s,
`memory.max` 8 GiB; every report carries its `host` block), same model, same 60 committed dev items,
same command — `python -u tools/e2_reproduce.py --suite quality --model Spark-X2.5-4B-Q8_0.gguf
--backend auto --runs 1 --threads 4 --cue <shape>` — and only the tree (pre-fix / post-fix) and
`--cue` move. Reports: `.e3d/bench_plain_<cue>.json` (pre-fix) and `.e3d/bench_templated_<cue>.json`
(post-fix). Every row carries its own `framing` and `prefix_tokens`.

### 4.1 The instrument, either way, on every cue shape

| arm | framing | agreement | Wilson 95 % | `low_mass` | refused at the cue | coverage median | above the 0.10 floor | choice · noul · score |
|---|---|---|---|---|---|---|---|---|
| `--cue shipped`, **pre-fix** | plain (prompt.py E1b framing) | 36/60 = 0.600 | 0.474–0.714 | 13/60 | 0/60 | 0.2711 | 47/60 | 16/24 · 16/18 · 4/18 |
| `--cue two_step`, **pre-fix** | plain (prompt.py E1b framing) | 27/60 = 0.450 | 0.331–0.575 | 29/60 | 8/60 | 0.1103 | 31/60 | 15/24 · 7/18 · 5/18 |
| `--cue json_field`, **pre-fix** | plain (prompt.py E1b framing) | 46/60 = 0.767 | 0.646–0.856 | 0/60 | 0/60 | 0.8544 | 60/60 | 21/24 · 17/18 · 8/18 |
| `--cue shipped`, **post-fix** | chat-template: spark2_5 / internal | 42/60 = 0.700 | 0.575–0.801 | 45/60 | 1/60 | 0.03847 | 15/60 | 20/24 · 16/18 · 6/18 |
| `--cue two_step`, **post-fix** | chat-template: spark2_5 / internal | 47/60 = 0.783 | 0.664–0.869 | 3/60 | 3/60 | 0.9708 | 57/60 | 21/24 · 15/18 · 11/18 |
| `--cue json_field`, **post-fix** | chat-template: spark2_5 / internal | 51/60 = 0.850 | 0.739–0.919 | 0/60 | 0/60 | 0.9997 | 60/60 | 23/24 · 17/18 · 11/18 |

**The pre-fix half is a re-measure, not a copy**: `.e3d/bench_plain_shipped.json` and
`.e3d/bench_plain_two_step.json` are item-for-item identical to the published
`.e3d/bench_shipped.json` / `.e3d/bench_two_step.json` of card `t_d90404ac` — every row's `correct`,
`reliability`, `coverage`, `answer` and `gold` compare equal — so **the only thing that moved
between the two halves of this table is the fix**, and the plain rows reproduce on demand.

### 4.2 What the framing moves

* **The prefix itself**: 102 → 119 tokens on `c01` (§1, reproduced live in §3). The same *state*
  becomes a different byte sequence, and the row the cue is read from moves.
* **The mass split inverts.** `--cue shipped`: `low_mass` 13/60 (plain) → 45/60 (chat template),
  coverage median 0.2711 → 0.03847, above the floor 47 → 15. The chat template parks the cue row on
  a bare newline, so the label mass read there is a tail — that is the *product's* actual behaviour,
  and it is why §2's published split (48 `ok` / 12 `low_mass`) cannot be read as a statement about
  the serving path. Re-derived with §2.1's shape, both framings:

| framing | reliability | items | correct | agreement | 95 % CI |
|---|---|---|---|---|---|
| plain (pre-fix) | `ok` | 47 | 31 | 0.6596 | 0.5167 – 0.7783 |
| plain (pre-fix) | `low_mass` | 13 | 5 | 0.3846 | 0.1771 – 0.6448 |
| chat template (post-fix) | `ok` | 15 | 11 | 0.7333 | 0.4805 – 0.8910 |
| chat template (post-fix) | `low_mass` | 45 | 31 | 0.6889 | 0.5433 – 0.8047 |

* **The agreement moves with it**: 36/60 → 42/60 on the same 60 items (+6), i.e. the plain framing
  was neither neutral nor kind to the product — it was a different prompt.
* **The cue switch reverses on the corrected instrument** (§5).

### 4.3 The two 35B-A3B models — in-container probes (capped scope)

**Scope, first.** §7's Tiel and Occamy quality rows are **[host]** rows under the fit plan's
placement; this box caps memory at 8 GiB, so a [host]-comparable re-run is not possible from here
and Tiel's corrected 60-item quality table is **routed, not claimed**. What is runnable is the
*paired* question — does the framing move these models? — on six committed dev items (the head of
`docs/evidence/tiel_chunks/devset_001.jsonl`: `c01 s01 n01 c02 s02 n02`, two per type), same box,
same command `tools/e3c_tiel_reproduce.py --suite quality --backend vulkan --threads 4
--gpu-layers <N>`, only the tree moves:

| model (probe) | framing | agreement | Wilson 95 % | `low_mass` | refused at the cue | coverage median | above the 0.10 floor | choice · noul · score |
|---|---|---|---|---|---|---|---|---|
| Occamy 1.0, **pre-fix** | plain (prompt.py E1b framing) | 3/6 = 0.500 | 0.188–0.812 | 6/6 | 0/6 | 0.03233 | 0/6 | 1/2 · 0/2 · 2/2 |
| Occamy 1.0, **post-fix** | chat-template: qwen35moe / internal | 1/6 = 0.167 | 0.030–0.564 | 6/6 | **6/6** | 4.02e-06 | 0/6 | 0/2 · 0/2 · 1/2 |
| Tiel-Coder 35B-A3B, **pre-fix** | plain (prompt.py E1b framing) | 3/6 = 0.500 | 0.188–0.812 | 3/6 | 3/6 | 0.2605 | 3/6 | 1/2 · 0/2 · 2/2 |
| Tiel-Coder 35B-A3B, **post-fix** | chat-template: qwen35moe / builtin | 0/6 = 0.000 | 0.000–0.390 | 6/6 | **6/6** | 1.94e-04 | 0/6 | 0/2 · 0/2 · 0/2 |

Raw reports: `docs/evidence/framing/{occamy,tiel}_{plain,templated}_probe.json`; placement receipts
next to them (`*_placement.json`, the sink with `device_log` dropped).

**The pairs are placement-matched, which is the only way a six-item probe can carry weight.**
Occamy: 7 layers, `kv_type` auto, no degrade, in both arms. Tiel: 4 layers in both arms — the
pre-fix arm got there by de-escalating from the 9-layer ask (`W_BACKEND_OOM`, `W_FIT_DOWNGRADE`),
the post-fix arm asked for 4 directly — with identical session parameters in both arms of each pair
(`n_ctx` 165, `n_prefix` 94, `n_seq_max` 3, `threads` 4, `kv_type_used` auto). What moves between
the arms is the framing, not the placement.

**What the probes say.** The direction is the 4B's, and harder: with the product's own prompt both
35B-A3B models collapse at the cue — 6/6 `low_mass` and **6/6 refusals** — where the plain framing
had them answering (3/6 and 3/6). Six items is a probe, not a table (a 0/6 Wilson interval reaches
0.39), so this is a direction to re-measure on the host; §7's published rows stay as they are.

## 5. The E3d re-check (card `t_d90404ac`)

`docs/BENCHMARKS.md` §8 used to read the cue switch as **inverted** in the bench ("`two_step` costs
agreement 36/60 → 27/60, doubles `low_mass` 13 → 29") and concluded that the cue's effect is
*framing-dependent*. That prose was the pre-fix instrument's: the arms sending the plain prompt,
where the shipped cue already sits on the answer. Corrected, the same three shapes say the
opposite, and they land on the probe's own numbers:

| statistic | probe (`e3d_cue_decision_4b.md` §1) | bench, post-fix |
|---|---|---|
| `shipped` agreement | 42/60 = 0.700 | 42/60 = 0.700 |
| `two_step` agreement | 46/60 = 0.767 | 47/60 = 0.783 |
| `json_field` agreement | 51/60 = 0.850 | 51/60 = 0.850 |
| paired `two_step` vs `shipped` | **KEEP** — risk difference +0.067, 95 % CI −0.033…+0.167 | +5 correct items, `low_mass` 45 → 3 |
| coverage above the 0.10 floor | 0.267 → 0.967 (`shipped` → `two_step`) | 15/60 → 57/60 |

**The E3d conclusion survives, and its premise is replaced**: `two_step` is not a cost to agreement
with a coverage windfall (the bench's old reading) nor a pure win (its own reading) — both
instruments now put it *slightly ahead of `shipped` on agreement, inside the paired CI, and far
ahead on the engine's own coverage verdict*, which is what §2's `decide()` verdict already said.
The default stays `shipped`: E3d froze it deliberately and this card is the instrument, not the
default. `docs/evidence/e3d_cue_decision_4b.md` §6 is regenerated with the framing markers and both
instruments, and `docs/BENCHMARKS.md` §8's prose is rewritten around the corrected table.

## 6. What is not claimed

* **The 35B quality rows are not re-published.** §4.3's probes are capped-scope (8 GiB, 4/7 layers,
  six items); §7.3/§7.4's [host] rows are untouched and now carry a pre-fix marker. A [host] re-run
  of Tiel's 60 items under the corrected instrument is the follow-up (35.6 s/batch there, cheap) and
  is where the "Tiel is not starved" reading should be re-derived.
* **The cue default does not move here** — E3d's decision, restated in §5.
* **Nothing is claimed about E2's published 38/60 vs today's plain 36/60** beyond what §7 already
  records: a same-framing difference of two items, attributed to fit/box drift.
* **The probe verdicts are paired over six items** (Wilson intervals printed above), and they are
  not a re-test of E3c's batch shape (`--suite batch`, `readout: sequence`), which this fix does not
  change and which §7.4's bench-vs-serving split lives on.

## Receipts

* parity gates: `docs/evidence/framing/red_offline_parity.txt`, `red_live_parity.txt`,
  `green_offline_parity.txt`, `green_live_parity.txt`
* the 4B arms: `.e3d/bench_{plain,templated}_{shipped,two_step,json_field}.json`
  (`bash .e3d/run_bench_arms.sh`; the plain arms come from the pre-fix tree)
* the 35B probes: `docs/evidence/framing/{tiel,occamy}_{plain,templated}_probe.json` +
  `*_placement.json`
* the E3d re-render: `docs/evidence/e3d_cue_decision_4b.md` (§6 regenerated) and
  `docs/evidence/e3d_cue_decision_4b.json`
* the code: the `fix(t_6de5fc53)` commit — `src/ggufone/bench/harness.py` (`LiveModel.decide`,
  the framing helpers), `src/ggufone/bench/suites.py` (row + report framing),
  `tests/test_bench_prompt_parity.py`, the live gate in `tests/test_bench_live.py`,
  `tests/fake_engine.py`
