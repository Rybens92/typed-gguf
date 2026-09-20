# E3d CI reading — auditor scorecard (card `t_fc037544`, audit of `t_d90404ac`)

**Verdict (TL;DR).** Every number the card asks me to check reproduces exactly from the committed
record, with an independent implementation and with an exact (seed-free) bootstrap — the marginal
agreement table, both paired readings, the coverage axis and the per-type split. The record is
internally consistent to the last derivation I can check offline, the evidence document regenerates
byte-for-byte, the bench arms recompute exactly, and I re-ran the 6-item serving-path check myself
(12/12 cells, byte-identical to the committed log). **On the two positions: the decision — default
frozen, mechanism shipped behind `--cue`, `json_field` excluded — is endorsed as taken; my one
disagreement with reading (a) is a wording one: the evidence doc's "What the numbers recommend" /
"The flip, if the second opinion agrees" implies the flip is licensed pending this reading. It is
not: on the card's own agreement rule nothing here is promotable, and the reliability axis is a
different criterion that needs an explicit decision. One text patch proposed (AWAITING APPROVAL);
zero numbers move.**

Card that settles it, verbatim: *"If two-step (or the opened field) wins by more than the CI noise:
propose the new default … If it does not hold: publish the negative result with the numbers and
leave the default alone."* Constraint 4: *"Do NOT touch Occamy's verdict: it is refused on all
at-the-cue shapes (30/30)."*

---

## 1. Independent recomputation (from `.e3d/full.json` only)

Reimplemented from scratch (`scripts/recompute_stats.py`): Wilson z=1.96, exact two-sided McNemar
(binomial tail), a Monte-Carlo bootstrap replicating the tool's exact convention (seed 20260919,
10 000 draws, order-statistic percentiles), **plus** an exact-DP bootstrap of the paired difference
(the Monte-Carlo target computed analytically — no seed at all) and 4 extra MC seeds × 100 000
draws. Result:

| statistic | card/evidence value | my recomputation | match |
|---|---|---|---|
| `shipped` 42/60 | 0.700 | 42/60 = 0.700000 | ✓ |
| `shipped` Wilson | 0.575–0.801 (card text says 0.802 — see F1) | 0.5749105–0.8010199 | ✓ |
| `two_step` 46/60 | 0.767 | 46/60 = 0.766667 | ✓ |
| `two_step` Wilson | 0.646–0.856 | 0.6456349–0.8556057 | ✓ |
| `json_field` 51/60 | 0.850 | 51/60 = 0.850000 | ✓ |
| `json_field` Wilson | 0.739–0.919 | 0.7388517–0.9190265 | ✓ |
| two_step discordant | a=3, b=7 | a=3, b=7 | ✓ |
| two_step McNemar p | 0.3438 | 0.343750 (exact) | ✓ |
| two_step risk diff + CI | +0.067, −0.033…+0.167 | +0.066667; MC-seed-replication, exact-DP and 4 own seeds × 100k all **[−0.033333, +0.166667]** | ✓ |
| json_field discordant | a=3, b=12 | a=3, b=12 | ✓ |
| json_field McNemar p | 0.03516 | 0.03515625 (exact) | ✓ |
| json_field risk diff + CI | +0.150, +0.033…+0.267 | +0.150000; **[+0.033333, +0.266667]** everywhere | ✓ |
| coverage above floor | 16 / 58 / 60 | 16 / 58 / 60 | ✓ |
| `low_mass` | 44 / 2 / 0 | 44 / 2 / 0 | ✓ |
| per type choice | 20/24 → 21/24 → 23/24 | 20 → 21 → 23 | ✓ |
| per type noul | 16/18 → 14/18 → 17/18 | 16 → 14 → 17 | ✓ |
| per type score | 6/18 → 11/18 → 11/18 | 6 → 11 → 11 | ✓ |

Extra numbers from the same recomputation (context for the ruling, not claims of the card):

* exact bootstrap tail mass: `P(resampled two_step diff ≤ 0) = 0.1284`;
  `P(json_field diff ≤ 0) = 0.0095` — the json_field exclusion is real but *thin* (see F5-ish note
  below).
* one-item fragility: move one discordant pair (json_field 3/12 → 4/11) → p = 0.1185 and the 2.5 %
  quantile lands on **0.0000 — the CI includes zero**. The json_field "clears the bar" verdict is
  one item wide.
* two_step extrapolation at the observed rates (5 % baseline-only, 11.7 % challenger-only):
  n=140 → 2.5 % quantile 0.0000 (borderline); **n=160 → +0.0063 (excludes zero)**; n=200 → +0.0100.
  This is the "larger item set" number the card asks for.
* paired McNemar on the **reliability axis** (not a card statistic — provided as input to any
  explicit criterion decision): `low_mass` 44→2 ⇒ p = 3.1e-11; 44→0 ⇒ p = 1.1e-13.
* correctness decomposition: on the 44 rows `shipped` calls `low_mass`, shipped is right 31 times,
  two_step 35; on the 16 rows shipped measured, both 11. The +4 net gains are concentrated in
  `score` (+4, 0 losses), `noul` loses 2, `choice` net 0.

## 2. Record-internal consistency (`.e3d/full.json`, all 60 items)

`scripts/consistency2.py` + `scripts/fixups.py` + `scripts/conf_check.py`:

* `probabilities == softmax(sequence_scores)` — **exact, 180/180, ≤1e-12** (mapped by first token;
  the dict order in the JSON is not wire order — the only apparent mismatch was my first checker's
  ordering bug).
* `confidence == normalized_peak(probabilities)` — **exact, 180/180, max diff 0.0**.
* `coverage == min(1, Σ first_mass)` — 900/900 checks (the 58 "newline" rows where the sum would
  exceed 1 are capped to 1.0 by `readout.coverage_from_scale` — the documented cap, not a defect;
  those are exactly the rows where all four labels collide on one first token).
* `got = argmax(probabilities)` (frozen lowest-index tie-break), `correct = (got == expected)`,
  `top_coverage = argmax(first_mass)`, `agrees = (top_coverage == got)`,
  `reliability = ok iff coverage ≥ floor` — all 180/180.
* devset correspondence: 60 ids in order, types and golds equal under the wire mapping
  (int→str, `True`/`False`→`yes`/`no`) — no real mismatch.
* prompt-bytes claims: `two_step` suffix tokens **identical** to `shipped` on all 60 items, readout
  position = cue+1 on all 60; `json_field` adds the opener to the suffix (+4 tokens on 42 items,
  +5 on 18 — the opener's token count) and moves the readout accordingly. `refused` counts: 0/0/0;
  closers: none; all 60 two_step items advanced (no refusal at the cue in this run) — consistent
  with the record's `advance: {rule: content}` on every item.
* advance semantics verified in code (`engine/decide.py::_advance_token`): `two_step` advances only
  when the cue row's argmax is **not** a turn-closer; a refusing cue never advances and its verdict
  is read exactly as `shipped` reads it. Constraint 4's protection is mechanical, and
  `tests/test_e3d_cue_switch.py::test_a_two_step_request_never_advances_past_a_refusal` gates it.

## 3. Regeneration (record → evidence doc)

* `python3 tools/e3d_cue_decision.py report --run .e3d/full.json --out … --json …` reproduces
  `docs/evidence/e3d_cue_decision_4b.md` **and** `.json` **byte-for-byte** under CPython 3.11
  (the box's system python — the version the committed artifacts were produced with).
* Working-tree hashes equal the HEAD blobs for `full.json`, the evidence `.md` and `.json`
  (17f5a109…, 107b8be7…, 5986ed42…).
* Caveat for automated diffs (F4): under CPython ≥3.12 the regenerated JSON differs on some `mean`
  fields in the **last ulp** (0.08833688777584976 vs …78) — Python 3.12+ uses compensated summation
  in `sum()`. Content-identical, byte different; pin 3.11 for byte-identity.
* `uv run --frozen pytest tests/test_e3d_cue_switch.py tests/test_e3d_cue_decision.py -q` →
  **37 passed** (19 + 18 collected — the card's counts are exact).

## 4. Bench arms (§6 / BENCHMARKS §8) — recomputed from the committed arm JSONs

`scripts/bench_verify.py`: `--cue shipped` 36/60 = 0.600 (Wilson 0.474–0.714), `low_mass` 13/60,
refused 0/60, median coverage 0.2711, above floor 47/60; `--cue two_step` 27/60 = 0.450
(0.331–0.575), `low_mass` 29/60, refused 8/60, median 0.1103, above floor 31/60 — the §8 table to
the digit, including the per-type row (16→15, 16→7, 4→5) and the sign reversal. The arms' own
overall blocks match the recomputation.
Cross-reference verified: E2's published row is 38/60 (agreement 0.6333) — §8's "36/60 vs 38/60,
two items off" is accurate.
The §6 c01 numbers have receipts in the implementer's session (export command below): a framing
probe measured **session plan** (plain framing) 102 tokens, tail `' database is reachable.\n'`,
cue-row mass **0.0367732 = E2's published 0.03677324** vs **handle plan** (chat template)
119 tokens, tail `'…<|Bot|></think>\n'`, mass **0.00696086** — both fresh measurements, same box.
Note for precision (F3): the Vulkan arms *in this section* read c01 = 0.0233 (plain framing, GPU
device visible), so "0.03677 (bench)" should be read as "the session-plan framing (reproducing
E2's CPU row)", not as the arms' own c01.

## 5. Serving-path check — re-run by the auditor

`uv run --frozen python tools/e3d_engine_check.py --record .e3d/full.json --items 6 --threads 4`
(CPU, devices hidden, same pinned runtime, same model): **12 ok, 0 different**, and every printed
cell (label, mass to the shown digits, reliability word) is **identical** to the committed
`.e3d/engine_check_serving.log` (`diff` of the 12 cell lines: no differences; log kept in
`logs/engine_check_rerun.log`). Model provenance: the Spark GGUF on this box hashes to
`5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2` — exactly the record's
`model_sha256`; size 4 375 021 152 bytes matches. This closes "the serving path reproduces the
probe" with my own execution, not with the implementer's log alone.

## 6. The ruling on the two positions

**(a) The implementer's reading — endorsed on the decision, with one wording caveat.**
The two decisive moves are correct:

1. *`json_field` clears the agreement bar but cannot be the default.* Verified: its opener changes
   the prompt bytes (+4/+5 tokens; the readout row moves), which (i) rewrites what every
   prompt-level table means (E1c label policy, E3b's cue×label grid, E3c's seven shapes — exactly
   the risk list in §5 of the evidence doc), and (ii) removes the at-the-cue refusal *row*: a
   refusing family would read as an answered row with a tiny mass, i.e. Occamy's verdict protocol
   (constraint 4 of the card) would be touched. Constraint 4 outranks the promotion rule's
   "propose the default" branch — a hard "do not touch" is not something a CI can clear.
2. *`two_step` does not take the default because its agreement gain is inside the noise.*
   Faithful to the card's rule ("If it does not hold: … leave the default alone").

**Caveat (the part of (a) I disagree with).** The evidence doc's §5 sentences *"What the numbers
recommend. `two_step` …"* and *"The flip, if the second opinion agrees."* read as if the flip were
licensed subject to this reading. It is not: on the card's own statistic the promotion bar is **not
met** (CI includes 0; p = 0.344), and the reliability axis (44→2 `low_mass`, p ≈ 3e-11) is a
*different* criterion the card does not name. Promoting on it is a governance decision for the
coordinator/user — not "what the numbers recommend". The mechanism shipped behind `--cue` is not a
promotion and is unaffected: shipping it was right, and the one-constant flip preparation is fine
as long as it stays unexercised. Proposal F2 fixes the wording (AWAITING APPROVAL).

**(b) "The card's rule is agreement-only, so a non-significant candidate must not be promoted at
all" — correct where it matters, incomplete where it doesn't.** Its demand is already satisfied
(nothing was promoted; the default is frozen). Its reading of the rule is the same one I endorse
for the *agreement* claim. Two notes: (i) it is not by itself a sufficient rationale for the frozen
default — `json_field` *is* agreement-significant, and only constraint 4 (plus the prompt-rewrite
costs) keeps it out; (ii) it should not be read as a blanket ban on any future default change on
another explicitly-adopted criterion (e.g. readout availability) — that would over-read a card
whose scope is "propose the new default *for this switch*, or publish the negative".

**What would change my mind** (per the card's request):

* a `two_step` paired agreement CI that excludes zero — on the observed rates, **≈140 items**
  reaches the boundary and **≈160** clears it (exact-DP extrapolation, §1);
* or an explicit coordinator/user decision to promote on the readout-availability axis, with the
  bench seam of §6 fixed first (per the implementer's own §6 argument) and the §5 invalidation
  list honored — then the flip is a decision, not an inference, and I'd support it;
* for `json_field`: no numbers would flip my ruling; only a change to the refusal-visibility
  protocol (e.g. the engine reading closers at the opener row) could make it default-eligible.

Confidence: the numbers — replica-exact, CERTAIN. The reading of the card — HIGH. The extrapolation
— MEDIUM (assumes stationary discordant rates).

## 7. Findings and proposals (nothing applied — AWAITING APPROVAL)

| id | sev | class | what (receipt) | fix |
|---|---|---|---|---|
| F1 | NICE | — | the task card's check list says `shipped` Wilson "0.575–**0.802**"; recomputed and committed value is 0.575–**0.801** (0.80102 rounds to 0.801). The evidence doc and JSON are correct. | none (card text is a kanban artifact; recorded here for the record) |
| F2 | IMPROVE | PROCEDURAL | evidence doc §5 phrasing implies the flip is licensed pending this reading (quotes in §6 above) | text patch in `tools/e3d_cue_decision.py::decision_section()` + regenerate `docs/evidence/e3d_cue_decision_4b.{md,json}` — exact replacement text in §7.1 |
| F3 | NICE | — | §6's "0.03677 (bench)" is the session-plan framing number that reproduces E2's row; the Vulkan arms' own c01 is 0.0233 — a reader can mistake one for the other | optional one-line parenthetical in the same generated §6 text + BENCHMARKS §8 (same regeneration) |
| F4 | NICE | TOOLING | byte-identical regeneration of the evidence JSON requires CPython 3.11; under 3.13 the `mean` fields differ in the last ulp (compensated `sum()` since 3.12). Also: the repo `.venv` was dangling when this audit started (uv rebuilt it — the same trap the E3d sweep notes record) | one line next to the reproduce commands: "byte-identity of the JSON holds under CPython 3.11; 3.12+ changes the last ulp of `mean` fields — diff with tolerance" |
| keep | — | — | see §8 | — |

### 7.1 F2 — exact replacement text (ready to paste into `tools/e3d_cue_decision.py`)

Old paragraph (lines ~419–423):

    "**What the numbers recommend.** `two_step` — the only candidate that leaves every prompt",
    "byte, every published prompt-level table and the refusal verdict intact, and takes the",
    "engine's own coverage verdict from a mostly-unusable row to a measured one. Its agreement",
    "gain is *not* significant (the paired CI above includes zero) and the evidence document",
    "says so: this is a readout-position fix, not an accuracy claim.",

New:

    "**What the data supports — and what it licenses.** `two_step` — the only candidate that",
    "leaves every prompt byte, every published prompt-level table and the refusal verdict",
    "intact, and takes the engine's own coverage verdict from a mostly-unusable row to a",
    "measured one. Its agreement gain is *not* significant (the paired CI above includes zero),",
    "so under the card's promotion rule (a paired agreement CI that excludes zero) **nothing",
    "here is licensed to become the default** — this is a readout-position fix, not an accuracy",
    "claim. The reliability axis (44 → 2 `low_mass`; exact McNemar 3.1e-11 on the same 60",
    "items) is a separate criterion the card does not state — promoting on it is a",
    "coordinator/user decision, not an inference from this table (auditor, card t_fc037544).",

Old paragraph (lines ~432–436, also fixes its stray `)`):

    "**The flip, if the second opinion agrees.** One constant,",
    "`schema.OPTION_DEFAULTS[\"cue\"]`",
    "+ `Options.cue`) and this document's default column; the mechanics (the readout, the",
    "refusal stop, the bench knob, the CLI flag, the payload key) are already gated by",
    "`tests/test_e3d_cue_switch.py`.",

New:

    "**What a flip would need (not licensed by this data).** One constant,",
    "`schema.OPTION_DEFAULTS[\"cue\"]`",
    "(plus `Options.cue`) and this document's default column; the mechanics (the readout, the",
    "refusal stop, the bench knob, the CLI flag, the payload key) are already gated by",
    "`tests/test_e3d_cue_switch.py`. A future flip needs either (a) a `two_step` agreement CI",
    "that excludes zero — at the observed discordant rates (3/60 vs 7/60) roughly a 140-160",
    "item set — or (b) an explicit decision to promote on the readout-availability axis",
    "instead, with the bench seam of section 6 fixed first.",

Then: `python3 tools/e3d_cue_decision.py report --run .e3d/full.json --out
docs/evidence/e3d_cue_decision_4b.md --json docs/evidence/e3d_cue_decision_4b.json` and
`uv run pytest tests/test_e3d_cue_decision.py -q` (the render tests may pin fragments of the text —
adjust the assertions in the same commit). Zero numbers move; the regenerated doc stays
byte-reproducible.

## 8. Keep & replicate

* the **deterministic, offline number generator** (record → analysis → doc) that regenerates
  byte-for-byte; a probe whose evidence is a first-class artifact;
* the **paired statistics discipline**: exact McNemar + seeded paired bootstrap, with the marginal
  Wilson intervals printed *next to* the paired reading precisely as the tempting wrong reading;
* the **constraint-aware design**: `two_step`'s conditional advance makes the refusal verdict
  mechanically untouchable, and there is a dedicated gate for it;
* the **risk note** enumerating exactly which published tables each candidate would invalidate —
  before any default moves;
* the **honest mutation note** ("attempted and not achieved", no score claimed beyond 74/50/25);
* the **framing bug carded separately** (t_6de5fc53) instead of scope-creeping into the E3d card;
* the **engine-vs-probe check as a tool** (`e3d_engine_check.py`) and its committed log — an
  instrument that documents where it may disagree (batching) and compares only the cells a
  published row is read from.

## 9. What I did not re-run / caveats

* I did not re-run the full 60-item probe (multi-hour, and the card's inputs are the committed
  record). The serving path was re-checked on 6 of 60 items (§5).
* `wall_s`, `cue_decode_s`, `prefill_ms` are the record's self-reports (timing, not correctness);
  not independently reproducible without re-running the probe.
* The mutation-sweep numbers are outside this card's question (not checked).
* The record's `devset` path (`/workspace/ggufone/...`) is the worker-container path of the same
  committed file; verified by content (ids/types/golds), not by path.
* Side effect of this audit: the repo `.venv` was dangling at start; uv (0.12.5) rebuilt it (now
  CPython 3.13.15, synced — `uv run pytest` works). No tracked file changed (`git status` clean
  for all E3d paths). See F4.

## 10. How to reproduce my checks

    # 1. independent statistics (pure stdlib, no repo imports)
    python3 scripts/recompute_stats.py
    # 2. record consistency (derivations, prompt bytes, devset, advance)
    python3 scripts/consistency2.py && python3 scripts/fixups.py && python3 scripts/conf_check.py
    # 3. regeneration (byte-identity needs CPython 3.11)
    python3 tools/e3d_cue_decision.py report --run .e3d/full.json --out /tmp/a.md --json /tmp/a.json
    diff docs/evidence/e3d_cue_decision_4b.md /tmp/a.md && diff docs/evidence/e3d_cue_decision_4b.json /tmp/a.json
    # 4. bench arms
    python3 scripts/bench_verify.py
    # 5. serving path (model + pinned runtime, CPU, ~6 min)
    uv run --frozen python tools/e3d_engine_check.py --record .e3d/full.json --items 6 --threads 4 \
      --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
      --runtime /var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
    # 6. the implementer's §6 receipts (session export)
    hermes -p code-tdd sessions export --session-id 20260919_190155_05a416 --format jsonl - > /tmp/e3d_session.jsonl
    # then grep 0.0367732 / "exactly E2"

Receipts: `logs/engine_check_rerun.log` (my run, 12/12), `logs/model.sha256`,
`scripts/` (all checkers). Session quotes for §6: frames "session plan → 0.0367732 = E2's
published" and "handle plan → 0.00696086".

_Proposals status: AWAITING APPROVAL (F2/F3/F4 recommended; nothing applied by the auditor.)_
