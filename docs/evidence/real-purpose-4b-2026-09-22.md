# Real-purpose test on the on-disk 4B — support-inbox triage (card `t_977ad206`, 2026-09-22)

**Question this answers:** what purposes do typed-decision engines (TypeSafe Jev / "System One",
here: this repository's `typed-gguf`) serve in the wild, and if a human used one for a single
concrete workplace purpose on a 4B GGUF *on this disk*, what would the results actually be —
reliable, or not?

Everything below is executed evidence: every engine call is in
`report.json` / `devset_crosscheck.json` with its exit-asserted command line, and the raw
responses' parsed fields sit next to it. Nothing here is quoted from a blog and presented as
measurement.

---

## 1. Survey — what this class of engine is used for (sources fetched 2026-09-22)

Return-shape column maps each use to one of this engine's three question types
(`choice` = one of N, `score` = position on an ordered 2–10 level scale, `noul` = p(yes)).

| # | purpose | return shape | source (fetched) |
|---|---|---|---|
| P1 | **Inbound support triage / routing** — department, queue, intent of a ticket | `choice` over the queue set (+ `noul` for "is it urgent") | <https://www.langchain.com/blog/building-a-harness-with-jev> ("support-ticket example", `is_urgent` noul), <https://www.datacamp.com/blog/system-one-models-jev> ("routing a support ticket … Choice over {billing, technical, sales, spam}") |
| P2 | **Urgency / priority scoring** of a message so code can prioritise a queue | `score` (levels) or `noul` (p(urgent)) | <https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e> (Score 2–10 ordered levels, `Noul` returns one number), <https://typesafe.ai/blog/introducing-system-one-models-and-jev> ("classify, route, score") |
| P3 | **Guardrail / tool-risk gating** before an agent acts | `choice` (read-only / reversible / destructive) or `noul` (block?) | <https://www.langchain.com/blog/building-a-harness-with-jev> (`AutoModeMiddleware`, "check tool calls for risky decisions … and block calls before the tool executes"), <https://flaviocopes.com/jev/> |
| P4 | **Model / handler routing** — pick the cheapest model that can do the job | `choice` with per-model `criteria` descriptions | <https://www.langchain.com/blog/building-a-harness-with-jev> (`ModelRouterMiddleware`, `ModelChoice`, "least costly model that can complete the task") |
| P5 | **Verifying generated text** — does each claim hold against the source; is a citation supported | `noul` per claim (flag low-probability ones) | <https://flaviocopes.com/jev/> ("checks each claim against the transcript with a Noul", cookbooks: "whether a quoted citation supports the claim"), <https://typesafe.ai/blog/introducing-system-one-models-and-jev> ("Verify everything. Score, judge, verify, guardrail") |
| P6 | **Bulk labelling / map-reduce over a dataset** (tag every row, cheaply) | `choice` (topic) or `score` (quality) | <https://flaviocopes.com/jev/> (1,018 papers over 24 topics for $0.08; 98,000 listings), <https://www.datacamp.com/blog/system-one-models-jev> (50M review rows) |
| P7 | **Retrieval triage** — relevance of a passage, and prompt-injection screening before it reaches the answering model | `noul` per query–passage pair | <https://flaviocopes.com/jev/> (TypeSafe cookbooks: re-rank BM25 shortlists, "hidden prompt injections") |

Class claims recorded for comparison in §5 (same sources): 70–500 ms end-to-end, 40–200× faster,
$0.042/MTok input with free output, "can't hallucinate"/0% structured-output errors, calibrated
confidence ("higher confidence means higher accuracy"), and ~68% accuracy on TypeSafe's own
4-workflow eval — which the vendor, and the two independent write-ups, all label *self-run and
unreproduced* (dev.to §caveats; datacamp "the open questions are whether an independent benchmark
confirms the accuracy parity").

---

## 2. Chosen purpose and protocol

**Purpose: support-inbox triage** (P1 + P2 + the escalation half of P5's "send the uncertain cases
to a human"). One human — a support lead — reads an inbound message and needs three decisions:
which queue owns it, how much the customer is affected, and whether a human must take it before an
automated reply goes out. Why this one: it is the class's canonical example (P1) and it exercises
all three return shapes in one request.

**Items:** `items.jsonl` — 30 synthetic workplace messages, 4 queues (billing 8 / technical 8 /
account 7 / policy 7), 16 escalate-true / 14 false, severity 9/13/8.
*Generation basis, stated honestly:* authored for this run on 2026-09-22; **not** production
traffic and **not** sampled from a public corpus. Each item was written with one decisive cue and
its gold labels were fixed from the written rubric below **before any model call**; the cue is
recorded per item in `items.jsonl`. Item difficulty is therefore a design choice, not a sampled
distribution — see §6.

**Label rubrics (also the text the model was given, `questions.json`):**

- `queue` (choice): **route to the function that must act, not by the first keyword** —
  `billing` charges/invoices/refunds/payment methods · `technical` product errors, bugs, outages,
  broken integrations · `account` login, access, seats, plans, permissions · `policy` terms, data
  protection, legal, abuse reports.
- `severity` (score, 3 levels): 0 "no customer impact stated" · 1 "impact stated but not blocking:
  delays, extra cost, duplicated work, or a regulatory deadline" · 2 "blocking or immediate: work
  stopped, service suspended, a deadline today or tomorrow, or an active security exposure".
- `escalate` (noul): true iff the message states (T1) a charge/refund/pricing dispute or a failed
  payment blocking service, (T2) legal/privacy/security matter, (T3) an explicit request for a
  person, or (T4) the customer's access at risk. Each true item records its trigger.

**Engine + model (verbatim from the responses):** `typed-gguf 0.1.0` · llama.cpp `b11026`
(bundle `linux-x64-vulkan`) · effective backend `vulkan`, `n_gpu_layers` 36 (fit plan:
`est_weights_bytes` 4,724,883,456 = 4,506 MiB, context 252 MiB, compute 285 MiB, budget
`budget_bytes` 6,138,363,904 = 5,854 MiB, host fingerprint `a8d49ce94b68e6f8`) ·
readout `sequence` · cue `json_instructed` · chat_format `role_split` (`json_contract: question`) ·
`n_ctx` per request 256 or 512 (fit cap 4096; states are 409–438 input tokens) ·
**`calibrated: false` on every call** · coverage floor 0.10 · model
`/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 B, `spark2_5`, Q8_0,
sha256 `5c2c3c19…dea2`) · warm keep host `674075b9cf1676ec` served all 30 calls (zero model loads
inside the run).

**Exact command per item** (one request per item — its own state, so its own prefill; the three
questions are answered in one pass):

```
uv run typed-gguf run --questions docs/evidence/real-purpose-4b-2026-09-22/questions.json \
  --state @<item>.txt --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
  --out <item>.json --keep-alive 10m
```

Driver: `run.sh` (30 calls, all `exit=0`, all recorded with wall ms in `report.json`).
Run dates 2026-09-22 13:24:37–13:25:13 UTC.

## 3. Results (n = 30, one run per item)

| question | accuracy | Wilson 95% | baseline |
|---|---|---|---|
| `queue` (choice, 4 options) | **26/30 = 86.7%** | 70.3 – 94.7% | majority class 26.7% |
| `escalate` (noul, p>0.5) | **27/30 = 90.0%** | 74.4 – 96.5% | majority class 53.3% |
| `severity` (score, 3 levels, argmax) | **24/30 = 80.0%** | 62.7 – 90.5% | — |
| `severity` mean absolute error | 0.267 level (score value 0.361) | | 28/30 within one level |

- **Queue errors are spread, not systematic:** one per class (billing→policy, technical→account,
  account→technical, policy→technical). The one *confident* error (b04, contract price vs quote,
  confidence 0.944) is arguably a rubric-boundary case — a human might route it to legal.
  The three others sit at 0.34–0.59 confidence.
- **Escalation errs only upward:** 16/16 T1/T2/T4 tickets held, 0 false negatives; the 3 misses
  (b02 card update, t01 total-endpoint outage, a03 offboarding) are *over*-escalations under my
  written policy — each one is a case a human could reasonably have queued for a person, so label
  and intuition genuinely diverge there.
- **Severity:** level 2 perfect (8/8), errors on the 0↔1 and 1↔2 boundaries; no collapse to a
  single level (contrast: the pre-v2 `shipped` cue row in `docs/BENCHMARKS.md` §2.1 collapsed
  `score` to level 0 for both models).
- **Honesty flags never fired:** `low_mass` 0/30, `low_confidence` 0/30, `cue.refused` 0/30 on
  all three questions; median coverage 0.99987 (queue), min 0.99422. The wrong answers are
  *confident-looking rows*: the flag says "this row is an answer", not "this answer is right".
- **Raw confidence vs correctness (uncalibrated — `calibrated: false`, no temperature/scale was
  fitted):** mean confidence 0.835 where the queue was right vs 0.586 where it was wrong; accuracy
  rises with the raw number — ≥0.4 → 89.3% (28 items), ≥0.5 → 92.0% (25), ≥0.6 → 95.7% (23).
  Those are raw probability-scale numbers, not calibrated probabilities. `escalate` decisiveness
  ≥0.99 → 8/8, ≥0.9 → 17/19; mean p(yes) 0.907 on true items vs 0.308 on false ones (one item,
  p04 = 0.545, sits just over the line).
- **Straight-through band** (queue confidence ≥ 0.5 **and** escalate decisiveness ≥ 0.9): 15/30
  tickets (50%) could auto-act, and in that band 13/15 were right on both questions — i.e. on this
  set the confidence gate did *not* buy accuracy (the confident b04 error lands inside it).
- **Timing (warm host, nothing generated):** one cold model load for the whole session
  1,183 ms; per item median prefill **67 ms**, candidate/questions **565 ms** (3 questions, 6
  forks), engine total **645 ms**; per CLI call wall clock median **1,006 ms** (client start-up
  included), **30 items in 35.6 s**. Total engine wall clock for this card's runs (purpose run
  36 s + cross-check 40 s) is ~76 s against the 45-minute budget.

## 4. Cross-check: the committed dev set, same path, same box

The repo ships a 60-item labelled dev set (`src/typed_gguf/bench/devset.jsonl`) with a published
4B row (`docs/BENCHMARKS.md` §2.3). Running all 60 through the *same* `run --questions` path, one
run per item, default policy, `--threads 4`: **50/60 = 83.3%** agreement (Wilson 72.0–90.7%),
choice 21/24, score 11/18, noul 18/18 — **60/60 items identical** (`got`, `correct`) to the
published row, and identical `prefix_tokens`. Cheap replication: 40 s vs the published recipe's
5-run bench, and it says the numbers in §3 are reproducible on this box, not a one-off.

## 5. Verdict — is this realistic and reliable for such a system?

**What a human could rely on it for (measured):** a first-pass triage *proposal* over a real
queue — 87% queue routing and 90% "should a person take this" on a 30-item set, at ~0.65 s per
ticket on a 4.3 GB model on one consumer GPU, with zero output-parsing risk and zero refusals.
The safety-relevant asymmetry is the strongest single result: it never missed a
money/legal/access-risk ticket, and its escalation errors were over-escalations. The raw
confidence *ordered* the decisions correctly (92% at ≥0.5 vs 87% overall), and severity was
monotone and never collapsed.

**What it is not (measured or unverifiable):** it is not an unattended router — with 30 items a
single confident error costs ~3 points of accuracy, and no flag fired on the four wrong tickets,
so **the engine's coverage/reliability flags cannot be used as a correctness alarm** (they answer
"is the row an answer", not "is the answer right"). The confidence numbers are uncalibrated on
this model (`calibrated: false`); 95% accuracy at confidence ≥0.6 is a measurement on 23 items,
not a calibrated promise. And this is a synthetic 30-item set with rubric-fixed labels: it can
only show the *mechanism* (typed decisions + probabilities + flags survive contact with a 4B), not
a production accuracy figure.

**Against the class's public claims:** the *shape* claims hold here — typed answers, no free text,
no refusals, probabilities on every decision, "cannot hallucinate" in the narrow sense the dev.to
write-up concedes ("it cannot return a value outside your schema; it can return the *wrong* valid
one" — §3 has four of those). But this run measures a **local** engine, not TypeSafe's hosted
model: the "70–500 ms / 40–200× faster" claim is *partly* checked — 645 ms engine-side per ticket
with 3 questions (and 173 ms per single-question decision on the dev set) on a 4B, i.e. the right
order of magnitude, but nothing here can reproduce a vendor latency or the 40–200× multiple, and
**calibration — the claim the write-ups call the most useful part — was not testable at all**: this
engine ships `calibrate` for exactly that and it was not run here, so §3's confidence numbers are
raw. Their ~68% workflow-accuracy claim is likewise a different eval (consensus-labelled, self-run)
and does not transfer to a 4B on authored triage items.

Conflict of interest to weigh: the 60-item cross-check is this repository's own set, and my 30
items were authored by the same run that measured them. Both limits are stated in §6 rather than
hidden — the cross-check's value is *replication* of a published row (identity 60/60), not
independence of labelling.

## 6. Limitations, or what could not be verified

1. **Item authoring** — the 30 items and their golden labels are the work of this run. Labels were
   fixed from the written rubric before any model call, but a different author would set different
   boundaries (b04, p05 and the b02/t01/a03 escalations are the visible seams). No second annotator
   was used.
2. **Difficulty is designed, not sampled** from real traffic; 30 items give ±10–17 points of
   Wilson width, so the point estimates above are indicative, not precise.
3. **One model, one box, one run per item.** Determinism was demonstrated on the dev set (60/60
   item-level identity with a published run, same prefix tokens), but cross-host variance in a
   Vulkan fit plan is not covered here.
4. **Not verified:** TypeSafe's latency/cost/accuracy figures (no API access in this run; the
   hosted model is waitlisted), the *calibrated* half of the calibration claim, and any purpose
   outside triage (P3–P7 above are surveyed, not measured).
5. **No engine bug surfaced**, so no product change was made or needed: this card added evidence
   only (`docs/` files).

## 7. Reproduce

```bash
cd <checkout>                                   # this commit
export TYPED_GGUF_HOME=<writable data home>     # keep socket, states, fit cache
export TYPED_GGUF_RUNTIME_DIR=<extracted b11026 llama.cpp bundle>   # rung 3: consume a runtime
export REAL_PURPOSE_BASE=/tmp/real-purpose-4b   # scratch dir (states, responses, run.log)
# purpose run (30 items, ~36 s warm; needs the 4B on disk, no network):
bash docs/evidence/real-purpose-4b-2026-09-22/run.sh   # BASE/MODEL/RUNTIME are env-overridable
REAL_PURPOSE_BASE=/tmp/real-purpose-4b python3 docs/evidence/real-purpose-4b-2026-09-22/analyze.py
# cross-check (60 items, ~40 s):
uv run typed-gguf bench --suite quality --model <...>/Spark-X2.5-4B-Q8_0.gguf \
  --backend vulkan --runs 1 --threads 4 --items 60 --json
```

`run.sh` exports the 30 item states from `items.jsonl` and asks the same three questions of every
item, so a rerun on this box is the same protocol; the committed `report.json` is the run this
document quotes.

Machine-readable evidence: `report.json` (30 items: gold, got, probabilities, confidence,
coverage, reliability, cue verdict, timings, and the exact command per item) and
`devset_crosscheck.json` (60 rows + the published comparison). Gate run for this card:
`env -u PYTHONPATH uv run --extra dev pytest -q tests/test_readout_math.py tests/test_e3.py` →
**77 passed**, exit 0 (engine behaviour unchanged — this card touched `docs/` only).
