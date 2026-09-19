# The corrected-instrument Tiel row — 60 committed dev items, measured [host] (`t_7c926398`)

Card `t_7c926398` (main-coder) · [host] run · tree `00265ea` (`ggufone-wt-t7c9`, branch `t7c926398-tiel-corrected`) — the committed
corrected instrument of card `t_6de5fc53`, unmodified. Companion of
`docs/evidence/e2_fix_t_6de5fc53_framing.md` §4.3 (the six-item probe this card turns
into a table).

**One sentence.** Re-measured on the host, with the framing `ggufone ask`/`run` actually sends,
Tiel-Coder-35B-A3B scores **22/60 = 0.367 [0.256–0.493]** on the same 60 committed dev items that
the published plain-framing row scored 31/60 = 0.517 [0.393–0.638] — and **59/60 rows refuse at the cue**
(`W_CUE_REFUSED`, a turn-closer where the label should sit) against 0/60 before.
The collapse the in-container probe predicted persists at full power; the plain row was a different prompt.

## 1. The pin

| what | value |
|---|---|
| file | `/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` |
| model name | `Tiel-Coder-35B-A3B (t_7c926398, corrected instrument)` |
| SHA-256 before (line 1) | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` |
| SHA-256 after (line 1) | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` |
| downloads | none — the file was already local |

The before/after pair is the campaign's own `sha256sum` (printed at the head and the tail
of `.t7c9/logs/campaign.log`); the two digests are identical, so nothing rewrote the weights
mid-run.

## 2. The instrument (what "corrected" means here)

Card `t_6de5fc53` fixed `LiveModel.decide`: it used to plan the executed context from the live
`ModelSession` (which resolves no chat template) while the product plans from the model handle.
Every published quality row was therefore the *plain* E1b framing. This campaign is the first 60-item
Tiel table measured with the resolved plan — the rows carry it:

| what | value |
|---|---|
| report `framing.labels` (first chunk's block) | `chat-template: qwen35moe / builtin` |
| framing verified **per row** over all 60 rows | `chat-template: qwen35moe / builtin` · mixed: False |
| per-row `prefix_tokens` | 74–109 tokens (60 rows) |
| pre-fix report's framing field | absent (the shape predates the fix) |

**Which surface rendered it (all 60 rows).** kind `{'builtin': 60}` · renderer `{'builtin': 60}` · source `{'llama_chat_apply_template': 60}` ·
thinking `{'suppressed': 60}` · warnings `{'W_TEMPLATE_FALLBACK': 60}`. The note the rows carry:
"the internal renderer rejected this template; the runtime's built-in family table rendered it".

The parity claim is not assumed from the label: the repository's live gate
(`tests/test_bench_live.py` → `::test_the_bench_sends_the_same_prompt_as_the_serving_path`) was run
on **this model file** for this card and passes — `item c01: serving prefix 109 tokens, bench
prefix 109 tokens, framing chat-template: qwen35moe / builtin` (`.t7c9/live_parity_tiel.txt`). The bench and
`ggufone ask` prefill byte-identical tokens on the 35B file, through the built-in fallback surface,
before any number below was measured.


Nothing else moved: same model file, same 60 committed dev items (the chunk slices are
byte-identical copies of `docs/evidence/tiel_chunks/devset_00{1..6}.jsonl`, `.t7c9/devset_sha.txt`),
same placement ask, same flags.

## 3. Environment and the scope that decides the numbers

```
host      Linux-7.2.4-ogc3.1.fc44.x86_64-x86_64-with-glibc2.43
GPU       NVIDIA GeForce RTX 3060 Ti, 8192 MiB
runtime   /home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan (pinned b11026)
ICD       VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json (libEGL_nvidia.so.0)
scope     systemd-run --user --unit=t7c9-tiel-campaign (MemoryMax=infinity)
```

`set -u` driver: `.t7c9/run_chunks_corrected.sh`; the campaign's own log prints `memory.max=max`
for the unit it ran in, because the kanban worker's own scope is capped at 4 GiB and a 20.8 GiB
model inside it re-reads its weights from disk forever (`.e3c_tiel/run_chunks.sh`'s note, measured on
card `t_a58f8b67`).

**Contention note.** The box is shared with sibling cards, but no second model was
resident during this campaign: VRAM read 7.4/8.2 GiB while chunk 001 ran (the per-chunk line in
`.t7c9/logs/campaign.log` is the *pre-chunk* reading, 1.5–1.9 GiB) and the per-chunk walls
(36–60 s for ten items) are the receipts for that claim.

## 4. Placement per chunk (the loader's own answer)

| chunk | items (choice/score/noul) | correct | ngl requested → used | degraded | kv_type_used | n_ctx | n_prefix | n_seq_max | load wall | chunk wall | median decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `report_001.json` | 10 (4/3/3) | 1/10 | 9 → 9 | False | auto | 184 | 93 | 5 | 23.761 s | 59.4 s | 1.3 s |
| `report_002.json` | 10 (3/4/3) | 2/10 | 9 → 9 | False | auto | 173 | 93 | 5 | 10.943 s | 42.5 s | 1.1 s |
| `report_003.json` | 10 (3/3/4) | 6/10 | 9 → 9 | False | auto | 154 | 82 | 3 | 5.232 s | 37.2 s | 1.2 s |
| `report_004.json` | 10 (4/3/3) | 6/10 | 9 → 9 | False | auto | 186 | 93 | 5 | 5.386 s | 37.2 s | 1.1 s |
| `report_005.json` | 10 (3/4/3) | 4/10 | 9 → 9 | False | auto | 151 | 75 | 5 | 5.941 s | 40.9 s | 1.2 s |
| `report_006.json` | 10 (7/1/2) | 3/10 | 9 → 9 | False | auto | 177 | 90 | 5 | 5.554 s | 36.1 s | 1.2 s |

The pre-fix campaign asked for the same 9 layers and got them in every chunk too, so the two rows
are placement-matched: **the framing is the only thing that moved between them.**

## 5. The 60 items, corrected instrument

| metric | corrected instrument (chat-template) | pre-fix row (plain, published) |
|---|---|---|
| framing | chat-template: qwen35moe / builtin | plain (prompt.py E1b framing) |
| overall | **22/60 = 0.367 [0.256–0.493]** | 31/60 = 0.517 [0.393–0.638] |
| choice | 7/24 = 0.292 [0.149–0.492] | 16/24 = 0.667 [0.467–0.820] |
| noul | 7/18 = 0.389 [0.203–0.614] | 7/18 = 0.389 [0.203–0.614] |
| score | 8/18 = 0.444 [0.246–0.663] | 8/18 = 0.444 [0.246–0.663] |
| rows `low_mass` | **57/60** | 14/60 |
| rows `measured` (≥ 0.10 floor) | **3/60** | 46/60 |
| refused at the cue (`W_CUE_REFUSED`) | **59/60** | 0/60 |
| coverage median | **0.0005828** | 0.2579 |
| coverage min · p25 · p75 · max | 1.442e-07 · 1.741e-05 · 0.003086 · 0.131 | 0.007883 · 0.1052 · 0.4172 · 0.8201 |
| cue row's top-token mass (median) | 0.6982 | 0.2993 |

**The cue verdicts are the mechanism, not a footnote.** 59 of 60 rows put a turn-closer on the cue row:
{"</think>": 32, "<think>": 1, "<|im_end|>": 26}. The readout sits at the last row of the prompt — the
start of the assistant turn — and under the model's own template the first thing Tiel wants to emit
there is `</think>` or `<|im_end|>`, not a label. The engine's own verdict says so (`W_CUE_REFUSED`),
and the agreement reads the argmax of a label set whose mass is a tail.

Reliability split (the §2.1 shape), both framings:

| framing | reliability | items | correct | agreement | 95 % CI |
|---|---|---|---|---|---|
| plain (pre-fix) | `low_mass` | 14 | 8 | 0.5714 | 0.3259 – 0.7862 |
| chat-template (corrected) | `low_mass` | 57 | 21 | 0.3684 | 0.2552 – 0.4982 |

### 5.1 The auxiliary arm: `--cue two_step` cannot engage on this model

The card asks for the shipped cue (the product's default) and that is §5. Because §5 raises the
obvious next question — *is the collapse the cue row itself, and does the knob the product already has
(`--cue two_step`, E3d) rescue it?* — the same 60 items were re-run once more at `--cue two_step` (same
model, same placement ask, same corrected instrument; driver `.t7c9/run_cue_arm.sh`).

| arm | agreement (Wilson 95 %) | `low_mass` | refused at the cue | rows identical to the shipped arm |
|---|---|---|---|---|
| `--cue shipped` (the card's row) | 22/60 = 0.367 [0.256–0.493] | 57/60 | 59/60 | — |
| `--cue two_step` | 22/60 = 0.367 [0.256–0.493] | 57/60 | 60/60 | 59/60 |

**The arm moves exactly 1 row of 60.** `decide._advance_token` (the committed E3d rule) refuses to advance past a
refused cue — "a cue the model closes never advances" — and Tiel closes 59 of
60 cues, so the shape has nothing to move; on the one row that did advance (`s09`) the row the
readout lands on is a turn-closer as well. The measured consequence: agreement is unchanged
(22/60 → 22/60), `low_mass` is unchanged (57 → 57), and refusals go
59 → 60.

* the one row that differs — `s09` (score): coverage 0.00116132 → 0.000553303; cue `content` (mass 0.3689) → `<|im_end|>` (mass 0.2148), refused False → True.

**What that buys the escalation.** A cue-shape change is *not* a remedy this model can reach, so
the "try `two_step` first" path is closed by measurement, not by opinion — the fix has to stop the
prefill before the row Tiel wants to close, which is exactly what card `t_4c48f40a` (E3e) is
measuring. This arm is auxiliary: it is not a table the card asked for, and it is not compared to the
4B's `two_step` arm across framings.

## 6. Paired against the pre-fix row (same 60 items, same placement)

* risk difference (corrected − plain): **-0.150** [-0.300…+0.000] (20000 bootstrap resamples,
  seed 20260919)
* discordant pairs: pre-fix-only correct **15**, corrected-only correct **6**
* exact McNemar (two-sided): **p = 0.07835**
* unpaired rows: 0 corrected-only ids, 0 pre-fix-only ids

Items the plain framing got and the corrected one loses:

| id | type | gold | plain got | corrected got |
|---|---|---|---|---|
| `c02` | choice | `billing` | `billing` | `escalation` |
| `c05` | choice | `security` | `security` | `support` |
| `c06` | choice | `fixed` | `fixed` | `changed` |
| `c07` | choice | `termination` | `termination` | `liability` |
| `c08` | choice | `business_hours` | `business_hours` | `weekend` |
| `c09` | choice | `spam` | `spam` | `harassment` |
| `c15` | choice | `cardiology` | `cardiology` | `neurology` |
| `c16` | choice | `travel` | `travel` | `meals` |
| `c20` | choice | `copyleft` | `copyleft` | `permissive` |
| `c22` | choice | `sms` | `sms` | `chat` |
| `c23` | choice | `identity` | `identity` | `search` |
| `s01` | score | `3` | `3` | `1` |
| `s02` | score | `3` | `3` | `0` |
| `s03` | score | `4` | `4` | `1` |
| `s16` | score | `0` | `0` | `1` |

Items the corrected framing gets and the plain one lost:

| id | type | gold | plain got | corrected got |
|---|---|---|---|---|
| `c11` | choice | `expedited` | `freight` | `expedited` |
| `c17` | choice | `midmarket` | `smb` | `midmarket` |
| `s08` | score | `1` | `4` | `1` |
| `s10` | score | `1` | `2` | `1` |
| `s14` | score | `1` | `2` | `1` |
| `s18` | score | `0` | `1` | `0` |

**Framing is the only variable.** Both rows ran the same 60 committed items, the same model file
(SHA-identical before and after), the same placement ask (9 layers, no degrade, `kv_type` auto in
every chunk), the same `--threads 4`, on the same host — and the pre-fix row is a committed
`quality` report of this repository (`docs/evidence/tiel_quality.json`), re-read here item for item.

## 7. What the reading is — and what it implies for the shipped cue

**Plainly: the collapse persists.** Six items said "the product's prompt breaks this model"; 60 items
say it with an interval that no longer reaches the published row: 22/60 = 0.367 [0.256–0.493] vs
31/60 = 0.517 [0.393–0.638], and the mechanism is visible per row — 59/60
rows refuse at the cue where the readout sits.

The E3c reading ("Tiel is not mass-starved, unlike Occamy") is a statement about the **plain**
instrument. On the prompt the product sends, Tiel behaves like Occamy in the six-item probe and worse
here: the mass at the cue row is a turn-closer's mass, so the word "starvation" understates it — the
model is closing the turn, not hesitating.

**What this does not decide.** The cue *policy* is not this card's to move. What the card changes is
the evidence under the policy question:

* the published Tiel row measured a prompt the product never sends, so any argument that read it as
  "the shipped cue is fine for 35B-A3B" was reading the wrong bytes;
* the six-item probe's direction is now a 60-item table, and the one cue knob the product already has
  (`--cue two_step`) is **measured inert on this model** (§5.1: it moves one row of sixty) — so "try a
  different cue shape" is not the escape hatch here;
* card `t_4c48f40a` (E3e) is measuring the instructed-JSON + roles-split policy, which attacks the same
  row from the other side: it stops the prefill before the row the model closes and reads the value row.
  That is the measurement this row points at; if it holds, the shipped *prompt shape* — not the model
  choice — is what has to change.

The default stays `shipped` until that card's own evidence lands (E3d's rule: the mechanism ships, the
default moves only with its own publication). This document is the corrected row, with its framing named,
next to the pre-fix one that keeps its marker.

## 8. What this does not claim

* no ranking between models — this is one model's row, and §7.7's three-way table keeps its own
  framing caveat;
* no mechanism beyond the cue verdicts: which of template bytes / `enable_thinking` suppression /
  the built-in family renderer produces the turn-closer is E3e's question;
* no claim about the other cue shapes on Tiel (not measured here);
* no throughput claim — the chunk walls are this box's, under sibling-card load.

## 9. Gates

| gate | command | result |
|---|---|---|
| reproduce driver exits 0 | `bash .t7c9/run_chunks_corrected.sh (systemd unit t7c9-tiel-campaign)` | 6/6 chunks exit=0, walls 36–60 s; raw log `.t7c9/logs/campaign.log` |
| oracle | `python3 docs/verify_runtime_contract.py` | exit 0 — `failures: 0  skips: 0` (`.t7c9/oracle.txt`) |
| test suite | `uv run --frozen --offline --extra dev pytest -q -rs` | 1244 passed, 45 skipped, 0 failed in 35.6 s (`.t7c9/gates.txt`; the 45th skip is `test_probe_pressure.py:312`, no container pid cgroup on the host) |
| ruff (the paths this card touches) | `uv run --frozen --offline --extra dev ruff check .t7c9` | All checks passed |
| live prompt-parity gate on **this** model file | `GGUFONE_BENCH_MODEL=<tiel.gguf> uv run --frozen --offline --extra dev pytest -q --run-network -s tests/test_bench_live.py::test_the_bench_sends_the_same_prompt_as_the_serving_path` | 1 passed in 38.80 s — `serving prefix 109 tokens, bench prefix 109 tokens` (`.t7c9/live_parity_tiel.txt`) |
| model SHA-256 before/after | `sha256sum <tiel.gguf> (campaign head and tail)` | identical — `9286a94c…cf17c` (`.t7c9/sha_lines.txt`) |
| dev-set slices | `sha256sum docs/evidence/tiel_corrected_chunks/devset_00*.jsonl` | byte-identical to the committed `tiel_chunks` slices (`.t7c9/devset_sha.txt`) |

## Receipts

* driver + analysis: `.t7c9/run_chunks_corrected.sh`, `.t7c9/analyse.py`, `.t7c9/render_doc.py`,
  `.t7c9/placement_facts.py`, `.t7c9/detail_facts.py`
* auxiliary `two_step` arm: `.t7c9/run_cue_arm.sh`, `.t7c9/tiel_cue_arm.py`, `.t7c9/analyse_arm.py`,
  `.t7c9/tiel_two_step_stats.json`
* per-chunk reports (corrected instrument): `docs/evidence/tiel_corrected_chunks/report_00{1..6}.json`
* per-chunk reports (`two_step` arm): `docs/evidence/tiel_corrected_chunks/two_step_report_00{1..6}.json`
* per-chunk placement sinks: `docs/evidence/tiel_corrected_chunks/placement_00{1..6}.json` and
  `.../two_step_placement_00{1..6}.json`
* dev-set slices (SHA-verified identical to `docs/evidence/tiel_chunks/`): `docs/evidence/tiel_corrected_chunks/devset_00{1..6}.jsonl` + `.t7c9/devset_sha.txt`
* merged rows: `docs/evidence/tiel_corrected_quality.json` (the card's row) and
  `docs/evidence/tiel_two_step_quality.json` (the auxiliary arm)
* statistics: `.t7c9/tiel_corrected_stats.json` (this doc is rendered from it)
* run logs: `.t7c9/logs/run_00{1..6}.log`, `.t7c9/logs/two_step_run_00{1..6}.log`, `.t7c9/logs/campaign.log`,
  `.t7c9/oracle.txt`, `.t7c9/gates.txt`, `.t7c9/live_parity_tiel.txt`
* the pre-fix row it pairs with: `docs/evidence/tiel_quality.json` + `docs/BENCHMARKS.md` §7.3/§7.4
  (pre-fix marker in §7.9)

