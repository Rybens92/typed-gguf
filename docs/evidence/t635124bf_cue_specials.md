# E3c FIX — the *second* silent refusal: a special token the catalogue cannot name

Card `t_635124bf` (code-tdd) · 2026-09-19 · branch `wt/t_635124bf`, first commit `331240c`
(the fix + gates), this document and its receipts after it.

The card asked for one thing: the E3c verdict (`engine/cue.py`, card `t_6c119626`) refused a cue row
only when the row's argmax was one of the **documented turn-closer strings** the session's tokenizer
encodes as a single token. Tiel-Coder's *serving-shaped* batch puts a **special token the catalogue
has never heard of** on top of the cue row on 19 of 20 items, so the rows read `0/20 measured` with
no reason attached — the second silent refusal.

## 1. The gap, read off the committed run

`.e3c_tiel/batch_response.json` is the pre-fix serving-shaped run (host, card `t_a58f8b67`,
`total_ms` 35 618 · `usage` 20 questions / 80 forks / 117 decode steps / 40 waves · state
`sha256:c206645c…` · 109 prefix tokens · `n_seq_max` 8). Its per-item cue block carries the two row
facts the verdict reads — the argmax id and its mass — and they say:

| what | count | detail |
|---|---|---|
| items whose cue argmax is token **248069** | **19/20** | `</think>` — `tokenizer.ggml.token_type` `USER_DEFINED`, **not** a catalogue string |
| items whose cue argmax is token **248046** | 1/20 | `<|im_end|>` — the one catalogue closer the report could name (c11) |
| `refused: true` before the fix | **1/20** | c11 only; every `</think>` row read `refused: false`, `closer: null` |
| mass of the top token | 0.5037 … 0.9916 | all 20 rows clear the 0.10 floor |
| `measured` before the fix | **0/20** | the same starvation the catalogue fix could not explain |

So the *detector* had a 19-item blind spot on the shape the CLI actually serves, while the
`low_mass` word (correct) carried no reason (wrong).

## 2. The rule (deliverable 1)

Two sources name the class, **one rule** decides (`engine/cue.py`):

1. the **catalogue** (`TURN_CLOSERS`) — the documented turn-closer *strings*, matched only when the
   session's own tokenizer encodes the string as one token (unchanged from `t_6c119626`);
2. the **vocabulary's own attribute table** — every token the model marks `CONTROL` or
   `USER_DEFINED` (`llama_token_get_attr`, mask `CONTROL | USER_DEFINED`). This is the class a
   string catalogue cannot enumerate, and the vocabulary itself is the authority on it.

A row is a refusal when its top token comes from either source **and dominates the row**:

```
refused  ==  (token is the argmax)  and  (closer is not None)  and  (mass >= floor)
```

* **dominating is one number, not two**: `floor` is the engine's own coverage floor
  (`OPTION_DEFAULTS["coverage_floor"]`, 0.10) — the threshold `reliability` calls a row's mass
  adequate with — and `decide.py` passes the request's *effective* floor in, so the verdict and the
  reliability word are read off the same number.
* **the block shape is unchanged**: `{refused, token, closer, mass[, hint]}`; `closer` names the
  token that refused (the vocabulary's own text, e.g. `</think>`), and a vocabulary that carries no
  text for its token is reported as `<special <id>>` rather than as nothing.
* **a non-enumerable session degrades to the pre-fix behaviour, byte for byte**:
  `closer_map()` duck-types on `session.special_tokens`, and a session (or bundle) without the
  attribute API gets the catalogue alone. Nothing matched by a hard-coded token id — the *session's*
  vocabulary decides, so ggufone cannot fire on a family it is not looking at.

**Deliberately widened, and worth a reviewer's eye:** before this card the catalogue path had **no**
floor — a closer that was the argmax counted however small its mass. The rule above applies the
floor to both classes, because "dominating" must not mean two things depending on which source
matched. The edge is one row wide (a closer at < 0.10 mass): pre-fix `refused: true`, now
`low_mass` with no refusal. Both sides are pinned
(`test_a_catalogue_closer_below_the_floor_is_not_called_dominating`,
`test_a_token_holding_exactly_the_floor_is_dominating`). The alternative position — keep the
catalogue floor-free and gate only the specials — is *not* shipped; it would let "the model closed
the turn" mean a 3 %-mass token in one branch and a 30 %-mass token in the other.

## 3. Why the attribute table is trustworthy (the cross-check)

The fix trusts llama.cpp's classification, so it is checked against each GGUF's own recorded
`tokenizer.ggml.token_type` array — **every** id, not a sample
(`.e3c_specials/vocab_attr_check.log`, `.e3c_specials/receipts/vocab_only_*.json`):

| model | vocabulary | `token_type` vs `llama_token_get_attr` | cost (vocab-only load) |
|---|---|---|---|
| Spark-X2.5-4B-Q8_0 | 131 072 ids (100 CONTROL, 13 USER_DEFINED) | **0 disagreements** | load 2.17 s · scan 82 ms · comparison 331 ms |
| Tiel-Coder-35B-A3B-UD-Q4_K_XL | 248 320 ids (33 specials) | **0 disagreements** | load 2.68 s · scan 822 ms · comparison 309 ms |

Tiel's block, read out of the file and out of llama.cpp, agreeing id by id:
`248044` `CONTROL`, `248045` `CONTROL`, `248046` `<|im_end|>` `CONTROL`, `248047` `<|endoftext|>`
`CONTROL`, `248068` `<think>` `USER_DEFINED`, **`248069` `</think>` `USER_DEFINED`**, … (33 ids).
The scan is per session and cached on the handle; the catalogue knows **3** ids in that vocabulary,
which is exactly why 16 of the 19 rows were invisible.

## 4. Tiel, serving shape, before → after (deliverable 2)

The before column is the committed pre-fix run (host, above). The after column is the fix's verdict
on **those same recorded rows** — the verdict is a pure function of (argmax id, mass, map, floor),
and the fix moves the map, not the row: `.e3c_specials/derive_verdicts.py` re-verdicts the response
through the engine's own `closer_map` + `REFUSAL_FLOOR` (`tiel_serving_verdicts.json`). §4.1 then
re-runs the shape live on this box: the rows come back identical and the verdicts match this column
item for item, which is what makes the column a measurement rather than an inference.

| id | cue token | catalogue before | vocabulary special | mass | dominating | refused before → after |
|---|---|---|---|---|---|---|
| c01 | 248069 | — | `</think>` | 0.9916 | yes | false → **true** |
| c02 | 248069 | — | `</think>` | 0.8196 | yes | false → **true** |
| c03 | 248069 | — | `</think>` | 0.9796 | yes | false → **true** |
| c04 | 248069 | — | `</think>` | 0.9054 | yes | false → **true** |
| c05 | 248069 | — | `</think>` | 0.8752 | yes | false → **true** |
| c06 | 248069 | — | `</think>` | 0.9457 | yes | false → **true** |
| c07 | 248069 | — | `</think>` | 0.8594 | yes | false → **true** |
| c08 | 248069 | — | `</think>` | 0.7311 | yes | false → **true** |
| c09 | 248069 | — | `</think>` | 0.7107 | yes | false → **true** |
| c10 | 248069 | — | `</think>` | 0.7894 | yes | false → **true** |
| c11 | 248046 | `<|im_end|>` | `<|im_end|>` | 0.5037 | yes | true → true |
| c12 | 248069 | — | `</think>` | 0.6817 | yes | false → **true** |
| c13 | 248069 | — | `</think>` | 0.9369 | yes | false → **true** |
| c14 | 248069 | — | `</think>` | 0.8289 | yes | false → **true** |
| c15 | 248069 | — | `</think>` | 0.9165 | yes | false → **true** |
| c16 | 248069 | — | `</think>` | 0.9616 | yes | false → **true** |
| c17 | 248069 | — | `</think>` | 0.9730 | yes | false → **true** |
| c18 | 248069 | — | `</think>` | 0.8970 | yes | false → **true** |
| c19 | 248069 | — | `</think>` | 0.8070 | yes | false → **true** |
| c20 | 248069 | — | `</think>` | 0.9493 | yes | false → **true** |

* **refusals before → after: 1/20 → 20/20.** 19 rows that were silently low-mass now carry
  `W_CUE_REFUSED`, `closer: </think>` and the `hint`; **0 refusals were removed** (the catalogue
  verdict on c11 is unchanged).
* **`measured` before → after: 0/20 → 0/20.** The fix names the refusal, it does not manufacture
  answer mass: those rows are still `low_mass` — now with a reason a caller can act on (the prompt
  shape, not the labels).
* the *only* difference is the verdict: `coverage`, `reliability` and the chosen answer are
  identical on all 20 rows (asserted by `derive_verdicts.py`'s receipt and by
  `batch_verdicts.py` on the live re-run below).

### 4.1 The live re-run on this box

Raw: `/work/t635/tiel-post/batch_response.json`, receipt and item-by-item comparison in
`.e3c_specials/tiel_serving_rerun.json`. The same 20 questions (byte-identical
`batch_questions.json`, `sha256:3506fd17…`), the **same state** (`sha256:c206645c…`), the same
flags (`--backend vulkan --threads 4 --items 20 --n-seq-max 8`, bundle
`b11026-linux-x64-vulkan`), `usage` identical to the host run (1175 input tokens, 117 decode steps,
40 waves) — run in this container (cgroup 8 GiB / 2 CPUs), where the weights cannot stay resident
and are re-read every wave, so the box costs 24.4 min where the host costs 35.6 s:

| | host (committed, pre-fix) | this box (live, post-fix) |
|---|---|---|
| `n_gpu_layers` | 9 | 8 (the fit plan saw the GPU busy) |
| `model_load_ms` · `prefill_ms` · `questions_ms` · `total_ms` | 9 851 · 7 879 · 27 715 · **35 618** | 31 470 · 205 200 · 1 255 510 · **1 460 910** |
| cue argmax per item | 248069 ×19, 248046 ×1 | **identical** (asserted, not eyeballed) |
| cue mass per item | 0.5037 … 0.9916 | **identical, bit for bit** |
| `coverage` · `reliability` · chosen answer | — | **identical on all 20** |
| label `probabilities` · `confidence` | — | 12 items differ in the last digits (6th significant digit at most), 8 in `confidence`: a different layer split sums in a different order — the same reason `answers_identical` is asserted on the *choice*, not on the float's last bit |
| **refused** | **1/20** (c11) | **20/20** — 19 rows `false → true`, 0 removed |
| `measured` | 0/20 | 0/20 |

So the fix's effect is measured, not inferred: on the identical rows the same engine refuses
everything the catalogue could not name, names it `</think>`, and leaves the answer, its coverage
and its reliability untouched. `.e3c_specials/derive_verdicts.py` reaches the same 20/20 from the
recorded rows alone, and the two agree item for item.

## 5. The default model, same shape (deliverable 3)

Spark-X2.5-4B-Q8_0 (CPU, `--backend cpu`, one state, `n_seq_max` 8, the same 20 questions),
pre-fix tree and post-fix tree, `4b_verdicts.json` + `mutmut`-style receipts:

* cue argmax on **all 20** items: token **198** = `Ċ` — a **newline**, `token_type` `NORMAL`,
  `special: false`, mass 0.9784 … 0.9978. Nothing for this detector to catch.
* **refusals 0/20 before → 0/20 after**; `measured` 0/20 → 0/20; `coverage` 2.3e-04 … 1.4e-03.
* So the *starvation* on the serving shape is not Tiel-specific (the 4B starves too, 0/20), but the
  **refusal class is**: the 4B's mass sits on a content token, and the honest verdict for it is
  `low_mass`, not "the model closed the turn".
* `coverage`, `reliability` and the answer are identical in the two runs (the fix is verdict-only).

## 6. The fixtures (deliverable 4)

`tests/test_e3c_cue_specials.py` — 19 gates, all of them on the fake-session seam
(`tests/fake_engine.py`), no model and no GPU needed:

| # | gate | pre-fix tree |
|---|---|---|
| a | `<|im_end|>` dominating → refused, `closer` named | passes (pin: behaviour that already existed) |
| b | an uncatalogued special (`</think>` at 413) dominating → refused, **named by the vocabulary** | **fails** |
| b' | a special the vocabulary carries no text for → refused, named `<special <id>>` | **fails** |
| c | a content token dominating → **not** refused, no `hint` | passes (pin) |
| — | the dominance floor is `OPTION_DEFAULTS["coverage_floor"]`; below-floor rows on both classes | fails |
| — | the boundary: `exp(0 - log 4) == 0.25` exactly → **dominating** (`>=`, not `>`) · one ULP of floor higher → not | fails |
| — | one row, two floors: the request's `coverage_floor` decides, and the warning follows it | fails |
| — | `closer_map` merges catalogue + specials; the catalogue wins a collision; no attribute API → catalogue alone | fails |
| — | the vocabulary scan: CONTROL/USER_DEFINED only, `[]` without the attribute API, mask constant | fails |
| — | the rendered quality table names the special where it used to read `ok`; the probe reads the same map | fails |

RED→GREEN, counted (`.e3c_specials/red_prefix.txt`): on the pre-fix tree **17 of 19 fail and the 2
pins pass** — the pins must pass on both trees, so the count is the honest one, not "all 19".

## 7. The suite, the lint, the Tier-M sweep (no tier declared on the card → M)

* `tests/test_e3c_cue_specials.py tests/test_e3c_cue_refused.py tests/test_e3c_cue_shapes.py` →
  **58 passed**.
* full `uv run pytest -q` → **1221 passed, 43 skipped** (104.8 s) — includes the two other E3c gate
  files and every earlier card's gates.
* `ruff check src/ggufone tools tests` → clean for every file this card touches (the two remaining
  `E501`s are in a sibling card's untracked `tests/test_e3d_cue_decision.py` /
  `tools/e3d_cue_decision.py`, which this card does not own).
* **Tier M: `src/ggufone/engine/cue.py`, 65 mutants, `--max-children 2`, two full sweeps**
  (`.e3c_specials/mutmut_run1.log`, `mutmut_run2.log`, driver `.e3c_specials/mutmut_sweep.sh`) —
  fresh sweeps, both metas deleted first, because mutmut 3.8 keys verdicts per function
  (`ggufone.engine.cue.x_cue_verdict__mutmut_15`) and this card moved both functions' bodies:
  * run 1 (the 16 gates as committed in `331240c`): **64 killed, 1 survived** —
    `ggufone.engine.cue.x_cue_verdict__mutmut_15` = `mass >= floor` → `mass > floor`, the boundary
    no gate sat on (`mutmut show` diff quoted in that log's context);
  * run 2 (with the three boundary/floor gates added): **65 killed, 0 survived, 0 pending,
    0 errored**.
  The survivor is what a Tier-M sweep is for: the mutation was *not* equivalent, it was a missing
  gate, and it is now pinned by `test_a_token_holding_exactly_the_floor_is_dominating`.

## 8. Where the change lives

| file | what moved |
|---|---|
| `src/ggufone/engine/cue.py` | `REFUSAL_FLOOR`, `closer_map()`, `single_token_closers(..., special=…)`, `cue_verdict(..., floor=…)` |
| `src/ggufone/engine/decide.py` | the engine resolves the map per session and passes the request's effective floor; `W_CUE_REFUSED` follows the verdict |
| `src/ggufone/engine/session.py` | `ModelHandle.special_tokens()` / `ModelSession.special_tokens()`, `tokenize`, the `n_vocab` guard |
| `src/ggufone/runtime/ctypes_binding.py` | `llama_token_get_attr` / `llama_token_get_text`, `SPECIAL_TOKEN_ATTRS`, `special_tokens()`, `tokenize()` — all optional bindings: a bundle without them keeps the pre-fix path |
| `tools/e3c_cue_shapes.py` | the probe's verdict column reads the same map (its `advance_token` rule deliberately keeps the catalogue alone — it defines the shapes) |
| `tests/test_e3c_cue_specials.py` | 19 gates (the card's fixtures + the boundary + the map + the scan + the report + the probe) |
| `pyproject.toml` | the Tier-M pair re-pointed at this card's gate file |

## 9. Receipts

```
.e3c_specials/red_prefix.txt              the gate file on the pre-fix clone: 17 failed, 2 passed
.e3c_specials/vocab_attr_check.log        the 131 072-id and 248 320-id every-id cross-checks
.e3c_specials/vocab_attr_check.py         the cross-check itself (vocab-only: no tensor is read)
.e3c_specials/receipts/vocab_only_*.json  the vocabularies as llama.cpp and the GGUFs report them
.e3c_specials/tiel_serving_verdicts.json  Tiel's 20 recorded rows, re-verdict-ed by the fix
.e3c_specials/tiel_serving_rerun.log      the live re-run vs the committed run + the row-identity check
.e3c_specials/tiel_serving_rerun.json     the same comparison, machine-readable (every field)
.e3c_specials/tiel_serving_rerun_response.json   the live post-fix response itself (20 answers)
.e3c_specials/4b_verdicts.json            the 4B before/after table (both runs, every row)
.e3c_specials/mutmut_run1.log             the sweep that found the boundary survivor
.e3c_specials/mutmut_run2.log             the sweep that kills all 65
.e3c_specials/mutmut_sweep.sh             the sweep driver (--max-children 2 + retry + census)
.e3c_specials/derive_verdicts.py          re-verdict a committed response through the engine's map
.e3c_specials/batch_verdicts.py           the same for two live responses (what moved, what did not)
.e3c_specials/row_identity.py             did the two boxes put the same row on the cue?
.e3c_specials/analyze_tiel.sh             the two commands above, wired to the two responses
.e3c_specials/run_batch.sh                the serving-shaped runner the re-runs used
.e3c_specials/census.py · tiel_record.py  the sweep census and the committed run's row read-out
```
