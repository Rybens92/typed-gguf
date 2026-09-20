# The [host] E3e probe — `role_split` + `json_instructed` on Tiel-Coder-35B-A3B (`t_9bcbecff`)

Card `t_9bcbecff` (main-coder) · [host] run · worktree `ggufone-wt-t9bcb`, branch `t9bcb-e3e-tiel` · instrument: the committed corrected instrument of card `t_7c926398`, unmodified, plus the two E3e policy switches of card `t_4c48f40a`.

**One sentence.** On the same 60 committed dev items, the same model file, the same placement ask and the same `--backend vulkan` instrument as the published corrected row, adding `--chat-format role_split --cue json_instructed --json-contract question` moves Tiel from 22/60 = 0.367 [0.256 – 0.493] to 53/60 = 0.883 [0.778 – 0.942] — paired difference +0.517 (0.353 – 0.680, exact McNemar p = 7.842e-07) — and it takes the collapse with it: `low_mass` 57/60 → 0/60, refusals at the cue 59/60 → 0/60.

## 1. The pin

| what | value |
|---|---|
| file | `/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` |
| SHA-256 before (first line of the campaign log) | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` |
| SHA-256 after (last line of the campaign log) | `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` |
| the pair is identical | `True` over `2` sha line(s) |
| downloads | none — the file was already local |
| dev-set slices | 6 slices, digests byte-identical to the baseline campaign's own receipt: `True` (`docs/evidence/t9bcbecff_tiel_chunks/devset_sha.txt`) |

## 2. The instrument (what moved, and the one thing that did not)

* **baseline** — `baseline: corrected instrument, shipped cue`: the corrected-instrument row of card `t_7c926398`, committed at `docs/evidence/tiel_corrected_quality.json`; **not re-measured here** (the card's rule), read from its own per-chunk reports.
* **challenger** — `challenger: role_split + json_instructed`: the same instrument plus exactly the two E3e flags (`--chat-format role_split` and `--cue json_instructed`, with the default `--json-contract question`).
* reproduce line of the challenger's own report: `uv run ggufone bench --suite quality --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf --backend vulkan --runs 1 --threads 4 --devset docs/evidence/tiel_corrected_chunks/devset_001.jsonl --cue json_instructed --chat-format role_split --gpu-layers 9 --json`

**Which path rendered the role split.** The family's own template is outside the internal renderer's subset (E3e recorded it `not-renderable` offline, `.e3e/role_render.json`), so the question is whether the built-in bridge can express the two-user-turn shape in a live run. It can, and the rows say so through the response's own surfaces:

| surface | value | rows | prefix chars |
|---|---|---|---|
| `engine.chat_format` | `{"contract": "question", "dropped": "", "kind": "role_split", "question_turn": "user"}` | 60 | 387–494 |
| `engine.template` | `{"family": "qwen35moe", "kind": "builtin", "renderer": "builtin", "source": "llama_chat_apply_template"}` | 60 | — |

so every challenger row prefilled the question into a **user** turn (`question_turn` `user`, `kind` `role_split`) rendered by the built-in bridge (`llama_chat_apply_template`), with nothing the shared prefix had to drop (`dropped` empty on every row) — the built-in bridge *does* express this family's two-user-turn shape, and `--template plain` was **not** needed (it is not used in this run).

## 3. Environment and the scope that decides the numbers

```
runtime   /home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
          (pinned b11026)
ICD       VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json (libEGL_nvidia.so.0)
scope     systemd-run --user --unit=t9bcb-campaign (memory.max=max)
driver    .t9bcb/run_chunks.sh → .t9bcb/tiel_e3e_arm.py (the committed observer + the flags)
```

The campaign log prints `memory.max=max` for the unit it ran in, because the kanban worker's own scope is capped at 4 GiB and a 20.8 GiB model inside it re-reads its weights from disk forever.

## 4. Placement per chunk (the loader's own answer), one backend named

| chunk | challenger correct | ngl req → used | degraded | kv_type_used | n_ctx | n_prefix | n_seq_max | load wall | chunk wall | median decision | challenger effective_backend | challenger compute buffers | baseline effective_backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `report_001.json` | 9/10 | 9 → 9 | False | auto | 206 | 95 | 5 | 4.484 s | 35.4 s | 1.22 s | `vulkan` | Vulkan0=10 · Vulkan_Host=10 | `vulkan` |
| `report_002.json` | 9/10 | 9 → 9 | False | auto | 195 | 95 | 5 | 4.189 s | 34.9 s | 1.25 s | `vulkan` | Vulkan0=10 · Vulkan_Host=10 | `vulkan` |
| `report_003.json` | 8/10 | 9 → 9 | False | auto | 172 | 84 | 3 | 4.190 s | 35.4 s | 1.27 s | `vulkan` | Vulkan0=10 · Vulkan_Host=10 | `vulkan` |
| `report_004.json` | 8/10 | 9 → 9 | False | auto | 208 | 95 | 5 | 4.261 s | 36.1 s | 1.26 s | `vulkan` | Vulkan0=10 · Vulkan_Host=10 | `vulkan` |
| `report_005.json` | 10/10 | 9 → 9 | False | auto | 173 | 77 | 5 | 4.641 s | 37.0 s | 1.38 s | `vulkan` | Vulkan0=10 · Vulkan_Host=10 | `vulkan` |
| `report_006.json` | 9/10 | 9 → 9 | False | auto | 199 | 92 | 5 | 5.177 s | 36.5 s | 1.29 s | `vulkan` | Vulkan0=10 · Vulkan_Host=10 | `vulkan` |

* the row shape carries no per-item `effective_backend`, so "per row" here is the **per-chunk report**: each chunk is one report and each report carries the attribution block (`devices` / `device_buffers` / `effective_backend`) read from the engine's own log — one backend named, the card's rule after `t_55de5779`.


* compute path per chunk (the engine's own buffer lines, challenger logs):

| chunk | compute lines | first line | devices in those lines |
|---|---|---|---|
| `001` | 20 | `~llama_context:    Vulkan0 compute buffer size is 232.3438 MiB, matches expectation of 232.3438 MiB` | Vulkan0=10, Vulkan_Host=10 |
| `002` | 20 | `~llama_context:    Vulkan0 compute buffer size is 229.5000 MiB, matches expectation of 229.5000 MiB` | Vulkan0=10, Vulkan_Host=10 |
| `003` | 20 | `~llama_context:    Vulkan0 compute buffer size is 227.6719 MiB, matches expectation of 227.6719 MiB` | Vulkan0=10, Vulkan_Host=10 |
| `004` | 20 | `~llama_context:    Vulkan0 compute buffer size is 231.3281 MiB, matches expectation of 231.3281 MiB` | Vulkan0=10, Vulkan_Host=10 |
| `005` | 20 | `~llama_context:    Vulkan0 compute buffer size is 228.2812 MiB, matches expectation of 228.2812 MiB` | Vulkan0=10, Vulkan_Host=10 |
| `006` | 20 | `~llama_context:    Vulkan0 compute buffer size is 227.9766 MiB, matches expectation of 227.9766 MiB` | Vulkan0=10, Vulkan_Host=10 |

* `W_BACKEND_MISMATCH` on any challenger row: **none** — every challenger report claims `vulkan` and its own engine log shows `Vulkan0` computing (the baseline reports the same claim; the pair is placement-matched: every challenger chunk asked for 9 layers and got them, no degrade).

## 5. The 60 items: challenger vs the corrected baseline

- **baseline (corrected instrument, shipped cue)** — 22/60 = 0.367 [0.256 – 0.493]; `low_mass` 57/60, `measured` 3/60, refusals 59/60
- **challenger (role_split + json_instructed)** — 53/60 = 0.883 [0.778 – 0.942]; `low_mass` 0/60, `measured` 60/60, refusals 0/60

### 5.1 By question type

| type | baseline | challenger | discordant (challenger-only / baseline-only) | difference | exact McNemar p |
|---|---|---|---|---|---|
| choice | 7/24 | 22/24 | 16 / 1 | 0.625 | 2.747e-04 |
| noul | 7/18 | 18/18 | 11 / 0 | 0.611 | 9.766e-04 |
| score | 8/18 | 13/18 | 9 / 4 | 0.278 | 0.267 |

### 5.2 The collapse, and where the mass went

| metric | baseline | challenger |
|---|---|---|
| rows `low_mass` | 57/60 | 0/60 |
| rows `measured` (≥ 0.10 floor) | 3/60 | 60/60 |
| rows refused at the cue | 59/60 | 0/60 |
| coverage min | 1.442e-07 | 0.717 |
| coverage p25 | 1.741e-05 | 0.997 |
| coverage p50 (median) | 5.828e-04 | 0.998 |
| coverage p75 | 0.003 | 0.999 |
| coverage max | 0.131 | 1.000 |
| prefix tokens | 74–109 | 76–111 |
| framing (per row) | chat-template: qwen35moe / builtin | chat-template: qwen35moe / builtin |
| framing mixed across rows | False | False |
| distinct framing surfaces per row | 1 | 1 |

**The refusals, with the tokens that closed them.**

| arm | refused rows | closures |
|---|---|---|
| baseline | 59/60 | `</think>` × 32, `<think>` × 1, `<|im_end|>` × 26 |
| challenger | 0/60 | — |

**The cue verdicts (`value_verdict` breakdown).**

| arm | verdicts |
|---|---|
| baseline | — |
| challenger | `answered` × 60 |

### 5.3 The auxiliary arms (same instrument, never the card's row)

The two levers are separable, so the same 60 items were measured once more for each half: the placement alone (`role_split` with the shipped cue) and the 4B's collapse cell (`role_split` with `two_step`). Same model file, same items, same placement ask, same single backend — the cells differ only in the policy flags. They are published as auxiliary: the card's row is the challenger.

| arm | correct | agreement (Wilson 95 %) | low_mass | refusals | coverage p50 | prefix tokens | vs baseline (paired) | vs challenger | effective_backend |
|---|---|---|---|---|---|---|---|---|---|
| `role_split_only` | 35/60 | 0.583 [0.457 – 0.699] | 60/60 | 60/60 | 2.488e-08 | 71–106 | +0.217 (p = 0.007) | +0.300 (p = 1.211e-04) | `vulkan` |
| `two_step_role_split` | 35/60 | 0.583 [0.457 – 0.699] | 60/60 | 60/60 | 2.488e-08 | 71–106 | +0.217 (p = 0.007) | +0.300 (p = 1.211e-04) | `vulkan` |

* **`two_step` cannot engage under the role split on this family**: the two auxiliary cells agree on **60/60** decisions, on 60/60 prefix-token counts, and on the coverage of every shared row (max |Δcoverage| = 0, identical to the printed digits). **0 winner flip(s)** — because every one of the 60 cue rows under the role split is refused (`low_mass` 60/60, refusals 60/60) and `decide._advance_token` never advances past a cue the model closed (the rule `t_7c926398` §5.1 measured on the shipped placement). The 4B's collapse cell (28/60) is the *other* branch of that rule: there the label sits **at** the cue row, the advance succeeds, and the row it reaches is not a label row.
* the instructed contract moves **22 of 60** winners on top of the placement (38/60 identical decisions) and is what turns the readout into a measured one (`low_mass` 60/60 → 0/60) — the placement by itself still leaves the readout where the model opens its think block, which is why the two cells above are `measured` 0/60.
* **the auxiliary agreements are not readable as accuracy**, and the table above should not be read that way: every row of both cells is `low_mass` (`measured` 0/60) with median coverage 2.488e-08 — under the role split with a cue that is not a contract, Tiel opens its think block at the readout row (`<think>` × 60 closers, mass ≈ 1.0) and the candidates' mass is a tail. Those numbers are stable (identical on a re-measure, §7) but they are guesses among candidates, which is what the `low_mass` floor exists to say. The challenger is the only cell in this document whose rows are `measured`.


## 6. Paired against §7.4.1 (same 60 items, item by item)

* risk difference (challenger − baseline): **+0.517 (0.353 – 0.680, exact McNemar p = 7.842e-07)**
* discordant pairs: challenger-only correct **36**, baseline-only correct **5**
* both correct **17**, neither correct **2**
* items paired: **60** — unpaired rows: 0 baseline-only id(s), 0 challenger-only id(s)
* the card's rule (the E3e unit): a challenger displaces the baseline only when the paired difference's 95 % interval excludes zero and the exact two-sided McNemar p < 0.05 — here **WIN**
* interval caveat (the tool's own words): Wald interval on the discordant counts (optimistic near zero discordance); the exact McNemar p is the test

Items the challenger wins that the corrected baseline lost:

| id | type | gold | baseline got | challenger got |
|---|---|---|---|---|
| `c01` | choice | `technical` | `support` | `technical` |
| `c02` | choice | `billing` | `escalation` | `billing` |
| `c04` | choice | `restricted` | `confidential` | `restricted` |
| `c05` | choice | `security` | `support` | `security` |
| `c06` | choice | `fixed` | `changed` | `fixed` |
| `c07` | choice | `termination` | `liability` | `termination` |
| `c08` | choice | `business_hours` | `weekend` | `business_hours` |
| `c09` | choice | `spam` | `harassment` | `spam` |
| `c14` | choice | `capital` | `non_deductible` | `capital` |
| `c15` | choice | `cardiology` | `neurology` | `cardiology` |
| `c16` | choice | `travel` | `meals` | `travel` |
| `c18` | choice | `standup` | `retrospective` | `standup` |
| `c19` | choice | `configuration` | `infrastructure` | `configuration` |
| `c20` | choice | `copyleft` | `permissive` | `copyleft` |
| `c22` | choice | `sms` | `chat` | `sms` |
| `c23` | choice | `identity` | `search` | `identity` |
| `n01` | noul | `yes` | `no` | `yes` |
| `n02` | noul | `yes` | `no` | `yes` |
| `n03` | noul | `yes` | `no` | `yes` |
| `n04` | noul | `yes` | `no` | `yes` |
| `n05` | noul | `yes` | `no` | `yes` |
| `n07` | noul | `yes` | `no` | `yes` |
| `n10` | noul | `yes` | `no` | `yes` |
| `n11` | noul | `yes` | `no` | `yes` |
| `n13` | noul | `yes` | `no` | `yes` |
| `n15` | noul | `yes` | `no` | `yes` |
| `n18` | noul | `yes` | `no` | `yes` |
| `s02` | score | `3` | `0` | `3` |
| `s03` | score | `4` | `1` | `4` |
| `s04` | score | `4` | `0` | `4` |
| `s05` | score | `0` | `1` | `0` |
| `s06` | score | `3` | `1` | `3` |
| `s12` | score | `3` | `1` | `3` |
| `s15` | score | `2` | `1` | `2` |
| `s16` | score | `0` | `1` | `0` |
| `s17` | score | `2` | `1` | `2` |

Items the baseline got and the challenger loses:

| id | type | gold | baseline got | challenger got |
|---|---|---|---|---|
| `c11` | choice | `expedited` | `expedited` | `freight` |
| `s07` | score | `1` | `1` | `2` |
| `s09` | score | `1` | `1` | `0` |
| `s10` | score | `1` | `1` | `3` |
| `s13` | score | `1` | `1` | `0` |

## 7. The two 6-item cells of the card's falsification order

**1. The collapse cell (`--chat-format role_split --cue two_step`), 6 items** — `two_step/role_split`: 1/6 correct, `low_mass` 6/6, refusals 6/6, closers `<think>` × 6.

| item | type | gold | got | correct | coverage | reliability | cue top token | top-token mass | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `c01` | choice | `technical` | `technical` | True | 6.890e-09 | `low_mass` | `<think>` | 1.0000 | W_CUE_REFUSED |
| `s01` | score | `3` | `4` | False | 5.769e-07 | `low_mass` | `<think>` | 1.0000 | W_CUE_REFUSED |
| `n01` | noul | `yes` | `no` | False | 3.906e-07 | `low_mass` | `<think>` | 1.0000 | W_CUE_REFUSED |
| `c02` | choice | `billing` | `escalation` | False | 2.008e-09 | `low_mass` | `<think>` | 1.0000 | W_CUE_REFUSED |
| `s02` | score | `3` | `0` | False | 7.408e-08 | `low_mass` | `<think>` | 1.0000 | W_CUE_REFUSED |
| `n02` | noul | `yes` | `no` | False | 1.522e-08 | `low_mass` | `<think>` | 1.0000 | W_CUE_REFUSED |

**2. The challenger (`role_split` + `json_instructed`), the same 6 items** — `json_instructed/role_split`: 5/6 correct, `low_mass` 0/6, refusals 0/6.

| item | type | gold | got | correct | coverage | reliability | cue top token | top-token mass | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `c01` | choice | `technical` | `technical` | True | 0.996 | `ok` | `token 69685` | 0.9419 | ok |
| `s01` | score | `3` | `4` | False | 0.997 | `ok` | `token 19` | 0.5759 | ok |
| `n01` | noul | `yes` | `yes` | True | 0.997 | `ok` | `token 9405` | 0.9005 | ok |
| `c02` | choice | `billing` | `billing` | True | 0.999 | `ok` | `token 37355` | 0.9989 | ok |
| `s02` | score | `3` | `3` | True | 0.926 | `ok` | `token 18` | 0.9168 | ok |
| `n02` | noul | `yes` | `yes` | True | 0.997 | `ok` | `token 9405` | 0.9089 | ok |

**Determinism receipt (both cells).** The six smoke items are also the first six rows of the 60-item campaign's chunk 001, so the same cells were measured twice — once in a 6-item run, once inside the 10-item chunk — and they agree row for row: collapse cell **0 winner flip(s) of 6**, challenger **0 of 6**, coverage identical to the printed digits on every shared row. Temperature 0 reads the same bytes the same way; what the auxiliary cells cannot do is put mass on a candidate (their agreement is a tail argmax, and it is *stably* so).

## 8. What the reading is — and what it is not

* the challenger's 53/60 against §7.4.1's 22/60 is measured on the same committed items, the same model file (SHA identical before and after), the same placement ask and the same single backend; the only difference is the two policy flags.
* the collapse the corrected row published is gone on this arm: refusals 59 → 0, `low_mass` 57 → 0, median coverage 5.828e-04 → 0.998.
* 60 items: a single cell's interval is still up to ~24 points wide, so the *paired* columns are what carry the claim; the type cells (18–24 items) are leads, not results.
* the auxiliary cells are **not** accuracy claims: they are `measured` 0/60 with median coverage 2.488e-08, so their agreement is the argmax of a tail (§5.3). What they establish is the *mechanism*: the placement moves the question out of the assistant turn, and `two_step` is inert under it on this family because every cue row is closed before the readout can advance.
* `score` remains the hardest type for this model even under the winning policy: 13/18 against 8/18 — the same reading E3c/E3d/E3e published.
* this is one model's row under a *policy*; it moves no default and it does not rank models. The comparability note of E3e applies unchanged: a cell measured under a role split and an instructed contract cannot be compared with rows measured on the shipped prompt bytes.
* the optional Occamy pass (§9) is a pair measured **entirely here** — its own shipped/answer-sheet cell against `role_split` + `json_instructed` — because Occamy has no row under the corrected instrument; it is not comparable with the container-era E3 row, and its numbers live in this document (this card's `docs/BENCHMARKS.md` block is the Tiel row).


## 9. The optional Occamy pass (the card's “(and Occamy)”)

Occamy (`Accio-Lab_occamy-1.0-Q4_K_L.gguf`, `/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf`) has **no published row under the corrected instrument**: the E3 Occamy row of `docs/evidence/e3_t_a431be85_occamy.md` was measured in a container before the framing fix, so it is not comparable with this pair. Both cells are therefore measured here — the same 60 committed items, the same placement ask and the same `--backend vulkan --threads 4` instrument — and compared with each other, paired by item.

- **Occamy, shipped placement + shipped cue (measured here)** — 26/60 = 0.433 [0.316 – 0.559]; `low_mass` 60/60, `measured` 0/60, refusals 60/60
- **Occamy, role_split + json_instructed (measured here)** — 54/60 = 0.900 [0.799 – 0.953]; `low_mass` 0/60, `measured` 60/60, refusals 0/60

* paired risk difference (challenger − its own baseline): **0.467 (0.332 – 0.601, exact McNemar p = 5.774e-08)** — discordant 29 challenger-only against 1 baseline-only, both correct 25, neither correct 5; the pair clears the card's E3e unit rule.
* refusals at the cue 60/60 → 0/60, `low_mass` 60/60 → 0/60, `measured` 0/60 → 60/60.

### 9.1 By question type

| type | shipped cue (measured here) | role_split + json_instructed | discordant (challenger-only / baseline-only) | difference | exact McNemar p |
|---|---|---|---|---|---|
| choice | 12/24 | 23/24 | 12 / 1 | 0.458 | 0.003 |
| noul | 7/18 | 18/18 | 11 / 0 | 0.611 | 9.766e-04 |
| score | 7/18 | 13/18 | 6 / 0 | 0.333 | 0.031 |

### 9.2 Placement per chunk (the loader's own answer)

| chunk | shipped cue correct | challenger correct | ngl req → used | degraded | n_ctx | chunk wall | effective_backend | challenger low_mass |
|---|---|---|---|---|---|---|---|---|
| `001` | 2/10 | 9/10 | 9 → 9 | False | 205 | 37.100 s | `vulkan` | 0/10 |
| `002` | 4/10 | 9/10 | 9 → 9 | False | 194 | 36.100 s | `vulkan` | 0/10 |
| `003` | 4/10 | 8/10 | 9 → 9 | False | 171 | 36.800 s | `vulkan` | 0/10 |
| `004` | 4/10 | 9/10 | 9 → 9 | False | 207 | 37.000 s | `vulkan` | 0/10 |
| `005` | 6/10 | 10/10 | 9 → 9 | False | 172 | 35.100 s | `vulkan` | 0/10 |
| `006` | 6/10 | 9/10 | 9 → 9 | False | 198 | 37.700 s | `vulkan` | 0/10 |

### 9.3 What the Occamy cells rendered through

| surface | value | rows |
|---|---|---|
| `engine.chat_format` | `{"contract": "question", "dropped": "", "kind": "role_split", "question_turn": "user"}` | 60 |
| `engine.chat_format` | `{"contract": null, "dropped": null, "kind": "answer_sheet", "question_turn": "assistant"}` | 60 |
| `engine.template` | `{"family": "qwen35moe", "kind": "gguf-renderer", "renderer": "internal", "source": "gguf:tokenizer.chat_template"}` | 120 |

* **the pin.** `/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf` — 24113674848 bytes, mtime `2026-09-18 11:18:55`, SHA-256 `633ae57faf731e863cc3ba7cb75396a1b1e377191730e7b0d7294eff55cdf757` (`.t9bcb/logs/occamy_sha.txt`); the same digest is in `docs/evidence/e3_environment.json` for this file — **True** — and its mtime precedes the pass, so the file the two cells loaded is the file that was already on disk, unmoved.
* the driver's own before/after hash pair did **not** reach this pass's log: the run went through `systemd-run` without a file redirect and only its status lines were journaled, so the digest above is a single measurement taken after the pass, not a pair.
* per-chunk reports + placement sinks: `docs/evidence/t9bcbecff_occamy_chunks/`; merged: `docs/evidence/t9bcbecff_occamy_base_quality.json`, `docs/evidence/t9bcbecff_occamy_e3e_quality.json`; log: `.t9bcb/logs/occamy.log`.


## 10. Gates

| gate | command | result |
|---|---|---|
| campaign driver exits 0 | `bash .t9bcb/run_chunks.sh` (unit `t9bcb-campaign`) | chunk 001 exit=0 (36 s), chunk 002 exit=0 (35 s), chunk 003 exit=0 (36 s), chunk 004 exit=0 (36 s), chunk 005 exit=0 (37 s), chunk 006 exit=0 (37 s) — raw log `.t9bcb/logs/campaign.log` |
| model SHA-256 before/after | `sha256sum <tiel.gguf>` (campaign head and tail) | identical — `9286a94c453c6a40ad51982c3dc88df4bba32fee9efad06e4588c83c059cf17c` |
| dev-set slices | `sha256sum docs/evidence/tiel_corrected_chunks/devset_00*.jsonl` | byte-identical to the baseline's receipt — `True` |
| 6-item smokes exit 0 | `.t9bcb/smoke.sh` (unit `t9bcb-smoke`) | see §7 |
| the two auxiliary arms exit 0 | `TAG=… bash .t9bcb/run_arm.sh` (unit `t9bcb-aux`) | role_split_only 35/60, two_step_role_split 35/60 — raw log `.t9bcb/logs/campaign_aux.log` |
| oracle | `python3 docs/verify_runtime_contract.py` | failures: 0  skips: 0 (`.t9bcb/oracle.txt`) |
| test suite | `uv run --frozen --offline --extra dev pytest -q -rs` | 1331 passed, 48 skipped in 33.71s (`.t9bcb/gates.txt`) |
| ruff (the paths this card touches) | `uv run --frozen --offline --extra dev ruff check .t9bcb` | All checks passed! (`.t9bcb/logs/gates_run.log`) |
| the suite again, after this document's render | `uv run --frozen --offline --extra dev pytest -q -rs -p no:cacheprovider` | 1331 passed, 48 skipped in 33.06s (`.t9bcb/logs/final_suite.txt`) — the only tree change after that run is this section's own text |
| the optional Occamy pass exits 0 | `bash .t9bcb/run_occamy.sh` (unit `t9bcb-occamy2`) | 12 chunk(s), all exit 0 — `True` — log `.t9bcb/logs/occamy.log` |

The suite's skip count is this host's, not a container's: the worker scope carries no container pid cgroup, so `test_probe_pressure.py` skips — the same skip the baseline card recorded on this box.

## Receipts

* campaign driver: `.t9bcb/run_chunks.sh`, `.t9bcb/tiel_e3e_arm.py` (the committed observer of `tools/e3c_tiel_reproduce.py` + the two flags + the surface counters)
* smokes: `.t9bcb/smoke.sh`, `.t9bcb/smoke_collapse.json`, `.t9bcb/smoke_challenger.json` and their placement sinks
* per-chunk reports + placement sinks: `docs/evidence/t9bcbecff_tiel_chunks/`
* merged challenger report: `docs/evidence/t9bcbecff_tiel_challenger_quality.json`
* merged auxiliary reports: `docs/evidence/t9bcbecff_tiel_role_split_quality.json`, `docs/evidence/t9bcbecff_tiel_two_step_quality.json`
* baseline: `docs/evidence/tiel_corrected_quality.json` + `docs/evidence/tiel_corrected_chunks/` (card `t_7c926398`, unchanged)
* statistics: `.t9bcb/stats.json` (this document is rendered from it by `.t9bcb/render_doc.py`)
* run logs: `.t9bcb/logs/campaign.log`, `.t9bcb/logs/run_00{1..6}.log`, `.t9bcb/logs/smoke_*.log`, `.t9bcb/logs/campaign_aux.log` (+ the two auxiliary arms' per-chunk logs)
* the same numbers in the benchmark document: `docs/BENCHMARKS.md` §7.4.2, spliced by this script from the same `stats.json`
* the optional Occamy pass: `.t9bcb/run_occamy.sh` → `.t9bcb/run_arm.sh`, `docs/evidence/t9bcbecff_occamy_chunks/`, the two merged reports, `.t9bcb/logs/occamy.log` + `.t9bcb/logs/occamy_sha.txt` (the pin, taken after the pass)
