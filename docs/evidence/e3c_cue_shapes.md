# E3c — the cue that isn't refused, and `W_CUE_REFUSED` (card `t_6c119626`)

E3b (`t_6952f0dd`) measured the label-policy question into a clean negative: on
`Accio-Lab/occamy-1.0` **no** rendering of the candidate name cleared the engine's 0.10 coverage
floor, because the cue row itself was `<|im_end|>` at p = 0.9976…1.00000 — the model closed the
assistant turn instead of answering. The engine reported only `low_mass`, which hides *why*.

E3c attacks both halves: the **verdict** (the engine now names the refusal) and the **shape** (put
the readout inside an answer, and measure it on both a model that answers and the model that
refuses).

## 1. What shipped (code)

| where | what | gates |
|---|---|---|
| `src/ggufone/engine/cue.py` | `single_token_closers(tokenize)` + `cue_verdict(row, scale, closers)`: the row's argmax classified through the **session's own tokenizer** (a closer counts only if this vocabulary encodes it as exactly one token — `<\|eot_id\|>` in a word-level vocabulary never fires) | `tests/test_e3c_cue_refused.py` |
| `src/ggufone/engine/decide.py` | `W_CUE_REFUSED` in `warnings` **and** the payload a flat list cannot carry — `{"refused", "closer", "mass", "hint"}` under the answer's `cue` key (choice, score and noul) | ditto |
| `src/ggufone/bench/harness.py` | `render_report` draws the **cue verdicts** table for the quality/calibration suites (item, type, cue top token, its mass, verdict) and the EOT-vs-candidate table | `tests/test_e3c_cue_shapes.py::test_the_verdict_table_*` |
| `src/ggufone/errors.py` | `W_CUE_REFUSED` in the warning catalogue | `test_the_warning_is_in_the_catalogue` |
| `tools/e3c_cue_shapes.py` | the probe: 7 cue shapes, five label renderings read off the same rows, the closers' mass, and the ranked readout (`labels.score_paths`) for named shape=label policies | `tests/test_e3c_cue_shapes.py` (21 gates) |
| `README.md`, `docs/TEMPLATES.md` | the warning is documented in the serving-path paragraph, and §4's `qwen35moe` row loses **[UNVERIFIED]**: measured, thinking suppressed, **cue refused**, plus the new *Cue shapes that put the readout mid-answer* guidance | `test_the_qwen35moe_row_is_no_longer_unverified`, `test_templates_*` |

The verdict is **diagnostic**: the shipped readout is unchanged. Choosing a different cue shape is a
quality decision with the same owner as the label policy (the labeled dev set) — E3c measures it and
publishes the numbers, it does not re-route production.

### Live capture (the serving path, not a fixture)

```
$ VK_DRIVER_FILES=/nonexistent/no-vulkan-icd.json uv run --frozen ggufone ask \
    --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
    --state @.e3c/state_occamy.txt --choice 'area=Which team owns this?:billing|technical|support'
…
"reliability": "low_mass",
"cue": {"refused": true, "token": 248046, "closer": "<|im_end|>", "mass": 0.999909,
        "hint": "docs/TEMPLATES.md §4 (the label policy, measured) — …"},
"warnings": ["W_BACKEND_MISMATCH", "W_LOW_MASS", "W_CUE_REFUSED"]
```

The whole response is `docs/evidence/e3c_cue_refused_ask.json`. `W_BACKEND_MISMATCH` is *correct*
here and not a bug of this card: with the Vulkan loader hidden the bundle's `vulkan` claim is
untenable, the log shows `CPU compute buffer size`, and the E3 fix (`t_80f1a4c6`) names the
contradiction — the same rule that makes the CPU-only probe runs claim `cpu` explicitly.

## 2. What was measured

Probe: `tools/e3c_cue_shapes.py run`, the 6 dev items E3b used
(`docs/evidence/e3_chunks/devset_001.jsonl --ids c01 c02 s01 s02 n01 n02`), `--threads 4`, all five
label renderings (`bare newline prefix suffix quoted` — read off the same rows, one forward pass
each), engine floor 0.10 (`labels.MASS_FLOOR`), ranked readout for `shipped=bare`,
`two_step_shipped=bare`, `answer_is=bare`.

The shapes are always **the shipped suffix plus an opener** (byte-for-byte, gated), so what changes
between two rows is the cue and nothing else — the question, the criteria and the instructions are
identical by construction:

| shape | what it does |
|---|---|
| `shipped` | the control: nothing appended, read at the cue (E3/E3b's shape) |
| `answer_is` | `The answer is ` appended (English) |
| `answer_colon` | `Answer: ` appended |
| `wybieram_pl` | `Zgodnie z opisem, wybieram: ` appended (Polish — the model is co-work post-trained, do not assume one language) |
| `json_field` | the field opened *for* the model: `{"choice": "` / `{"severity": "` / `{"answer": "` |
| `two_step_shipped` | the shipped suffix byte-identical, one model-chosen **content** token decoded, read at the **second** decision point (the generalized `newline`) |
| `two_step_answer_is` | `The answer is ` + one model token, read after it |

The two-step shapes advance on the row's own argmax **excluding the turn-closers**, with the
engine's frozen lowest-index tie-break — a shape can never "advance" onto the closer it is supposed
to get past.

### The 4B inverse control (positive)

`Spark-X2.5-4B-Q8_0` — the model E2 measured answering at the cue — CPU, `--hide-devices`,
`--gpu-layers 0`. Full tool-generated tables: `docs/evidence/e3c_cue_shapes_4b.md`.

| shape | items above floor | ranked `bare` readout |
|---|---|---|
| `shipped` (control) | 0/6 | 3/6 correct (0.500), **6/6 `low_mass`** |
| `answer_is` | 2/6 | 4/6 correct (0.667), 4/6 `low_mass` |
| `answer_colon` | 2/6 | — |
| `wybieram_pl` | 2/6 | — |
| `json_field` | **6/6** | — |
| `two_step_shipped` | **6/6** | 5/6 correct (0.833), **0/6 `low_mass`** |
| `two_step_answer_is` | 4/6 | — |

**Verdict: positive.** On a model that answers, the shape is the lever: the two-step readout takes
the same 6 items from 0/6 above the floor and 3/6 correct (with every answer `low_mass`) to 6/6 and
5/6, and opening the JSON field does it in one shot. The plain openers triple the label mass without
clearing the floor — they move the tip of the prompt, not the readout.

### Occamy 1.0 (the family that refuses): a clean negative

`Accio-Lab_occamy-1.0-Q4_K_L` — the model E3b measured — 6 items, CPU, `--hide-devices`,
`--gpu-layers 0` (see the placement note below). Full tool-generated tables:
`docs/evidence/e3c_cue_shapes_occamy.md`.

| shape | items above floor | refused | the argmax at the readout row |
|---|---|---|---|
| `shipped` (the control) | 0/6 | **6/6** | `<|im_end|>`, p = 0.9999…1.0000 |
| `answer_is` | 0/6 | **6/6** | `<|im_end|>`, p = 0.9841…0.9997 |
| `answer_colon` | 0/6 | **6/6** | `<|im_end|>`, p = 0.9783…0.9994 |
| `wybieram_pl` | **1/6** | **6/6** | `<|im_end|>`, p = 0.5643…0.9999 |
| `json_field` | 0/6 | **6/6** | `<|im_end|>`, p = 0.9130…0.9997 |
| `two_step_shipped` | 0/6 | 1/6 | a newline (`\n\n` ×3, `\n` ×2), p = 0.5441…1.0000 |
| `two_step_answer_is` | 0/6 | 1/6 | a newline (`\n\n` ×3, `\n` ×2), p = 0.7232…0.9968 |

Mean `bare` coverage: 2.20e-06 (`shipped`) … 8.76e-02 (`wybieram_pl`) for the at-the-cue shapes,
and 3.95e-08 / 6.30e-07 for the two-steps; the single item above the floor (`wybieram_pl`) is still
a refused row — the floor and the verdict are different questions, and the verdict is the stronger
one. Ranked readout (`labels.score_paths`): `shipped=bare` 1/6 correct (0.167), `answer_is=bare`
4/6 (0.667), `two_step_shipped=bare` 3/6 (0.500) — **all three with 6/6 `low_mass`**, so those
"correct" counts are restricted-softmax tie-breaks, not knowledge.

**Verdict: a clean negative, and a named one.** Across the 6 items the five at-the-cue shapes are
refused **30/30** — the forced partial answers (`The answer is `, `Answer: `, `Zgodnie z opisem,
wybieram: `) and the opened JSON field included: whatever follows the cue, this model's argmax is
`<|im_end|>` (p = 0.5643…1.0000). So the closer's mass *is* the EOT-vs-candidate number the card
asked for, and the candidate label never gets a chance (`W_CUE_REFUSED`). The two-step shapes do get
past the closer — they never advance onto one, so only 2/12 of their cells are refused — but they
land on whitespace: the model's first content token is `</think>` (the closer E1c's empty-block
strip removed from the prefix), and the row after it is a newline at p = 0.5441…1.0000, 0/6 above
the floor.

Only one shape×item cell in the whole sweep cleared the floor (`wybieram_pl`, 1/6) and that row was
refused too. So on this family the honest answer is: **none of the seven shapes lifts the coverage
floor**, the shape lever that works on the 4B does not transfer, and one step past the cue is not
enough. The mechanism is testable — the strip leaves the model wanting to emit
`<|im_end|>`/`</think>`/newlines — and it is the trigger for the follow-up at the end, not a claim
here.

## 3. Reproduction

```bash
# the 4B control (CPU by construction: the box's GPU belongs to a sibling campaign)
bash .e3c/run_4b.sh                      # -> .e3c/spark4b.json + .e3c/spark4b.md

# the Occamy sweep, resumable, one dev item per invocation (23 GB model)
bash .e3c/run_occamy_items.sh all        # -> .e3c/occamy_<id>.json + .md
python3 .e3c_scratch/merge_runs.py .e3c/occamy_*.json .e3c/occamy.json .e3c/occamy.md
```

`GGUFONE_RUNTIME_DIR` points at the shipped Vulkan bundle for both runs (`b11026-linux-x64-vulkan`);
`--hide-devices` makes the CPU placement deterministic by pointing the Vulkan loader at an ICD that
does not exist.

### Placement note (recorded, not hidden)

E3b's Occamy numbers were taken with **7 GPU layers** offloaded. This run is **CPU**
(`--backend cpu --gpu-layers 0`) because the GPU is held by a sibling campaign (743 MiB free, and a
Vulkan context reserves ~1 GB of *device* memory even with zero layers offloaded — that is what
OOMs). The label mass at a fixed row is a weight-level quantity, and the engine's own device log
line (`CPU compute buffer size = …`) is quoted in the report header, so the reader can see which
device computed. The cue verdict itself (`<|im_end|>` at p = 0.99998) reproduces E3b's GPU number.

### SHA pins

The probe hashes both models before measuring (`model_sha256` in every record) and the tree's
hashes are re-checked after the sweep; the models are published copies under
`/var/home/rybens/.hermes/models/` and are not modified by any run (read-only mmap).

| model | bytes | sha256 | before/after |
|---|---|---|---|
| `Accio-Lab_occamy-1.0-Q4_K_L.gguf` | 24 113 674 848 | `633ae57faf731e863cc3ba7cb75396a1b1e377191730e7b0d7294eff55cdf757` | identical (and equal to E3's published digest) |
| `Spark-X2.5-4B-Q8_0.gguf` | 4 375 021 152 | `5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2` | identical |

Machine-readable: `docs/evidence/e3c_cue_sha256_after.txt` and `e3c_cue_sha256_receipt.json`
(`{"identical": true, "published_matches": true}`). Careful with the names: the older
`e3c_sha256_before.txt` / `e3c_sha256_after.txt` / `e3c_sha256_receipt.json` in the same directory
belong to E3's host campaign (card `t_6d2e084d`), not to this card.

## 4. Limits and what is left open

* **One step is not a mechanism.** `two_step_*` reads after exactly one model-chosen content token.
  On Occamy that lands on a whitespace loop; whether the answer mass appears after a *later* row
  (e.g. after the model's own `</think>` + blank line) is a measurement nobody has taken — the
  trigger for a follow-up card, not a claim here.
* **`correct` at `low_mass` is a tie-break.** The ranked readout's agreement column is only
  knowledge when the coverage cleared the floor; the reports print the `low_mass` share next to it
  for exactly that reason (Occamy c01: 1/1 `low_mass` while the ranked winner happened to match the
  expected label).
* **The 4B control is a control, not a dev-set verdict.** 6 items, and the ranked readout's CI is
  wide (0.436–0.970 for the two-step policy). It shows the shape moves the readout, not that it
  beats the shipped policy on quality — that needs the labeled dev set.
* **CPU vs GPU placement** (above).
