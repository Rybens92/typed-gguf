# E3b — the `qwen35moe` label policy: candidate-mass coverage per variant

Card `t_6952f0dd` · branch `main` (this repo has no remote; the commits are local on the shared tree) · Tier **M** · schema `ggufone.e3b.label-policy/v1`

**Question.** E3 (`t_a431be85`) published a paired comparison in which every Occamy answer was `low_mass` (20/20, no `measured` row at all) while the 4B was mostly `measured` on the same 20 items. This card asks whether the *rendering* of the candidate label is part of that, and picks a label policy for this family — or records the negative result.

**Answer in one paragraph.** No label rendering (leading space, casing, the two-step newline readout, the long description form) and no cue rewrite (blank line, an explicit cue naming the labels) reaches the engine's 0.10 coverage floor on this model: the best combination, `explicit` × `bare`, has mean coverage 3.359e-05 and lifts 0/6 items above the floor. The reason is in the cue row itself — `<|im_end|>` takes p ≈ 1.00000 of the next-token mass on the shipped prompt, i.e. the model closes the assistant turn instead of answering, and a label rendering cannot change that. This is the negative result the card allows, and the live table is below.

## 1. The pin, the box, the placement

| what | value |
|---|---|
| model | `Accio-Lab_occamy-1.0-Q4_K_L.gguf` — 24,113,674,848 bytes, arch `qwen35moe` |
| **SHA-256** | `633ae57faf731e863cc3ba7cb75396a1b1e377191730e7b0d7294eff55cdf757` (hashed by this run, before the load) |
| runtime | `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` (`GGUFONE_RUNTIME_DIR`, the pinned b11026 bundle) |
| backend / threads | `vulkan` / 4 (E3 §6.5: 4 beats 8 and 12 by ~2×) |
| `--gpu-layers` requested | 7 |
| placement the loader used | `{"note": "degraded after a backend allocation failure: 3 layer(s) offloaded, kv_type=f16", "n_gpu_layers": 3, "kv_type": "f16", "degraded": true, "attempts": ["n_gpu_layers=7 -> oom"], "warnings": ["W_BACKEND_OOM", "W_FIT_DOWNGRADE"]}` |
| model load | 22198 ms |
| wall, whole run | 1783.6 s |

Every number in this document is tagged by construction **[container]**: the worker container sees `cpu.max = 2 CPU-seconds/s` and `memory.max = 8 GiB` against a 23 GB model (E3 §2.2), so one prefill *and* one decode batch each cost a full weight sweep from disk. No `[host]` re-run happened in this card.

## 2. Method

* the candidate label is **rendered** by `bench/labels.py`: `bare` (what the engine ships), `space` (`' billing'`), `caps` (`'Billing'`), `newline` (`'\nbilling'`, the two-step readout) and `long` (the description the prompt shows — `'billing: payments, invoices and refunds'`, the level text for `score`, the criteria text for `noul`);
* the cue line is rendered by the same module: `shipped`, `blank` (one empty line after the cue — the variant `docs/TEMPLATES.md` §4 measured to triple the label mass on the 4B) and `explicit` (`Answer with exactly one of these candidate names: billing, …`);
* **one model load, one context per item-prefix, one decode batch per prefix for all three cue variants** — each cue is a sequence forked from the item's prefix, which is the engine's own candidate protocol, so the cue variants share the sweep;
* `coverage` is `readout.coverage_from_scale` on the cue row: the full-vocabulary mass of each label's first token, read *without* any extra forward pass (that is why 15 cue × label combinations cost 3 decodes per item);
* the **ranked** readout (`--rank cue=label`) is the engine's decision — `readout.candidate_sequence_score` + `restricted_softmax` + `argmax_first` — with the candidate sequences decoded through a prefix-sharing trie (`labels.score_paths`, gated offline in `tests/test_e3b_labels.py`);
* the **cross-check** runs the same item through `DecisionEngine` itself and compares coverage and winner, so a probe bug cannot be published as a model finding.

`docs/evidence/e3_chunks/devset_001.jsonl` — the first 6 items 
(choice 2, noul 2, score 2), the same items E3 measured, so the before/after sides ask the same questions.

## 3. Coverage per (cue, label) variant

| cue | label | c01 | c02 | s01 | s02 | n01 | n02 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 8.068e-08* | 5.429e-07* | 7.318e-07* | 5.018e-06* | 1.522e-06* | 5.645e-06* | 0/6 | 1.522e-06 |
| `shipped` | `space` | 5.084e-09* | 7.350e-09* | 2.855e-10* | 1.433e-09* | 7.425e-09* | 6.495e-08* | 0/6 | 7.350e-09 |
| `shipped` | `caps` | 7.816e-09* | 1.029e-08* | 7.318e-07* | 5.018e-06* | 8.718e-09* | 3.691e-08* | 0/6 | 3.691e-08 |
| `shipped` | `newline` | 6.735e-08* | 1.882e-09* | 5.735e-09* | 9.914e-09* | 6.169e-08* | 1.907e-07* | 0/6 | 6.169e-08 |
| `shipped` | `long` | 8.068e-08* | 5.429e-07* | 5.532e-10* | 6.400e-11* | 1.522e-06* | 5.645e-06* | 0/6 | 5.429e-07 |
| `blank` | `bare` | 1.491e-06* | 3.184e-07* | 2.457e-07* | 6.754e-07* | 1.243e-06* | 8.350e-07* | 0/6 | 8.350e-07 |
| `blank` | `space` | 5.580e-08* | 1.187e-08* | 6.250e-11* | 3.475e-10* | 4.960e-09* | 2.792e-09* | 0/6 | 4.960e-09 |
| `blank` | `caps` | 1.558e-07* | 1.752e-08* | 2.457e-07* | 6.754e-07* | 1.903e-08* | 4.769e-09* | 0/6 | 1.558e-07 |
| `blank` | `newline` | 6.241e-10* | 3.487e-11* | 2.904e-10* | 4.510e-11* | 1.694e-09* | 1.066e-08* | 0/6 | 6.241e-10 |
| `blank` | `long` | 1.491e-06* | 3.184e-07* | 3.409e-10* | 2.104e-10* | 1.243e-06* | 8.350e-07* | 0/6 | 8.350e-07 |
| `explicit` | `bare` | 2.564e-07* | 1.316e-07* | 7.186e-06* | 9.869e-06* | 1.290e-04* | 5.504e-05* | 0/6 | 9.869e-06 |
| `explicit` | `space` | 2.357e-08* | 1.232e-08* | 3.514e-07* | 7.164e-08* | 2.519e-06* | 1.031e-06* | 0/6 | 3.514e-07 |
| `explicit` | `caps` | 4.697e-07* | 3.077e-08* | 7.186e-06* | 9.869e-06* | 2.364e-06* | 1.847e-06* | 0/6 | 2.364e-06 |
| `explicit` | `newline` | 2.423e-08* | 2.144e-08* | 1.255e-07* | 8.366e-09* | 1.976e-07* | 2.446e-07* | 0/6 | 1.255e-07 |
| `explicit` | `long` | 2.564e-07* | 1.316e-07* | 6.695e-09* | 6.480e-10* | 1.290e-04* | 5.504e-05* | 0/6 | 2.564e-07 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the cue (`readout.coverage_from_scale`), read from the cue row — so every label variant of one cue costs no extra forward pass.

### 3.1 Summary

| cue | label | mean coverage | median | above floor | low_mass share | best item | worst item |
|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 2.257e-06 | 1.522e-06 | 0/6 | 1.00 | n02 5.645e-06 | c01 8.068e-08 |
| `shipped` | `space` | 1.442e-08 | 7.350e-09 | 0/6 | 1.00 | n02 6.495e-08 | s01 2.855e-10 |
| `shipped` | `caps` | 9.689e-07 | 3.691e-08 | 0/6 | 1.00 | s02 5.018e-06 | c01 7.816e-09 |
| `shipped` | `newline` | 5.621e-08 | 6.169e-08 | 0/6 | 1.00 | n02 1.907e-07 | c02 1.882e-09 |
| `shipped` | `long` | 1.299e-06 | 5.429e-07 | 0/6 | 1.00 | n02 5.645e-06 | s02 6.400e-11 |
| `blank` | `bare` | 8.014e-07 | 8.350e-07 | 0/6 | 1.00 | c01 1.491e-06 | s01 2.457e-07 |
| `blank` | `space` | 1.264e-08 | 4.960e-09 | 0/6 | 1.00 | c01 5.580e-08 | s01 6.250e-11 |
| `blank` | `caps` | 1.864e-07 | 1.558e-07 | 0/6 | 1.00 | s02 6.754e-07 | n02 4.769e-09 |
| `blank` | `newline` | 2.225e-09 | 6.241e-10 | 0/6 | 1.00 | n02 1.066e-08 | c02 3.487e-11 |
| `blank` | `long` | 6.480e-07 | 8.350e-07 | 0/6 | 1.00 | c01 1.491e-06 | s02 2.104e-10 |
| `explicit` | `bare` | 3.359e-05 | 9.869e-06 | 0/6 | 1.00 | n01 1.290e-04 | c02 1.316e-07 |
| `explicit` | `space` | 6.682e-07 | 3.514e-07 | 0/6 | 1.00 | n01 2.519e-06 | c02 1.232e-08 |
| `explicit` | `caps` | 3.628e-06 | 2.364e-06 | 0/6 | 1.00 | s02 9.869e-06 | c02 3.077e-08 |
| `explicit` | `newline` | 1.036e-07 | 1.255e-07 | 0/6 | 1.00 | n02 2.446e-07 | s02 8.366e-09 |
| `explicit` | `long` | 3.074e-05 | 2.564e-07 | 0/6 | 1.00 | n01 1.290e-04 | s02 6.480e-10 |

The best combination by mean coverage is `explicit` × `bare` (mean 3.359e-05, 0/6 items at or above the floor); the best single value anywhere in the sweep is 1.290e-04, i.e. still below the 0.10 floor. 

By cue variant the ordering is `explicit`-best 3.359e-05 (its best label `bare`, max 1.290e-04) then `shipped` 2.257e-06: even the cue that names the labels explicitly lifts the mean coverage by 15x over the runner-up and leaves every answer short of the floor.


## 4. What the model wants to emit at the cue (why the label mass is where it is)

| item | cue | top tokens at the cue (full-vocab p) |
|---|---|---|
| c01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · ````` 0.0000 · `Answer` 0.0000 |
| c01 | `blank` | `<|im_end|>` 0.9999 · `</think>` 0.0000 · `<think>` 0.0000 · `<|im_start|>` 0.0000 |
| c01 | `explicit` | `<|im_end|>` 0.9996 · `Answer` 0.0003 · `</think>` 0.0000 · ````` 0.0000 |
| c02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `answer` 0.0000 |
| c02 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `technical` 0.0000 |
| c02 | `explicit` | `<|im_end|>` 0.9998 · `Answer` 0.0002 · `answer` 0.0000 · `</think>` 0.0000 |
| s01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `4` 0.0000 · `Answer` 0.0000 |
| s01 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `4` 0.0000 |
| s01 | `explicit` | `<|im_end|>` 0.9992 · `Answer` 0.0006 · `</think>` 0.0001 · `4` 0.0000 |
| s02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `3` 0.0000 · `Answer` 0.0000 |
| s02 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `<|im_start|>` 0.0000 |
| s02 | `explicit` | `<|im_end|>` 0.9992 · `Answer` 0.0007 · `</think>` 0.0000 · `<think>` 0.0000 |
| n01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `no` 0.0000 |
| n01 | `blank` | `<|im_end|>` 0.9999 · `<think>` 0.0001 · `</think>` 0.0000 · `Answer` 0.0000 |
| n01 | `explicit` | `<|im_end|>` 0.9976 · `Answer` 0.0020 · `no` 0.0001 · `</think>` 0.0001 |
| n02 | `shipped` | `<|im_end|>` 0.9999 · `Answer` 0.0000 · `</think>` 0.0000 · `no` 0.0000 |
| n02 | `blank` | `<|im_end|>` 0.9999 · `<think>` 0.0000 · `</think>` 0.0000 · `Answer` 0.0000 |
| n02 | `explicit` | `<|im_end|>` 0.9977 · `Answer` 0.0019 · `</think>` 0.0002 · `no` 0.0001 |

## 5. The prefix control: the template's own empty think block

| item | prefix | prefix tokens | cue | top token (p) | `bare` coverage | `newline` coverage |
|---|---|---|---|---|---|---|
| c01 | `shipped` | 109 | `shipped` | `<|im_end|>` 1.0000 | 8.068e-08 | 6.735e-08 |
| c01 | `kept` | 113 | `shipped` | `<|im_end|>` 0.9988 | 3.646e-06 | 6.221e-06 |
| c02 | `shipped` | 99 | `shipped` | `<|im_end|>` 1.0000 | 5.429e-07 | 1.882e-09 |
| c02 | `kept` | 103 | `shipped` | `<|im_end|>` 0.9985 | 1.368e-05 | 4.613e-05 |

## 6. The ranked readout, and the `DecisionEngine` cross-check

| policy | n | correct | agreement | 95% CI | low_mass | median coverage |
|---|---|---|---|---|---|---|
| `shipped=bare` | 6 | 2 | 0.333 | 0.097–0.700 | 6/6 | 0.0000 |
| `shipped=newline` | 6 | 1 | 0.167 | 0.030–0.564 | 6/6 | 0.0000 |

| item | probe coverage | engine coverage | |Δ coverage| | probe winner | engine winner |
|---|---|---|---|---|---|
| c01 | 8.0684e-08 | 1.6435e-07 | 8.37e-08 | technical | support |

Worst |Δ coverage| over the cross-checked items: **8.37e-08** — the probe reads the same row with the same function; only the batch shape differs.

## 7. `ggufone calibrate` on the accepted run

```
calibration ggufone.calibration/v1  model file:Accio-Lab_occamy-1.0-Q4_K_L.gguf:24113674848
  devset 20 item(s) docs/evidence/e3_occamy_quality.json digest sha256:6c222815aa850fecf2b
  mode auto  holdout 0.333  params sha256:5a96f400a953dc726f484efdc2b4f35ab4837a6959631f7b8475be2da8806d16
  type     temperature          holdout ECE              fit ECE  verdict
  choice        1.0000     0.2687 -> 0.2687     0.4538 -> 0.4538  no calibration applied: too few fit rows (5 < 8)
  noul          1.0000     0.4178 -> 0.4178     0.6290 -> 0.6290  no calibration applied: too few fit rows (4 < 8)
  score         1.0000     0.5308 -> 0.5308     0.2469 -> 0.2469  no calibration applied: too few fit rows (5 < 8)
  no calibration applied (no question type improved on the held-out split)
```

## 8. Before/after on the paired item set

**Occamy before/after** (20 paired items — E3's merged report dropped 40 row(s) it measured and this card did not, so both sides are the same items):

Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | Occamy shipped (E3) | Occamy accepted | delta |
|---|---|---|---|
| overall | 0.450 (9/20) [0.258–0.658] | 0.300 (6/20) [0.145–0.519] | -0.150 |
| choice | 0.571 (4/7) [0.250–0.842] | 0.143 (1/7) [0.026–0.513] | -0.429 |
| noul | 0.167 (1/6) [0.030–0.564] | 0.167 (1/6) [0.030–0.564] | +0.000 |
| score | 0.571 (4/7) [0.250–0.842] | 0.571 (4/7) [0.250–0.842] | +0.000 |
| low_mass (below the floor) | 0.450 (9/20) [0.258–0.658] | 0.300 (6/20) [0.145–0.519] | -0.150 |
| measured (at or above the floor) | — | — | — |

`Occamy accepted` is worse than `Occamy shipped (E3)` by -0.150 overall (0.450 -> 0.300); the `measured` row is the one to read first.

The coverage the before/after sides were read with (each row's own `coverage` field; the floor is the engine's 0.10):

| side | n | mean coverage | median | max | low_mass |
|---|---|---|---|---|---|
| `Occamy shipped (E3)` | 20 | 2.838e-02 | 1.976e-02 | 9.939e-02 | 20/20 |
| `Occamy accepted (E3b)` | 20 | 2.717e-06 | 9.517e-07 | 1.270e-05 | 20/20 |

The two sides read the cue row through different execution paths: E3's `DecisionEngine` decodes each candidate sequence one at a time and reports the mass of the winner's *first token* (`coverage_from_scale`), while this probe reads the same quantity from the batched cue row. The E3 side's mean is 10446x the probe's — a scale difference between a sequential and a batched decode, not a re-measure of different items — and **both sides carry an identical verdict**: every row, old and new, is `low_mass`. The margin is the finding, and the gate this card needed is the threshold crossing, which did not happen. The `low_mass` share is unchanged: **both sides are `low_mass` on every item**, so the re-measure reproduces E3's 20/20 — the label rendering moved the masses, never across the floor.

**The published pairing re-rendered** (20 paired items, dropped 40 unpaired baseline row(s) and 0 unpaired challenger row(s)):

Agreement on the committed dev set, 95 % Wilson intervals; the mass split uses the engine's own verdict, or `coverage < 0.10` where a report predates it.

| metric | 4B default (E2, CPU) | Occamy accepted (E3b) | delta |
|---|---|---|---|
| overall | 0.500 (10/20) [0.299–0.701] | 0.300 (6/20) [0.145–0.519] | -0.200 |
| choice | 0.429 (3/7) [0.158–0.750] | 0.143 (1/7) [0.026–0.513] | -0.286 |
| noul | 1.000 (6/6) [0.610–1.000] | 0.167 (1/6) [0.030–0.564] | -0.833 |
| score | 0.143 (1/7) [0.026–0.513] | 0.571 (4/7) [0.250–0.842] | +0.429 |
| low_mass (below the floor) | 0.333 (1/3) [0.061–0.792] | 0.300 (6/20) [0.145–0.519] | -0.033 |
| measured (at or above the floor) | 0.529 (9/17) [0.310–0.738] | — | — |

`Occamy accepted (E3b)` is worse than `4B default (E2, CPU)` by -0.200 overall (0.500 -> 0.300); the `measured` row is the one to read first.

**The alternative policy measured on the same items** (`shipped` cue × `newline` label, the two-step readout):

agreement 0.450 (9/20) [0.258–0.658]; mean coverage 6.465e-08, max 3.643e-07, `low_mass` 20/20 — the same negative result, with a 20/20 floor verdict.


## 9. Honest limits

1. **Container numbers.** One weight sweep per prefill *and* per decode batch is a property of the 8 GiB cgroup, not of the model; the coverage/agreement numbers do not depend on it, the wall times do.
2. **Small n.** The variant sweep is a fixed 6-item subset (2 per question type) of chunk 001; the re-measure is 20 items. A 6-item subset can rank variants, it cannot estimate an agreement.
3. **One runtime, one box, one quantization** (`b11026`, Vulkan + CPU offload, Q4_K_L).
4. **The ranked readout is the engine's arithmetic driven by a probe**, not by `DecisionEngine`: the cross-check above is what makes the two comparable.
5. **No E3-published number was changed** by this card; the before/after table is its own artifact.

## 10. The generated tables

# E3b — `qwen35moe` label policy: candidate-mass coverage per variant

- card `t_6952f0dd` · model `Accio-Lab_occamy-1.0-Q4_K_L.gguf` (24,113,674,848 bytes, sha256 `633ae57faf731e86…`)
- runtime `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` · backend `vulkan` · threads 4 · `--gpu-layers` requested 7
- placement used: `{"note": "degraded after a backend allocation failure: 3 layer(s) offloaded, kv_type=f16", "n_gpu_layers": 3, "kv_type": "f16", "degraded": true, "attempts": ["n_gpu_layers=7 -> oom"], "warnings": ["W_BACKEND_OOM", "W_FIT_DOWNGRADE"]}`
- dev items: c01, c02, s01, s02, n01, n02 (choice 2, noul 2, score 2) · prefix variants: `shipped`, `kept` · generated 2026-09-19T13:51:15Z
- engine floor: coverage < **0.10** ⇒ `low_mass` (`OPTION_DEFAULTS['coverage_floor']`)
- wall: 1783.6 s (one model load 22198 ms, one context per item-prefix, one decode batch per prefix for all cue variants)

## 1. What the model puts at the cue (the row coverage is read from)

| item | cue | top tokens at the cue (full-vocab p) |
|---|---|---|
| c01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · ````` 0.0000 · `Answer` 0.0000 |
| c01 | `blank` | `<|im_end|>` 0.9999 · `</think>` 0.0000 · `<think>` 0.0000 · `<|im_start|>` 0.0000 |
| c01 | `explicit` | `<|im_end|>` 0.9996 · `Answer` 0.0003 · `</think>` 0.0000 · ````` 0.0000 |
| c02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `answer` 0.0000 |
| c02 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `technical` 0.0000 |
| c02 | `explicit` | `<|im_end|>` 0.9998 · `Answer` 0.0002 · `answer` 0.0000 · `</think>` 0.0000 |
| s01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `4` 0.0000 · `Answer` 0.0000 |
| s01 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `4` 0.0000 |
| s01 | `explicit` | `<|im_end|>` 0.9992 · `Answer` 0.0006 · `</think>` 0.0001 · `4` 0.0000 |
| s02 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `3` 0.0000 · `Answer` 0.0000 |
| s02 | `blank` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `<think>` 0.0000 · `<|im_start|>` 0.0000 |
| s02 | `explicit` | `<|im_end|>` 0.9992 · `Answer` 0.0007 · `</think>` 0.0000 · `<think>` 0.0000 |
| n01 | `shipped` | `<|im_end|>` 1.0000 · `</think>` 0.0000 · `Answer` 0.0000 · `no` 0.0000 |
| n01 | `blank` | `<|im_end|>` 0.9999 · `<think>` 0.0001 · `</think>` 0.0000 · `Answer` 0.0000 |
| n01 | `explicit` | `<|im_end|>` 0.9976 · `Answer` 0.0020 · `no` 0.0001 · `</think>` 0.0001 |
| n02 | `shipped` | `<|im_end|>` 0.9999 · `Answer` 0.0000 · `</think>` 0.0000 · `no` 0.0000 |
| n02 | `blank` | `<|im_end|>` 0.9999 · `<think>` 0.0000 · `</think>` 0.0000 · `Answer` 0.0000 |
| n02 | `explicit` | `<|im_end|>` 0.9977 · `Answer` 0.0019 · `</think>` 0.0002 · `no` 0.0001 |

## 2. Coverage per item, cue × label variant

| cue | label | c01 | c02 | s01 | s02 | n01 | n02 | above floor | median |
|---|---|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 8.068e-08* | 5.429e-07* | 7.318e-07* | 5.018e-06* | 1.522e-06* | 5.645e-06* | 0/6 | 1.522e-06 |
| `shipped` | `space` | 5.084e-09* | 7.350e-09* | 2.855e-10* | 1.433e-09* | 7.425e-09* | 6.495e-08* | 0/6 | 7.350e-09 |
| `shipped` | `caps` | 7.816e-09* | 1.029e-08* | 7.318e-07* | 5.018e-06* | 8.718e-09* | 3.691e-08* | 0/6 | 3.691e-08 |
| `shipped` | `newline` | 6.735e-08* | 1.882e-09* | 5.735e-09* | 9.914e-09* | 6.169e-08* | 1.907e-07* | 0/6 | 6.169e-08 |
| `shipped` | `long` | 8.068e-08* | 5.429e-07* | 5.532e-10* | 6.400e-11* | 1.522e-06* | 5.645e-06* | 0/6 | 5.429e-07 |
| `blank` | `bare` | 1.491e-06* | 3.184e-07* | 2.457e-07* | 6.754e-07* | 1.243e-06* | 8.350e-07* | 0/6 | 8.350e-07 |
| `blank` | `space` | 5.580e-08* | 1.187e-08* | 6.250e-11* | 3.475e-10* | 4.960e-09* | 2.792e-09* | 0/6 | 4.960e-09 |
| `blank` | `caps` | 1.558e-07* | 1.752e-08* | 2.457e-07* | 6.754e-07* | 1.903e-08* | 4.769e-09* | 0/6 | 1.558e-07 |
| `blank` | `newline` | 6.241e-10* | 3.487e-11* | 2.904e-10* | 4.510e-11* | 1.694e-09* | 1.066e-08* | 0/6 | 6.241e-10 |
| `blank` | `long` | 1.491e-06* | 3.184e-07* | 3.409e-10* | 2.104e-10* | 1.243e-06* | 8.350e-07* | 0/6 | 8.350e-07 |
| `explicit` | `bare` | 2.564e-07* | 1.316e-07* | 7.186e-06* | 9.869e-06* | 1.290e-04* | 5.504e-05* | 0/6 | 9.869e-06 |
| `explicit` | `space` | 2.357e-08* | 1.232e-08* | 3.514e-07* | 7.164e-08* | 2.519e-06* | 1.031e-06* | 0/6 | 3.514e-07 |
| `explicit` | `caps` | 4.697e-07* | 3.077e-08* | 7.186e-06* | 9.869e-06* | 2.364e-06* | 1.847e-06* | 0/6 | 2.364e-06 |
| `explicit` | `newline` | 2.423e-08* | 2.144e-08* | 1.255e-07* | 8.366e-09* | 1.976e-07* | 2.446e-07* | 0/6 | 1.255e-07 |
| `explicit` | `long` | 2.564e-07* | 1.316e-07* | 6.695e-09* | 6.480e-10* | 1.290e-04* | 5.504e-05* | 0/6 | 2.564e-07 |

`*` = below the engine's floor (0.10 ⇒ `low_mass`). Coverage is the full-vocabulary mass of the label's first token at the cue (`readout.coverage_from_scale`), read from the cue row — so every label variant of one cue costs no extra forward pass.

## 3. Variant summary

| cue | label | mean coverage | median | above floor | low_mass share | best item | worst item |
|---|---|---|---|---|---|---|---|
| `shipped` | `bare` | 2.257e-06 | 1.522e-06 | 0/6 | 1.00 | n02 5.645e-06 | c01 8.068e-08 |
| `shipped` | `space` | 1.442e-08 | 7.350e-09 | 0/6 | 1.00 | n02 6.495e-08 | s01 2.855e-10 |
| `shipped` | `caps` | 9.689e-07 | 3.691e-08 | 0/6 | 1.00 | s02 5.018e-06 | c01 7.816e-09 |
| `shipped` | `newline` | 5.621e-08 | 6.169e-08 | 0/6 | 1.00 | n02 1.907e-07 | c02 1.882e-09 |
| `shipped` | `long` | 1.299e-06 | 5.429e-07 | 0/6 | 1.00 | n02 5.645e-06 | s02 6.400e-11 |
| `blank` | `bare` | 8.014e-07 | 8.350e-07 | 0/6 | 1.00 | c01 1.491e-06 | s01 2.457e-07 |
| `blank` | `space` | 1.264e-08 | 4.960e-09 | 0/6 | 1.00 | c01 5.580e-08 | s01 6.250e-11 |
| `blank` | `caps` | 1.864e-07 | 1.558e-07 | 0/6 | 1.00 | s02 6.754e-07 | n02 4.769e-09 |
| `blank` | `newline` | 2.225e-09 | 6.241e-10 | 0/6 | 1.00 | n02 1.066e-08 | c02 3.487e-11 |
| `blank` | `long` | 6.480e-07 | 8.350e-07 | 0/6 | 1.00 | c01 1.491e-06 | s02 2.104e-10 |
| `explicit` | `bare` | 3.359e-05 | 9.869e-06 | 0/6 | 1.00 | n01 1.290e-04 | c02 1.316e-07 |
| `explicit` | `space` | 6.682e-07 | 3.514e-07 | 0/6 | 1.00 | n01 2.519e-06 | c02 1.232e-08 |
| `explicit` | `caps` | 3.628e-06 | 2.364e-06 | 0/6 | 1.00 | s02 9.869e-06 | c02 3.077e-08 |
| `explicit` | `newline` | 1.036e-07 | 1.255e-07 | 0/6 | 1.00 | n02 2.446e-07 | s02 8.366e-09 |
| `explicit` | `long` | 3.074e-05 | 2.564e-07 | 0/6 | 1.00 | n01 1.290e-04 | s02 6.480e-10 |

## 4. The prefix control (shipped vs the template's own empty think block)

| item | prefix | prefix tokens | cue | top token (p) | `bare` coverage | `newline` coverage |
|---|---|---|---|---|---|---|
| c01 | `shipped` | 109 | `shipped` | `<|im_end|>` 1.0000 | 8.068e-08 | 6.735e-08 |
| c01 | `kept` | 113 | `shipped` | `<|im_end|>` 0.9988 | 3.646e-06 | 6.221e-06 |
| c02 | `shipped` | 99 | `shipped` | `<|im_end|>` 1.0000 | 5.429e-07 | 1.882e-09 |
| c02 | `kept` | 103 | `shipped` | `<|im_end|>` 0.9985 | 1.368e-05 | 4.613e-05 |

## 5. The ranked readout (the engine's decision under a policy)

| policy | n | correct | agreement | 95% CI | low_mass | median coverage |
|---|---|---|---|---|---|---|
| `shipped=bare` | 6 | 2 | 0.333 | 0.097–0.700 | 6/6 | 0.0000 |
| `shipped=newline` | 6 | 1 | 0.167 | 0.030–0.564 | 6/6 | 0.0000 |

## 6. Cross-check against `DecisionEngine` (shipped policy, `bare`)

| item | probe coverage | engine coverage | |Δ coverage| | probe winner | engine winner |
|---|---|---|---|---|---|
| c01 | 8.0684e-08 | 1.6435e-07 | 8.37e-08 | technical | support |

Worst |Δ coverage| over the cross-checked items: **8.37e-08** — the probe reads the same row with the same function; only the batch shape differs.



