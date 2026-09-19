bash: fork: retry: Resource temporarily unavailable
bash: fork: retry: Resource temporarily unavailable
# Templates: how ggufone renders a prompt (E1c)

This document is the reference for `engine/template.py` (A-E1c-1/2/3/9). It answers four
questions: which template wins, what we can render, how thinking is suppressed, and what each
supported family expects at the decision position.

Everything below was executed on this box; the numbers/snippets marked **[executed]** are
reproducible with the commands in §8. Anything not measurable here is tagged **[UNVERIFIED]** or
**[recon]** exactly as `SPEC.md` §"Honest tags" requires.

---

## 1. The resolution chain (A-E1c-1)

`CHAIN` in `engine/template.py` is the machine-readable copy of this list, and
`CHAIN_LABELS` the human one. The order is **not** a preference — it is the order in which
sources are *consulted*:

| # | step | renderer | when it is used | warning |
|---|---|---|---|---|
| 1 | `gguf-renderer` | the internal Jinja-subset renderer on the GGUF's `tokenizer.chat_template` | whenever the template parses (the normal case for the pinned models) | — |
| 2 | `builtin` | `llama_chat_apply_template` (the bundle's built-in family table) | step 1 rejected the template **and** the runtime recognises the family | `W_TEMPLATE_FALLBACK` |
| 3 | `user` | `--template plain \| <builtin-name> \| <path> \| <inline text>` | nothing above resolved, or the operator forced a choice | — |
| 4 | `error` | — | none of the above | `E_TEMPLATE_UNRESOLVED` (exit 3) |

Rules that follow from the order:

* **Step 1 is the only step that can honour `enable_thinking`-style keyword arguments.** The C
  API of `llama_chat_apply_template` (b11026) has no kwargs, so the fallback renders whatever the
  built-in family emits, and §3's *strip* guarantee is what keeps the prompt thinking-free.
* **An explicit override short-circuits steps 1–2.** `--template chatml` means "use chatml", not
  "use chatml if the model's template fails"; the operator asked for it by name. Automatic
  resolution (no `--template`) never consults step 3 before steps 1–2.
* **`--template plain`** is the documented escape hatch to E1b's model-agnostic framing
  (`engine/prompt.py`): no chat template at all. It exists so that a model with a broken or
  unknown template can still be used without patching ggufone.
* **Step 4 always carries the fix.** The message names the unsupported construct *and* the flag
  that fixes it, e.g. for `{%- include … %}`:

  ```
  E_TEMPLATE_UNRESOLVED: no template could render this prompt. The GGUF tokenizer.chat_template
  uses unsupported template construct 'include' at line 1 ({%- include 'x.jinja' %}); libllama's
  built-in templates (llama_chat_apply_template) do not match it either. Fix: pass
  --template <builtin-name|path-to-jinja|plain> (options.template in a request), or use a model
  whose template is inside the supported subset (40 filters, if/for/set/macro, `is` tests).
  ```

### Where each step is observable

`engine.template` in every native response reports `{kind, renderer, source, family, thinking,
warnings, notes, explicit}` — so a caller never has to guess which path ran **[executed]**:

```json
{"kind": "gguf-renderer", "renderer": "internal", "source": "gguf:tokenizer.chat_template",
 "family": "spark2_5", "thinking": "suppressed", "warnings": [], "notes": [], "explicit": false}
```

---

## 2. What the internal renderer supports

A deliberately small, documented subset of Jinja — enough for the templates the families ship,
with **no silent degradation**: an unsupported construct raises `UnsupportedTemplate`, which is
what moves the chain to step 2/3.

Supported: `{{ expr }}` output, `{# comment #}`, whitespace control (`{%-`, `-%}`, `{{-`, `-}}`),
`if`/`elif`/`else`, `for … in … [if cond]`, `set` (plain, dotted, namespace), `macro`/`endmacro`
(including default arguments), attribute access, indexing and slicing (`messages[::-1]`),
calls, `+|~|-|*|/|//|%` operators, comparisons, `and`/`or`/`not`, `in`, `is` tests
(`defined`, `undefined`, `none`, `string`, `mapping`, `iterable`, `number`, `sequence`,
`boolean`, `true`, `false`, `even`, `odd`, `callable`), the `loop` object
(`index/index0/revindex/first/last/length/previtem/nextitem`), `namespace()`,
`raise_exception()`, a 40-entry filter table (`default`, `trim`, `tojson`, `string`, `length`,
`split`, `rstrip`, `lstrip`, `replace`, `first`, `last`, `join`, `items`, `keys`, `values`,
`sort`, `reverse`, `int`, `float`, `round`, `abs`, `upper`, `lower`, `list`, `count`, `safe`),
inline conditionals (`{{ 'a' if x else 'b' }}`) and list literals.

Rejected on purpose (each names itself in the error): `include`, `import`, `extends`, `block`,
`filter`-blocks, `range()`/other Jinja globals we do not implement, and any filter or test
outside the tables above.

**Live proof that the subset covers the pinned models** **[executed]**: both the `spark2_5`
(4 556 chars) and the `qwen35` (7 816 chars) `tokenizer.chat_template` parse and render through
step 1 — `tests/test_templates.py::test_a_real_gguf_template_renders_through_chain_step_one`.

---

## 3. Thinking suppression (A-E1c-2)

The predicate is `no_open_think(text)`: after the last rendered byte the model is **not** inside a
thinking block (`<think>`/`<|think|>` opened but not closed, ending at the tail). It is asserted on
the rendered string *and* on the model's own detokenized ids, so "provably" is a measurement, not
an assumption.

Three modes, recorded in `engine.template.thinking`:

| mode | mechanism | families |
|---|---|---|
| `suppressed` | the template honours `enable_thinking=false` and closes the block itself | `spark2_5`, `qwen35`, `qwen35moe` |
| `marker` | the template has no switch: the documented soft marker is appended to the last user turn | `k2-horizon` (see §4 caveats) |
| `stripped` | the template ignores the switch and ends with an open `<think>`: the trailing opener is removed | any family, as a last resort |

The engine renders with `enable_thinking=false` **by default**; `--thinking` / `options.thinking`
turns the block back on (`thinking: "on"` in the response) and is what the tests use to prove the
switch is real rather than a no-op.

Two extra guarantees the tests pin:

* the **empty closed block** (`<think>\n\n</think>\n\n`) that Qwen-family templates emit for
  `enable_thinking=false` is stripped when the family policy says so
  (`strip_empty_think_block`), leaving a generation prompt with no think-opener at all;
* the suppression is applied to the **bytes that are tokenized** — the same function builds the
  prefix and the question suffix, so a fork can never see a different prompt than the prefill.

### A-E1c-3: the readout is unaffected by anything after the cue

The engine decodes no generated token. Every branch's readout row is the position of the *cue*
(the question suffix's last token); a model that would start reasoning right after the cue
contributes logits at positions the engine never reads. Two tests hold this down:

* offline — a fake session whose row function puts a huge logit on a degenerate continuation
  token changes `coverage` (the full-vocab diagnostic) but not one bit of
  `choice`/`probabilities`/`confidence`/`score`/`noul`;
* live — the candidate label is read from the row at the cue, and the response's `decode_steps`
  equals the candidate tokens actually decoded (`tests/test_engine_fork.py`,
  `tests/test_fit_live.py::test_the_live_answer_carries_the_template_and_the_fit_plan`).

---

## 4. The family table

`FAMILIES` in `engine/template.py` is the code copy; this table is the documentation A-E1c-9
requires. "Template source" is where the string comes from, "thinking" the policy row,
"label policy" what the decision position expects.

| family (arch) | template source | thinking | label policy | caveats |
|---|---|---|---|---|
| `spark2_5` | GGUF `tokenizer.chat_template` (4 556 chars, `<｜start▁of▁sentence｜>` + `<\|User\|>`/`<\|Bot\|>` roles) **[executed]** | hard: `enable_thinking=false` renders `<\|Bot\|></think>` — a *closed* block | option name / level number as plain text right after the assistant header | the special tokens are single tokens in this vocabulary; no soft marker exists (there is no `/no_think` convention here) |
| `qwen35` | GGUF `tokenizer.chat_template` (7 816 chars, `<\|im_start\|>`/`<\|im_end\|>` + `enable_thinking`) **[executed]** | hard: `enable_thinking=false` renders `<\|im_start\|>assistant\n<think>\n\n</think>\n\n`; ggufone strips the empty block | option name / level number after the assistant header | hybrid SSM+attention: recurrent state makes `seq_cp` the interesting case (`n_rs_seq=0`, §6); the template also parses historical `</think>` spans — we never send assistant turns, so that path is inert |
| `qwen35moe` | same Qwen3.5 template family, **measured here** on `Accio-Lab_occamy-1.0` (24 GB Q4_K_L, 48 experts) and `Tiel-Coder-35B-A3B` **[executed]**: chain step 1, thinking suppressed (the empty block is stripped, so the cue is `<\|im_start\|>assistant\n`) | hard (inherited) | option name / level number after the assistant header — **but the cue is refused**: `<\|im_end\|>` holds 0.99998 of the cue row's mass on the shipped prompt shape, so the label never gets a chance (`W_CUE_REFUSED`, §4.2) | the official Qwen3.5 templates are shared across flavors and differ only in the `enable_thinking` default (`qwen3_5_think_training.jinja` for larger models, `…_nothink…` for ≤2B); the expert layout changes the **fit plan**, not the prompt |
| `k2-horizon` | Kimi-K2 lineage: `<\|im_system\|>…<\|im_middle\|>` roles, no `enable_thinking` in the template (**[recon]**, from the published `moonshotai/Kimi-K2-Thinking` `chat_template.jinja`) | soft: no template switch | option name / level number after `<\|im_assistant\|>assistant<\|im_middle\|>` | **thinking is controlled by the serving stack, not the prompt** (`thinking.type` on Moonshot's API — this was *the* design of K2-Thinking). ggufone therefore appends the documented `/no_think` soft marker *and* keeps the strip guarantee; treat the marker as advisory for this family. The llama.cpp bundle also ships a `kimi-k2` built-in, so chain step 2 covers the shape if our renderer ever rejects a variant |

Notes that apply to every row:

* the **question id is never sent to the model** — labels are the option names / level numbers /
  `yes`·`no` the response reports back (SPEC §2.6);
* `readout: "single_token"` scores only the first token of the label while the *prompt stays
  byte-identical* (`sequence` vs `single_token` is a readout choice, never a prompt choice);
* candidate labels that tokenize to the same sequence are `E_CANDIDATE_COLLISION` (exit 2)
  regardless of family.

### Choosing a template by hand

```
ggufone run … --template plain                 # E1b framing, no chat template
ggufone run … --template chatml                # a built-in name (needs a runtime)
ggufone run … --template /path/family.jinja    # a Jinja file in the supported subset
ggufone run … --template '{{ messages[0].content }}'   # inline template text
```

`--template` values are validated at request time: an unknown name that is not a file and not
template text is `E_TEMPLATE_UNRESOLVED` with the accepted forms listed.

### The label policy, measured (and why it is conservative)

The policy above is the *bare* label — the exact option name / level number / `yes`·`no` the
response reports back — asked for by the cue. That is a choice, and it was measured on the pinned
model before being kept **[executed]** (probe: state = the payment incident, 4 options,
`threads=4`, CPU):

| cue ending | top token at the cue | candidate mass | `billing` first-token p | restricted softmax |
|---|---|---|---|---|
| `Answer with exactly one candidate name:` + `\n` (shipped) | `\n` (logit 0.15) | 0.0114 | 0.0026 | b 0.22 / t 0.06 / s 0.05 / su 0.67 |
| … + `\n\n` | `Answer` (−6.97) | **0.0340** | **0.0324** | b 0.95 / t 0.03 / s 0.02 / su 0.00 |
| cue, no newline | `\n` (−2.35) | 0.00001 | 6e-6 | b 0.60 / t 0.15 / s 0.24 / su 0.01 |
| no cue at all | `\n` (0.07) | 0.000000 | 0.0 | b 0.22 / t 0.42 / s 0.32 / su 0.04 |

Read carefully: this model puts ~94 % of the next-token mass on a *newline* after any instruction
line, so **the label mass at the cue is small (1–3 %)** — which is exactly what `coverage` reports
as `low_mass` in the E2E runs. The blank-line variant roughly triples the label mass and sharpens
the restricted distribution, but it also moves the tip of every prompt and therefore every pinned
E2E number. E1c therefore **records** the measurement instead of silently changing the readout:
choosing between the bare cue, the blank-line cue and a two-step readout (score the label after
the model's own newline) is a *quality* decision, and E2 owns the labeled dev set that can measure
which one agrees better with a human. `docs/evidence/e1c_t_c8e36cad_resolver_fit.md` §3 keeps the
numbers.

### Cue shapes that put the readout mid-answer

That open question has since been closed on both sides — the *label* half by E3b (`t_6952f0dd`): on
`Accio-Lab/occamy-1.0` the cue row is `<|im_end|>` at p = 0.9976…1.00000, so no rendering of the
candidate name clears the 0.10 floor (best 4.72e-03, 0/6 items on every one of the 15 cue × label
combinations) — and the *shape* half by E3c (card `t_6c119626`). The reason is not the label:
**the cue row itself closes the assistant turn**, and the engine now says so instead of reporting
`low_mass` **[executed]**:

* `engine/cue.py` classifies the row the coverage is read from through the **session's own
  tokenizer**: it is a refusal when the row's argmax is a turn-closer that this vocabulary encodes
  as exactly **one token** (`<|im_end|>`, `</s>`, `<|endoftext|>`, `eos`). A closer the tokenizer
  splits (`<|eot_id|>` → `eot`+`id`) is never reported as one.
* `decide.py` raises **`W_CUE_REFUSED`** and publishes what a flat warnings list cannot carry —
  `{"refused": true, "closer": "<|im_end|>", "mass": 0.999984, "hint": …}` under the answer's
  `cue` key — and `harness.render_report` draws a **cue verdicts** table for the quality suites, so
  a `low_mass` row names the closer, its mass and this section instead of standing anonymous.
* the verdict is **diagnostic**: E3c does not re-route the readout. Picking a different cue shape is
  a quality decision with the same owner as the label policy above (the labeled dev set).

Measured per shape (probe `tools/e3c_cue_shapes.py`, the 6 dev items E3b used, all five label
renderings read off the same rows, engine floor 0.10, `--threads 4`). The **4B control** first —
the model E2 measured answering at the cue:

| shape | the readout sits | items above floor | ranked `bare` readout |
|---|---|---|---|
| `shipped` (the control) | at the cue | 0/6 | 3/6 correct (0.500), 6/6 `low_mass` |
| `answer_is` | at the cue, after `The answer is ` | 2/6 | 4/6 correct (0.667) |
| `answer_colon` | at the cue, after `Answer: ` | 2/6 | — |
| `wybieram_pl` | at the cue, after `Zgodnie z opisem, wybieram: ` | 2/6 | — |
| `json_field` | at the cue, inside an opened field (`{"choice": "`) | **6/6** | — |
| `two_step_shipped` | after the model's own first content token | **6/6** | 5/6 correct (0.833), 0/6 `low_mass` |
| `two_step_answer_is` | after `The answer is ` + one model token | 4/6 | — |

So the shape *is* the lever on a model that answers: the two-step readout (score the label after the
model's own first token — the generalized `newline`) takes the same 6 items from 0/6 above the floor
and 3/6 correct to 6/6 and 5/6, and opening the JSON field does it in one shot. The plain openers
only triple the mass without clearing the floor.

Occamy — the family that refuses — is a different story, and a **clean negative**: all five
at-the-cue shapes (the forced openers in both languages *and* the opened JSON field) are refused
**30/30** shape×item cells, with `<|im_end|>` the argmax at p = 0.56…1.00. The two-step shapes do
get past the closer (only 2/12 of their cells are refused) but land on whitespace: the model's
first *content* token is `</think>` (the closer E1c's empty-block strip removed from the prefix) and
the row after it is a newline at p = 0.54…1.00 — 0/6 items above the floor. One cell in the whole
sweep cleared the floor (`wybieram_pl`, 1/6) and its row was refused too. Tables:
`docs/evidence/e3c_cue_shapes_occamy.md`; both runs' raw records are kept next to it.

---

## 5. The fit plan in one paragraph (A-E1c-4/5/6)

`ggufone fit [<model>]` returns `{n_gpu_layers, n_ctx, kv_type, n_seq_max, est_weights_bytes,
est_kv_bytes, est_total_bytes, backend, source}`. `source` is `llama-fit-params` when the bundle's
tool ran (SPEC 2.10 flags: `--fit on --fit-target MiB --fit-ctx N --fit-print on`, plus the plan's
`-c`/`-ngl`/`-b`), else `estimate` + `W_FIT_ESTIMATED`. The plan is cached per
`(model sha256, host fingerprint)` under `$GGUFONE_HOME/fit/` and applied on load
(`n_gpu_layers`, `kv_type`, the `n_ctx` ceiling) unless `--no-fit`.

Two accounting facts worth knowing before reading a plan:

* **KV is a unified cache.** `kv_unified=True` (mandatory for the fork engine) means the KV
  memory is `n_ctx` cells total — *not* `n_ctx × n_seq_max`. Sequence count is a concurrency
  bound, and the fit plan's `n_seq_max` is a floor the request can only raise.
* **Two per-element numbers exist on purpose.** The E1a conservative planner
  (`registry/recommend.py`, SPEC 2.4, mirrored by the oracle) charges 1 byte/element for both
  `q8_0` and `q4_0` — an upper bound. The fit plan uses the real ggml ratios (f16 2, q8_0 34/32,
  q4_0 18/32) because A-E1c-6 cross-checks it against measured RSS. On this box **[executed]**:
  plan `est_total = 4685 MiB` vs measured load RSS `4251–4795 MiB` (**−9.3 % … +2.3 %**, inside
  the ±20 % target), `tests/test_fit_live.py::test_the_fit_estimate_cross_checks_against_measured_load_rss`.

Over budget, the ladder is fixed: `kv_type` moves `f16 → q8_0 → q4_0` first (each step emits
`W_KV_TYPE_DOWNGRADE`), *then* `n_ctx` shrinks towards `--fit-ctx`; a plan whose **weights** alone
exceed the budget is reported as `insufficient` rather than silently truncated.

---

## 6. Hybrid models and `n_rs_seq` (A-E1c-7)

For the hybrid families (`qwen35`, `qwen35moe`) the fork has to move the **recurrent state**
along with the KV cells. `llama_memory_seq_cp` does both, which is why:

* `kv_unified = true` is mandatory (PoC pitfall 2 — without it the cross-stream copy trips
  `GGML_ASSERT(is_full)`);
* ggufone leaves `llama_context_params.n_rs_seq` at **0** (the llama.cpp default = one recurrent
  state per sequence, allocated on demand). The PoC and E1b/E1c runs use 0, and the
  fork-equivalence gate (A-E1b-2/A-E1c-7) is measured with it: `max |Δ| = 0.000e+00` on both
  `qwen35` and `spark2_5` (the latter is the pinned default model).

---

## 7. Reproducing the evidence

```bash
export GGUFONE_RUNTIME_DIR=$HOME/.hermes/runtime/b11026-linux-x64-cpu   # the pinned bundle

# chain + suppression on the real templates (no network):
uv run pytest -q --run-network tests/test_templates.py -s

# the plan, the binary's table and the RSS cross-check:
uv run pytest -q --run-network tests/test_fit_live.py -s

# by hand:
uv run ggufone fit $HOME/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --json
uv run ggufone fit --help

# every E1c test with the network disabled (A-E1c-10): see tools/e1c_offline_gate.py
uv run python tools/e1c_offline_gate.py
```

`docs/evidence/e1c_t_*.md` records the raw outputs (gate table + receipts + the E2E run).

---

## 8. Known limits (all of them)

1. The internal renderer is a **subset**. A template outside it costs one fallback step
   (`W_TEMPLATE_FALLBACK`) or an explicit `--template`.
2. `llama_chat_apply_template` **ignores kwargs**: on the built-in path, `enable_thinking` cannot
   be passed, so suppression there relies on the strip guarantee.
3. The `k2-horizon` marker is **advisory** — that family's thinking is controlled by the serving
   stack, and ggufone only guarantees the prompt-level predicate.
4. `qwen35moe`'s **template is measured here** (Occamy 1.0, Tiel-Coder), but its **cue is refused**
   on the shipped prompt shape: `<|im_end|>` holds 0.99998 of the cue row's mass, so every answer
   comes back `low_mass` with `W_CUE_REFUSED` no matter how the label is rendered (§4). The cue
   shapes that move the readout past it are measured in §4 and in
   `docs/evidence/e3c_cue_shapes.md`.
5. The fit plan's `est_*` numbers describe **this host at plan time**; a different container
   limit or a busy GPU invalidates them — hence the host fingerprint in the cache key.
6. `_empty closed_ think blocks are stripped for the Qwen families by policy. If a future
   variant depends on the literal empty block, `strip_empty_think_block` in the family policy is
   the one place to change (and the live tests in §7 will notice).
