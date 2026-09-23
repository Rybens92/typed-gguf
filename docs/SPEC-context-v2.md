# SPEC — context sizing v2: standard 32k, grow into the room, shrink gracefully

Status: **APPROVED — ratified by the owner on 2026-09-23**. §11 is resolved (decisions recorded
there) and this document is the implementation contract for the v2 policy. Implementation card:
`t_ca1d4231` (typed-gguf, kanban board of this box).
Card: `t_3d521f79` (typed-gguf, kanban board of this box).
Date: 2026-09-23. Ratified: 2026-09-23. Author: code-spec. Host: `NVIDIA GeForce RTX 3060 Ti 8192 MiB`, 31 GiB RAM,
pinned runtime `llama.cpp b11026` (linux-x64-vulkan) at
`/home/rybens/.local/share/typed-gguf/runtime/b11026-linux-x64-vulkan`.
Probe evidence: `/tmp/t_context_v2/` (index in Appendix A; every number below is quoted from a
file there or from a source line cited as `path:line`).

Why a separate document and not a SPEC.md section: SPEC.md is the frozen v0.1.1 contract whose §6
holds **[executed]** reference values (including `recommend_quant` numbers this policy would
re-pin). A v2 policy that amends §2.2/§2.10 deserves its own ratifiable document; when approved,
SPEC.md gets one pointer line, not a rewrite. This card writes **only** this file.

---

## 0. Decision summary (for the owner)

The owner's policy: *standard 32k context; when the box has room, load with more; when 32k does
not fit, shrink — KV ladder first, then context, with a warning*.

What the probes say (all measured on this box, 2026-09-23):

1. **The binary route is closed.** llama.cpp's own `--fit`/`--fit-ctx` search exists only in the
   CLI tools' argument handling, not in `libllama` (no fit field in `llama_context_params`,
   `src/typed_gguf/runtime/ctypes_binding.py:90-127`), so our ctypes loader cannot delegate to it.
   Probing it anyway (probe C2/C2b) shows it picks **37 376** ctx for the 4B at f16 with
   `--fit-ctx 32768`, and **37 632** with the default floor 4096 — i.e. the binary's own honest
   answer for this box.
2. **`-c 0` means "the model's own window" = 1 048 576 on this model** (`spark2_5.context_length`).
   Leaving ctx unset would ask for a 36 GiB KV cache; the binary's fit ran, projected 42 234 MiB
   against 6 721 MiB free, and only then shrank. So "delegate by leaving it unset" is a
   non-starter at the loader level.
3. **Our KV math is 4× wrong for this arch — it ignores sliding-window attention.** The 4B declares
   `spark2_5.attention.sliding_window = 512` with a 3:1 SWA pattern (27 SWA layers, 9 global).
   llama.cpp's real iswa cache holds `n_ctx` cells for the 9 global layers and
   `window + n_ubatch` cells for the 27 SWA layers. Measured: 252 MiB at 4 096 (our formula says
   576), 1 260 MiB at 32 768 (ours says 4 608), 354.4 MiB at 32 768 at q4_0 (ours says 1 296).
   Exact accounting in §3; it is reproduced by every load probe.
4. **Consequence for the policy on this box:** with the SWA-corrected math and the existing
   512 MiB reserve, `f16` holds ~27 000 ctx and `q8_0` ~53 000 — so the v2 default plan is
   **53 511 @ q8_0 at free 6 760 MiB (52 601 at free 6 743; the number tracks live free memory)**
   — the 32k standard met and grown ~1.6×; measured loadable: 5 332 MiB self, 1 409 MiB free after.
   Today's conservative math answers **≈26 700 @ q4_0** for the same request (26 719 at free
   6 760 MiB; 26 290 at free 6 714 in probe P1) — the standard misses and the KV is two rungs down.
   That delta is what this spec buys.
5. **Shrink-below-32k is a first-class warning, not an error** (`W_CTX_BELOW_STANDARD`, §5.4), and
   a request that cannot fit the loaded context still fails loudly (`E_CTX_TOO_SMALL`, kept).

Mechanism chosen: **(A) compute the v2 target in our code**, on the existing ladder machinery,
with the KV accounting corrected to what the runtime actually allocates (§4). Two owner decisions
remain (§11): the overhead-reserve policy (it is what puts this box on `q8_0` instead of `f16` at
32k) and whether `recommend-quant` follows the new standard in this card.

---

## 1. Goal

Replace the fixed 4 096-token context target with a policy: a plan aims at **32 768 tokens** by
default, **grows** up to `min(model window, largest ctx the box holds at the chosen KV)` when there
is room, and **shrinks gracefully** — KV ladder first, then context below the standard with a named
warning — when there is not. The per-request guard is unchanged: the engine never allocates more
than the plan, never truncates a prompt, and says `E_CTX_TOO_SMALL` with token counts when the
loaded context cannot hold a request.

Outcome target: on the owner's current box and model, a plain `typed-gguf ask` loads with a
context ≥ 32 768 that the *measured* runtime behaviour justifies, and `typed-gguf fit --json`
explains the number (`standard_n_ctx`, `ctx_limit`, warnings, notes).

---

## 2. Verified starting point (code citations)

| fact | where |
| --- | --- |
| `DEFAULT_N_CTX = 4096`, `DEFAULT_N_SEQ_MAX = 8`, `DEFAULT_FIT_TARGET_MB = 1024`, `MIN_CTX_FLOOR = 512`, `OVERHEAD_BYTES = 512 MiB` | `src/typed_gguf/runtime/fit.py:69-75` |
| KV per-token formula (all layers, SPEC 2.4): `n_layer * n_kv_head * (key_len + value_len) * type_bytes`; real ggml sizes f16=2, q8_0=34/32, q4_0=18/32 B/element | `fit.py:67-68`, `fit.py:361-364` |
| `ModelFacts` reads `context_length` into `n_ctx_train` but **nothing sizes from it** | `fit.py:136-138` |
| `estimate_plan` shrinks only: ladder f16→q8_0→q4_0, then `n_ctx` down to `floor` (`max(floor, min(requested, shrunk))`) | `fit.py:367-431` (`:373`, `:382-400`) |
| `run_llama_fit_params` passes `--fit on --fit-target MIB --fit-ctx min(min_ctx, 512) --fit-print on -c <plan> -ngl <layers>` | `fit.py:750-782` (`:769-773`) |
| `plan_for_model` chain: cache → binary table → estimate; `n_ctx: int = DEFAULT_N_CTX`; the binary probe is **seeded with `estimate_plan`'s shrunken answer** (§3.1) | `fit.py:886-920` (`:909-916`) |
| `replan_for_host` re-validates a cached plan against live free memory and only ever shrinks | `fit.py:594-633` |
| The engine caps **every** request at the plan: `n_ctx_cap = plan.n_ctx` | `src/typed_gguf/cli.py:1008-1012`, `:1034` |
| `plan_context`: `n_ctx = options.n_ctx or (prefix + max_question + CONTEXT_MARGIN=32)`, then `min(n_ctx, n_ctx_cap)` | `src/typed_gguf/engine/decide.py:249-279` (`:275-277`, `CONTEXT_MARGIN` at `:52`) |
| Loader: `params.n_ctx = int(plan.n_ctx)` (a `ContextPlan`), `params.n_batch = max(512, plan.n_ctx)`, `kv_unified=True` | `src/typed_gguf/engine/session.py:714-740` (`:718-719`) |
| Guard: `needed = prefix + max_question + context_margin`; `needed > meta.n_ctx` → `E_CTX_TOO_SMALL` naming all three numbers and "raise options.n_ctx" | `decide.py:522-545` (`:535-541`) |
| The keep key: model path+sha, `backend`, `n_ctx` (**the request's option**, not the plan's), `n_seq_max`, `kv_type`, `threads`, fit flags | `src/typed_gguf/keep/identity.py:87-128` (`:107-128`), SPEC.md §2.12 (`:461`), README §Sessions (`README.md:299`) |
| Routing aims at its own default: `DEFAULT_ROUTE_CTX = 4096` | `src/typed_gguf/calibration/routing.py:45`, `:62-67` |
| `recommend-quant` aims at its own default: `recommend.DEFAULT_N_CTX = 4096` | `src/typed_gguf/registry/recommend.py:32`, `cli.py:799` |

**Today, live, on this box** (`/tmp/t_context_v2/e0b_ask_nctx32768_defaultplan.json`):
`typed-gguf ask … --n-ctx 32768` loads with `engine.n_ctx = 4096` and `engine.fit.n_ctx = 4096` —
the plan cap wins over an explicit 32k request. A 32k request is therefore unsatisfiable today,
which is the second half of the owner's problem.

---

## 3. The finding: the KV estimate must model sliding-window attention

`spark2_5` carries `attention.sliding_window = 512` and a bool pattern
`[T,T,T,F,…]` (`ModelFacts` read on the 4B: 36 layers = **27 SWA + 9 global**). llama.cpp's
`llama_kv_cache_iswa` allocates two caches (`/tmp/t_context_v2/c1_cli_c32768_f16.log`):

```
llama_kv_cache: size = 1152,00 MiB ( 32768 cells,  9 layers, 1/1 seqs), K (f16): 576, V (f16): 576
llama_kv_cache: size =  108,00 MiB (  1024 cells, 27 layers, 1/1 seqs), K (f16):  54, V (f16):  54
```

The SWA cache size is `window + n_ubatch` cells — measured 1 024 at `-ub 512` and **768 at
`-ub 256`** (`d1_c32768_ub256.log`). The exact model, verified against four independent loads:

```
b(kv_type)          = n_kv_head * (key_len + value_len) * bytes_per_element(kv_type)
                        = 4 * 512 * {2, 34/32, 18/32} = 4096 | 2176 | 1152  B per layer per cell
kv_bytes(n_ctx, kv) = n_global * b * n_ctx
                    + n_swa    * b * (sliding_window + n_ubatch)          # n_ubatch = 512
```

| n_ctx | today (all 36 layers) f16 / q8_0 / q4_0 | real f16 / q8_0 / q4_0 | measured |
| --- | --- | --- | --- |
| 4 096 | 576.0 / 306.0 / 162.0 MiB | **252.0** / 133.9 / 70.9 MiB | 252 MiB (`c0b.log`) |
| 32 768 | 4 608.0 / 2 448.0 / 1 296.0 | **1 260.0** / 669.4 / **354.4** MiB | 1260 (`c1`), 354.38 (`d2`) |
| 65 536 | 9 216.0 / 4 896.0 / 2 592.0 | 2 412.0 / 1 281.4 / 678.4 | — |
| 131 072 | 18 432.0 / 9 792.0 / 5 184.0 | 4 716.0 / 2 505.4 / **1 326.4** | 1296+30.38 (`d3`) |

Numbers recomputed by `/tmp/t_context_v2/v2_plan_probe.py` and quoted verbatim from
`v2_plan_probe.out`; the 4 096/32 768/131 072 rows are *measured* loads, not estimates. The
"today" column is `kv_bytes_per_token` (`fit.py:361-364`) — the formula `estimate_plan` uses, and
the one the rung test in `_kv_from_budget` (`fit.py:842-844`) uses for **every** rung except the
number it merely *reports* for f16 (where it substitutes the binary's measured context,
`fit.py:845-846`; so `fit --json` at the 4 096 default already shows the honest 252 MiB —
`e0b_ask_nctx32768_defaultplan.json` has `est_kv_bytes = 264241152` while still deciding on the
576 MiB figure).

### 3.1 The conservative number does not just pick a rung — it picks the question

`plan_for_model` (`fit.py:909-916`) seeds the binary measurement with the **estimate's own answer**:

```
preliminary = estimate_plan(model, host, n_ctx=<requested or 4096>, ...)   # fit.py:910-912
plan        = run_llama_fit_params(..., n_ctx=preliminary.n_ctx, ...)      # fit.py:913-916
```

So on this box the binary is only ever asked about the ctx that the 4×-over formula could afford:
the coordinator's dry probe and our reproductions of `fit --n-ctx 32768` all answer ~25 500-26 300
tokens at q4_0 (25 556 / 26 290 measured), and the *table* that comes back is measured at that ctx.
The binary's own capability — 37 376 tokens at f16 with the same model and box (`c2` log) — is never
reached, and the plan's `_kv_from_budget` then confirms q4_0 by the same over-counting formula. Both
seams must move for v2 to work: the KV math (`estimate_plan`, `_kv_from_budget`) **and** the seed
(`plan_for_model` must probe at the policy's ctx, not at the shrunken estimate's).

P1's own note proves the seed: `memory table from b11026-linux-x64-vulkan/llama-fit-params
(model 4506 MiB, context 1035 MiB, compute 306 MiB)` — 1 035 MiB is the **f16** KV for 26 290 cells,
i.e. the binary was asked at the shrunken ctx, not at 32 768. And the same plan *reports*
`est_kv_bytes = 1 090 298 880` (1 040 MiB — our q4_0 formula) while the binary's measured q4_0 KV at
that ctx is ≈290 MiB: the over-count is reported as well as used.

**Why this is a v2 blocker, not a cleanup:** `estimate_plan` charges 147 456 B/token at f16 —
4× the real global cost — so on this box it answers "32k does not fit, q4_0 and 26 719 tokens"
(`p1_nctx32768.json`, `v2_plan_probe.out`) while the box in fact loads 32k at f16 with 1 200 MiB
still free (`c1_cli_c32768_f16.log`, post-load breakdown). A policy that says "standard 32k" and a
planner that cannot see 32k fitting is a policy that would silently never fire. Non-SWA models
keep today's formula byte-for-byte (AC-7).

---

## 4. Mechanism decision

Three candidates from the card, decided with probes:

**(B) delegate to llama.cpp `--fit on` with ctx unset — rejected.**
* `llama_context_params` has no fit field (`ctypes_binding.py:90-127`), and `libllama.so` exports no
  fit symbol (`nm -D --defined-only libllama.so | grep -i fit` → empty). The fit search lives in
  the CLI's common argument layer; a ctypes loader cannot reach it.
* `-c 0` is not "let the binary choose" at the loader level: it is "the model's own window".
  Measured: the first fit projection ran at `n_ctx = 1048576` for 42 234 MiB against 6 721 free
  (`c2_fit_ctx32768_ngl99.log`). Asking the C API for that is a guaranteed failure.

**(C) hybrid "compute ours, pass `--fit-ctx` down as a safety net" — reduced to a no-op, kept.**
`--fit-ctx` is the *floor* of the binary's own shrink search, not a target. In our measurement path
we already pass `--fit-ctx` (`fit.py:771`); with an explicit `-c` the binary cannot move ctx
anyway (probe B2: `-c 32768` → table at 32 768, no search). No safety net to add.

**(A) compute the v2 target in our code — chosen.**
Deterministic, fork-proof (our loader is the only thing that allocates), visible in `fit --json`,
already the documented chain (cache → binary table → estimate) and already re-validated against
live free memory (`replan_for_host`). The binary's table stays what it is today: the *reported*
cross-check (`source: "llama-fit-params"`, `notes`), not the ctx authority.

The binary's own answer is corroboration, not a mechanism: with `--fit-ctx 32768` it chose
**37 376** and with the default floor **37 632** at f16 (`c2*` logs). Our v2 rule (same box, same
model, `OVERHEAD_BYTES = 512`) answers 26 987 at f16 / 53 511 at q8_0 (26 504 / 52 601 on an
earlier free-VRAM reading) — deliberately more conservative than the runtime, by the reserve policy
in §11-D2.

---

## 5. Exact semantics

### 5.1 Constants and fields

| name | value | meaning |
| --- | --- | --- |
| `fit.STANDARD_N_CTX` (new) | `32768` | the policy standard a plan aims at when the user pins nothing |
| `fit.DEFAULT_N_CTX` (unchanged) | `4096` | the shrink **floor** (`--fit-ctx` default) and the low-level default of `estimate_plan(...)`; its docstring must say so |
| `fit.DEFAULT_N_CTX_TARGET` | — | *not* introduced; one standard constant only |
| `W_CTX_BELOW_STANDARD` (new warning) | — | the plan had to land below `STANDARD_N_CTX`; carries the numbers in `notes` |
| `FitPlan.standard_n_ctx: int` (new field, default `0`) | — | the standard in force for this plan; `0` when none applies |
| `FitPlan.ctx_limit: str` (new field, default `""`) | — | one of `"standard"`, `"grown"`, `"shrunk"`, `"pinned"`, `"window"`, `""` (legacy/estimate) |

Both fields join `FIT_FIELDS`/`to_dict`/`from_dict` (`fit.py:62-65`, `:335-358`) and the
`typed_gguf.fit/v1` payload. `from_dict` must accept payloads **without** them (cache from an
older version reads as `0`/`""`; a cache hit then re-validates as today).

### 5.2 The target (no explicit `--n-ctx`)

```
window      = model.n_ctx_train if model.n_ctx_train > 0 else None    # None = no window cap
standard    = STANDARD_N_CTX = 32768
target      = min(standard, window) if window is not None else standard
```

If `window` is `None`, a plan may grow past 32 768 freely; if `window < standard` the target *is*
the window and `ctx_limit = "window"` (a note names the model's own ceiling; **not** a warning —
nothing degraded).

### 5.3 KV ladder, then grow (the main path)

Walk `KV_DOWNGRADE_ORDER = (f16, q8_0, q4_0)` exactly as today (`fit.py:66`, `:382-400`), with the
corrected `kv_bytes(n_ctx, kv)` from §3:

1. For each rung: `cap(kv) = min(max_fit(kv), window or ∞)` where `max_fit` is the largest `n_ctx`
   with `weights + OVERHEAD_BYTES + kv_bytes(n_ctx, kv) ≤ fit_budget(host, fit_target_mb)`.
2. The first rung with `cap ≥ target` wins the ladder. On it:
   * a **pin** (`--n-ctx N`) → `chosen_n_ctx = min(N, cap)`;
   * no pin → `chosen_n_ctx = cap` — i.e. the plan **grows** to the largest context the box holds
     at that rung (already window-capped), not merely up to the standard.
   A rung below the top carries `W_KV_TYPE_DOWNGRADE` as today.
3. `ctx_limit`: `"standard"` when `chosen == standard`, `"grown"` when `chosen > target`,
   `"window"` when the window capped it, `"pinned"` on an explicit value. A note records the
   arithmetic, e.g.
   `n_ctx 32768 -> 53511: the box holds more (fit-target 1024 MiB kept free, kv_type q8_0)`.
4. Nothing above the floor fits any rung: keep q4_0 (the last rung), shrink
   `chosen = max(floor, min(target, max_fit(q4_0)))`, warnings `W_KV_TYPE_DOWNGRADE` (if the rung
   moved) + **`W_CTX_BELOW_STANDARD`** (new), notes with the numbers and the existing fix hints
   ("raise --fit-target, lower --fit-ctx or use a smaller quant"). Growth is **not** attempted on a
   rung chosen this way.
5. The plan may not exceed the model window: `chosen ≤ window`. The floor is
   `min(floor, window)` (a 2 048-window model is planned at 2 048, never 4 096).
6. `--kv-type X` pins the rung to start the ladder at (existing `kv_start`, `fit.py:494-502`);
   `warnings` for the rung it lands on, as today.

### 5.4 Shrink semantics (first-class)

A plan below `STANDARD_N_CTX` is **valid** and loadable; it is reported, never raised:

* warning `W_CTX_BELOW_STANDARD` (machine-readable) — the plan is below the standard;
* note with `requested -> chosen`, the budget, the KV rung and the fixes;
* `ctx_limit = "shrunk"` (or `"window"` when the model itself is the reason);
* the existing `insufficient` note fires unchanged when the weights alone exceed the budget (a plan
  that will page) — v2 does not change that path;
* the load proceeds; a request that does not fit the *loaded* context still fails loudly with
  `E_CTX_TOO_SMALL` (`decide.py:535-541`), extended by one hint line (§6.3).

### 5.5 Explicit values (unchanged meaning — recommended)

| flag | after v2 |
| --- | --- |
| `--n-ctx N` on `fit`/`run`/`ask` | a **pin**: `chosen = min(N, cap)` (also never above the model window), **no grow**, `ctx_limit = "pinned"`; if it does not fit, shrink to fit with today's note, never below the floor; `--n-ctx 32768` on this box now really loads 32 768 instead of being capped at 4 096 |
| `--fit-ctx N` | unchanged: the shrink **floor** (default `DEFAULT_N_CTX = 4096`); a user who wants "never below 32k" passes `--fit-ctx 32768` — today's code already does that (probe E1: `ask --n-ctx 32768 --fit-ctx 32768` loaded 32 768) |
| `--fit-target MIB` | unchanged: the margin kept free; it is the *only* operator-controlled reserve |
| `--n-seq-max N` | unchanged; `kv_unified=True` means KV cells are `n_ctx`, not `n_ctx × n_seq_max` (`fit.py:30-36`, test-pinned) |
| `--no-fit` | unchanged: no plan, no cap, SPEC 2.2's request formula applies literally (`cli.py:1034`, `decide.py:275`) |

### 5.6 The plan is a ceiling *and* a load size

* `plan.n_ctx` stays `n_ctx_cap` for every request (`cli.py:1011`).
* **New:** when the request does not pin `options.n_ctx`, the *load* uses the plan's `n_ctx`
  (i.e. `ContextPlan.n_ctx = plan.n_ctx`), not `prefix + question + margin`. When the request pins
  a value, the load is `min(pin, plan.n_ctx)` (today's behaviour).
* The standard is a **plan default, not a request ceiling**: a request may use any amount up to the
  loaded context (the guard compares against `meta.n_ctx`).
* So the standard/grown context is what is actually allocated (the owner's "load the model with a
  bigger context"), and the guard still protects a request that needs more than the result.
* The `--no-fit` path is untouched: `n_ctx = prefix + max_question + CONTEXT_MARGIN` (SPEC 2.2
  literal).

### 5.7 Cost of a plan (unchanged except the KV term)

`est_kv_bytes = kv_bytes(chosen_n_ctx, chosen_kv)` (§3 — SWA-aware when the model declares SWA,
else the SPEC 2.4 formula), `est_weights_bytes` as today (tensor index for `estimate`; the table
sum for `llama-fit-params`), `est_total_bytes = weights + kv + OVERHEAD_BYTES`. `budget_bytes`,
`host_fingerprint`, `created_at`, `replan_for_host` semantics (shrink-only, live free memory) are
unchanged.

---

## 6. Interactions

### 6.1 SPEC.md §2.2 — what this changes, explicitly

SPEC.md line 176 documents the load as `n_ctx = prefix tokens + longest question suffix + margin
(32)`. Under v2 that line becomes conditional, and the amendment (to be written into SPEC.md when
this document is ratified) is:

> **v2 (amends §2.2).** With a fit plan applied (the default), the context is sized by the plan:
> `n_ctx = plan.n_ctx` (standard 32 768, grown when the box holds more, shrunk with
> `W_CTX_BELOW_STANDARD` when it does not), unless the request pins `options.n_ctx`, in which case
> `n_ctx = min(options.n_ctx, plan.n_ctx)`. The formula `prefix + longest question + margin` stays
> the rule for `--no-fit` and the *request-fit guard*: a request whose needs exceed the loaded
> context raises `E_CTX_TOO_SMALL` with the three token counts — never a truncation. `plan.n_ctx`
> remains the ceiling (`A-E1c-5`, SPEC.md:662).

Nothing else in §2.2 moves: `n_batch = max(512, n_ctx)`, `n_ubatch = 512`, `n_seq_max` waves,
`kv_unified = True`, `type_k/type_v` from the plan, `flash_attn` "auto" (with the engine enabling
it when the KV is quantized, `session.py:730-733`) are unchanged.

### 6.2 Session key / warm host

The keep key keeps its meaning (user intent): `n_ctx` in the key is the request's *option*
(`identity.py:119`), so a v2 load whose size comes from the plan does **not** re-key. Consequences
to accept and document:

* Two calls with identical flags share one host whose loaded `n_ctx` is whatever the plan said when
  it was spawned. If free VRAM drifts and a later plan would be smaller/bigger, the *resident* host
  is not swapped (`replan_for_host` is load-time only, shrink-only — `fit.py:594-633`) and the
  response reports the **actual** loaded size (`engine.n_ctx`, `decide.py:519`), so drift is visible,
  never hidden.
* A request that needs more than the resident host holds gets the typed `E_CTX_TOO_SMALL` — the fix
  hint must say how to get a bigger host: pin `--n-ctx <bigger>` (which *does* change the key →
  swap). No silent reload, no truncation.
* Because the load size now follows the plan, `--n-ctx` and the fit flags are the user's handles on
  the load; the key already covers both.

### 6.3 Message/test-surface changes

* `E_CTX_TOO_SMALL` text: keep the existing counts sentence; append
  `; this host is loaded with n_ctx=<meta.n_ctx> (plan …); pass --n-ctx <needed> to reload bigger`.
  (Tests pin the code and the counts, not the tail — check `tests/test_engine_fork.py:335`.)
* `fit --json` gains `standard_n_ctx` and `ctx_limit`; `fit` (human output) prints
  `n_ctx 53511 (standard 32768, grown from the box's free memory)`.

### 6.4 Routing and the quant recommender

* `calibration/routing.py:45` `DEFAULT_ROUTE_CTX = 4096` → **the shared standard**
  (`fit.STANDARD_N_CTX`), so `--route auto` aims at the same context the loader will use; otherwise
  a route plan would cap requests at 4 096 while `fit` plans 32k+. Consequence to re-pin in
  `tests/test_routing.py`: routing's conservative bound (1 B/element, × `n_seq_max`) now spends more
  on ctx, so on this box the router considers smaller quants/aliases than before. That is the
  intended semantics, not a regression.
* `registry/recommend.py:32` `DEFAULT_N_CTX = 4096` (used by `recommend-quant`, `cli.py:799`):
  **left alone in this card** (recommended) because the oracle (`docs/verify_runtime_contract.py`)
  and SPEC.md §6 carry **[executed]** `recommend_quant` values re-derived from that default
  (`SPEC.md:383-386`). Unifying it is a follow-up card that re-pins those values — see §11-D3.

---

## 7. Blast radius and migration list

Legend: **M** = migrate, **K** = keep (test input is explicit; nothing to do), **F** = frozen
evidence (do not touch), **R** = review (assert the new numbers).

### 7.1 Source

| file:line | today | migration |
| --- | --- | --- |
| `runtime/fit.py:9`, `:40` (module docstring: `"n_ctx": 4096`, "down to `--fit-ctx`, default 4096") | prose | M: describe the v2 policy (standard/grow/shrink) and the SWA KV model |
| `runtime/fit.py:70-75` constants | `DEFAULT_N_CTX = 4096` | M: add `STANDARD_N_CTX = 32768`; keep `DEFAULT_N_CTX` as the floor with a docstring |
| `runtime/fit.py:100-152` `ModelFacts` | no SWA facts | M: read `attention.sliding_window` + `attention.sliding_window_pattern` (both optional, default off); add `n_swa_layers`/`n_global_layers`/`sliding_window` |
| `runtime/fit.py:361-364` `kv_bytes_per_token` | all-layer formula | M: add a model-aware `kv_bytes(model, n_ctx, kv_type)` implementing §3; keep the old function for non-SWA models and for the SPEC 2.4 oracle path |
| `runtime/fit.py:367-431` `estimate_plan` | shrink-only, `floor` | M: `n_ctx=None` → the policy of §5.2-5.4; an int → the pin of §5.5; ladder logic and `insufficient` note unchanged |
| `runtime/fit.py:477-479` `_kv_bytes_for` | all-layer | M: delegate to the SWA-aware function (used by `degrade_ladder`) |
| `runtime/fit.py:835-849` `_kv_from_budget` | all-layer | M: same delegation (SWA models only) |
| `runtime/fit.py:62-65` `FIT_FIELDS`/`to_dict`/`from_dict` | 9 fields | M: add `standard_n_ctx`, `ctx_limit` (back-compatible reads) |
| `runtime/fit.py:886-920` `plan_for_model` | `n_ctx: int = DEFAULT_N_CTX`; **seeds the binary probe with `estimate_plan`'s shrunken answer** (§3.1) | M: default `None` = policy; probe the binary at the *policy's* ctx, not at the estimate's; `replan_for_host(..., min_ctx=…)` unchanged (shrink-only) |
| `runtime/fit.py:841-846` `_kv_from_budget` rung test | rung chosen by the all-layer formula; the binary's measured context only *reported* for f16 | M: SWA-aware formula for every rung; keep substituting `binary_context` as the reported f16 number |
| `runtime/fit.py:750-782` `run_llama_fit_params` | `-c <plan>` | K: signature keeps `n_ctx: int`; callers pass the resolved value. `--fit-ctx max(min_ctx, 512)` unchanged |
| `engine/decide.py:249-279` `plan_context` | request-sized | M: §5.6 load-sizing rule |
| `engine/decide.py:522-545` `_guard_context` | unchanged | M: only the tail hint of §6.3 |
| `engine/session.py:714-740` `_context_params` | uses `ContextPlan.n_ctx` | K (no change needed once `plan_context` returns the load size); `n_batch = max(512, n_ctx)` stays (§8.4) |
| `cli.py:1008-1012`, `:1034` | cap wiring | K (already the cap; the load size now equals it by default) |
| `cli.py:1224-1233` `_with_fit_options` | fills kv/n_seq_max | K unless the implementer prefers to fill `n_ctx` here instead of in `plan_context` (one seam, not two) |
| `calibration/routing.py:45` | `DEFAULT_ROUTE_CTX = 4096` | M: `= fit.STANDARD_N_CTX` (§6.4) |
| `registry/recommend.py:32` | `DEFAULT_N_CTX = 4096` | K in this card (D3) |
| `bench/harness.py:342-522` | explicit `n_ctx` params | K: bench rows stay reproducible from their flags |

### 7.2 Tests

| test | what it pins | migration |
| --- | --- | --- |
| `tests/test_fit_live.py:150-175` `test_the_plan_is_applied_on_load_unless_no_fit` | `fit.n_ctx < 32768` for a 32768 request; `engine.n_ctx == fit.n_ctx` | **M (live)**: v2 semantics — `engine.n_ctx == fit.n_ctx` stays true for the *unpinned* call; the pinned call asserts `engine.n_ctx == 32768`, `fit.n_ctx ≥ 32768`, `← pin wins, no grow` |
| `tests/test_fit_live.py:121-133` `test_the_estimate_alone_still_answers_the_contract` | `est_kv == 147456 × n_ctx` on the real 4B (SWA!) | **M**: assert against `fit.kv_bytes(model, n_ctx, kv)`; keep a non-SWA fixture assertion for the 147456 formula |
| `tests/test_fit.py:115-120` `test_the_kv_estimate_is_the_unified_cache_not_the_worst_case` | 147456 × 4096 | K (fixture has no SWA → formula unchanged) |
| `tests/test_fit.py:139-160` ladder tests | explicit `n_ctx=4096` | K |
| `tests/test_fit.py:124-129` determinism with default `n_ctx` | no value assertion | K (R: run and eyeball the new n_ctx in the payload) |
| `tests/test_fit.py:186-194` `test_over_budget_downgrade_is_an_e1c5_gate…` | default call + tight budget | R: `small` now plans at the window cap (32 768 for the fixture); `tight` gains `W_CTX_BELOW_STANDARD` — the existing asserts still hold |
| `tests/test_e1c_mutation_pins.py` (32 hits) | explicit `n_ctx=4096` in every formula/boundary pin | K, **except** `:416-423` (default-`n_ctx` calls: assert against the new default), `:750-775` (a GGUF fixture with `context_length=4096` → window-capped plan at 4 096 — add an explicit assertion for that, it is now a *feature*) |
| `tests/test_fit_free_vram.py:189-234` | explicit `n_ctx=4096` + `plan_from_binary` pins | R: `plan_from_binary` still reports the table's `est_*`; its `n_ctx` now comes from the caller's resolved value |
| `tests/test_fit_oom_recovery.py:137-170` | explicit `n_ctx=4096` | K |
| `tests/test_bench_placement.py:87-104` | explicit `n_ctx=4096` | K |
| `tests/test_keep.py` (13), `tests/test_keep_cli.py:227-254` | key fields with explicit `n_ctx=4096` | K (key semantics unchanged, §6.2) |
| `tests/fake_engine.py:51`, `tests/fake_keep_host.py:76` | fake `n_ctx = 4096` defaults | R: any engine test that now expects a plan-sized load must pass the size explicitly |
| `tests/test_engine_fork.py:335` | `E_CTX_TOO_SMALL` code | R: the appended hint must not break the message assertions |
| `tests/test_schema.py:220`, `tests/test_e3b_evidence.py:53`, `tests/test_calibration.py:887`, `tests/test_bench_teardown_crash.py:195`, `tests/test_gguf_header.py:114` | fixtures/other 4096s | K |
| `tests/test_routing.py:164-247`, `tests/test_policy_v2.py` | routing ctx | R/M per §6.4 (`:247` is symbolic; scenario tests must re-pin to the new router outputs) |
| `tests/test_recommend_quant.py` (8) | explicit `n_ctx` args | K (recommend-quant unchanged in this card) |

### 7.3 Docs and frozen evidence

| artifact | migration |
| --- | --- |
| `SPEC.md:176` (§2.2 context block) | M: the amendment text of §6.1 (only after ratification) |
| `SPEC.md:659-664` (A-E1c-4/5/6), `:461` (§2.12 key list), `:103` (glossary) | R: add the standard/grow/shrink sentence; the key list keeps "n_ctx" (unchanged meaning) |
| `SPEC.md:383-386`, `:289`, §6 reference values | **F**: untouched (recommend-quant values, response samples). A `recommend-quant` migration is D3 |
| `README.md:299` (Sessions/key), `:335` (`fit` row), `:357` (`fit --json` sample) | M: describe the v2 plan fields and the standard |
| `README.md:167`, `docs/BENCHMARKS.md:835-1041`, `docs/evidence/**` | **F**: measured rows at explicit `n_ctx` — untouched; add nothing |
| `docs/verify_runtime_contract.py:509-519` | **F** for the `recommend_quant` cases (explicit n_ctx); if D3 is approved, re-pin in that card |
| `docs/evidence/real-purpose-mimo-9b-2026-09-22.md` (and its `fit/*.json`) | **F**: a frozen run receipt at `--n-ctx 4096`; do not edit |
| `docs/evidence/e1c_t_c8e36cad_resolver_fit.md:39` | R: its live claim "cap wins over a 32768 request" becomes "a pin wins, the plan sizes the rest" — the receipt is historical, so add a *new* receipt in the implementation card rather than editing it |

Proposed evidence receipts for the implementation card (committed small, ~15 KB total):
`docs/evidence/context-v2/` with the four `fit --json` plans, the two llama-cli KV extracts
(`grep -E "llama_kv_cache: size|memory breakdown"`), the engine `ask` responses, and the
`v2_plan_probe` output. The full logs stay out of git (`/tmp` is wiped).

---

## 8. Costs and risks (measured)

### 8.1 KV cost per type (the table in §3)

At 32 768 the real KV is 1 260 MiB (f16) / 669 MiB (q8_0) / 354 MiB (q4_0) for this 4B on this box;
at 131 072 it is 4 716 / 2 505 / 1 326 MiB. The SWA part is a **constant** (108/57/30 MiB) that does
not grow with ctx; growth is priced by the 9 global layers only (36 864 B/token at f16).

### 8.2 What the box can hold (vs the plan's own answers)

| rung | real max (llama.cpp's own fit / loads) | our plan's max (512 MiB reserve) |
| --- | --- | --- |
| f16 | 37 376 (binary fit) / 38 000 loads with 1 015 MiB free | 26 987 |
| q8_0 | ≥ 52 736 (loaded, 1 409 MiB free) | 53 511 |
| q4_0 | ≥ 131 072 (loaded, 1 046 MiB free) | 103 807 |

(Our column moves with live free VRAM: 26 504 / 52 601 / 102 088 at free 6 743 MiB — the same
policy answer on a busier desktop.)

Our reserve is the reason the two columns differ; measured compute on this box is 103 MiB
(llama-cli, `-b 32768`/`-ub 512`) to 261 MiB (our engine at 32k with q4_0 KV + flash attention,
`Vulkan0 compute buffer size is 261.0000 MiB` in the E1 response) plus 43-75 MiB `Vulkan_Host`.
The 512 MiB reserve is therefore 2-5× the measured compute — that is D2.

### 8.3 `n_gpu_layers` with a bigger ctx

With the corrected KV, the v2 answer's own arithmetic spends the budget by construction: weights
(our tensor-index sum, 4 167 MiB) + KV (1 040 MiB at 52 601 q8_0) + overhead (512) = 5 719 MiB —
and the box really loads it (measured `v1`: 5 332 MiB self including compute buffers, 1 409 MiB
free after). The *reported* `est_total_bytes` on the binary path is larger, because it sums the
table's weights (4 506 MiB) instead of the tensor index. With the *old* math the same request lost
a layer (35/36, E1 response) because it charged 1 296 MiB of KV for a 354 MiB cache. On a busier
desktop `replan_for_host` shrinks the plan (layers first — `degrade_ladder` is unchanged) and the
v2 ctx follows down.

### 8.4 `n_batch`, `n_seq_max`, waves

`params.n_batch = max(512, plan.n_ctx)` (`session.py:719`) means a 52 601-token plan sets
`n_batch = 52 601` — measured harmless at 32k (compute buffer 103 MiB at `-b 32768` in llama-cli;
261 MiB through our engine, which also enables flash attention for the quantized KV). `n_seq_max`
stays 8 (waves unchanged) and does **not** multiply KV under `kv_unified=True`.

### 8.5 Warm-host cost

Measured with the 32k plan on this box (`W1`/`W2`, `keep status`):
cold call wall **6.29 s** (`model_load_ms 1 065`), warm call wall **0.71 s**, one host serving
2 requests, key `n_ctx=32768 fit_ctx=32768 fit_cache=false` (digest `18620e5f01e55052`). For
reference, the same box/README reports cold 17.50 s / warm 2.58 s at the 4k-era plan — the numbers
are host- and measurement-method dependent; the pair measured here is the one to re-measure in the
implementation card.

### 8.6 Risks

| risk | mitigation |
| --- | --- |
| Growth is aggressive on big-VRAM cards (the rule's only ceiling is the model window and the box's own free memory; this model's 1 M window would want ~19 GiB of q8_0 KV — or ~36 GiB at f16 — for a small request) | the `--fit-target` margin is still kept free (so the desktop keeps its headroom); `--n-ctx N` pins a ceiling; a `--ctx-max` cap is the follow-up (§12) |
| Our max-fit is *estimated*; the runtime may allocate slightly more (measured: asked 52 601 → 52 736 cells) | the fit-target margin absorbs it (1 409 MiB free after the v2 load); `replan_for_host` re-checks at load; the load-time KV ladder turns an OOM into a degradation (`session._init_context`) |
| The runtime's SWA cells formula (`window + n_ubatch`) is an observed behaviour; a future build could change it | AC-6 pins the formula against a live load; the value is only ever used with the fit-target margin as buffer |
| A plan-sized load on a small model (8k window) still costs KV for a tiny request | that is the 32k standard applied honestly; `--n-ctx` pins smaller |
| Two same-key calls can see different loaded sizes over time (drift) | documented in §6.2; the response reports the actual size |
| The reserve/overhead policy is the difference between `f16` and `q8_0` at 32k on this box | D2 — the owner decides; the warning `W_KV_TYPE_DOWNGRADE` makes the rung visible either way |

---

## 9. Acceptance criteria (testable, numbered)

Criticality: 🔴 needs careful owner review · 🟡 business logic · 🟢 display/rounding.

**AC-1 (🟡) Default target.** Given a model whose window is ≥ 32 768 and a host whose free memory
holds 32 768 at the top rung, when a plan is computed without `--n-ctx`, then `plan.n_ctx ≥ 32768`,
`plan.standard_n_ctx == 32768`, `plan.ctx_limit ∈ {"standard","grown"}`, and no
`W_CTX_BELOW_STANDARD`.

**AC-2 (🟡) Grow.** Given the same model on the measured box (4B Q8_0, SWA 512, 6 760 MiB free,
`--fit-target 1024`), when `fit --json` runs without flags, then `n_ctx == max_fit("q8_0")`
(**53 511** at free 6 760 MiB; 52 601 at free 6 743 — the number tracks live free VRAM),
`kv_type == "q8_0"`, `ctx_limit == "grown"`, `warnings == ("W_KV_TYPE_DOWNGRADE",)` (the ladder had
to leave f16 to reach the standard — the rung moved, so the warning fires; if D2 moves the reserve
and the plan lands on f16, the assertions become `n_ctx ≈ 26987`, `kv_type == "f16"`,
`warnings == ()`), and a note records `32768 -> 53511`.
*(Numbers are host-state dependent; the test asserts the relation `chosen == max_fit(kv)` and, on
the pinned fingerprint, the exact value.)*

**AC-3 (🟡) Ladder before shrink.** Given a budget that holds 32 768 at q8_0 but not at f16, when
the plan is computed, then `kv_type == "q8_0"`, `n_ctx ≥ 32768`, and `W_KV_TYPE_DOWNGRADE` is
present.

**AC-4 (🔴) Graceful shrink.** Given a budget that cannot hold 32 768 at any rung above the floor,
when the plan is computed, then `n_ctx < 32768`, `n_ctx ≥ min(floor, window)`, `kv_type == "q4_0"`,
`W_CTX_BELOW_STANDARD` and `W_KV_TYPE_DOWNGRADE` are present, a note carries `requested -> chosen`
and the budget, and the plan is still returned (no exception).

**AC-5 (🟡) Window cap.** Given `n_ctx_train == 8192`, when the plan is computed without pins, then
`n_ctx == 8192`, `ctx_limit == "window"`, and there is **no** `W_CTX_BELOW_STANDARD`; given
`n_ctx_train == 2048` and floor 4096, then `n_ctx == 2048` (never above the window, never at the
floor if the window is smaller).

**AC-6 (🔴) SWA KV accounting.** Given `ModelFacts` for the 4B (36 layers, 4 kv heads, 256/256,
window 512, 27/9 pattern) and `n_ubatch = 512`, then `kv_bytes(4096, f16) == 264241152`,
`kv_bytes(32768, f16) == 1321205760`, `kv_bytes(32768, q4_0) == 371589120`,
`kv_bytes(131072, q4_0) == 1390804992` bytes — each equal to a live load's `llama_kv_cache`
sum from the probe logs (±1 MiB for cell rounding). With `n_ubatch = 256` the SWA term uses
768 cells. (Executed arithmetic: `/tmp/t_context_v2/ac6_check.py` → all four pins `OK`.)

**AC-7 (🟡) Non-SWA models unchanged.** For a model without `attention.sliding_window`, the KV math
and every existing formula pin is byte-identical to today (e.g. the tiny 36-layer fixture still
charges 147 456 B/token at f16).

**AC-8 (🟡) Pin semantics.** `--n-ctx 8192` → `n_ctx == 8192`, `ctx_limit == "pinned"`, no grow even
when the box holds more; `--n-ctx 65536` on the measured box → `kv_type == "q4_0"`, `n_ctx == 65536`;
`--n-ctx 32768` → `n_ctx == 32768` (today it is capped to 4096 — this is the live regression to
flip).

**AC-9 (🟡) Load sizing.** With a plan at `n_ctx = P` and no pinned `options.n_ctx`, an `ask`/`run`
reports `engine.n_ctx == P` (not `prefix + question + margin`); with `options.n_ctx = 8192`, it
reports `8192`; with `--no-fit`, it reports the request formula (SPEC 2.2 literal) and no
`engine.fit`.

**AC-10 (🔴) Guard unchanged.** A request whose `prefix + longest question + margin` exceeds the
loaded context raises `E_CTX_TOO_SMALL` naming prefix, longest question, margin and the loaded
`n_ctx`, and the message names the reload handle (`--n-ctx`). No truncation happens in any case.

**AC-11 (🟡) Plan fields.** `fit --json` carries `standard_n_ctx` and `ctx_limit`; `FIT_FIELDS` +
`to_dict`/`from_dict` round-trip; a cached payload written before v2 loads as
`standard_n_ctx = 0, ctx_limit = ""` and is re-validated exactly as today.

**AC-12 (🟢) Honest reporting.** `fit` (human output) prints the chosen `n_ctx` with its standard and
whether it grew/shrank; the response's `engine.fit` block is the effective plan (already the case)
and includes the new fields.

**AC-13 (🟡) Cache and re-validation.** A cached plan is re-validated against live free memory and
may only shrink (unchanged), and the shrunken plan keeps the v2 fields consistent
(`ctx_limit` reflects the new number).

**AC-14 (🟡) Routing consistency.** `routing.DEFAULT_ROUTE_CTX is fit.STANDARD_N_CTX`; a route plan
computed for a request without `options.n_ctx` plans for ≥ 32 768 or explains why not.

**AC-15 (🟢) Docs.** README's Sessions/`fit` rows and SPEC.md's §2.2/§2.10/§2.12 statements match the
implemented semantics; the frozen receipts listed as **F** in §7.3 are byte-identical after the
change (`git diff --stat` shows no `docs/evidence/**`, no `docs/BENCHMARKS.md`).

**AC-16 (🔴) Live end-to-end at the standard.** On this box, `typed-gguf fit --json` for the 4B and
then `typed-gguf ask` (no pins) produce `engine.n_ctx == engine.fit.n_ctx ≥ 32768`, the load
completes, and the response's `engine.n_ctx` equals the loaded context read back from the runtime
(`meta.n_ctx`); a request needing ~6 000 tokens (≈25 KB of state text) is answered, where the same
call against today's 4 096 cap raises `E_CTX_TOO_SMALL`.

---

## 10. Test strategy

**Unit (offline, no model, no runtime; the bulk of the pins).** A fake `ModelFacts` factory with
and without SWA; `host_facts` with injected `device_probe`; then AC-1…AC-8, AC-11, AC-13 as plain
unit tests on `estimate_plan`/`plan_for_model`/`degrade_ladder`/`replan_for_host`. AC-6 gets a
table of the four measured KV numbers, so a regression in the SWA formula fails loudly.

**Integration (fake engine / offline CLI).** AC-9/AC-10/AC-12 through `decide.plan_context`,
`DecisionEngine._guard_context` and `cli.decide_payload` with the existing fake session; AC-14 in
`tests/test_routing.py`.

**Live (`@pytest.mark.model`, this box, opt-in gate as today).** AC-16 plus a live SWA assertion:
compute the plan, then assert the `llama_kv_cache` sizes the engine logs match
`fit.kv_bytes(plan.n_ctx, plan.kv_type)` within one cell block; re-run the probe commands in
Appendix A and diff the KV table. Update `tests/test_fit_live.py` per §7.2.

**Docs gate.** Extend the existing docs tests where they check README/SPEC wording (that they still
name the semantics the code implements); no new gate for prose beyond the checks already in
`tests/test_e1c_mutation_pins.py` (`WARNING_CODES` must gain `W_CTX_BELOW_STANDARD`).

**Out of the suite:** the frozen evidence files and `docs/BENCHMARKS.md` are read-only inputs;
a `git diff --stat` inside the docs test is the cheapest guard (AC-15).

---

## 11. Owner decisions — **Resolved 2026-09-23** (ratified with `Status: APPROVED`)

Every decision below was answered by the owner on 2026-09-23 and is binding for the implementation
card (`t_ca1d4231`). The recommended option won in all five; the "Recommended" wording is kept so
the reasoning stays auditable.

* **D1 — Load = plan size? — RESOLVED 2026-09-23: YES.** The default `ask`/`run` loads the model at
  the plan's context (32 768+, on this box ~53 000) instead of `prefix + question + margin`. Benefit:
  any request up to the standard works, and the standard is real. Cost: every cold start allocates
  the KV (669 MiB if the plan stops at the standard q8_0; 1 040 MiB at the measured growth answer
  52 601) for the host's lifetime. Escape hatches: `--n-ctx` pins, `--no-fit` restores request
  sizing. Implementation: §5.6.
* **D2 — Overhead reserve. — RESOLVED 2026-09-23: keep `OVERHEAD_BYTES = 512 MiB`.** (v2 answer on
  this box: `q8_0`, ~53 500 ctx — more context, near-lossless KV.) Cutting it to the measured
  compute (~260 MiB) would answer `f16`, ~32 500 ctx — better KV precision, less context, and
  200 MiB less headroom; revisit with the measurements in §8.2 once the v2 default has been used
  for a while.
* **D3 — Does `recommend-quant` follow the standard? — RESOLVED 2026-09-23: NO, not in this card.**
  Its default ctx (4096) is part of SPEC §6's executed reference values. A dedicated card re-pins
  `recommend_quant` + the oracle + SPEC §6 + BENCHMARKS prose together. The path stays frozen here
  (§7.1 `registry/recommend.py:32` = K).
* **D4 — Warning name. — RESOLVED 2026-09-23: `W_CTX_BELOW_STANDARD`** (says what happened); it
  joins `WARNING_CODES` (§5.1, §7.3).
* **D5 — Growth ceiling. — RESOLVED 2026-09-23: ship the rule as written** — the ceiling is
  `min(model window, box capacity)`; a `--ctx-max N` cap is explicitly **not** in this card (it is
  the first follow-up, §12).

---

## 12. Out of scope / follow-ups

* `--ctx-max N` (a growth ceiling that does not pin the exact size) — follow-up card.
* Any change to `OVERHEAD_BYTES`/weights accounting (D2) — follow-up card with its own re-pins.
* `recommend-quant` / oracle / SPEC §6 re-pins (D3).
* Growth-on-re-validation (`replan_for_host` growing a cached plan when the desktop freed memory) —
  today the plan is grown only when it is (re)built; refreshing is `fit --no-cache`.
* Version bump / release notes — explicitly not this card (nor the SPEC card).
* HTTP/MCP surfaces: nothing in them names a ctx default (`api/http.py`, `api/mcp.py` untouched).

---

## Appendix A — probe inventory (`/tmp/t_context_v2/`)

| file | what it holds |
| --- | --- |
| `p0_default.json` | `fit --json --no-cache` today: 4 096 @ f16, 36/36 layers |
| `p1_nctx32768.json` | today with `--n-ctx 32768`: **26 290 @ q4_0**, `W_KV_TYPE_DOWNGRADE` |
| `p2_nctx1m.json` | today with `--n-ctx 1048576`: 26 467 @ q4_0 (the shrink is requested-ctx independent) |
| coordinator's dry probe (card body, 2026-09-23 11:46Z) | `fit --n-ctx 32768` at vram_free 6 714 MiB → **25 556 @ q4_0**, est_kv ≈ 1 010 MiB, est_total ≈ 5 823 MiB against a 5 690 MiB budget — our re-runs put the same command at 26 290 / 26 719 (the estimate *is* the answer, §3.1; the ≥budget est_total is the table-weights vs tensor-weights gap, §8.3) |
| `e0b_ask_nctx32768_defaultplan.json` | today's live cap: `ask --n-ctx 32768` → `engine.n_ctx 4096` |
| `e1_ask_32k.json` | engine at the standard via `--n-ctx 32768 --fit-ctx 32768`: `engine.n_ctx 32768`, q4_0, 35/36, load 1 047.94 ms |
| `w1_cold.json`, `w2_warm.json` | warm-host pair at the 32k plan: wall 6.29 s / 0.71 s |
| `b1_fitparams_c4096.txt`, `b2_fitparams_c32768.txt` | the binary's own tables: context 248 / 1 256 MiB |
| `c0b.log` | real load `-c 4096` f16: KV 144+108 MiB, compute 75 |
| `c1_cli_c32768_f16.log` | real load `-c 32768` f16: KV 1 152+108, self 5 530, free 1 200 |
| `c2_fit_ctx32768_ngl99.log`, `c2b_fit_ctx4096_ngl99.log` | the binary's `--fit` search: 1 048 576 → **37 376** / **37 632** |
| `d1_c32768_ub256.log` | SWA cells = `window + n_ubatch` (768 at 256) |
| `d2_c32768_q4kv.log` | real load 32k q4_0: KV 324+30.4 MiB |
| `d3_c131072_q4kv.log` | real load 131 072 q4_0: KV 1 296+30.4, free 1 046 |
| `d4_c32768_b32768.log` | `-b 32768` compute at 32k: 103 MiB |
| `v1_c52601_q8kv.log`, `v2_c38000_f16.log` | the v2 answer (52 601 q8_0) and the f16 edge (38 000) load for real: free 1 409 / 1 015 |
| `v2_plan_probe.py`, `v2_plan_probe.out` | the executed arithmetic of §3/§8 (run: `uv run python /tmp/t_context_v2/v2_plan_probe.py`) |
| `ac6_check.py`, `ac6_check.out` | AC-6's four pinned byte values, recomputed from the formula (`python3 /tmp/t_context_v2/ac6_check.py` → all four `OK`) |

Responses, logs and `fit --json` payloads are quoted from these files; the exact rerun commands
are in each file's first lines (`# cmd:` style) except the `fit`/`ask` probes, which are one-liners
over `uv run typed-gguf … --no-cache/--no-fit-cache`.

## Appendix B — executed reference values

Verbatim `v2_plan_probe.out` (2026-09-23; host free 6 760 MiB, budget 5 736 MiB):

```
model: {"n_ctx_train": 1048576, "weights_MiB": 4167.2, "layers": 36, "kv_head": 4, "key": 256, "value": 256}
swa: {"window": 512, "n_swa_layers": 27, "n_global_layers": 9, "swa_cells_at_ubatch512": 1024, "per_layer_bytes": {"f16": 4096.0, "q8_0": 2176.0, "q4_0": 1152.0}}
host: {"vram_free_MiB": 6760, "budget_MiB": 5736}

KV bytes at n_ctx (today's formula vs measured-formula), MiB:
   n_ctx  f16_today  f16_real  q8_today  q8_real  q4_today  q4_real
    4096      576.0     252.0     306.0    133.9     162.0     70.9
   32768     4608.0    1260.0    2448.0    669.4    1296.0    354.4
   65536     9216.0    2412.0    4896.0   1281.4    2592.0    678.4
  131072    18432.0    4716.0    9792.0   2505.4    5184.0   1326.4

largest n_ctx this box holds per KV (window cap applied):
    f16: raw fit   26987  window-capped   26987  (total 5736 MiB of 5736 MiB)
   q8_0: raw fit   53511  window-capped   53511  (total 5736 MiB of 5736 MiB)
   q4_0: raw fit  103807  window-capped  103807  (total 5736 MiB of 5736 MiB)

today's estimate_plan(n_ctx=32768):
  n_ctx 26719 kv q4_0 est_kv_MiB 1057 layers 36 warnings ('W_KV_TYPE_DOWNGRADE', 'W_FIT_ESTIMATED')

v2 semantics (proposed) on the same box:
  {"n_ctx": 53511, "kv_type": "q8_0", "standard_n_ctx": 32768, "ctx_limit": "grown", "warnings": ["W_KV_TYPE_DOWNGRADE"]}
  {"n_ctx": 65536, "kv_type": "q4_0", "standard_n_ctx": 32768, "ctx_limit": "pinned", "warnings": ["W_KV_TYPE_DOWNGRADE"]}  <- explicit --n-ctx 65536
  {"n_ctx": 8192, "kv_type": "f16", "standard_n_ctx": 32768, "ctx_limit": "pinned", "warnings": []}  <- explicit --n-ctx 8192 (no grow)
  {"n_ctx": 4096, "kv_type": "f16", "standard_n_ctx": 32768, "ctx_limit": "pinned", "warnings": []}  <- explicit --n-ctx 4096 (today's default pin)
```

The four KV pins in AC-6 are from `ac6_check.out`:

```
PIN (4096, 'f16') 264241152 want 264241152 OK
PIN (32768, 'f16') 1321205760 want 1321205760 OK
PIN (32768, 'q4_0') 371589120 want 371589120 OK
PIN (131072, 'q4_0') 1390804992 want 1390804992 OK
```
