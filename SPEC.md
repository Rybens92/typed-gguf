# SPEC: typed-gguf — GGUF-native typed decision engine (Jev-like, no fine-tuning)

- Task: t_7bcff796 (code-spec) | Tier: L | Repo: `Rybens92/typed-gguf` (MIT; this box's checkout is the dir formerly called `ggufone`) | Date: 2026-09-17
- Renamed: the public name is `typed-gguf` (card t_5f9c15fe, 2026-09-20) — distribution, import
  package (`typed_gguf`), console script, env vars (`TYPED_GGUF_*`) and the default data home were
  formerly `ggufone`; `docs/evidence/` keeps the pre-rename spelling on purpose (receipts are
  records of the runs that produced them).
- Status: DRAFT — everything in §2 is frozen by the coordinator brief + operator updates; §8 lists
  S-1..S-12 for ratification before E1a implementation starts.
- Oracle: `docs/verify_runtime_contract.py` (executed; exit 0 == every pinned number reproduced).
  Run: `python3 docs/verify_runtime_contract.py` (offline) and with `TYPED_GGUF_RUNTIME_DIR=<runtime>` (live).
- Evidence: `docs/evidence/` (captured 2026-09-17) + the executed PoC `docs/evidence/poc-ctypes-20260917.py`.
- Supersession: the operator update of 2026-09-17 15:55 (distribution) and the PoC report of 16:05
  (ctypes pitfalls) supersede the packaging wording in the original card. Both are incorporated here.
- Serve wave (2026-09-24; cards `t_a51b1205` → `t_f5d8b6c7` → `t_d88b4be0`): §2.8 gains `serve`'s
  flags and the `runtime update|rollback` surface, §2.9 becomes the full TypeSafe-compatible HTTP
  wire (recon: the official `typesafe-sdk` 0.7.1 source + its OpenAPI-derived models — thread
  `state/groupchat/typed-gguf-e1.md` entry 2026-09-24 (wieczór); the coordinator's fleet reference
  for this project, `bot-fleet-dispatch/references/`, § "Jev-compatibility"), §2.12 gains the
  serve→host client rule, §5 gains E5 (acceptance + the test/host-gate plan), §7 gains R14–R15 and
  §8 gains S-13..S-17. The `serve`/`mcp` "specified, not shipped" wording is **not** moved here —
  the implementation card owns that flip.

Every number below is tagged:
- **[executed]** — reproduced by the oracle in this repo, right now;
- **[recon]** — measured by the coordinator on this box, recorded in `docs/evidence/poc_report.json`
  or in the thread, not re-run by the oracle;
- **[sdk-0.7.1]** — read out of the official `typesafe-sdk` 0.7.1 Python source (its
  `_schemas/models.py` is generated from `https://api.typesafe.ai/openapi.json`; recon 2026-09-24).
  Every wire claim in §2.9 carries this tag; none is quoted from vendor prose or from memory;
- **[target]** — a number we intend to measure in a later milestone;
- **[UNVERIFIED]** — evidence missing; must not be quoted as fact.

---

## 1. WHAT / WHY / OUT-OF-SCOPE / TOP-LEVEL ACCEPTANCE

**WHAT.** `typed-gguf` is a local decision engine over GGUF models. Input: a *state* (text or structured
data) plus a map of *typed questions* (`choice` | `score` | `noul`). Output: *typed answers* with full
probability distributions and a confidence scalar — no generated text, no parsing, no hosted service,
no fine-tuning. The engine evaluates every question against the same state in **one prefill** of a
shared prefix, then **forks the sequence state** (`llama_memory_seq_cp`) per question and reads the
answer from the logits of a small, fixed set of candidate tokens (restricted softmax). Model weights
are never updated: any GGUF llama.cpp can load is a valid backend, default
`XHToken/Spark-X2.5-4B-GGUF:Q8_0`. The runtime is the **official llama.cpp release bundle** (pinned
`b11026`) driven through `ctypes` — the end user never compiles anything.

**WHY.** Text generation is the wrong interface for machine-consumed judgement: slow, nondeterministic,
unparseable. Trained "System One"-style models proved the mechanic (`rorshopping/parallel-decisions`,
`TheoLeeCJ/openjev` and others) but require a service or a fine-tune. The PoC in this repo proves the
mechanic works **on frozen, stock GGUFs** through llama.cpp's own sequence API: fork readout is
mathematically identical to sequential decode (max |Δ| = 0.00e+00 **[recon]**, `docs/evidence/poc_report.json`),
the shared prefix is paid for once, and each extra question costs only its own suffix decode.

**OUT-OF-SCOPE (v1).** Fine-tuning / LoRA / RLCD / distillation of any kind; hosted service; any
claim of TypeSafe parity (the `typesafe` format is an *adapter*, §2.6); GUI; multimodal input;
training-data generation; cloud inference APIs; model conversion/quantization (we consume GGUF, we
do not produce it).

**TOP-LEVEL ACCEPTANCE (frozen; each is testable in the card that owns it).**

- **A1 No-compile install.** On a machine with no compiler reachable, `typed-gguf init` yields a working
  runtime and answers a question end-to-end. Enforced by running `init` with a poisoned `PATH` that
  hides `cc/gcc/g++/clang/nvcc/cmake/ninja` (§4, A-E1a-2).
- **A2 No fine-tuning.** The repository contains no training code, no training dependency, and no CLI
  path that mutates weights. Enforced by an executed gate (oracle §D + `tests/test_no_finetune.py`).
- **A3 No text generation.** The engine never constructs a sampler chain and never produces free-form
  text; answers come from logits at fixed candidate positions (§2.3). Enforced by a grep gate on
  `llama_sampler_` and a decode-count spy test.
- **A4 Fork equivalence.** Per-question probabilities from the forked readout equal those from a
  sequential decode on a fresh context: `max |Δ| ≤ 1e-3` (PoC: 0.0 **[recon]**).
- **A5 Determinism.** Same model + same backend + `threads=1` + same request ⇒ byte-identical answers
  JSON after stripping `timings` (asserted by hash).
- **A6 Oracle gate.** `python3 docs/verify_runtime_contract.py` → exit 0 offline, and exit 0 **with no
  skips** when a runtime is installed.
- **A7 Unit gate.** `uv run pytest -q` runs with no network and no model download; everything real is
  behind `@pytest.mark.network` / `@pytest.mark.model`.
- **A8 TypeSafe adapter.** In `--format typesafe` the response contains exactly the documented keys
  (§2.6) — verified against fixtures captured from `docs.typesafe.ai` on 2026-09-17.
- **A9 Math invariants.** Every answer: `sum(probabilities) == 1 ± 1e-6`; `confidence ∈ [0,1]`;
  `score ∈ [0, K-1]`; `noul ∈ [0,1]` and carries no `confidence`; `choice == argmax(probabilities)`
  with deterministic tie-break (lowest index).
- **A10 Registry discipline.** `models pull` downloads exactly the selected quant file, resumes after
  interrupt, verifies SHA-256, and atomically publishes the registry entry; `models rm` deletes both.
- **A11 Pre-flight, never a crash.** Model-arch vs runtime-capability mismatch produces
  `E_MODEL_ARCH_UNSUPPORTED` with actionable text; no NULL-pointer crash path (PoC pitfall 1).
- **A12 Scope discipline.** A milestone's `git diff` touches only the paths §3 assigns to it.
- **A13 Baseline recorded.** E1a records this box's prefill tok/s (CPU + Vulkan), warm decision ms,
  first-call shader-compile cost and the measured KV footprint vs the conservative bound — numbers
  reported, not gated.
- **A14 Honest tags.** Any numeric claim in this SPEC is `[executed]`, `[recon]`, `[target]` or
  `[UNVERIFIED]`; a claim whose tag is missing is a bug in this document.

---

## 2. FROZEN CONTRACT

### 2.1 Domain model

```
State        the shared context (text, or structured data rendered into text). One state per request.
Question     {id, type ∈ {choice, score, noul}, instructions, criteria}
             choice: criteria = {option_name: description|null}          (max 255 options)
             score:  criteria = [level_description, ...]                  (2..10 levels, ordered)
             noul:   criteria = {true: str, false: str} (optional)
             instructions/criteria values may be string | object | array (structured guidance).
Candidate    the token sequence the model's next tokens would spell for one option/level/answer.
Readout      logits at the candidate position(s) → restricted softmax → probabilities.
Fork         duplication of the state's sequence memory (KV + recurrent state) into a new seq_id.
Prefill      the single forward pass over the shared prefix (state + framing), done once per state.
Wave         a batch of branches decoded in one llama_decode call (bounded by n_seq_max).
Coverage     candidate probability mass under the FULL vocab softmax (not renormalized).
Confidence   a scalar in [0,1] computed from the shape of `probabilities` (§2.4).
Fit plan     device-fit parameters for one (model, host): n_gpu_layers, n_ctx, kv_type, n_seq_max.
Alias        a registry name → {path, sha256, arch, quant, size, fit plan}.
```

Naming note: `noul` is the question type name used by the adapter target; the engine treats it as
"probability of yes" and exposes it under the same key.

### 2.2 Runtime binding contract (ctypes → official llama.cpp release)

**D-1 (operator decision, 2026-09-17 15:55).** Primary runtime = **official llama.cpp release bundle**
from GitHub Releases, pinned. The bundle carries shared libraries (`libllama.so`, `libggml*.so`) and
tools; `typed-gguf init` downloads the variant matching the host. `llama-cpp-python` is demoted to an
**optional compatibility backend**; our own CI-built wheels are the planned **fallback** for platforms
with no official asset — future work in v0.1.0, no wheel workflow ships (§4; the dispatch-only stub
was dropped before the public tag). Rationale: official assets need no compiler, ship the pinned build
(≥ `b10828`, so `spark2_5` works), and expose everything through a stable C ABI.

Pinned release **[executed]** (`docs/evidence/llama_cpp_release_b11026.json`):

| Asset | Bytes | Use |
|---|---|---|
| `llama-b11026-bin-ubuntu-x64.tar.gz` | 16 855 810 | linux-x64 CPU (verified locally) |
| `llama-b11026-bin-ubuntu-vulkan-x64.tar.gz` | 30 294 625 | linux-x64 Vulkan |
| `llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz` | 168 811 114 | linux-x64 CUDA |
| `llama-b11026-bin-ubuntu-cuda-13.3-x64.tar.gz` | 149 113 548 | linux-x64 CUDA |
| `llama-b11026-bin-win-cpu-x64.zip` | 18 439 911 | windows-x64 CPU |
| `llama-b11026-bin-win-vulkan-x64.zip` | 31 766 385 | windows-x64 Vulkan |
| `llama-b11026-bin-win-cuda-12.4-x64.zip` | 254 193 665 | windows-x64 CUDA |
| `llama-b11026-bin-macos-arm64.tar.gz` | 11 156 751 | macOS arm64 (Metal) |
| `llama-b11026-bin-macos-x64.tar.gz` | 11 204 942 | macOS x64 |

Release facts **[executed]**: tag `b11026`, published `2026-09-17T13:31:47Z`, 33 assets; the
`ubuntu-x64` archive has 61 entries, SHA-256
`219cf1c726bae1da4289b96a6378314d5485c6bc74c43891a4203e30906afb06`, and contains
`libllama.so`, `libggml.so`, `libggml-base.so`, `llama-server`, `llama-fit-params`, `llama-tokenize`
plus 38 shared libraries. `llama-cli --version` prints
`version: 0.4.1-dev (build 11026, commit b49650adb)` on **stderr** **[executed]**.

**Load order (mandatory; PoC pitfall 1).**

```
ctypes.CDLL(<rt>/libggml.so, RTLD_GLOBAL)      # ggml first
ggml_backend_load_all_from_path(<rt>)          # REQUIRED before any model load
llama_backend_init()
model = llama_model_load_from_file(path, llama_model_default_params())
ctx   = llama_init_from_model(model, ctx_params)   # ctx_params.kv_unified = True  (pitfall 2)
```

**Required symbols** (32 from `libllama.so` + 2 from `libggml.so`; all resolved by the oracle on
`b11026` **[executed]**):

```
llama_backend_init llama_backend_free llama_model_default_params llama_context_default_params
llama_model_load_from_file llama_model_free llama_init_from_model llama_free
llama_model_get_vocab llama_model_meta_val_str llama_model_chat_template
llama_model_n_layer llama_model_n_embd llama_n_ctx llama_n_seq_max llama_get_memory
llama_memory_seq_cp llama_memory_seq_rm llama_memory_seq_keep
llama_state_seq_get_size llama_state_seq_save_file llama_state_seq_load_file
llama_batch_init llama_batch_free llama_batch_get_one llama_decode llama_get_logits_ith
llama_tokenize llama_token_to_piece llama_vocab_n_tokens
llama_chat_apply_template llama_synchronize
ggml_backend_load_all ggml_backend_load_all_from_path
```

Header-level pins for `include/llama.h` @ `b11026` (1645 lines) are in the oracle; the struct layouts
for `llama_model_params`, `llama_context_params` and `llama_batch` are pinned by field order in
`docs/evidence/poc-ctypes-20260917.py` — that file is the **reference implementation to transplant**
into `src/typed_gguf/runtime/` (it loads a model, tokenizes, prefills, forks, batch-decodes two branches
and reads logits — all verified on this box).

**Context parameters typed-gguf sets explicitly** (never inherited blindly):

```
n_ctx        with a fit plan (the default): plan.n_ctx — standard 32 768, grown when the box holds
             more, shrunk with W_CTX_BELOW_STANDARD when it does not; with --no-fit (or no plan):
             prefix tokens + longest question suffix + margin (32)     [v2 amendment, see below]
n_batch      >= sum of candidate tokens in the largest wave (default 512)
n_ubatch     512
n_seq_max    1 + max_candidates_per_question  (waves, §2.3); never above the model's capability
kv_unified   True   (MANDATORY — pitfall 2: with False, seq_cp across per-seq KV streams trips
                     GGML_ASSERT(is_full) and kills the process)
n_rs_seq     0 in E1 (recurrent-state rollback is not used); documented if set
type_k/type_v from the fit plan (f16 | q8_0 | q4_0); q8_0 halves the KV footprint
threads      physical cores by default; threads=1 is the determinism mode
flash_attn   "auto"
offload_kqv  per fit plan
no_perf      False (we need timings)
```

**Context sizing v2 (ratified 2026-09-23) — `docs/SPEC-context-v2.md`.** The `n_ctx` row above is
amended by that document's §6.1: with a fit plan applied (the default), the context is sized by the
plan — `n_ctx = plan.n_ctx` (standard 32 768, grown when the box holds more, shrunk with
`W_CTX_BELOW_STANDARD` when it does not), unless the request pins `options.n_ctx`, in which case
`n_ctx = min(options.n_ctx, plan.n_ctx)`. The `prefix + longest question + margin` formula stays the
rule for `--no-fit` and for the *request-fit guard*: a request whose needs exceed the loaded context
raises `E_CTX_TOO_SMALL` with the three token counts — never a truncation. `plan.n_ctx` remains the
ceiling (`A-E1c-5`). Full policy, constants, semantics and acceptance criteria: `docs/SPEC-context-v2.md`.

**Capability probe (pre-flight, no model load).** `arch` is read from the GGUF metadata (§2.7) and the
runtime must prove it can run it: the probe scans `libllama.so` for the arch's implementation symbol
(`llama_model_spark2_5` — 17 exported symbols in `b11026` **[executed]**) and parses the build number
(≥ `b10828` for `spark2_5`). On failure: `E_MODEL_ARCH_UNSUPPORTED` naming the arch, the runtime build
and the fix (`typed-gguf init --force`, or pull a build ≥ b10828).

**Optional compatibility backend.** `typed-gguf[llamacpp]` selects `llama-cpp-python`. Documented as
degraded: its vendored llama.cpp (2026-09-04) and the local builds (b10679/b10715) predate
`spark2_5`, its wheels carry no Vulkan backend, and it does not expose `chat_template_kwargs` /
`enable_thinking`, `n_cpu_moe` or `fit`. It is never on the critical path.

### 2.3 Engine mechanics (the part that makes it fast and deterministic)

1. **Prompt assembly.** `system framing` (constant, model-agnostic) + `state` + per-question framing
   (instructions + criteria rendered as candidate labels) + `answer cue`. The prefix (framing + state)
   is byte-identical for every question of a request.
2. **Prefill once.** The shared prefix is decoded on `seq_id = 0`.
3. **Fork.** For every question, `llama_memory_seq_cp(mem, 0, seq_id_q, 0, prefix_len)` copies the
   prefix state (KV cells + recurrent state on hybrid models) to that question's sequence.
4. **Candidate scoring.** Each candidate is a token sequence (an option label, a level phrase, a
   bracket letter — readout policy per model family, §5/E1c). Its score is
   `z_c = sum(log p(tok_i | prefix, tok_<i)) / L ** length_norm` (default `length_norm = 1.0`;
   `0.0` gives the plain log-prob sum). Candidates are decoded in **one batch** with the question's
   suffix: tokens of all branches in a single `llama_batch`, each token tagged with its `seq_id`,
   `logits=1` only on each branch's last token. `llama_get_logits_ith(ctx, i)` is indexed by
   **token position within the batch** (pitfall 3) — the branch's last token index selects its row.
5. **Restricted softmax.** `p_c = exp(z_c / T) / sum_j exp(z_j / T)` over that question's candidates
   only. `sum(p) == 1 ± 1e-6`.
6. **Diagnostics.** `coverage = sum over candidates of softmax_full(logits)[c]` — computed from the
   full vocab row *before* renormalization. `coverage < options.coverage_floor` (default 0.10) ⇒
   `reliability = "low_mass"` + `W_LOW_MASS`. Low mass is never hidden by renormalization.
7. **Answer derivation.** `choice = argmax(p)` (tie ⇒ lowest candidate index);
   `score = Σ i·p_i` over ordered levels; `noul = p_yes`; `confidence` per §2.4.
8. **No sampling.** No sampler chain, no token loop, no text. The only `llama_decode` calls are the
   prefill and the bounded candidate batches (`waves × branches`), asserted by a spy test.
9. **Waves.** A request whose branch count exceeds `n_seq_max - 1` is split into waves; sequences are
   released (`llama_memory_seq_rm`) and re-forked per wave; results must be identical to a single-wave
   run (E1b-5).
10. **State reuse.** With `state_id` set (default: SHA-256 of prefix tokens), the prefix state is saved
    once (`llama_state_seq_save_file`) and reloaded for later requests, so a repeated state costs no
    prefill. Determinism holds because the saved state is byte-frozen.

### 2.4 Math contract (all values **[executed]** by the oracle)

```
restricted_softmax([2,1,0])        = 0.665241, 0.244728, 0.090031   (T = 1.0; sums to 1)
confidence_normalized_peak(p)      = (max(p) - 1/K) / (1 - 1/K),  K = len(p); 1.0 if K <= 1
score_weighted_mean(p)             = Σ i·p_i
score([0, .7, .3]) = 1.30          score([0, .94, .06]) = 1.06     score([1,0,0]) = 0.0
coverage(candidates={0,1})         = 0.880797   (full logits [2,1,0,-1])
candidate_sequence_score([-1,-1,-1], length_norm=1.0) = -1.0        (length-neutral)
kv_bytes_per_token(spark2_5)       = 36 layers × 4 kv_heads × (256 + 256) × 2 B = 147456 B  (f16)
                                     q8_0: 73728 B     q4_0: 73728 B
conservative_plan(weights, kv, ctx, seq) = weights + kv_per_token · ctx · seq + 512 MiB
```

`confidence_normalized_peak` reproduces 7 of the 8 confidence values printed in the adapter target's
docs within ±0.02 **[executed]**; the 8th (`quickstart/department`: p = [0.159, 0.84, 0.001],
documented 0.596, ours 0.760) is inconsistent with every spread-based statistic we could fit and is
recorded as an outlier in §2.6 — **typed-gguf makes no parity claim**; we always expose the raw
`probabilities` so a caller can apply its own statistic.

### 2.5 Native schema

**Request (native).**

```json
{
  "state": "<string | object | array>",
  "model": "<alias | path | repo[:quant]>",
  "questions": {
    "<id>": {
      "type": "choice | score | noul",
      "instructions": "<string | object | array>",
      "criteria": {"<option>": "<desc|null>", ...} | ["<level>", ...] | {"true": "...", "false": "..."}
    }
  },
  "format": "native | typesafe",            // default "native"
  "options": {                              // ALL optional; defaults in brackets
    "temperature": 1.0, "length_norm": 1.0, "readout": "sequence | single_token" ["sequence"],
    "confidence_mode": "normalized_peak | entropy | margin" ["normalized_peak"],
    "cue": "shipped | two_step | json_field | json_instructed" ["json_instructed"],
    "chat_format": "answer_sheet | role_split" ["role_split"],
    "json_contract": "question | system" ["question"],
    "thinking": false,
    "n_ctx": null, "n_seq_max": null, "kv_type": "auto | f16 | q8_0 | q4_0" ["auto"],
    "coverage_floor": 0.10, "seed": 0, "threads": null, "backend": "auto",
    "state_id": null, "state_cache": true, "save_state": false, "max_waves": null,
    "strict": false
  }
}
```

**Response (native).**

```json
{
  "model": "<alias>",
  "engine": {"runtime": "llama.cpp b11026", "backend": "vulkan", "readout": "sequence",
             "kv_unified": true, "n_ctx": 4096, "n_seq_max": 9, "prefix_tokens": 812,
             "state_id": "sha256:…", "prefill_reused": true},
  "answers": {
    "<id>": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.61, "technical": 0.33, "sales": 0.06},
             "confidence": 0.415, "legend": {"billing": "Payments, invoicing, refunds", …},
             "coverage": 0.93, "reliability": "ok | low_mass | low_confidence",
             "decode_steps": 3}
  },
  "usage": {"input_tokens": 900, "output_tokens": 12, "questions": 4, "forks": 9,
            "prefill_tokens": 812, "decode_steps": 12, "waves": 1},
  "timings": {"model_load_ms": 0, "prefill_ms": 430, "questions_ms": 41, "total_ms": 471},
  "warnings": []
}
```

Rules: `strict=false` ⇒ unknown keys inside `options` produce `W_UNKNOWN_OPTION`; unknown keys at the
top level are always `E_UNKNOWN_KEY`. The three prompt-policy options (`cue`, `chat_format`,
`json_contract`) default to the shipped v2 cell (`json_instructed` / `role_split` / `question`): a
request that names none of them gets exactly that cell — the policy the v2 rows in
`docs/BENCHMARKS.md` were measured under — and the older cells stay reachable by naming them.
`usage.output_tokens` counts **decode steps consumed by the readout** (we generate nothing; the
field keeps the wire shape honest and is documented as such). Numbers are rounded to 6 significant
decimals in JSON to keep runs byte-comparable.

**Error catalog** (exit code 2 for user errors, 3 for runtime/model errors, 4 for internal):
`E_UNKNOWN_KEY`, `E_STATE_EMPTY`, `E_QID_INVALID`, `E_Q_TYPE_UNKNOWN`, `E_CHOICE_CRITERIA`,
`E_CHOICE_TOO_MANY` (>255), `E_SCORE_LEVELS` (outside 2..10), `E_NOUL_CRITERIA`, `E_CANDIDATE_COLLISION`,
`E_MODEL_NOT_FOUND`, `E_MODEL_ARCH_UNSUPPORTED`, `E_RUNTIME_MISSING`, `E_RUNTIME_SYMBOLS`,
`E_RUNTIME_BUILD_OLD`, `E_CTX_TOO_SMALL`, `E_SEQ_MAX_EXCEEDED`, `E_PREFILL_FAILED`, `E_DECODE_FAILED`,
`E_GGUF_CORRUPT`, `E_SHA256_MISMATCH`, `E_DOWNLOAD_FAILED`, `E_AMBIGUOUS_QUANT`, `E_TEMPLATE_UNRESOLVED`,
`E_HF_AUTH_REQUIRED`, `E_INSUFFICIENT_DISK`, `E_REGISTRY_CORRUPT`, `E_STATE_LOAD_FAILED`,
`E_BACKEND_OOM`, `E_ROLE_SPLIT_UNSUPPORTED`, `E_UPDATE_UNAVAILABLE`. `errors.ERROR_CODES` is that
frozen list; the bench adds
its own exit-2 family on top (`E_BENCH_USAGE`, `E_BENCH_SUITE`, `E_BENCH_MODEL`, `E_BENCH_BACKEND`,
`E_BENCH_QUICK`, `E_BENCH_COMPARE`, `E_BENCH_CHILD`, `E_BENCH_STATE`, and the `E_LABEL_*`
label-rendering codes) — extra codes only, never a redefinition of one above.
`E_UPDATE_UNAVAILABLE` is the serve-wave addition (§2.8): the update/rollback cannot apply here — the
active runtime is `$TYPED_GGUF_RUNTIME_DIR`-managed (not ours to replace), the resolved upstream
release carries no bundle under this host's pinned asset name, or there is no retained `previous`
bundle to roll back to. Exit 2; the message names which of the three.
Warnings: `W_LOW_MASS`, `W_LOW_CONFIDENCE`, `W_UNKNOWN_OPTION`, `W_TRUNCATED_STATE`,
`W_KV_TYPE_DOWNGRADE`, `W_VULKAN_WARMUP`, `W_TEMPLATE_FALLBACK`, `W_FIT_ESTIMATED`, `W_FIT_DOWNGRADE`,
`W_BACKEND_OOM`, `W_BACKEND_MISMATCH`, `W_CUE_REFUSED`, `W_JSON_EMPTY_VALUE`, `W_JSON_WRONG_FIELD`,
`W_ESCALATED`, `W_CTX_BELOW_STANDARD` (`errors.WARNING_CODES`). The last one is v2
(2026-09-23, `docs/SPEC-context-v2.md` §5.4): the fit plan had to size the context below the
standard 32 768 because the box could not hold it — valid and loadable, the numbers ride in the
plan's `notes`.

### 2.6 TypeSafe adapter (`--format typesafe`)

Shape pins captured from `docs.typesafe.ai` on 2026-09-17 **[executed against fixtures]**:

- request: top-level `state`, `model`, `questions`; per-question `type` + `instructions` + `criteria`
  (`criteria` map for choice, ordered array for score, optional `{true,false}` for noul).
- response: top-level `model`, `answers`, `usage{input_tokens, output_tokens}` only; per answer
  `type` + (`choice` | `score` | `noul`) + `probabilities` + `confidence` (choice/score) and
  `legend` (score; level number as a string key).
- limits: choice ≤ 255 options; score 2..10 levels; question ids are never sent to the model.
- `score` is the probability-weighted mean of level numbers (`1.30 = 0×0.0 + 1×0.70 + 2×0.30`).

Adapter mapping: native `state`/`questions` are already a superset — the adapter passes them through,
translates `model: "jev-latest"` (and any other non-registry alias) to the configured default alias,
and **drops** every native-only key (`engine`, `timings`, `coverage`, `reliability`, `decode_steps`,
`warnings` are not emitted; `usage` is reduced to the two documented counters). Nothing is renamed
inside `answers` except that probabilities are keyed by option/level name exactly as in the request.
Documented outlier (§2.4) is reproduced verbatim in the fixture table with a note; no parity claim.

### 2.7 Model registry & acquisition

**Paths (XDG).** `models/` (files), `registry.json` (aliases), `runtime/<tag>-<variant>/` (the llama.cpp
bundle), `runtime.json` (active runtime + probe results), `states/` (saved prefix states),
`calibration.json`. Env overrides: `TYPED_GGUF_HOME`, `TYPED_GGUF_RUNTIME_DIR` (read-only consumption of an
existing runtime).

**Pull semantics.** `typed-gguf models pull <repo>[:quant]` (default repo = the pinned default model):
1. resolve the repo via the HF API (offline-capable: a pinned metadata snapshot is committed for the
   default model); 2. select exactly one file (`<repo>:Q8_0` → the single `*Q8_0*.gguf`; bare repo →
   `recommend_quant`; several matches → `E_AMBIGUOUS_QUANT` listing them); 3. download with HTTP range
   resume into `<name>.part`, optional `hf_transfer`/Xet acceleration when importable; 4. verify
   SHA-256 against the HF tree API `lfs.oid`, then atomic `os.replace`; 5. read the GGUF header
   (metadata only, never tensor data) for `general.architecture` + `general.file_type`; 6. write the
   registry entry `{alias, path, sha256, arch, quant, size, added_at, fit_plan}`.

Pinned default **[executed]**: `XHToken/Spark-X2.5-4B-GGUF` @ `902d865994943ab9235670e24f01846ee06091f2`,
Q8_0 = 4 375 021 152 B, SHA-256 `5c2c3c19…9dea2`; Q4_K_M = 2 600 224 352 B `adfcfa19…4028`;
F16 = 8 229 920 352 B `8cecf405…e1a14`. The `lfs.oid == sha256(file)` identity was verified by
downloading a real LFS object from the same repo **[executed]**
(`docs/evidence/hf_lfs_oid_semantics.json`), and the default model's local copy on this box hashes to
exactly the pinned SHA-256 **[executed]**.

**GGUF metadata reader** (pure Python, header only): version, `n_tensors`, `n_kv`, KV map with arrays.
Verified against real files **[executed]**: the default model reports `general.architecture = spark2_5`,
`file_type = 7` (Q8_0), `block_count = 36`, `head_count = 16`, `head_count_kv = 4`, `key_length = 256`,
`value_length = 256`, `context_length = 1048576`, `embedding_length = 2560`; Qwen3.5-0.8B reports
`qwen35`, 24 layers, `file_type = 15` (Q4_K_M). `file_type` → quant: 7 = Q8_0, 15 = Q4_K_M (from the
`llama_ftype` enum in the pinned header).

**`recommend_quant(vram, ram, n_ctx, n_seq_max)`** = largest quant, then largest KV type, whose
**conservative** plan (`weights + kv·ctx·seq + 512 MiB ≤ budget`) fits VRAM (10% margin); else the
largest that fits RAM (20% margin, CPU placement); else `insufficient` + warning. Executed reference
**[executed]**: on 8 GiB VRAM / 31 GiB RAM / `spark2_5` KV:
- `n_ctx=4096, n_seq_max=8` → Q8_0 + **q8_0** KV, GPU, est. total 7.33 GB (f16 KV would need 9.74 GB);
- `n_ctx=2048, n_seq_max=4` → Q8_0 + **f16** KV, GPU, est. total 6.12 GB;
- `n_ctx=32768, n_seq_max=16` → `insufficient` (the conservative bound is deliberately pessimistic:
  it charges every fork a private `n_ctx`; E1a measures the real unified-cache footprint, A-E1a-8).

### 2.8 CLI surface

```
typed-gguf init [--backend auto|cpu|vulkan|cuda|metal] [--force] [--dry-run]
typed-gguf doctor [--json]                     # exit 0 ok / 2 warnings / 1 failures
typed-gguf models search <query>
typed-gguf models pull <repo[:quant]> [--file NAME] [--no-verify] [--jobs N]
typed-gguf models use <alias> | ls [--json] | rm <alias> | verify [<alias>] | recommend-quant [--vram GIB]
typed-gguf run --questions q.json [--state s.txt|--state-json f] [--model alias] [--format native|typesafe]
              [--out r.json] [--state-id ID]
              [--cue shipped|two_step|json_field|json_instructed] [--chat-format answer_sheet|role_split]
              [--json-contract question|system] [--thinking] [--keep-alive <dur|0>]
typed-gguf ask --state <text|@file> --choice "id=instr:opt1|opt2" --score "id=instr:l0|l1|l2"
              --noul "id=instr" [--keep-alive <dur|0>]
typed-gguf serve [--host 127.0.0.1] [--port 8088] [--format native|typesafe] [--keep-alive <dur|0>]
typed-gguf runtime update [--check|--dry-run] [--tag TAG] [--backend auto|cpu|vulkan|cuda|metal] [--json]
typed-gguf runtime rollback [--json]
typed-gguf mcp                                  # stdio JSON-RPC for MCP clients
typed-gguf bench --suite latency|throughput|quality|calibration|determinism [--model alias] [--json]
typed-gguf fit [<model>] [--print] [--no-cache]
typed-gguf calibrate [--model alias] [--dry-run]
typed-gguf keep status [--json] | stop [--json]   # the warm engine host (§2.12)
typed-gguf version [--json]
```

`--cue`, `--chat-format` and `--json-contract` are the prompt-policy switches of §2.5 (`--thinking`
the opt-in thinking switch); `run`, `ask` and `bench` all accept them, so a published row's cell is
reproducible from the CLI.

**`serve` (serve wave, 2026-09-24).** The HTTP surface of §2.9. Both surface families are mounted at
every `--format`: the flag picks only the **default response format of `/v1/decide`** (`/v1/systemone`
is always the TypeSafe projection — a client that sets its base URL must not depend on a server
flag), and `--keep-alive` is the §2.12 window passed on every served decision.

**`runtime update` / `runtime rollback` (serve wave, 2026-09-24).** Owner task: *"żeby dało się
zaktualizować llama.cpp które się instaluje poprzez init"* — refresh the bundle `init` installed. The
runtime ladder itself is unchanged (`$TYPED_GGUF_RUNTIME_DIR` > data-home `runtime.json` > a scan of
`<home>/runtime/*`) and `init` keeps installing the pinned bundle (§2.2/§4). `runtime update` is the
explicit, **never automatic** way to move the *installed* bundle to a newer **official** llama.cpp
release. It never rewrites `runtime.lock`, never touches models/registry/calibration, and no per-fork
branch is ever special-cased (a fork stays a rung-3 runtime: `TYPED_GGUF_RUNTIME_DIR`, §2.2). The
pinned bundle stays what `init` installs and what the oracle's pins speak about (S-16).

1. **Resolve.** Current runtime = `finder.find_runtime()`, its build from the `runtime.json` record
   when the record names that directory. Rung 1 (`$TYPED_GGUF_RUNTIME_DIR` set) → refuse
   (`E_UPDATE_UNAVAILABLE`: that runtime is managed outside typed-gguf — unset the variable to
   update the installed one); `--check` reports the same refusal (it never queries upstream for a
   runtime we will not switch). Nothing installed at all → `E_RUNTIME_MISSING` (the fix is `init`
   first). Otherwise `--check`/`--dry-run` stops after resolve + target: it prints current vs target
   (tag, build, dir, size) and touches nothing — with no network it answers `E_DOWNLOAD_FAILED`
   (exit 3) naming the URL it could not reach, and leaves no partial state.
2. **Target.** Without `--tag`, the newest official release *that actually carries this host's
   bundle*: `GET https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20` (plain HTTPS
   GET, `Accept: application/vnd.github+json`, `User-Agent: typed-gguf/<version>`), first entry whose
   asset list contains the retagged name. **Not `releases/latest`**: the recon of 2026-09-24 found
   `releases/latest` = `v0.5.0` with a single `nightly-tag.txt` asset, while the per-build
   `bNNNNN` releases (marked prerelease) carry the 35 platform bundles **[recon: live GitHub API]**.
   With `--tag TAG`, that one release is fetched by tag and must carry the same name. The host
   variant comes from `init`'s own detection (`--backend` overrides; **no fallback ladder** — a GPU
   bundle that fails its probe aborts the update instead of silently changing the backend). The
   asset name is the pinned one retagged (`llama-b11026-bin-ubuntu-vulkan-x64.tar.gz` →
   `llama-b11160-…`), and the download URL is `runtime.lock`'s own template with the new tag
   (verified live 2026-09-24: that URL answered `302 → 200`, 30 943 538 B **[recon]**). No name match
   → `E_UPDATE_UNAVAILABLE` naming the naming rule: upstream renamed the asset, we never guess.
   Target tag == current tag → `updated: false` ("already at `<tag>`"), exit 0.
3. **Stage.** Disk-space precheck (`E_INSUFFICIENT_DISK`), download into `<home>/downloads/<asset>`
   with `init`'s resume/size semantics, extract into `<home>/runtime/.pending-<asset>` (the same
   path-safety filter and flatten rule), verify the lock's `required_files`, then run the **lock
   probe** in a child (`capability.probe_runtime`): required tools, the 34 required symbols, the
   build number, the backend list — plus the arch rule of §2.2 (`spark2_5` needs build ≥ `b10828`).
   A failure removes the staging directory and leaves the working runtime untouched
   (`E_RUNTIME_SYMBOLS` / `E_RUNTIME_BUILD_OLD` / `E_MODEL_ARCH_UNSUPPORTED` / `E_DOWNLOAD_FAILED`,
   per the cause).
4. **Switch.** Stop the resident keep host first (`keep stop`, drain — abort, old runtime intact, if
   it cannot be stopped: a process that dlopen'd the old libraries must not survive the switch),
   `os.replace` the staged directory into its final `<home>/runtime/<tag>-<variant>/`, then rewrite
   `runtime.json` **once**, atomically (`finder.write_runtime_record`), with the new record plus a
   `previous` block (`dir`, `tag`, `build`, `installed_at`; all existing record keys stay). That
   record write **is** the switch point: everything before it leaves `runtime.json` byte-identical,
   so a failed download, a failed probe, or a `kill -9` anywhere earlier keeps the old runtime
   active; nothing is deleted on the way (the previous bundle stays on disk — that is the point).
5. **Report.** `--json` carries `updated`, `from`/`to` (tag, build, dir), asset + sha256 + url,
   probe summary (`tools`, `symbols_ok`, `build`, `backends`) and the `previous` dir; the human
   output states the same in one line. `doctor`/`version` then report the new build as its own
   probe found it.

`runtime rollback` flips the record back to the retained `previous` bundle (atomic record write, no
download, no probe of the old bundle beyond its presence); with no `previous` recorded, or with the
directory gone, it refuses (`E_UPDATE_UNAVAILABLE`) and changes nothing. Rollback never requires a
human to repair state by hand, and neither command ever deletes a bundle.

### 2.9 HTTP surface (`serve`) + MCP surface

**`typed-gguf serve`** is a stdlib HTTP server (`http.server`, no third-party runtime dependency, no
telemetry, `--host 0.0.0.0` prints a warning) that answers **from the same warm engine host the CLI
uses** (§2.12). Every decision is a keep-client request: same model file + SHA, same fit plan, same
calibration, same readout, same 6-significant-decimal rounding (§2.5) — the served numbers **are**
the CLI's numbers. The server holds no model handle of its own and never a second one: a request for
a model the resident host does not hold swaps (stop → load → new countdown) exactly like a warm CLI
call, and decisions serialize through the host socket, one at a time, in arrival order (§2.12).
`/health` and `/v1/models` never touch the host and never wait behind a decision.

Default bind `127.0.0.1:8088` (`--host`, `--port`; S-8). Four routes, the TypeSafe pair mounted
regardless of `--format`:

| Route | Format | Purpose |
|---|---|---|
| `GET /health` | ours | liveness + what is resident; never loads a model |
| `GET /v1/models` | TypeSafe | the registry aliases plus the compat name `jev-latest` |
| `POST /v1/systemone` | TypeSafe | the drop-in decision route |
| `POST /v1/decide` | native | the §2.5 request/response, verbatim |

`--format native|typesafe` (default `native`) is only the default response format of `/v1/decide`;
a request body's own `format` overrides it per call. `/v1/systemone` always answers the §2.6
projection — a client that sets its base URL must not depend on a server-side flag.

**`GET /health`** → `200`, exactly these five keys:

```json
{"status": "ok", "service": "typed-gguf", "version": "<package version>",
 "model": "<resident alias | null>", "warm": true}
```

`model`/`warm` are read from the keep ledger (no ping, no load): `model: null` + `warm: false` mean
no host is resident — the next decision will pay the load.

**`GET /v1/models`** → `200` (`{"models": [{"name", "description", "release_date"}, …]}`):

- `name` — every alias in `registry.json`, **sorted by name** (a deterministic wire, independent of
  pull order), plus `jev-latest`, which every request may use and which resolves to the registry's
  `current` alias. An empty registry lists `[]` (and `/v1/systemone` then answers `422`, naming
  `models pull`).
- `release_date` — the model **file's mtime** as a UTC `%Y-%m-%d`: when *this copy* was written on
  this box. We do not know an upstream release date and do not invent one.
- `description` — one auto line, no marketing: `"<arch> <quant> GGUF (<human size>), local;
  typed-gguf's own engine"`, arch/quant/size from the registry entry.

**`POST /v1/systemone`** request — exactly the SDK's body [sdk-0.7.1]:

```json
{"state": "<string | object | array>",
 "model": "<alias | jev-latest>",
 "questions": {"<name>": {"type": "noul | choice | score",
                          "instructions": "<string | object | array>",
                          "criteria": {"<label>": "<desc|null>", "…": null} | ["<level>", "…"] | {"true": "…", "false": "…"}}}}
```

- These three keys **only**. An unknown top-level key is `422 extra_forbidden`, never silently
  ignored (a silently dropped instruction is a wrong answer; the SDK's `extra_body` escape hatch
  therefore does not extend this surface).
- `model` must resolve through the registry: an alias, or `jev-latest` (→ `current`). A path or
  `repo[:quant]` reference is **rejected** (`422`) — a remote client must not point the server at
  files.
- `state`, `questions` and each question's `criteria` are validated by the engine's own rules and
  codes (§2.5): `E_STATE_EMPTY`, `E_Q_TYPE_UNKNOWN`, `E_CHOICE_CRITERIA`, `E_CHOICE_TOO_MANY`
  (>255), `E_SCORE_LEVELS` (2..10 levels), `E_NOUL_CRITERIA`… Our limits are the engine's (§2.6);
  a client that leans on the vendor's looser ones gets a typed `422`, never a wrong answer. The
  code text rides in the error's `msg`.
- `instructions` may be text, an object or an array (the engine renders JSON content exactly as it
  does for `run`); `criteria` is the option map (choice), the ordered level list (score), or the
  optional `{true, false}` (noul).

**`POST /v1/systemone`** response `200` — the §2.6 projection and nothing else:

```json
{"model": "<resolved alias>",
 "answers": {"<name>": {"type": "noul", "noul": 0.93}
                    | {"type": "choice", "choice": "billing", "confidence": 0.415,
                       "probabilities": {"billing": 0.61, "technical": 0.33, "sales": 0.06}}
                    | {"type": "score", "score": 1.3, "confidence": 0.55,
                       "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                       "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}},
 "usage": {"input_tokens": 812, "output_tokens": 12}}
```

- `model` echoes the **resolved** alias (what answered), never the raw request string.
- Answer key sets are exactly the documented ones: `type` + value key (`noul`/`choice`/`score`) +
  `probabilities` + `confidence` for choice/score, plus `legend` for score; **`noul` carries no
  `confidence`** (the SDK's own answer model has no such field). Native-only keys (`engine`,
  `timings`, `coverage`, `reliability`, `decode_steps`, `warnings`) are dropped exactly as §2.6
  says; nothing inside `answers` is renamed.
- `probabilities` are the engine's calibrated readout **as-is** — the adapter never renormalizes
  (they already sum to 1 ± 1e-6, §2.4) and never rescales `confidence`.
- `legend` is keyed by the level number as a string, matching the SDK's `dict[int, str]` coerce rule
  and §2.6's fixtures.
- `usage` is the §2.6 projection's own two counters, verbatim from the native response (§2.5):
  `input_tokens` = the request's real prompt token count (the shared prefix + every question suffix,
  i.e. the native `usage.input_tokens`); `output_tokens` = the native `usage.output_tokens`, i.e.
  **decode steps consumed by the readout** (§2.5's own definition). `prefill_tokens` is deliberately
  *not* reported as `input_tokens`: it counts the work *this call paid* and is 0 on a warm state
  cache — as an input count it would be a lie. We generate no text and bill nothing; the two counters
  are the real ones.

**`POST /v1/decide`** is the §2.5 native wire verbatim (request and response, including `format` and
`options`); its errors are the native `{"error": {"code", "message"}}` with the §2.5 code, mapped
exit 2 → `400`, exit 3 → `503`, exit 4 → `500`.

**Errors on the TypeSafe routes are FastAPI-shaped** — the shape the SDK's own `HTTPValidationError` /
`ValidationError` models describe and its `extract_message` reads [sdk-0.7.1]:

| Condition | Status | Body |
|---|---|---|
| body is not JSON / not an object | `422` | `{"detail": [{"type": "json_invalid", "loc": ["body"], "msg": "…"}]}` |
| a required key missing (`state`/`model`/`questions`) | `422` | `{"detail": [{"type": "missing", "loc": ["body", "<key>"], "msg": "Field required"}]}` |
| unknown top-level key | `422` | `{"detail": [{"type": "extra_forbidden", "loc": ["body", "<key>"], "msg": "…"}]}` |
| unknown model, or an engine validation code (exit 2) | `422` | `{"detail": [{"type": "value_error", "loc": ["body", "…"], "msg": "E_…: …", "input": …}]}` |
| engine/runtime failure, any exit-3 code | `500` | `{"detail": "E_…: <message>"}` |
| unknown route | `404` | `{"detail": "Not Found"}` |

`loc` follows the FastAPI path convention (`["body", "questions", "<name>", "criteria"]`), and `msg`
always carries the typed code, so an SDK user reading `TypeSafeAPIError` sees **which** rule failed.
A `5xx` is retried by the SDK's default policy (2 retries, backoff, 30 s budget [sdk-0.7.1]); our
failures are deterministic, so a retry yields the same error — a caller who dislikes that passes its
own `RetryPolicy`.

**Auth.** `Authorization: Bearer <anything>` is accepted, and a **missing** header is accepted too (a
local server has no tenants). The SDK refuses to construct a client without a key, so the docs say:
set `TYPESAFE_API_KEY` to any non-empty string. The value is never logged, never echoed, and never
put into an error body.

**Cold start** (the SDK's default per-request timeout is 10 s [sdk-0.7.1]). A cold decision pays the
model load inside that request (the keep-host spawn, §2.12) — the same cost as `ask`'s first call
(~1–3 s for the 4B on this box; a CPU box or a cold shader cache can exceed 10 s, §6 **[recon]**).
Docs advise running `serve` long-lived: it passes `--keep-alive <dur|0>` (default 600 s; precedence
flag > env > default, §2.12) on every decision, so traffic keeps the host warm and the host still
exits by itself after the window; `--keep-alive 0` means every request pays the load. Where a load
exceeds a client's timeout, the SDK's retry budget (30 s) absorbs it or the client raises its timeout
— documented, not hidden.

**Logging.** One line per request on stderr (route, status, `served_by`, total ms, request id);
never the `Authorization` value, never a request body by default.

MCP (stdio, JSON-RPC 2.0 — unchanged by this wave: `initialize`, `tools/list`, `tools/call`): tools
`typed_gguf_decide` (state + questions → answers), `typed_gguf_models_list`, `typed_gguf_models_pull`,
`typed_gguf_runtime_status`, `typed_gguf_fit`. Tool schemas mirror §2.5; no tool ever triggers a
network call except `models_pull`.

### 2.10 Calibration, fit and routing (E2.5)

- **Confidence modes**: `normalized_peak` (default), `entropy` (`1 − H/log K`), `margin`
  (`(p1 − p2)/(1 − p2)`), all in [0,1]; each is measured in the calibration suite.
- **Calibration**: fit a per-(model, question-type) temperature/scale on a committed labeled dev set,
  store in `calibration.json`, apply at readout. Accept only if ECE (or agreement at a fixed
  confidence threshold) improves on a held-out split; otherwise report "no calibration applied".
- **Fit**: `typed-gguf fit` runs the bundle's `llama-fit-params` (`--fit on`, `--fit-target MiB`,
  `--fit-ctx N`, `--fit-print on` **[executed: flags present in b11026]**) and adds typed-gguf's own
  `n_seq_max`/KV-type math; the plan is cached per (model sha256, host fingerprint) and applied on load
  unless `--no-fit`.
- **Routing**: `--route auto` picks (alias, kv_type, n_ctx, n_seq_max) from the registry under the
  device budget, honoring arch capability; low-confidence answers can escalate to a larger model or a
  decomposed question set (opt-in, `max_escalations` default 1, always logged).

### 2.11 No-fine-tuning guarantee

The critical path never mutates weights: no optimizer, no LoRA/QLoRA adapter loading, no RLCD, no
distillation, no dataset generation. Core has **zero third-party runtime dependencies** (stdlib only);
the optional extras are `network` (accelerated download), `llamacpp` (compat backend) and `dev`.
Enforced by: the oracle §D dependency scan, `tests/test_no_finetune.py` (no `torch|peft|trl|unsloth|
bitsandbytes|deepseed|accelerate|lightning` in `pyproject`), a repo-wide grep gate for `llama_sampler_`
in `src/`, and the absence of any training entry point in the CLI.

### 2.12 Warm engine host — keep-alive, idle unload, swap (E4)

`run`/`ask` do not unload the model when the call returns: the call that pays the load leaves a
**keep host** behind — a detached child process (own session, `setsid`, survives its parent) holding
the loaded model and answering later decisions over a **unix socket** in the data home
(`$TYPED_GGUF_HOME/keep/`, mode 0600, never TCP). A later call with the same identity skips the cold
start. This host is the warm core the `serve`/`mcp` surfaces (§2.9) reuse.

- **One host, one model, one data home.** At most one host is alive per data home, and it holds at
  most one model. A request whose key differs from the resident host's **stops that host before the
  new model loads**, so a swap never keeps two models in RAM/VRAM.
- **The key** is the identity: the resolved model path plus its SHA (the registry's recorded sha256,
  else the file's own size+mtime) plus the placement-affecting options (`backend`, `n_ctx`,
  `kv_type`, `n_seq_max`, `threads`, fit flags). `keep status` prints the key it holds; a differing
  key is a swap, never a wrong answer.
- **The window.** The host exits itself after `keep_alive` seconds without a request — the countdown
  restarts on every request. Default **600 s**. Configured by `--keep-alive <dur|0>` on `run`/`ask`
  or `$TYPED_GGUF_KEEP_ALIVE`, spelled as bare seconds (`600`) or with a unit (`10m`, `90m`, `2.5s`);
  precedence **flag > env > default**; `--keep-alive 0` reproduces the pre-E4 behaviour exactly
  (answer inline, nothing resident). No settings file: the window belongs to the call that starts the
  host, so it travels with the request that pays for it.
- **Requests are serialized** inside the host: one decision at a time, in arrival order. Two clients
  sharing a host never run concurrently against one model handle.
- **Exit discipline** follows `runtime/teardown.py`: the host closes the handle and ends through
  `os._exit`, so the NVIDIA ICD teardown SIGSEGV of t_57cc0179 cannot take the device down with it.
  The socket is unlinked and the ledger entry removed on the way out; debris a `kill -9` left behind
  (a socket where nothing answers, a stale pid) is probed and replaced on the next call.
- **Fallback.** If a host cannot be had — no `AF_UNIX` on this platform, a spawn that never becomes
  ready, a host that dies mid-request — the client cleans up its ledger entry and answers **inline on
  that same call**, naming the reason in `engine.keep.fallback`. A crash *inside* the host is a typed
  error on the wire, rebuilt as the product's own exception type. The CLI never wedges on a host.
- **Who answered** is in the response of every call that went through the keep path:
  `engine.keep.served_by` (`"host"` / `"inline"`), the host pid, the load *that call paid* — the
  spawning call waited for it, so it reports the host's one-time `model_load_ms` in its own
  `timings.model_load_ms`, and a warm answer reports `0.0` — the idle seconds left, and the fallback
  reason when there was one. `--keep-alive 0` never enters the keep path: its response is the pre-E4
  one verbatim, with no `engine.keep` block. `keep status` reports the pid, the model, the key,
  loaded/idle seconds, the placement, and the device the **engine's own log** proved (the t_603a35a
  / t_80f1a4c6 honesty rule).
- **Platforms.** The host needs `AF_UNIX`, i.e. Linux/macOS. Where it is missing (Windows), `run`/
  `ask` answer inline with the named warning `W_KEEP_UNAVAILABLE`; no daemon is attempted.
- **`serve` is a host client, nothing else** (serve wave, 2026-09-24; §2.9). Every decision it
  answers is a keep-client request against this socket: the same key, the same swap discipline (one
  model, one home), the same arrival-order serialization, the same inline fallback when no host can
  be had. The server keeps no engine state of its own — which is why `keep status` / `keep stop`
  mean exactly what they say while `serve` runs, and why a served warm answer is the CLI's answer.
  Two rules follow for the implementation cards: `serve` passes its `--keep-alive` on every decision
  (the window belongs to the call that pays for the host, above), and `runtime update` stops the host
  **before** it switches bundles — a process that dlopen'd the old libraries must never keep
  answering after a new build is installed (§2.8).

---

## 3. Repo layout & scaffold (created with this SPEC)

```
typed_gguf/
  pyproject.toml            # uv-managed; requires-python >=3.11; core deps = [] (stdlib only)
  runtime.lock              # the pinned runtime: release tag, per-platform asset names/sizes/sha256,
                            # required symbols, mandatory call order, default-model pin
  README.md                 # what it is, quickstart, no-compile/no-finetune guarantees
  LICENSE                   # MIT
  .gitignore                # models/, .venv/, *.gguf, *.so, runtime/
  SPEC.md                   # this document
  docs/
    verify_runtime_contract.py     # the oracle (executed; exit 0)
    evidence/                      # captured facts + PoC + listings (§6)
    TEMPLATES.md                   # (E1c) family → template source → thinking mode → label policy
    BENCHMARKS.md                  # (E2) measured tables
  tools/
    capture_evidence.py            # re-captures docs/evidence/*.json from the network
  src/typed_gguf/
    __init__.py  __main__.py  cli.py  schema.py  errors.py
    engine/{prompt,readout,session,decide}.py
    runtime/{finder,ctypes_binding,capability,install,fit}.py
    registry/{gguf,hf,store,recommend}.py
    calibration/{stats,calibrate,routing}.py
    api/{http,mcp}.py
    bench/{harness,suites}.py
    keep/{identity,state,host,client}.py        # (E4) the warm host: key, ledger, server, client
  tests/
    test_scaffold.py               # E0: package imports, layout, version
    test_runtime_contract.py       # thin wrapper: oracle sections as pytest cases (E1a)
    test_no_finetune.py            # A2 gate
    test_gguf_header.py test_registry_store.py test_recommend_quant.py      (E1a)
    test_readout_math.py test_schema.py test_typesafe_adapter.py test_engine_fork.py test_cli.py (E1b)
    test_templates.py test_fit.py (E1c) | test_bench.py (E2) | test_calibration.py test_routing.py (E2.5)
    test_keep.py test_keep_host.py test_keep_client.py test_keep_cli.py (E4, offline)
    test_keep_live.py              # E4 live: cold vs warm, idle unload, swap, orphans (--run-network)
  .github/workflows/
    ci.yml                    # lint + unit gate + oracle (offline and live per platform)
    runtime-matrix.yml        # downloads each pinned asset, runs the oracle live section (per OS)
  state/fights/<fight>/     # committed receipts of an evidence fight (scorecard + logs + scripts)
```

Nothing in this layout is aspirational: E1a owns `runtime/`, `registry/`, `cli.py init|doctor|models`,
E1b owns `schema.py`, `engine/`, `cli.py run|ask`, E1c owns the template resolver + `fit`, E2 owns
`bench/` + `docs/BENCHMARKS.md`, E2.5 owns `calibration/`, E3 owns no new source (runs + docs).

---

## 4. Packaging & distribution — "the user never compiles"

**Ladder (in order, each rung fully automated before the next is offered).**

1. **Prebuilt (primary).** `typed-gguf init` detects OS/arch/GPU, maps to the pinned asset table (§2.2),
   downloads it (resume + size/entry verification), extracts into `runtime/<tag>-<variant>/`, then runs
   the probe (symbols, build number, backend list) and a warm-up decode. No compiler on the path.
   GPU detection: `nvidia-smi` + driver version → CUDA asset; else Vulkan ICD present (`libvulkan.so.1`
   / `vulkaninfo`) → Vulkan asset; else CPU. `--backend` overrides; `--dry-run` prints the plan.
2. **Automated source build (fallback).** When no asset matches (exotic arch, no official variant): a
   scripted cmake+ninja build with progress and a hard timeout, producing the *same* runtime dir
   layout, logged with the exact command line for reproducibility.
3. **Guided manual.** Print the exact steps, accept a user-provided runtime dir via
   `TYPED_GGUF_RUNTIME_DIR`, and verify it with the same probe (so a hand-built runtime is first-class,
   just not automatic).

**Rules.** `runtime.lock` (repo root, data) is the single source of truth for the pinned release;
`src/typed_gguf/runtime/pins.py` (typed access) reads it and the oracle asserts both against
`docs/evidence/`. Never commit binaries to git (only pins, sizes and hashes; `.gitignore` covers `*.so`,
`*.dylib`, `*.dll`, `*.gguf`). The runtime is reported by `typed-gguf doctor --json`/`version --json`.
Runtime-vs-model arch pre-flight always runs before load (A11). `runtime.json` (in the data dir)
records the asset name, its SHA-256 (computed at install; GitHub publishes no per-asset digest, so our
own pin is the source of truth), the probe results and the warm-up timing.

**CI.** `ci.yml`: ruff + `uv run pytest -q` + oracle offline on linux. `runtime-matrix.yml`: per
platform (ubuntu cpu/vulkan, windows cpu, macos arm64) download → extract → oracle **live** section →
smoke decision on a tiny GGUF; a job fails if the live section skips. A `llama-cpp-python` wheel
matrix for platforms with no official asset (exotic arch, downstream re-distribution) stays **future
work**: the dispatch-only `wheels-fallback.yml` stub was dropped before the public v0.1.0 tag rather
than shipped half-built (release review N2, card `t_a25bd190`). Rung 1 (the pinned prebuilt bundle) is
the only automated rung in v0.1.0; rung 2 (the automated source build) is not implemented yet, and a
host with no matching asset uses rung 3 — its own build, consumed through `TYPED_GGUF_RUNTIME_DIR`
and probed by `typed-gguf doctor`.

---

## 5. Milestones

### E1a — runtime + model registry (no engine yet)

Deliverable: `typed-gguf init`, `typed-gguf doctor`, `typed-gguf models {search,pull,use,ls,rm,verify,
recommend-quant}`, `registry/`, `runtime/`, tests.

- **A-E1a-1** Oracle live section green: `TYPED_GGUF_RUNTIME_DIR=<rt> python3 docs/verify_runtime_contract.py`
  → exit 0 with no `SKIP` in section B.
- **A-E1a-2** `typed-gguf init` succeeds with `PATH` poisoned to hide `cc/gcc/g++/clang/nvcc/cmake/ninja`
  (or on a machine without them) and with `TYPED_GGUF_OFFLINE_CACHE` pointing at a pre-downloaded bundle;
  wall clock ≤ 180 s (warm connection) **[target]**; `--dry-run` prints asset choice + destination.
- **A-E1a-3** `doctor` verifies: bundle present, `libllama.so` SHA-256 recorded, all 34 required
  symbols resolve, build == pinned tag and ≥ `b10828`, `llama-fit-params --help` exits 0, backend list
  contains the expected accelerator. `--json` schema is stable; exit codes 0/2/1 pinned.
- **A-E1a-4** `models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0` → exactly one file, 4 375 021 152 B,
  SHA-256 `5c2c3c19…9dea2`, registry entry written; **resume test**: kill at ~10%, re-run, sha matches,
  bytes re-fetched ≤ one chunk beyond the interruption point; `@pytest.mark.network` only.
- **A-E1a-5** Quant selection: `:Q8_0` → `*Q8_0*.gguf`; bare repo → `recommend-quant`; two matches →
  `E_AMBIGUOUS_QUANT` listing candidates; unknown quant → actionable error.
- **A-E1a-6** Registry: alias resolution from `--model`, `ls --json` fields (alias/path/size/sha/arch/
  quant), `rm` deletes file+entry, corrupt `registry.json` → recoverable error without data loss.
- **A-E1a-7** `recommend-quant` reproduces the executed table of §2.7 for the pinned scenarios.
- **A-E1a-8** Memory report: measured KV footprint at `n_ctx=2048`, `n_seq_max ∈ {1,4}` vs the
  conservative bound (report-only, recorded in the E1a evidence file); the recommender stays
  conservative until E2.5 has the measurement.
- **A-E1a-9** Arch pre-flight: a model whose arch the runtime lacks → `E_MODEL_ARCH_UNSUPPORTED`
  (no crash, no NULL); the message names arch, runtime build and the fix.
- **A-E1a-10** GGUF reader unit tests over the pinned local files (skipped when absent) and over
  synthetic headers (v2/v3, arrays, nested arrays, truncated file → `E_GGUF_CORRUPT`).
- **A-E1a-11** Gate: `uv run pytest -q` green offline; oracle offline green; `git diff` limited to §3
  paths for E1a.
- **A-E1a-12** Baseline numbers recorded (A13) in `docs/evidence/e1a_baseline.json`.

### E1b — engine core (schema + fork readout + CLI)

Deliverable: `schema.py`, `engine/*`, `cli.py run|ask`, typesafe adapter, tests.

- **A-E1b-1** Invariants (property tests over synthetic logits): `sum(p) = 1 ± 1e-6`, `confidence ∈
  [0,1]`, `score ∈ [0, K-1]`, `noul ∈ [0,1]` with no `confidence`, deterministic argmax tie-break.
- **A-E1b-2** Fork equivalence on a real model: max |Δ| ≤ 1e-3 vs sequential decode, on a hybrid
  (`qwen35`) **and** a pure-attention model; the PoC number (0.0 **[recon]**) is reproduced or beaten.
- **A-E1b-3** One prefill per state: `usage.prefill_tokens` counts the prefix once for N questions;
  with a warm `state_id` the second call reports `prefill_reused=true` and `prefill_ms ≈ 0`
  (threshold: ≤ 5 ms **[target]**); per-question marginal cost ≈ its own suffix decode.
- **A-E1b-4** Determinism: 3 runs, `threads=1` → identical answers JSON after stripping `timings`
  (SHA-256 equality asserted).
- **A-E1b-5** Waves: `n_seq_max=4`, 8 questions × 4 candidates → ≥2 waves, never above the cap,
  results identical to the single-wave run for the same questions.
- **A-E1b-6** Both readouts: `sequence` and `single_token` implemented; switching changes only the
  readout math; candidate sequences must be pairwise distinct → `E_CANDIDATE_COLLISION`.
- **A-E1b-7** Coverage + reliability: coverage from the full-vocab row; below `coverage_floor` →
  `reliability=low_mass` + `W_LOW_MASS` (tested with a synthetic low-mass logits fixture).
- **A-E1b-8** State save/load: `llama_state_seq_save_file` → fresh context → same answers (≤1e-3);
  corrupt/truncated state file → `E_STATE_LOAD_FAILED`, cache invalidated, no crash.
- **A-E1b-9** No-generation gate: no `llama_sampler_` in `src/`; decode spy asserts
  `decode_calls == 1 + waves` (one prefill + one batch per wave) with `logits=1` exactly on branch
  last tokens.
- **A-E1b-10** CLI end-to-end on a real GGUF (marked `model`): `run` from files and `ask` from flags;
  JSON on stdout; exit codes 0/2/3/4 pinned.
- **A-E1b-11** Typesafe adapter: fixtures from the doc captures produce byte-comparable key sets
  (`answers` keys exactly `type` + value key + `probabilities` + `confidence` (+ `legend` for score));
  native mode additionally exposes `engine`/`timings`.
- **A-E1b-12** Oracle section D green: transplanted `readout` functions match the mirror exactly.
- **A-E1b-13** Error paths: empty state, unknown type, 256 options, 1 or 11 levels, bad criteria shape
  → pinned `E_*` codes, no traceback reaches the user.
- **A-E1b-14** Perf record (report-only): prefill tok/s CPU; warm 4-candidate choice ms on this box for
  Qwen3.5-0.8B CPU — the recon's 14–20 ms with a warm cache is the reference point **[recon]**, this
  milestone publishes its own measured number as **[target]** (no gate; correctness before speed).

### E1c — reasoning resolver + fit

Deliverable: template resolution chain, thinking suppression, `typed-gguf fit`, `docs/TEMPLATES.md`,
end-to-end run on the pinned default model.

- **A-E1c-1** Resolution chain, ordered and documented: (1) GGUF `tokenizer.chat_template` rendered by
  the internal renderer (supported subset), (2) `llama_chat_apply_template` built-ins,
  (3) user override, (4) `E_TEMPLATE_UNRESOLVED` with the fix. Fallback emits `W_TEMPLATE_FALLBACK`.
- **A-E1c-2** Thinking suppression for `enable_thinking`-style templates (Spark/Qwen families): the
  rendered prompt provably contains no think-opener; soft-switch families get the documented marker.
- **A-E1c-3** Post-cue degenerate output (a model that would start reasoning after the cue) never
  affects the readout: answers come from the cue position, proven with a synthetic logits fixture.
- **A-E1c-4** `typed-gguf fit` returns `{n_gpu_layers, n_ctx, kv_type, n_seq_max, est_weights_bytes,
  est_kv_bytes, est_total_bytes, backend, source, standard_n_ctx, ctx_limit}`; `source` is
  `llama-fit-params` when the binary ran, `estimate` otherwise (`W_FIT_ESTIMATED`); cached per
  (model sha, host fingerprint). **v2 (2026-09-23, `docs/SPEC-context-v2.md`):** without `--n-ctx`
  the plan aims at the standard `n_ctx = 32 768`, grows to the largest context the box holds at the
  top rung that reaches it (bounded by the model's own window), and shrinks below the standard with
  `W_CTX_BELOW_STANDARD` when nothing above the floor (`--fit-ctx`) fits; `--n-ctx N` is a pin
  (`min(N, cap)`, never grown) and `ctx_limit` reports which of these happened.
- **A-E1c-5** Fit plan applied on load unless `--no-fit`; over-budget plans downgrade kv_type first
  (`f16 → q8_0 → q4_0`) with `W_KV_TYPE_DOWNGRADE`. **v2:** with a plan applied the *load* uses the
  plan's `n_ctx` unless the request pins `options.n_ctx` (then `min(pin, plan.n_ctx)`); the plan
  stays the request ceiling and a request over the *loaded* context still raises `E_CTX_TOO_SMALL`
  (never a truncation). The KV estimate models sliding-window attention (`window + n_ubatch` cells
  on the SWA layers) and is byte-identical to the all-layer formula for models without a window.
- **A-E1c-6** The fit plan's estimate is cross-checked against the measured load RSS within ±20%
  **[target]**.
- **A-E1c-7** Equivalence (A-E1b-2) holds for the hybrid `qwen35` **and** the default `spark2_5`
  model; recurrent-state handling (`n_rs_seq=0`) documented.
- **A-E1c-8** End-to-end: the pinned default model answers the 4 documented example question sets on
  this box; probabilities + confidence + timings recorded as evidence.
- **A-E1c-9** `docs/TEMPLATES.md` covers `spark2_5`, `qwen35`, `qwen35moe`, `k2-horizon` with template
  source, thinking mode, candidate-label policy and known caveats.
- **A-E1c-10** No network in any decision path (asserted by running all E1c tests with network disabled).

### E2 — benchmarks

Deliverable: `bench/`, `docs/BENCHMARKS.md`, committed labeled dev set, JSON reports.

- **A-E2-1** `bench --suite latency`: `model_load_ms`, prefill tok/s at {256, 2k, 8k}, per-question ms
  at {2, 4, 10} candidates, wave scaling N=1..16, warm cache ms; p50/p95 over ≥5 runs; single command
  reproduces the published table.
- **A-E2-2** `--suite throughput` per backend (CPU, Vulkan, CUDA when available) with the same model.
- **A-E2-3** `--suite quality` on a committed dev set (≥50 items, ≤200 tokens, ≥3 question types):
  exact-match agreement per type + overall with Wilson CIs.
- **A-E2-4** `--suite calibration`: reliability bins, ECE, confidence/coverage correlation.
- **A-E2-5** `--suite determinism`: 3 repeats, byte-identical (timings stripped), per backend.
- **A-E2-6** Recon numbers re-measured side by side and tagged: prefill ~2k tokens 5.9 s CPU /
  0.43 s Vulkan, warm decision 14–20 ms, first Vulkan call 23–30 s shader compile (all **[recon]**;
  re-measure and publish as `[executed]` or explain the delta).
- **A-E2-7** Benchmarks never touch the registry and never hit the network.
- **A-E2-8** `runtime-matrix.yml` includes one non-linux smoke job running the engine (not just the
  oracle).

### E2.5 — auto-calibration + routing

Deliverable: `calibration/`, `typed-gguf calibrate`, `--route auto`, escalation.

- **A-E2p5-1** `calibrate` fits per-(model, question-type) parameters on the labeled set, writes
  `calibration.json`, `--dry-run` prints the table.
- **A-E2p5-2** Calibration is accepted only if ECE (or agreement at a fixed threshold) improves on a
  held-out split; otherwise the tool reports "no calibration applied" and stores nothing.
- **A-E2p5-3** All three confidence modes are measured in one calibration report; the default stays
  `normalized_peak` unless a mode wins by a documented margin.
- **A-E2p5-4** `--route auto` picks (alias, quant, kv_type, n_ctx, n_seq_max) within the device budget
  (asserted against the fit plan) and echoes the reason in `engine` (budget fit, arch capability,
  availability).
- **A-E2p5-5** Escalation is opt-in, logged, bounded by `max_escalations` (default 1), and improves
  agreement on the held-out set when enabled **[target]**; the no-escalation path stays the default.
- **A-E2p5-6** Calibration artifacts are reproducible: same set + same model ⇒ identical params hash.
- **A-E2p5-7** The router never routes a model to a runtime lacking its arch (capability pre-flight).
- **A-E2p5-8** Routing/escalation decisions are present in the response (`engine.route.reason`) and in
  the audit log when `--audit DIR` is set.

### E3 — 27–35B runs

Deliverable: measured runs + docs. No new source required (config only).

- **A-E3-1** A 27B-class model and a 35B-A3B MoE run on this box (8 GB VRAM, 31 GB RAM) with published
  fit plans (MoE expert offload via tensor buffer overrides where needed); prefill tok/s, decision ms
  and peak RSS recorded.
- **A-E3-2** A 20-question batch completes without OOM; waves adapt when `n_seq_max` is constrained.
- **A-E3-3** The E2 quality suite runs on both models; the comparison table (4B default vs 27B vs 35B)
  is published with CIs.
- **A-E3-4** Recommended quant/routing for 8 GB VRAM published (measured, not guessed).
- **A-E3-5** All artifacts are stock GGUFs; SHA-256 recorded; no fine-tuning anywhere (A2 gate re-run).
- **A-E3-6** Long-context smoke: a 32k-token state completes a decision on the 27B/35B hybrid models
  (CPU acceptable) with a recorded time.

### E4 — warm engine host (keep-alive, idle unload, swap)

Deliverable: `src/typed_gguf/keep/`, the `ask`/`run` auto-spawn, `keep status|stop`, `--keep-alive`,
plus the offline pins and the real-model gates.

- **A-E4-1** A second `ask` on the same model inside the window pays **no** model load: the response
  reports `engine.keep.served_by = "host"` and `timings.model_load_ms = 0.0`, and the warm call's
  engine log carries no load line. Cold vs warm wall-clock quoted on the 4B (the warm one ≈
  decision-only).
- **A-E4-2** Idle unload: with `--keep-alive 5s` the host process is gone after the window (asserted
  by pid), and the next ask is cold again.
- **A-E4-3** Model switch A → B → A: B's host is stopped before A loads, `keep status` shows one host
  at a time, and the device is freed between swaps (no two models resident).
- **A-E4-4** No orphans: a `kill -9` of the client mid-request leaves no host behind; a socket left by
  a dead host is cleaned up on the next call rather than blocking it.
- **A-E4-5** The key is honored: a request differing in any placement-affecting option
  (backend/`n_ctx`/`kv_type`/`n_seq_max`/threads/fit) gets a swap, never an answer from the resident
  host.
- **A-E4-6** `--keep-alive 0` reproduces the pre-E4 behaviour (inline answer, nothing resident), and
  the precedence chain is flag > env > default in both spellings of `0`.
- **A-E4-7** Both exit paths (idle timer, `keep stop`) close the handle and end through `os._exit`
  (the `runtime/teardown.py` discipline); `keep status` reports placement and the device the engine's
  own log proved.

### E5 — `serve` (TypeSafe-compatible HTTP) + `runtime update` (serve wave, 2026-09-24)

Deliverable: the stdlib HTTP server behind `typed-gguf serve` (§2.9), the serve→keep-host client
path, `typed-gguf runtime update|rollback` (§2.8), and the offline pins + host legs below. Wire and
semantics: §2.8/§2.9; recon: the thread entry of 2026-09-24 (wieczór) + the `typesafe-sdk` 0.7.1
source [sdk-0.7.1] (a copy lives at `/tmp/tssdk/src/typesafe_sdk` on this box; the live handshake
demo at `/tmp/ts_demo.py`). Implementation cards: `t_f5d8b6c7` (serve), `t_d88b4be0` (runtime update).

- **A-E5-1** **Wire.** An in-process server (no network) answers the four routes with the exact key
  sets of §2.9: the served request bytes are the SDK's (mixed noul/choice/score in one call, all
  three question types), and the field lists are pinned in a committed fixture (e.g.
  `tests/fixtures/typesafe_sdk_0_7_1_fields.json`) derived from the SDK source — request keys,
  per-type answer keys, `ModelMetadata` keys, error shapes — so "no invented fields" is a test, not
  a promise.
- **A-E5-2** **Mapping.** Served answers are byte-equal to `schema.render_response(native_result,
  format="typesafe")` for the same engine result (no serve-local projection, no renormalization,
  noul without `confidence`); `model` echoes the resolved alias; `usage` passes the native counters
  through.
- **A-E5-3** **Errors.** Malformed JSON, missing field, unknown top-level key, unknown model, and
  each engine validation `E_*` answer exactly the §2.9 table (status + FastAPI-shaped body, code
  text in `msg`); a sentinel `Authorization` value never appears in the server's log.
- **A-E5-4** **Same engine.** For a fixed state + questions the served answer equals the same
  request through `ask`/`run` on the same host (offline via the fake host, live on the 4B): same
  model, same fit plan, same calibration source, same 6-sig-fig numbers.
- **A-E5-5** **Cold/warm/serialization.** First request cold-loads through the keep host
  (`served_by = "host"`), warm requests reuse it, `keep stop` still works, two concurrent requests
  serialize in arrival order, and no host/socket/ledger entry is left behind after `serve` exits.
- **A-E5-6** **`runtime update --check`** prints current vs target (tag, build, dir, size) and
  changes nothing — online and offline (offline: a typed message, no partial state).
- **A-E5-7** **A successful update** (fake release metadata + a fake bundle fixture): the record's
  `dir`/`tag`/`build` move to the staged bundle, `previous` names the old one, `doctor` reports the
  new build, and `runtime rollback` moves the record back; both directions pinned.
- **A-E5-8** **Failure paths** (deterministic, offline, fake broken bundles): failed download, a
  bundle missing a required file/symbol, a bundle whose build is below the arch rule, a switch
  refused because the host would not stop — each leaves `runtime.json` and the working runtime
  byte-identical, and removes only the staging debris.
- **A-E5-9** **Refusals.** `runtime update` on a `$TYPED_GGUF_RUNTIME_DIR`-managed runtime and on an
  empty data home answer `E_UPDATE_UNAVAILABLE` / `E_RUNTIME_MISSING`; `runtime rollback` with no
  `previous` answers `E_UPDATE_UNAVAILABLE`; nothing changes in any of the three.
- **A-E5-10** **Gates.** The offline suite stays green with the new code in the tree (the live
  download leg is `@pytest.mark.network`, skipped by name offline); the SPEC-reading gates and the
  docs gates are green; the impl cards' own README/`--help` flips are theirs, not this card's.

**Test plan.** Offline pins (CI shape) — the wire and mapping gates drive the in-process server with
the recorded SDK requests; the update gates drive `install`-level functions against synthetic locks
and tar bundles (the convention of `tests/test_runtime_install.py`), with the GitHub API leg behind a
fixture the offline tests never touch. Host gates the coordinator runs:

```bash
# (b) runtime update — the pinned b11026 -> the newest asset-bearing release, doctor green, build changed
cd "$(git rev-parse --show-toplevel)"   # the repo root; the checkout's directory name is not part of the contract
uv run typed-gguf doctor --json | tee /tmp/upd-before.json        # record the current build
uv run typed-gguf runtime update --check                          # current vs target; touches nothing
uv run typed-gguf runtime update                                  # download -> probe -> atomic switch
uv run typed-gguf doctor --json | tee /tmp/upd-after.json         # green; build differs from before
uv run typed-gguf runtime rollback && uv run typed-gguf doctor --json | tee /tmp/upd-back.json

# (a) serve + the real SDK on the 4B — mixed noul/choice/score, answers parsed by SDK 0.7.1
cd "$(git rev-parse --show-toplevel)"   # the repo root; the checkout's directory name is not part of the contract
python3 -m venv /tmp/ts-venv && /tmp/ts-venv/bin/pip install 'typesafe-sdk==0.7.1'
uv run typed-gguf serve --host 127.0.0.1 --port 8088 &            # warm host via §2.12
TYPESAFE_API_KEY=local TYPESAFE_BASE_URL=http://127.0.0.1:8088 \
  /tmp/ts-venv/bin/python tools/host_gate_serve.py                # exits non-zero on any mismatch
uv run typed-gguf keep stop
```

`tools/host_gate_serve.py` (the impl card commits it; `tools/host_gate_*.sh` is the house convention)
creates/uses that venv, fails loudly on any SDK version other than 0.7.1, prints the three parsed
answers and exits non-zero on a mismatch. Both host legs are the acceptance; the container rehearses
the offline halves only.

---

## 6. Reference values (executed)

From `python3 docs/verify_runtime_contract.py` (this box, 2026-09-17; live section against the
downloaded `llama-b11026-bin-ubuntu-x64` bundle):

```
PoC                                  qwen35 fork Δ=0.0
llama.cpp release                    b11026 (2026-09-17T13:31:47Z)
assets / ubuntu-x64 bytes            33 / 16855810
default model pin                    XHToken/Spark-X2.5-4B-GGUF:Q8_0 4375021152 B
llama-cli version                    version: 0.4.1-dev (build 11026, commit b49650adb)
softmax([2,1,0])                     0.665241, 0.244728, 0.090031
confidence choice/returns-ticket     0.4000 vs doc 0.39
confidence choice/return_reason      1.0000 vs doc 1.0
confidence choice/shipping_issue     0.5375 vs doc 0.53
confidence choice/requested_resolution 0.1600 vs doc 0.16
confidence choice/tone               0.8800 vs doc 0.88
confidence score/bug_severity        0.5500 vs doc 0.54
confidence score/spinner+examples    0.9100 vs doc 0.91
score = Σ i·p_i                      1.30 / 1.06 / 0.0 reproduced
coverage(cands={0,1})                0.880797
candidate z (-1.0 x3, norm=1)        -1.0000
spark2_5 KV/token/seq (f16|q8_0)     147456 | 73728 B
recommend_quant 8GiB                 Spark-X2.5-4B-Q8_0.gguf kv=q8_0 total=7.33 GB
recommend_quant 8GiB small-ctx       Spark-X2.5-4B-Q8_0.gguf kv=f16 total=6.12 GB
recommend_quant huge ctx             None kv=None placement=insufficient
local Spark arch                     spark2_5 file_type=7 (MOSTLY_Q8_0)
Qwen3.5-0.8B header                  qwen35 layers=24
```

Live probes additionally: 32/32 required `libllama.so` symbols resolved via ctypes, 2/2 `libggml.so`
loader symbols, `libllama.so` carries the `spark2_5` implementation, `llama-fit-params` present, local
`Spark-X2.5-4B-Q8_0.gguf` SHA-256 == the HF `lfs.oid` pin.

Other **[recon]** numbers this SPEC relies on (not re-run by the oracle; sources in the thread and
`docs/evidence/poc_report.json`): model load 0.67 s; prefill 44 tok = 279 ms CPU; one batched decode of
2 branches (63 tok) = 397 ms CPU; prefill ~2k tokens = 5.9 s CPU / 0.43 s Vulkan (~4.6k tok/s); warm
decision 14–20 ms; first Vulkan call 23–30 s (shader compilation); stock 7B ≈ 73.8 % vs Jev 86.6 % on
the vendor's public eval; frozen Qwen3.5-4B ≈ 0.845 vs 0.883 modal agreement on the alignable subset
(the last two are the aspiration to re-measure on our stack — **[target]**, never a claim).

---

## 7. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Official release assets move/vanish; upstream API drift | Pin tag + asset names + sizes + own SHA-256 in `runtime/pins.py`; mirror the pinned bundles into our own GitHub release (still no compile); the oracle fails loudly when an asset disappears |
| R2 | First Vulkan call pays 23–30 s shader compilation **[recon]** | `init` and `serve` perform a warm-up decode (recorded as `warmup_ms`); `W_VULKAN_WARMUP` documents it; benchmarks report warm and cold |
| R3 | `spark2_5` KV is fat: 144 KiB per token per fork (f16, 36×4 heads × 256) | Conservative planner + `kv_type` downgrade chain + waves + A-E1a-8 measurement; default recommendation on 8 GB is Q8_0 weights + q8_0 KV (7.33 GB est.) |
| R4 | Frozen models may put little mass on candidate tokens (masked readout looks confident) | `coverage` from the full-vocab row + `reliability` + `W_LOW_MASS`; E2 measures agreement, not just confidence; per-family candidate policy in E1c |
| R5 | A model ignores the answer cue and "wants" to talk | Sequence scoring with `length_norm`, stronger cue variants, per-family policy (§2.3/E1c), E2 quality suite as the arbiter |
| R6 | No jinja in the C API → arbitrary chat templates | Resolution chain (§5/E1c-1) + conformance tests against `llama_chat_apply_template` built-ins + user override + actionable error |
| R7 | CUDA did not load on this box; `nvcc/glslc/ninja` absent | Vulkan is the measured default; CPU always works; CI matrix covers CUDA builds; `doctor` reports the detected backend honestly |
| R8 | Float drift across backends/thread counts | Determinism pinned to (runtime, backend, `threads=1`); cross-backend equality is never promised; answers rounded to 6 significant decimals |
| R9 | The adapter target's docs change | Shapes pinned from captures in the oracle/fixtures; adapter marked best-effort; the quickstart confidence outlier is documented instead of papered over |
| R10 | "No text generation" is misread (we do bounded decode) | Precise wording (§2.3/§2.11) + grep/spy gates; `usage.output_tokens` documented as decode steps |
| R11 | State may carry PII/sensitive data | Local-only by default; HTTP binds 127.0.0.1; no telemetry; states cached only with `state_cache=true`; `--audit` is opt-in and documented |
| R12 | Windows/macOS file naming and loader differences | `finder` handles `.so/.dylib/.dll` + `RTLD_GLOBAL` differences; CI smoke job per platform; the oracle's live section is the acceptance |
| R13 | `n_seq_max` mis-sizing causes silent wrong answers | Waves never exceed the cap; the fork-equivalence test runs per model family; a `E_SEQ_MAX_EXCEEDED` guard instead of clamping |
| R14 | A drop-in client's timeout (SDK: 10 s/request [sdk-0.7.1]) versus a cold model load | `serve` routes every decision through the keep host (§2.12) and refreshes its window; the docs advise a long-lived `serve` (or one warm `ask`) before the first SDK call; the SDK's own retry budget (30 s) absorbs a typical cold load |
| R15 | Upstream llama.cpp renames/retags release assets, or a milestone release (`v0.5.0`) sits at `releases/latest` with no bundles **[recon 2026-09-24]** | `runtime update` never uses `releases/latest`: it takes the newest release whose asset list carries the host's retagged pinned name, probes the bundle against `runtime.lock` before any switch, refuses with `E_UPDATE_UNAVAILABLE` when the name is gone, and keeps the previous bundle for `runtime rollback` |

---

## 8. Decisions for ratification (S-1..S-12, plus the serve wave S-13..S-17)

- **S-1** Name `typed-gguf` (package + CLI + repo). Rename before publication = 1 commit.
- **S-2** Runtime: **ctypes → official llama.cpp bundle** primary; `llama-cpp-python` optional compat;
  own CI wheels only as fallback (operator decision, incorporated).
- **S-3** `confidence` = normalized peak by default, with `entropy`/`margin` modes; **no parity claim**
  with the adapter target; the documented outlier is recorded (§2.4/§2.6).
- **S-4** Memory planning stays conservative (`n_ctx × n_seq_max`) until A-E1a-8 measures the unified
  cache; the recommender may relax only with recorded evidence.
- **S-5** Default readout = `sequence` scoring with `length_norm = 1.0` (not single-token letters).
- **S-6** Default `kv_type = auto` with the downgrade chain `f16 → q8_0 → q4_0`.
- **S-7** Escalation off by default in E1/E2, enabled in E2.5 (`max_escalations = 1`).
- **S-8** HTTP default `127.0.0.1:8088`, no auth, `--host` change warns.
- **S-9** MCP tool names frozen: `typed_gguf_decide`, `typed_gguf_models_list`, `typed_gguf_models_pull`,
  `typed_gguf_runtime_status`, `typed_gguf_fit`.
- **S-10** Quality dev set: we author ≥50 items ourselves (no vendor eval reuse); provenance documented
  in `docs/BENCHMARKS.md`.
- **S-11** E2 quality is report-only in v1 (no minimum agreement threshold); a floor is set after the
  first honest measurement, with the vendor-eval numbers as `[target]` reference only.
- **S-12** State caching on (`state_cache=true`), `save_state=false` (opt-in persistence).

**Serve wave (2026-09-24, cards `t_a51b1205` → `t_f5d8b6c7` → `t_d88b4be0`) — settled here so the
implementation cards do not re-litigate them.**

- **S-13** The served surface is exactly §2.9: four routes, the TypeSafe wire as `typesafe-sdk` 0.7.1
  reads it [sdk-0.7.1], FastAPI-shaped errors, any/absent `Authorization` accepted and never logged,
  FastAPI-shaped `422` bodies. No third-party runtime dependency: stdlib server.
- **S-14** A served decision is a keep-host request — one model, one home, one at a time, swap
  discipline included (§2.12). `serve` never loads a model itself and never projects an answer
  locally: the numbers are `render_response(..., format="typesafe")`'s (§2.6).
- **S-15** `model: "jev-latest"` resolves to the registry's `current` alias; `/v1/models`
  `release_date` is the file's mtime (UTC date) and `description` is one honest auto line — no
  invented upstream metadata. `usage` is the native projection's two counters (`input_tokens` = the
  request's prompt tokens, `output_tokens` = decode steps); nothing is billed, nothing is fabricated.
- **S-16** `runtime update` moves the *installed* bundle to a newer official release; it never
  rewrites `runtime.lock`, never runs automatically, and knows no per-fork branches. The pinned
  bundle stays what `init` installs and what the oracle's pins speak about.
- **S-17** Update safety: staging → lock probe → stop-the-host → atomic record switch; the previous
  bundle is retained and `runtime rollback` returns to it; any failure leaves the working runtime
  byte-identical; refusals are one code, `E_UPDATE_UNAVAILABLE`.

---

## 9. Definition of done for this card + handoff to E1a (code-tdd)

Done when: `SPEC.md` committed; scaffold committed (`pyproject.toml`, `README.md`, `LICENSE`,
`src/typed_gguf/*` stubs, `tests/test_scaffold.py`, workflows, `docs/evidence/*`,
`docs/verify_runtime_contract.py`); `uv run pytest -q` green; the oracle exits 0 (offline and live);
the project's coordination thread (live during development, deliberately not part of this
repository) carries this milestone list.

**E1a entry conditions for code-tdd:** S-1..S-12 ratified (or defaults accepted); the oracle is the
first test to make green; `docs/evidence/poc-ctypes-20260917.py` is transplanted into
`src/typed_gguf/runtime/ctypes_binding.py` (structs verbatim — they are verified); no implementation may
add a third-party runtime dependency; `typed-gguf init` must never invoke a compiler (A1).
