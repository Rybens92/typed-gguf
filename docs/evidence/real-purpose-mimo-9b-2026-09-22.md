# Real-purpose test on MiMo-V2.6-Distill-Qwen-9B (Q4_K_M) **fully GPU-resident** — the same 30 support-inbox items (card `t_d199e09c`, 2026-09-22)

**Question this answers** (owner, Telegram 2026-09-22): test
`bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF` at a Q4 quant **so that the whole model fits on the
graphics card** (Q4_K_M if possible, "żeby nie tracić na jakości"), then run the *same* test as the
4B arm (`t_977ad206`) and the Tiel arm (`t_0d5db2ef`) and say whether the bigger model is better,
by how much, and whether the full-GPU speed is real.

Everything below is executed evidence: every engine call's exit code, wall clock and exact command
line are in `run.log` (and `report_mimo.json`), the per-item comparison against the committed 4B
run is in `compare_vs_4b.json`, and the residency proof is `proof/residency_proof.txt`. The numbers
in §2–§5 are computed from those files by `report_mimo.py`. Nothing here is quoted from a model
card and presented as measurement.

---

## 1. Protocol — identical to the 4B arm; the model (plus two placement knobs) is the variable

**Inputs are the frozen 4B inputs, byte-identical** (copies live next to this document; the
originals under `docs/evidence/real-purpose-4b-2026-09-22/` were not edited):

| file | sha256 |
|---|---|
| `items.jsonl` (30 messages, gold queue/severity/escalate + the decisive cue per item) | `68a8984e9ee6147cd0cbce712957d0314e93a5a1cac28e5e43c1ee46e0d362ba` |
| `questions.json` (the three question texts + label rubrics) | `b3b72582b30933854c0941ccd03b06171f332d69715c289732498ab7ff4802ff` |

**One request per item, three questions answered in one pass** (queue `choice` over 4 labels,
severity `score` over 3 levels, escalate `noul`):

```
uv run typed-gguf run --questions <this dir>/questions.json --state @<item>.txt \
  --model <MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf> \
  --threads 4 --fit-target 512 --out <item>.json --keep-alive 10m
```

**Two documented deviations from the 4B arm — both placement/performance knobs, neither a decision
knob** (same prompt, same labels, same scoring, same readout):

1. **`--threads 4`** — the same deviation the Tiel arm made and for the same reason: the engine
   default is `os.cpu_count()` (24), while the worker sandbox has a 2-CPU cgroup quota. `--threads
   4` is the setting the repository's published Tiel rows use (`docs/BENCHMARKS.md` §7.4/§7.4.2).
2. **`--fit-target 512`** — *this is the knob that makes the owner's request possible*, and the
   choice is argued in §2: at the engine's default margin (1024 MiB) the fit plan offloads 29 of
   the model's 32 layers; at 512 MiB it offloads **32/32 — the whole model on the card**. Nothing
   else about the plan changes (n_ctx 4096, n_seq_max 8, `kv_type auto` → `q4_0` by the ladder).

**Everything else is the 4B arm's protocol:** readout `sequence`, cue `json_instructed`,
`chat_format role_split` (contract `question`), `calibrated: false`, coverage floor 0.10, default
fit planning, warm `keep` host (one model load for the whole run, no reloads), one call per item,
exit codes asserted by the driver (a run that cannot fail is not a gate: `run_mimo.sh` exits 1 on
any non-zero exit **or** an exit-0 call that leaves no payload).

**What the engine reported, verbatim, on every request** (from the responses; `report_mimo.json`
carries the full blocks and the `protocol_checks.pass` block asserts each line):

| what | reported |
|---|---|
| engine | `typed-gguf 0.1.0` · runtime `llama.cpp b11026` (bundle `linux-x64-vulkan`) · backend `vulkan`, effective `vulkan`, devices `CPU, CPU_Mapped, Vulkan0, Vulkan_Host` |
| model | `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf` — 5 841 049 120 B, sha256 `4bca6f18c73f72270c7a20c2ea2bea581de8246e318714277120369d34048c81` (= the HF LFS `sha256` for that file, `hf_api_manifest.json`) |
| GGUF facts | `general.architecture = qwen35`, `general.name = MiMo V2.6 Distill Qwen 9B`, base model `Qwen/Qwen3.5-9B` (the HF repo is tagged `mimo_v2`; the container's own architecture field is `qwen35`) · 32 blocks · `context_length` 262 144 (declared) · 16 heads, 4 KV heads, key/value length 256 · 427 tensors · quant `Q4_K - Medium` |
| renderer / fallback | `template: {kind: builtin, renderer: builtin, source: llama_chat_apply_template, family: qwen35, thinking: suppressed}` + **`W_TEMPLATE_FALLBACK` on every request** — "the internal renderer rejected this template; the runtime's built-in family table rendered it" (the same bridge the Tiel arm reported); `chat_format: role_split` (contract `question`, question turn `user`, prefix 365–473 chars) |
| placement | fit `n_gpu_layers **32 / 32**`, `degraded: false`, `attempts: []`, `cpu_only: false`, `kv_type q4_0`, `n_ctx 4096`, `n_seq_max 8`; fit warnings `W_KV_TYPE_DOWNGRADE` only; fit source `llama-fit-params`; host fingerprint `a8d49ce94b68e6f8` |
| readout / cue | `readout: sequence`, `cue: json_instructed` (every item) |
| n_ctx per request | 256 (all 30 items; the plan's 4096 cap was never the binding limit) |
| honesty flags | `calibrated: false` (raw probabilities, nothing fitted) · cue verdicts `answered` on every question · `low_mass` / `low_confidence` / `refused` never fired (§3) |
| serving | one warm keep host served every call (`served_by: host`, `threads: 4`, pid 68542, one model load) |

Run window: **2026-09-22T16:26:57Z – 16:27:34Z UTC, all 30 items**, one model load
(`calls_paying_a_model_load: 1`).

**Where this ran.** The kanban worker sandbox mounts the host's `~/coding-pipeline` at `/work`, so
each `/work/t_d199e09c/...` path in `run.log` is `~/coding-pipeline/t_d199e09c/...` on the
operator host; the GPU is the host's RTX 3060 Ti, passed through to the container (`nvidia-smi`
inside the sandbox reports the same device). The model file lives in the card workspace
(`~/coding-pipeline/t_d199e09c/models/`) because `~/.hermes/models` is read-only inside the
sandbox — production placement is a host step and is still pending.

## 2. Full-residency proof — the owner's actual ask

**The card's GPU is 8 GB and the quant is 5.84 GB of weights.** Fit plans for the same model and
host, at 4096 ctx unless stated (all probes are committed under `fit/`):

| fit invocation | layers offloaded | plan | verdict |
|---|---|---|---|
| Q4_K_M `--n-ctx 4096` (engine default margin 1024 MiB) | **29 / 32** | kv q4_0, budget 5850 MiB, warnings `W_KV_TYPE_DOWNGRADE`, `W_FIT_DOWNGRADE` | not fully resident |
| Q4_K_M `--n-ctx 4096 --fit-target 512` ← **this arm** | **32 / 32** | kv q4_0, budget 6342 MiB, `W_KV_TYPE_DOWNGRADE` | **fully resident** |
| Q4_K_M `--n-ctx 8192 --fit-target 512` (longer-context probe) | 31 / 32 | budget 6342 MiB | not fully resident at 8K |
| Q4_K_M `--n-ctx 8192 --fit-target 256` (longer-context probe) | 32 / 32 | budget 6599 MiB | fully resident at 8K, 256 MiB margin only |
| Q4_K_S (5.48 GB) `--n-ctx 4096`, default margin | **31 / 32** | kv q4_0, model 5218 MiB, budget 5842 MiB | not fully resident either |
| IQ4_XS (5.23 GB) `--n-ctx 4096`, default margin | **32 / 32** | kv **q8_0**, model 4974 MiB, budget 5841 MiB | fully resident at the default margin |

**Why Q4_K_M and not a smaller Q4 file — measured, not estimated.** The fallback ladder was probed
for real (both smaller files downloaded from the same HF repo, sha256 checked against the manifest):
at the engine's default 1024 MiB margin the largest Q4-family file of this model that is fully
resident is **IQ4_XS (5.23 GB)** — its immediate neighbour Q4_K_S (5.48 GB) still needs 31/32 layers.
The owner asked for the *largest* Q4 that fits entirely on the card, so this arm kept Q4_K_M and
bought the residency with the placement knob that exists for exactly this: `--fit-target`. The cost
is stated plainly: 512 MiB of margin instead of 1024 MiB, and the KV cache at q4_0 instead of the
q8_0 a smaller quant could afford. The measurement below shows what actually remains free.

Alternative, if the default margin must be preserved: **IQ4_XS is the fully-resident-at-default
choice** (5.23 GB of weights, kv q8_0) — it was fit-probed here but **not** run through the 30
items; this arm's 30-item numbers are Q4_K_M at `--fit-target 512`.

**The proof, verbatim from `proof/residency_proof.txt`:**

```
-- nvidia-smi before the load (desktop/voice-app floor already counted) --
NVIDIA GeForce RTX 3060 Ti, 8192 MiB, 1038 MiB, 6854 MiB

-- fit plan for the placement this load will use --
cmd: uv run typed-gguf fit <model> --print --json --no-cache --n-ctx 4096 --fit-ctx 4096 --fit-target 512
{"n_gpu_layers": 32, "n_ctx": 4096, "kv_type": "q4_0", "n_seq_max": 8,
 "est_weights_bytes": 5829033984, "est_kv_bytes": 150994944, "est_total_bytes": 6534725632,
 "budget_bytes": 6650068992, "source": "llama-fit-params", "warnings": ["W_KV_TYPE_DOWNGRADE"],
 "notes": ["memory table from b11026-linux-x64-vulkan/llama-fit-params (model 5559 MiB, context 177 MiB, compute 529 MiB)"]}

-- one real request through the driver's exact invocation --
cmd: uv run typed-gguf run --questions .../questions.json --state @.../items/t07.txt --model <model> --threads 4 --fit-target 512 --out .../t07.json --keep-alive 1m
exit=0 wall_ms=8063

-- nvidia-smi while the model is resident --
8192 MiB, 5907 MiB, 1985 MiB

-- keep host's own placement block (keep status --json) --
{"model": "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M", "state": "running", "pid": 69762, "requests": 1,
 "model_load_ms": 1456.2386369998421,
 "placement": {"note": "fit plan: 32 layer(s) offloaded, kv_type=q4_0", "n_gpu_layers": 32,
               "kv_type": "q4_0", "degraded": false, "attempts": [], "warnings": [], "cpu_only": false},
 "devices": {"devices": ["CPU", "CPU_Mapped", "Vulkan0", "Vulkan_Host"], "effective_backend": "vulkan"}}

-- the response's own engine block (from the payload it wrote) --
"engine.n_gpu_layers": 32, "engine.placement": {"n_gpu_layers": 32, "kv_type": "q4_0",
    "degraded": false, "attempts": [], "cpu_only": false, "note": "fit plan: 32 layer(s) offloaded, kv_type=q4_0"},
"engine.fit": {"n_gpu_layers": 32, "n_ctx": 4096, "kv_type": "q4_0", ...}
```

Read that as: the plan's `n_gpu_layers` (32) equals the model's total layer count (32, from the
GGUF header), the loading host also reports 32 offloaded with `degraded: false` and no retry
attempts, and while loaded the card shows **5907 MiB used / 1985 MiB free** against a 1038 MiB
pre-load floor — i.e. ~1.9 GiB of the card still free with the whole model, its KV cache and its
compute buffer resident. Nothing was reduced after the fact; the only downgrade in the plan is the
KV cache type (`W_KV_TYPE_DOWNGRADE`: f16 → q4_0, 144 MiB instead of 512 MiB at 4096 ctx — the
measured q4_0 and q8_0 plans are 144 and 272 MiB for this model's 32 layers × 4 KV heads × 512 dim), which
is a *plan* choice, not an allocation failure — `degraded: false` means the loader never had to
give up layers.

## 3. Results (n = 30, one run per item)

| question | accuracy | Wilson 95% | baseline |
|---|---|---|---|
| `queue` (choice, 4 options) | **27/30 = 90.0%** | 74.4 – 96.5% | majority class 26.7% |
| `escalate` (noul, p>0.5) | **25/30 = 83.3%** | 66.4 – 92.7% | majority class 53.3% |
| `severity` (score, 3 levels, argmax) | **23/30 = 76.7%** | 59.1 – 88.2% | — |
| `severity` mean absolute error | 0.267 level (score value 0.306) | | 29/30 within one level |

- **Queue errors (3):** b03 billing→account (conf 0.433); b04 billing→policy (conf 0.657); p05 policy→technical (conf 0.412)
- **Escalate errors (5):** b03 gold True (T1)→False (p 0.173); b07 gold True (T1)→False (p 0.193); a01 gold True (T4)→False (p 0.486); p05 gold True (T2)→False (p 0.176); a03 gold False (T-)→True (p 0.976) — **false negatives 4, over-escalations 1**
- **Escalation safety (the headline since Tiel):** of the **13 tickets whose gold escalation trigger is money or legal (T1/T2)**, the model sent **10 to a human and missed 3** (b03, b07, p05) — recall **10/13 = 76.9%**. The three misses were *confidently* wrong (p_yes 0.17–0.19, not near the 0.5 gate); the fourth false negative is a01 (T4, access at risk) at p 0.486 — 0.014 below the gate.
- **Severity errors (7):** b02 0→2 (1.611); b06 0→1 (0.804); t05 2→1 (1.122); t08 0→1 (0.686); a03 0→1 (0.781); a04 2→1 (1.369); p03 0→1 (0.599)
- **Honesty flags:** `low_mass` 0/30, `low_confidence` 0/30, `refused` 0/30 on all three questions; coverage median 0.99901 (`queue`), 0.99976 (`severity`), 0.99946 (`escalate`), minima 0.98994/0.99254/0.99258. Confidence does separate right from wrong on `queue` (mean 0.823 correct vs 0.501 wrong), and `escalate` probabilities separate the cohorts (mean p_yes 0.731 on gold-True vs 0.273 on gold-False); nothing warned on any of the 30 tickets.

## 4. Comparison against the committed 4B arm (and the Tiel sibling, for context)

`compare_vs_4b.json` — same 30 items, per-item, keyed by id:

| metric (n = 30) | **MiMo 9B Q4_K_M** | 4B (`Spark-X2.5-4B-Q8_0`) | Tiel-Coder-35B (9/40 layers) |
|---|---|---|---|
| `queue` accuracy | **27/30 = 90.0%** | 26/30 = 86.7% | 28/30 = 93.3% |
| `severity` accuracy | 23/30 = 76.7% | **24/30 = 80.0%** | 23/30 = 76.7% |
| `escalate` accuracy | 25/30 = 83.3% | **27/30 = 90.0%** | 24/30 = 80.0% |
| escalate false negatives (of the 16 gold-True) | 4 | **0** | 4 |
| — of which **money/legal (T1/T2, 13 tickets)** | 3 (recall 10/13 = 76.9%) | **0 (13/13 = 100%)** | 4 (recall 9/13 = 69.2%) |
| over-escalations (of the 14 gold-False) | **1** | 3 | 2 |
| net vs the 4B on the same 30 items | `queue` **+1**, `severity` −1, `escalate` −2 | — | `queue` +2, `severity` −1, `escalate` −3 |

- **Agreement with the 4B**: `queue` 27/30 identical (90.0%) — of the 3 disagreements, MiMo was right on 2 and the 4B on 1; `severity` 22/30 identical (73.3%) — MiMo right on 3, 4B right on 4, both wrong on 1; `escalate` 24/30 identical (80.0%) — MiMo right on 2, **4B right on 4**, never both wrong.
- **The safety gate is where this model loses**, and it is the metric that decided the Tiel arm too: the 4B missed **zero** escalations on this set (it errs toward over-escalating — 3 gold-False tickets sent to a human, versus 1 for MiMo), while MiMo missed 4 — 3 of them exactly the money/legal tickets the policy says must reach a human (a duplicate-charge dispute b03, a charge dispute b07, an abuse report p05). Trading one over-escalation for three missed money/legal tickets is the wrong direction for this gate.
- Direction vs the 35B: MiMo is one item worse at routing, one item better at the escalation gate, identical on urgency — and ~6× faster (below).

## 5. Timing and the full-GPU speed sample

| what | MiMo 9B Q4_K_M (all 32 layers on the GPU) | 4B Q8_0 (all 36) | Tiel 35B (9/40) |
|---|---|---|---|
| model load (once) | **1 471 ms** (warm page cache; 3 466 ms for the first-ever load) | 1 183 ms | 18 783 ms |
| per-item engine time (median) | **573 ms** (prefill 100 ms + questions 463 ms) | 645 ms | 5 500 ms |
| per-item CLI wall (median) | **1 006 ms** (min 949, max 8 051 — the max is the item that paid the load) | 1 006 ms | 6 088 ms |
| whole 30-item run (CLI wall, sum) | **37.3 s** | 35.6 s | 231.5 s |
| serving | one warm host, 1 load in 30 calls | one warm host | one warm host (host run) |

**Fully-resident generation speed, method quoted:** instruments are the bundle's own
`llama-bench` (`b11026`, `linux-x64-vulkan`) with
`-m <model> -ngl 99 -p 0 -n 128 -r 3 -t 4 -ctk q4_0 -ctv q4_0 -fa 1 -o json` — i.e. **tg128: 128
tokens generated after an empty prompt, all 32 layers on the Vulkan device, the same KV type the
run's plan chose, flash attention on** (the engine's own ctypes binding enables it,
`src/typed_gguf/engine/session.py:731`, so the bench matches the engine rather than the defaults).
Result: **59.81 tok/s** (sd 0.08; samples 59.90 / 59.77 / 59.76; `speed/llama_bench_tg128.json`).
The run itself generates no free text (readout `sequence` scores fixed label candidates), so tg
cannot be read off the 30 items; the bench completion is the smallest honest instrument for the
"is the full-GPU speed real?" question. Cross-check that it is *not* an artifact of an idle card:
while the model was resident the 30-item CLI run held the card at 5907 MiB used and still finished
each item in ~1.0 s.

## 6. Plain-language verdict (owner-facing)

- **It fits — fully.** The 9B model runs with every one of its 32 layers on the card, no fallback,
  and ~1.9 GB of the 8 GB card is still free while it answers.
- **The speed is real.** ~1.0 s per ticket — the same as the small 4B model and about **6× faster**
  than the 35B (which has to keep most of itself in system memory); generation speed is ~60 tokens
  per second, measured with llama.cpp's own bench at the same full-GPU placement.
- **Quality: a mixed bag, and mostly a wash.** Routing is slightly better than the 4B (27 vs 26 of
  30), urgency is slightly worse (23 vs 24), and the "send to a human" decision is worse (25 vs 27).
- **The safety gate is the problem.** It missed **3 money/legal tickets** (a duplicate charge, a
  charge dispute, an abuse report) that the policy says must reach a person; the small 4B missed
  none on this set (it sent 3 non-urgent tickets to a human instead). For this gate a missed
  escalation costs far more than a wasted one.
- **Bottom line:** worth keeping as a fast, small-footprint model for routing/urgency work, but
  **not** as the model that decides escalations — the 4B stays on that gate.

## 7. Limitations, or what could not be verified

1. **The margin is a choice, not a free lunch.** Full residency at 4096 ctx only happens at
   `--fit-target 512` (1024 MiB margin → 29/32 layers; the fully-resident alternative at that
   margin is the smaller IQ4_XS quant, §2). Measured headroom with the whole configuration
   resident was 1985 MiB free of 8192 MiB, so the placement is comfortable in practice — but a
   heavier desktop (this box's pre-load floor is ~1038 MiB) would eat into that margin first. At
   8192 ctx even 512 MiB is not enough (31/32); 8K full residency needs a 256 MiB margin and was
   **only fit-probed, not run** — the 30 items here are 4096-ctx placements, and no 30-item run was
   made on Q4_K_S or IQ4_XS either (their rows are fit plans, not triage results).
2. **`W_KV_TYPE_DOWNGRADE` is part of why it fits.** The plan moves the KV cache from f16 to q4_0
   (144 MiB instead of 512 MiB at 4096 ctx). That is a cache-precision choice inside the plan, not
   an allocation fallback (`degraded: false`), but it is a quality-relevant detail and is reported
   rather than hidden.
3. **n = 30 with ±10 points of Wilson width** — the point estimates are indicative; 3–4 items
   decide every delta in §4. The escalation-recall gap (0/13 vs 3/13) is the one difference large
   enough to act on, and even that rests on three tickets.
4. **Item authoring** is the 4B card's: 30 synthetic messages with labels frozen before any model
   call; no second annotator. The 4B arm's seams (b04, p05, the a03 over-escalation) apply here
   unchanged.
5. **The renderer path is the shared fallback.** Every request carried `W_TEMPLATE_FALLBACK`
   (the internal renderer rejected the model's shipped chat template; the runtime's built-in
   `qwen35` family table rendered it) — the same bridge the Tiel arm used. The readout itself is
   unaffected (`cue: json_instructed`, all cue verdicts `answered`), but any prompt-shape claim
   about this model is a claim about that fallback rendering.
6. **Not verified:** the image path (mmproj is deliberately unused — text only), the model's tools
   for anything outside this 4-label triage, the `calibrated` half of the class's claims (nothing
   fitted here), and any placement outside this one box (8 GB card, Vulkan, the pinned b11026
   bundle). No engine bug surfaced, so this card adds `docs/` only.

## 8. Evidence inventory, reproduction, gate

| file | what it is |
|---|---|
| `run_mimo.sh` | the driver: one `typed-gguf run` per item, exit codes asserted, wall ms + exact command per item in `run.log`, exit 1 if any item fails or writes no payload |
| `report_mimo.py` | builds `report_mimo.json` + `compare_vs_4b.json` and prints every number quoted above |
| `export_states.py`, `items.jsonl`, `questions.json` | the frozen protocol inputs (byte-identical to the 4B receipt) |
| `run.log` | 30 lines, `exit=0` on every one, with wall ms and the exact command |
| `report_mimo.json` | full engine/fit/timing/answer blocks per item + `protocol_checks.pass` |
| `compare_vs_4b.json` | per-item MiMo vs 4B (and Tiel) comparison, agreement/disagreement, escalation safety cohorts |
| `proof/residency_proof.txt`, `proof/keep_status.json`, `proof/t07.json`, `proof/fit_4096_target512.json` | the full-residency paste in §2 |
| `fit/*.json` | the six fit probes: Q4_K_M at the default margin, at 512 MiB, at 8192 ctx with 512 and with 256, plus Q4_K_S and IQ4_XS at the default margin (`q4ks_default.json`, `iq4xs_default.json`) |
| `speed/llama_bench_tg128.json(.stderr)`, `speed_sample.json` | the tg128 sample and its stated method |
| `hf_api_manifest.json` | the HF repo manifest this arm's sha256/size/quant lines were checked against (Q4_K_M, Q4_K_S and IQ4_XS all verified byte-for-byte) |

**Reproduce** (from the repository root; needs the model file and the b11026 bundle):

```
REPO=. MODEL=~/coding-pipeline/t_d199e09c/models/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf \
REAL_PURPOSE_BASE=/tmp/mimo-run bash docs/evidence/real-purpose-mimo-9b-2026-09-22/run_mimo.sh   # 30 calls, ~40 s
REAL_PURPOSE_BASE=/tmp/mimo-run python3 docs/evidence/real-purpose-mimo-9b-2026-09-22/report_mimo.py
```

Gate run for this card: `env -u PYTHONPATH uv run --extra dev pytest -q tests/test_readout_math.py`
— docs-only change, no engine code touched.
