# typed-gguf

**Formerly `ggufone`** — the public name changed on 2026-09-20 (card t_5f9c15fe): the import package is
`typed_gguf`, the command is `typed-gguf`, the default data home is `~/.local/share/typed-gguf`.
Files under `docs/evidence/` keep the old spelling: they are records of runs, not names.

**System-One-style typed decisions on any GGUF** — inspired by the typed-decision interface (Jev);
no affiliation, no parity claim.

**GGUF-native typed decision engine** — `state` + typed questions → typed answers with full
probability distributions and confidence, computed locally on **frozen** GGUF models.

- **One pass over a shared prefix.** The state is prefilled once; every question forks the sequence
  state (`llama_memory_seq_cp`) and costs only its own short suffix decode.
- **No text generation.** Answers are read from logits over a fixed candidate set (restricted
  softmax), never sampled or parsed.
- **No fine-tuning. Ever.** Any GGUF llama.cpp can load is a valid backend; weights are never updated.
  The core has zero third-party runtime dependencies (stdlib only).
- **No compiler, ever.** `typed-gguf init` downloads a pinned official llama.cpp release bundle
  (shared libraries driven through `ctypes`). Users never build anything.
- **One bundle per process.** Probes (`init`'s fallback chain, `doctor`, the warm-up) run in a
  disposable child (`typed_gguf.runtime.probe_child`) and come back as JSON. Third-party GPU
  libraries and drivers stay out of the command process, so their teardown cannot take the
  command down with it.

Default model: [`XHToken/Spark-X2.5-4B-GGUF`](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF)
`Q8_0`, pinned by size and SHA-256.

## Quickstart

v0.1.0 installs from this repository (there is no PyPI release yet) and needs nothing but
Python 3.11+ — the core is stdlib-only and no compiler is ever invoked:

```bash
git clone https://github.com/Rybens92/typed-gguf && cd typed-gguf
uv sync                # or: python3 -m venv .venv && .venv/bin/pip install .

uv run typed-gguf init                                      # pinned runtime, ~30 MB
uv run typed-gguf models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0   # 4.38 GB, SHA-256 verified
```

Every command below is written `uv run typed-gguf …`: neither install line puts the console script
on your `PATH` (`uv run` uses the project's `.venv`; with a plain venv it is `.venv/bin/typed-gguf`).
`python -m typed_gguf …` is the third spelling of the same entry point.

`init` and `pull` print what they actually verified (run on this box, 2026-09-20):

```
installed: True
variant: linux-x64-vulkan
backend: vulkan
working_backend: vulkan
build: 11026
asset: llama-b11026-bin-ubuntu-vulkan-x64.tar.gz
asset_sha256: 1b40310bf4d47c2c84853ebb4ccaf4dcbd992596cd1c2f610be6a0532a874708
asset_verified: True
bytes_fetched: 30294625
symbols_ok: True
rung: prebuilt
```

```
pulled spark-x2.5-4b-q8_0 -> ~/.local/share/typed-gguf/models/Spark-X2.5-4B-Q8_0.gguf
  repo      XHToken/Spark-X2.5-4B-GGUF@902d86599494
  file      Spark-X2.5-4B-Q8_0.gguf (4.38 GB)
  sha256    5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2 (verified against lfs.oid)
  license   apache-2.0
  arch      spark2_5  quant Q8_0  [quant]
```

`uv run typed-gguf doctor` reads both back — exit `0` = ready, `2` = works with warnings, `1` =
broken. A `2` is not a failed install: on this box `doctor` reports the documented CUDA→Vulkan
pre-flight fallback (an NVIDIA card, a Vulkan bundle) and `model.present` until the pull lands.

Then ask — one state, three typed questions:

```bash
uv run typed-gguf ask \
  --state "The billing page is blank for every user since 09:12. The incident channel is quiet, no deploy in the last 24h, and the API is healthy on synthetic traffic." \
  --choice "area=Which team owns this?:billing|technical|platform" \
  --score  "severity=How severe?:cosmetic|annoying|critical" \
  --noul   "page=Should we page the on-call engineer?"
```

The real response (this box, Vulkan, no policy flags — trimmed only where the `…` marks it):

```json
{
  "model": "spark-x2.5-4b-q8_0",
  "engine": {
    "runtime": "llama.cpp b11026",
    "backend": "vulkan", "backend_source": "bundle",
    "devices": ["CPU", "CPU_Mapped", "Vulkan0", "Vulkan_Host"],
    "effective_backend": "vulkan",
    "readout": "sequence",
    "cue": "json_instructed",
    "chat_format": {"kind": "role_split", "question_turn": "user", "contract": "question",
                    "prefix_chars": 507, "dropped": "\n"},
    "template": {"kind": "gguf-renderer", "renderer": "internal", "source": "gguf:tokenizer.chat_template", "family": "spark2_5",
                 "thinking": "suppressed"},
    "n_ctx": 256, "n_seq_max": 8, "kv_type": "f16", "n_gpu_layers": 36,
    "prefix_tokens": 104, "state_id": "sha256:a5b7…", "prefill_reused": false
  },
  "answers": {
    "area": {
      "type": "choice",
      "choice": "billing", "probabilities": {"billing": 0.73758, "technical": 0.171766, "platform": 0.0906544}, "confidence": 0.606369, "coverage": 0.999454, "reliability": "ok",
      "cue": {"refused": false, "token": 79, "closer": null, "mass": 0.841117, "verdict": "answered", "next": null},
      "decode_steps": 5
    },
    "severity": {
      "type": "score",
      "score": 0.683022, "probabilities": {"0": 0.42261, "1": 0.471758, "2": 0.105632}, "confidence": 0.207637, "coverage": 0.999991, "reliability": "ok", "legend": {"0": "cosmetic", "1": "annoying", "2": "critical"},
      "cue": {"refused": false, "token": 30, "closer": null, "mass": 0.471754, "verdict": "answered", "next": null},
      "decode_steps": 3
    },
    "page": {
      "type": "noul",
      "noul": 0.00470045, "probabilities": {"yes": 0.00470045, "no": 0.9953}, "coverage": 0.999993, "reliability": "ok",
      "cue": {"refused": false, "token": 1643, "closer": null, "mass": 0.995293, "verdict": "answered", "next": null},
      "decode_steps": 2
    }
  },
  "usage": {"input_tokens": 255, "output_tokens": 10, "questions": 3, "forks": 8, "prefill_tokens": 104, "decode_steps": 10, "waves": 4},
  "timings": {"model_load_ms": 6332.95, "prefill_ms": 783.573, "questions_ms": 473.311, "total_ms": 1273.17},
  "warnings": []
}
```

Read it as: **the decision** (`choice: billing`, `score: 0.68` on a 0–2 scale, `noul: 0.0047`),
**the full distribution** (`probabilities`, with `legend` naming the levels of a `score`),
**how concentrated it is** (`confidence`) and **whether the row the answer was read from is really
an answer** (`reliability: "ok"`, `cue.verdict: "answered"`). `usage.decode_steps` counts decode
steps consumed by the readout — nothing is generated. `engine` is the receipt for the rest:

- `engine.cue: "json_instructed"` and `engine.chat_format.kind: "role_split"` are the product
  **defaults** since policy v2 (2026-09-20) — `json_instructed` says the ask line is a JSON
  contract, `role_split` says the question is its own user turn rendered by the model's own chat
  template (`question_turn: "user"`). The pre-v2 cell stays one flag away
  (`--cue shipped --chat-format answer_sheet`) — see `docs/TEMPLATES.md` §4.
- `engine.template` says which template produced the bytes: `kind: "gguf-renderer"` /
  `renderer: "internal"` is the model's own `tokenizer.chat_template` rendered by our renderer,
  `family: "spark2_5"`, `thinking: "suppressed"` (proved on the bytes — `docs/TEMPLATES.md` §3).
  Templates outside the internal Jinja subset fall through the chain to llama.cpp's built-in
  templates (`renderer: "builtin"`, `W_TEMPLATE_FALLBACK`) or to `--template`, and a template
  nothing resolves is `E_TEMPLATE_UNRESOLVED` with the fix in the message.

### The same request as a file

`run` takes a whole request (`--questions q.json`, with `--state`/`--state-json` to override the
state; SPEC §2.5 is the schema):

```json
{
  "state": "The billing page is blank for every user since 09:12. …",
  "questions": {
    "area": {"type": "choice", "instructions": "Which team owns this incident?",
             "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs, crashes, API errors"}},
    "severity": {"type": "score", "instructions": "How severe is it for the customer?",
                 "criteria": ["cosmetic", "annoying", "critical"]},
    "page": {"type": "noul", "instructions": "Should we page the on-call engineer now?"}
  }
}
```

`--out r.json` writes the response to a file; `--format typesafe` emits the adapter's shape.

### Reuse a prefix across calls

```bash
uv run typed-gguf run --questions q.json --state-id billing-incident --save-state   # first call
uv run typed-gguf run --questions q.json --state-id billing-incident                # prefill_reused: true
```

The state file lives under `$TYPED_GGUF_HOME/states/` (default `~/.local/share/typed-gguf`,
`$XDG_DATA_HOME/typed-gguf` when that is set). The second call reports `prefill_reused: true` and
`prefill_tokens: 0`; everything else in the two responses is identical (the two runs differ only in
the plan block's `engine.fit.budget_bytes`, which re-reads free memory, and in the two prefill
fields above).

### The TypeSafe-compatible shape

```bash
uv run typed-gguf run --questions q.json --state-id billing-incident --format typesafe
```

```json
{
  "model": "spark-x2.5-4b-q8_0",
  "answers": {
    "area": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.766797, "technical": 0.220731, "platform": 0.0124716},
             "confidence": 0.650196},
    "severity": {"type": "score", "score": 1.14801,
                 "probabilities": {"0": 0.0825367, "1": 0.686919, "2": 0.230545},
                 "confidence": 0.530378,
                 "legend": {"0": "cosmetic", "1": "annoying", "2": "critical"}},
    "page": {"type": "noul", "noul": 0.000165299,
             "probabilities": {"yes": 0.000165299, "no": 0.999835}}
  },
  "usage": {"input_tokens": 287, "output_tokens": 10}
}
```

Native-only keys (`engine`, `timings`, `coverage`, `reliability`, `decode_steps`, `warnings`) are
dropped by the adapter on purpose, `usage` is reduced to its two documented counters, and an alias
that is not in the registry (`model: "jev-latest"` and friends) is translated to the configured
default. **No parity claim**: the confidence statistic is ours, and the one documented outlier in
the adapter target's docs is reproduced as-is in `docs/verify_runtime_contract.py` (SPEC §2.6).

## Interfaces

| command | what it does | milestone |
| --- | --- | --- |
| `typed-gguf init [--backend auto\|cpu\|vulkan\|cuda\|metal] [--dry-run]` | downloads, verifies, extracts and probes the pinned llama.cpp bundle; `--dry-run` prints the plan | E1a |
| `typed-gguf doctor [--json]` | checks the bundle (files, symbols, build, `llama-fit-params`, backends, accelerator, recorded SHA) and the registry; exit 0 ok / 2 warnings / 1 broken | E1a |
| `typed-gguf models search <q>` / `pull <repo[:quant]>` / `use <alias>` / `ls [--json]` / `rm <alias>` / `verify [alias]` / `recommend-quant [--vram GiB]` | the model registry: resume + SHA-256 verified downloads, the model author's license recorded with the file, and a quant recommendation for a VRAM budget | E1a |
| `typed-gguf fit [<model>] [--json]` | the fit plan for this host (`n_gpu_layers`, `n_ctx`, `kv_type`, `n_seq_max`, `est_*` bytes), cached per (model SHA-256, host fingerprint) and applied on load unless `--no-fit` | E1c |
| `typed-gguf run --questions q.json [--state …] [--format native\|typesafe] [--out r.json]` | a whole request from a file | E1b |
| `typed-gguf ask --state … --choice/--score/--noul "id=instruction:labels"` | the same engine from the command line | E1b |
| `typed-gguf bench --suite latency\|throughput\|quality\|calibration\|determinism --model <path.gguf>` | reproduces the tables in `docs/BENCHMARKS.md`; never touches the registry and never opens a socket | E2 |
| `typed-gguf calibrate [--dry-run]` | fits the per-(model, question-type) temperature/scale on the committed dev set and keeps it only if a held-out split improves | E2.5 |
| `typed-gguf version [--json]` | versions, the pinned runtime tag, the installed runtime and the data home | E0 |
| `typed-gguf serve` / `typed-gguf mcp` | the HTTP (`/health`, `/v1/models`, `/v1/decide`, `/v1/systemone`) and MCP (`typed_gguf_decide`, `typed_gguf_models_list`, `typed_gguf_models_pull`, `typed_gguf_runtime_status`, `typed_gguf_fit`) surfaces are **specified in SPEC §2.9 but not implemented in v0.1.0**: both commands exit 3 with the milestone pointer | — |

`python -m typed_gguf <command>` is the same CLI. Exit codes (SPEC §2.5): `0` ok, `2` user error,
`3` runtime/model error, `4` internal (`doctor` adds `2` for "works, with warnings" and `1` for
"broken"). Every command accepts `--help`.

**One interface, three shapes.** `--format native` (default) is the full response above;
`--format typesafe` is the compatibility adapter; and the same request/response pair is what
`typed-gguf run --questions` and `typed-gguf ask` build internally, so anything the CLI can ask can
be driven from a file.

## Model families and templates

The resolver chain is ordered and every step is published in the response
(`engine.template`): **1.** the model's own `tokenizer.chat_template`, rendered by the internal
renderer; **2.** `llama_chat_apply_template` — llama.cpp's built-in family templates
(`renderer: "builtin"`, with `W_TEMPLATE_FALLBACK`); **3.** an explicit `--template plain | <builtin
name> | <path.jinja> | <inline text>`; **4.** `E_TEMPLATE_UNRESOLVED` with the accepted forms in the
message. `docs/TEMPLATES.md` §1–§2 is the reference.

| family (arch) | measured on | template source | thinking | questions' placement |
| --- | --- | --- | --- | --- |
| `spark2_5` | Spark-X2.5-4B-`Q8_0` (the default) | the GGUF's own template, internal renderer, 5/5 role-split checks | hard-suppressed: `enable_thinking=false` renders a *closed* block, so no think-opener is ever present | role split renders; the shared prefix stops one byte before the template's own trailing newline (`engine.chat_format.dropped`) |
| `qwen35` | Qwen3.5-0.8B-UD-`Q4_K_XL` (dense) | the GGUF's own template, internal renderer, 5/5 | hard-suppressed: the empty block is stripped from the generation prompt | role split renders |
| `qwen35moe` | Occamy 1.0 (24 GB `Q4_K_L`, 48 experts) and Tiel-Coder-35B-A3B (20.8 GB) | same Qwen3.5 family template; Occamy's renders internally (5/5), **Tiel's own template is outside the internal Jinja subset** → the LLAMA built-in bridge renders it (`renderer: "builtin"`, `W_TEMPLATE_FALLBACK`) | hard (inherited) | role split renders for Occamy; for Tiel the live run goes through the built-in bridge (`--template plain` is the offline fallback) |
| `k2-horizon` | no GGUF of this family has been run here — the row is **[recon]** from the published `moonshotai/Kimi-K2-Thinking` template | `<\|im_system\|>`/`<\|im_middle\|>` roles, no `enable_thinking` switch | **soft**: the `/no_think` marker is advisory — that family's thinking is controlled by the serving stack | the `kimi-k2` built-in covers variants our renderer rejects |
| anything else | — | the chain above | — | if the role split cannot be rendered, the run is refused by name (`E_ROLE_SPLIT_UNSUPPORTED`, naming `--chat-format answer_sheet` / `--template plain`) instead of rendering a conversation nobody described |

Notes that apply to every family: the **question id is never sent to the model** (labels are the
option names / level numbers / `yes`·`no` the response reports back); `--readout single_token`
scores only the first token of the label while the prompt stays byte-identical; two candidates that
tokenize to the same sequence are `E_CANDIDATE_COLLISION` (exit 2); and a family with a hard
thinking switch is verified by *stripping the bytes*, not by trusting the kwargs
(`docs/TEMPLATES.md` §3).

## Fit: what this host can actually hold

```bash
uv run typed-gguf fit                                  # the default model, human-readable
uv run typed-gguf fit Spark-X2.5-4B-Q8_0 --json        # {n_gpu_layers, n_ctx, kv_type, n_seq_max,
                                                      #  est_weights_bytes, est_kv_bytes,
                                                      #  est_total_bytes, backend, source}
```

`source` is `llama-fit-params` when the bundle's own tool produced the numbers, `estimate`
otherwise (with `W_FIT_ESTIMATED`). The plan is cached per `(model sha256, host fingerprint)` and
applied on load — `run`/`ask` honour it unless `--no-fit` is passed. Over budget, `kv_type` walks
`f16 → q8_0 → q4_0` (each step warns `W_KV_TYPE_DOWNGRADE`) before the context shrinks; the
estimate is cross-checked against measured load RSS within ±20 % on this box
(`docs/TEMPLATES.md` §5).

## Limitations and known issues

- **Questions inside one request are decided sequentially.** The parallelism is *per candidate
  within a question* (branches packed into waves of `n_seq_max − 1`); there is no cross-question
  batched decode yet, so a request's decode cost grows additively with the number of questions.
  `usage.waves` and `usage.decode_steps` report what a request actually cost.
- **The calibration, routing and latency/throughput/determinism tables are pre-policy-v2** until the
  optional E2-v2 campaign re-measures them. They stay published as the policy their own report
  names — the `shipped` cue + `answer_sheet` cell, i.e. the pre-v2 policy — and `docs/BENCHMARKS.md`
  marks every such row rather than mixing it with a v2 one. What *is* v2: the 4B quality row (§2.3),
  the Tiel row (§7.4.2) and the Occamy pair.
- **`serve`/`mcp` are specified, not shipped** (SPEC §2.9); the CLI is the only interface in v0.1.0.
- **Exotic-platform wheels are future work.** The primary distribution is the pinned prebuilt
  llama.cpp bundle (SPEC §4, rung 1) and rung 1 is the only automated rung in v0.1.0: a
  `llama-cpp-python` wheel matrix for platforms with no official asset was scoped but is not built by
  this release, and the dispatch-only `wheels-fallback.yml` stub was dropped rather than shipped
  half-built. On such a host, point `TYPED_GGUF_RUNTIME_DIR` at a runtime you built yourself and
  `typed-gguf doctor` probes it (rung 3).
- **Thinking suppression is prompt-level.** It is proved on the rendered bytes for the families with
  a real switch; for `k2-horizon` the `/no_think` marker is advisory, because that family's thinking
  is a serving-stack setting (SPEC §2.11, `docs/TEMPLATES.md` §8).
- **Quality numbers come from our own 60-item dev set** (`src/typed_gguf/bench/devset.jsonl`, authored
  in this repo, provenance recorded). They are measurements with Wilson intervals, not a vendor
  comparison; `docs/BENCHMARKS.md` §4 lists what the tables deliberately do not claim.
- **Big models on small boxes are box physics.** A 23–24 GB MoE on an 8 GB-VRAM host decides at
  ~0.3 tok/s decode (measured, §6.5) and its fit plan offloads only what free VRAM allows; a host
  that can keep the weights resident turns the same command into a compute-bound run. Read
  `docs/BENCHMARKS.md` §6/§7 before blaming the engine.
- **A cached fit plan is never re-expanded.** The plan is cached per (model SHA-256, host
  fingerprint) under `$TYPED_GGUF_HOME/fit/` and re-checked against free device memory on every
  load, but that check only walks the plan *down*: a plan degraded for one busy run stays degraded
  after the device frees up (only its `budget_bytes` refreshes). Drop the cache — `--no-fit-cache`
  on a request, `typed-gguf fit --no-cache`, or deleting `$TYPED_GGUF_HOME/fit/` — when free memory
  returns.
- **No CUDA row exists in the published tables** (no CUDA device was reachable when they were
  measured); each table says `measured: false` with the reason instead of omitting the backend.
- **`typed-gguf` is not affiliated with TypeSafe** and makes no parity claim; the `typesafe` output
  format is a compatibility adapter (§2.6).

## Status

`SPEC.md` is the contract (milestones E1a → E3 with numbered acceptance criteria); every number in
it carries a `[executed]` / `[recon]` / `[target]` / `[UNVERIFIED]` tag. `docs/BENCHMARKS.md`
carries the measured tables and `docs/TEMPLATES.md` the template/family contract.

- **E1a (done)** — runtime + model registry: `init` (pinned prebuilt bundle, no compiler on the
  path), `doctor`, `models {search,pull,use,ls,rm,verify,recommend-quant}`, the GGUF header reader,
  the conservative fit planner, and the oracle green with its live section.
- **E1b (done)** — the engine core: `run`/`ask`, `--format native|typesafe`, the fork readout (one
  prefill per state, waves bounded by `n_seq_max`, prefix states cached on disk), the schema/error
  catalog and the TypeSafe adapter. Numbers: `docs/evidence/e1b_perf.json`,
  `docs/evidence/e1b_t_34abf324_engine.md`.
- **E1c (done)** — the template resolver, provable thinking suppression and `fit`, with
  `docs/TEMPLATES.md` as the family contract. Numbers: `docs/evidence/e1c_e2e.json`,
  `docs/evidence/e1c_t_c8e36cad_*.md`.
- **E2 (done)** — the five benchmark suites over the pinned runtime, the committed 60-item labeled
  dev set, and `docs/BENCHMARKS.md`; one command reproduces each table
  (`python3 tools/e2_reproduce.py --suite <name> --model <path.gguf>`).
- **E2.5 (done)** — calibration, `--route auto` and bounded escalation, measured in
  `docs/BENCHMARKS.md` §5.
- **E3 (done)** — the 27–35B runs on this box: Occamy 1.0 (`qwen35moe`, 24 GB), the three-way
  comparison, threads/placement recommendations and the Tiel-Coder row (§6–§7).
- **E3e + policy v2 (done, 2026-09-20)** — the two switches that fix the question's placement and
  the ask line (`role_split`, `json_instructed`) are the defaults; measured grounds in
  `docs/BENCHMARKS.md` §9 and §7.4.2.
- **Not in v0.1.0** — the HTTP/MCP serving surface (SPEC §2.9) and the optional E2-v2 re-measurement
  of the pre-v2 tables.

## Verification

```bash
uv run pytest -q                                    # unit gate (offline: live tests are skipped)
uv run pytest -q --run-network                      # + real HF downloads / real GGUF headers
uv run pytest -q --run-network tests/test_engine_fork.py tests/test_cli.py   # fork equivalence,
                                                    # waves, determinism, state save/load, CLI e2e
uv run pytest -q --run-network tests/test_templates.py tests/test_fit_live.py  # E1c: the real
                                                    # templates + the real fit plan (RSS ±20%)
uv run python tools/e1c_offline_gate.py             # every E1c test with the network disabled
python3 docs/verify_runtime_contract.py             # oracle: pinned facts + formulas
TYPED_GGUF_RUNTIME_DIR=<runtime> python3 docs/verify_runtime_contract.py   # + live ctypes probes
```

The suite is offline by default (A7): live gates are marked and **skip by name** when they cannot
run, and a run that skipped one exits non-zero rather than looking green — including the
`@pytest.mark.needs_fork` gates, which skip under a starved pid cgroup
(`pids.max` shared with sibling sandboxes) and say so instead of reporting "the backend does not
load on this host".

`docs/evidence/e1a_baseline.json` records this box's measured numbers (prefill tok/s, warm decode
ms, KV footprint vs the conservative bound, the pull/resume/SHA transcript).

## Verifying a claim

Every number in `SPEC.md` is tagged: `[executed]` (reproduced in this repo right now), `[recon]`
(measured once, source recorded), `[target]` (to be measured later) or `[UNVERIFIED]` (evidence
missing; must not be quoted as fact — the tag is currently unused). Anything in the E1a evidence
file is `[executed]` unless it says otherwise, and any statement whose tag is missing is a bug in
the document.

## License

MIT. The bundled/used llama.cpp release is MIT as well; model licenses are the model authors'.
`typed-gguf` is not affiliated with TypeSafe; the `typesafe` output format is a compatibility adapter.

## Credits and attribution

`typed-gguf` stands on other people's work, and says so:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) — the inference runtime and the
  C ABI this project drives through `ctypes` (`libllama.so` / `libggml*.so`, pinned release
  `b11026`). `typed-gguf init` downloads the official release bundle; nothing here is built from
  llama.cpp source.
- **[TypeSafe](https://docs.typesafe.ai)** — the documented `state` + typed `questions` →
  typed `answers` wire shape that the `--format typesafe` adapter mirrors, and the interface whose
  "System One"-style typed decisions this project reimplements on frozen GGUFs. `typed-gguf` makes
  **no parity claim** with it: the confidence statistic is our own and the documented quickstart
  outlier is reproduced as-is in `docs/verify_runtime_contract.py`.
- **[rorshopping/parallel-decisions](https://github.com/rorshopping/parallel-decisions)** and
  **[TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev)** /
  **[bnsd55/openjev](https://github.com/bnsd55/openjev)** — the "System One" decision-readout
  idea (score a fixed candidate set instead of generating text) that this project exists to
  reproduce on stock, frozen GGUFs. Their trained models are the reference point, not a
  dependency.
- **[harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)** —
  RLCD-style training that demonstrated decision behaviour in a small model; cited as prior
  art for the *frozen* variant we build.
- **[XHToken/Spark-X2.5-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF)**
  (Apache-2.0) — the default model, used unmodified and pinned by size + SHA-256.

The `license` field of every pulled model is recorded in the registry and printed by
`typed-gguf models pull` / `models ls --json`, so the model author's terms travel with the file.
