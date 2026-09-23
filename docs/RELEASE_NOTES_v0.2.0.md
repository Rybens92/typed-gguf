# typed-gguf v0.2.0 — typed decisions on any GGUF

> Release notes for the v0.2.0 build. What is new here is **context sizing v2** — the context a call
> plans, grows into and loads with, including the `--n-ctx` pin — the three fixes listed below it,
> and a step-by-step walkthrough of a first query in `README.md`. Everything the earlier releases
> measured (the quality tables, the warm-host numbers, the install receipts) is carried forward
> unchanged; every number in it is already published inside this repository and marked where it was
> measured. v0.1.1 is live on PyPI as `typed-gguf`.

## What this is

`typed-gguf` turns a model that can only *generate text* into one that *answers typed questions*.
You give it a **state** (text or structured data) and a map of **typed questions** — `choice`,
`score`, `noul` (a yes/no with a probability) — and it gives back typed **answers** with full
probability distributions, a confidence scalar and a reliability verdict. No prompt engineering,
no output parsing, no sampling: the answer is read from the logits over a fixed candidate set.

Three properties are the point of the project:

- **No fine-tuning, ever.** Any GGUF that llama.cpp can load is a valid backend; weights are never
  touched (the core has zero third-party runtime dependencies and there is no training code in the
  repository — gates in `tests/test_no_finetune.py` and the oracle).
- **No compiler, ever.** `typed-gguf init` downloads a pinned official llama.cpp release bundle
  (shared libraries driven through `ctypes`) and verifies it. Users never build anything.
- **No text generation.** No sampler chain, no token loop: `decode_calls = 1 prefill + waves`, and
  the parallel readout is *exactly* the sequential one — the proof-of-concept measured
  `max |Δ| = 0.00e+00` against a fresh sequential decode on the same context (SPEC §2.4, A4).

## The headline of this build: context sizing v2

The context a call runs in decides what it can read, and it used to be sized from the request
(`prefix + question + margin`) — so two questions about the same long state could be sized
differently, and a plan was a number nobody could read. It is now planned from the box:

- **The standard is 32 768 tokens.** With no `--n-ctx`, a plan aims at the standard whatever size
  the request itself is.
- **It grows when there is room.** The plan takes the largest context this box holds at the top KV
  rung that reaches it, and never above the model's own window. On the 8 GB-VRAM reference box the
  default plan is **49 763** tokens — `n_ctx: 49763 (standard 32768, grown from the box's free
  memory)`.
- **It shrinks gracefully, with a warning.** When the box cannot hold the standard, the KV rung
  steps down first (`f16 → q8_0 → q4_0`, each step naming `W_KV_TYPE_DOWNGRADE`), then the context
  itself goes below the standard and the plan carries `W_CTX_BELOW_STANDARD`. Measured live: a
  budget that cannot hold the standard (`--fit-target 5200`) plans 4 096 tokens at `q4_0`,
  `ctx_limit shrunk`, warning present — a smaller answer, never a broken one.
- **`--n-ctx` is a pin.** A pinned window is answered as asked (`min(pin, plan)`), the plan reports
  `ctx_limit pinned`, and the pin is what re-keys a warm host. `--no-fit` restores the
  request-sized behaviour for anyone who wants the old arithmetic.
- **The load follows the plan.** A call that pins nothing loads at the plan's context instead of at
  its own request size, so the plan and the load are one number — plus at most one 256-cell block
  of the runtime's own cache padding.
- **Nothing is truncated, silently or otherwise.** A request that does not fit the loaded context
  fails with `E_CTX_TOO_SMALL`, naming the `--n-ctx` that fixes it.
- **Windowed models are charged honestly.** The KV estimate models sliding-window attention
  (`window + n_ubatch` cells on the windowed layers), so a sliding-window model is no longer
  charged roughly four times its real cache. Models without a window keep the previous formula byte
  for byte.

`fit` prints the plan in one line and `fit --json` carries the machine-readable form
(`standard_n_ctx`, and `ctx_limit` = `standard` / `grown` / `shrunk` / `pinned` / `window`), cached
per (model SHA-256, host fingerprint) under the data home. The policy itself is written down in
`docs/SPEC-context-v2.md`; the live receipts are in `docs/evidence/context-v2/README.md` — the
grown default, the shrunk one, a `--n-ctx 32768` pin that really plans 32 768, and a ~6 000-token
request answered at `engine.n_ctx` 50 688 against a plan of 50 620 (the runtime pads the cache to a
256-cell block, +68 cells; the fit-target margin absorbs it).

## The warm engine host (unchanged since v0.1.1)

`run`/`ask` no longer pay the load twice. The first call leaves a **keep host** behind — a detached
child process holding the loaded model and answering over a `0600` unix socket in the data home
(`$TYPED_GGUF_HOME/keep/`, never TCP) — and the next call to the same model is answered warm.
Measured on the operator box (4B `Q8_0`, pinned Vulkan bundle, `tests/test_keep_live.py`, 7 gates in
297.82 s, 2026-09-20):

- **Cold → warm, one host pid 35915.** Cold **17.50 s** (`timings.model_load_ms` **2280 ms**) →
  warm **2.58 s** (load **0 ms**): the warm call is decision-only, no second load, no second fit.
- **Idle unload.** With `--keep-alive 5s` the host was gone **5.3 s** after the window, and the
  device went 6384 → **2314** MiB (resident) → **6409** MiB (unloaded): the unload really frees.
- **Model switch A → B → A.** Pids 37279 → 37591 → 38083: the old host is stopped *before* the new
  one loads, so a swap never holds two models.

The knobs:

- **The window** is 600 s (10 min) by default, configurable with `--keep-alive <dur|0>` (`10m`,
  `30s`, `1h`) on `run`/`ask` or with `TYPED_GGUF_KEEP_ALIVE`. Precedence: **flag > env > default**.
  `--keep-alive 0` is the pre-E4 path exactly — answer inline and unload, so the keep path never
  runs and the response carries no `engine.keep` block; it is also the escape hatch for an A/B
  workflow on a box that cannot hold two models.
- **One model at a time.** One host per data home, keyed by the resolved model plus its SHA and the
  placement-affecting options; a request for a different key gets a swap, never a wrong answer.
- **The control surface** is `typed-gguf keep status [--json]` / `keep stop [--json]`: `status`
  reports the pid, the model, the key it is holding, idle seconds left, and the device the engine's
  own log proved.
- **Who answered** travels in the response: `engine.keep.served_by` is `"host"` or `"inline"`, with
  the host's pid, its one-time `model_load_ms` and the named reason for a fallback. The call that
  spawned the host reports the load it waited for; a warm answer reports `0.0`. If a host cannot be
  reached, the client answers inline on that same call — the CLI never wedges on a host.

Receipts: `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md` (the gate table, the two bugs the live
gates found, the Tier-M sweep over `src/typed_gguf/keep`); offline pins in `tests/test_keep*.py`.

## The three fixes in this build

- **A cached plan can no longer cap a later call.** The plan cache is keyed by model and host, so
  it holds one answer: the one for a request that pins nothing. A request carrying its own knobs
  (`--n-ctx`, `--kv-type`, `--fit-target`) is now computed from scratch instead of being answered
  from — or written into — that entry, and a request that pins nothing recomputes whenever the
  cached entry answers *less* than the box would plan now. An entry written while the desktop was
  busy no longer holds every later load below the standard, and `--no-fit-cache` is an escape hatch
  rather than a requirement.
- **The live KV check reads the plan's own warning list.** The live gate compared a plan-level
  step-down against the request's warning list; those two lists answer different questions (what
  the plan sized against what the load did), and the comparison could pass or fail for the wrong
  reason. A verification fix, not a behaviour change.
- **`typed-gguf doctor` no longer dies on a fallback record.** The human-readable report indexed a
  key the report never builds, so any record carrying a fallback reason raised `KeyError`,
  truncated the report and exited 4. It reads the report's own key now, and prints `none` when
  there is no probe to ask.

## Measured highlights

**Quality on our own 60-item labeled dev set** (`src/typed_gguf/bench/devset.jsonl`, authored in
this repository, provenance recorded; Wilson 95 % intervals; `docs/BENCHMARKS.md`):

| row | model | agreement | notes |
| --- | --- | --- | --- |
| default row, policy v2 (§2.3) | Spark-X2.5-4B-`Q8_0` (the default model) | **50/60 = 0.833** (0.720–0.907) | measured with **no policy flag at all**, item-for-item identical to the E3e arm it was spliced from |
| best E3e cell (§9) | same 4B | 51/60 = 0.850 | `json_instructed` + `answer_sheet`; the pre-v2 cell (`shipped` + `answer_sheet`) reads 42/60 |

**The rescue that decided the defaults.** Two switches — `--chat-format role_split` (the question
becomes its own user turn, rendered by the model's own chat template) and `--cue json_instructed`
(the ask line *is* the JSON contract, and the assistant turn is prefilled with the opened field) —
move both 35B-A3B hybrids out of a collapse, on the very same 60 items
(`docs/BENCHMARKS.md` §7.4.2, `docs/evidence/e3e_role_split_t_9bcbecff.md`):

| model (arch `qwen35moe`) | pre-v2 policy (corrected instrument) | `role_split` + `json_instructed` | paired result |
| --- | --- | --- | --- |
| Tiel-Coder-35B-A3B (20.8 GB) | 22/60 = 0.367 (0.256–0.493) · 59/60 cue refusals · coverage median 5.8e-04 | **53/60 = 0.883** (0.778–0.942) · 0 refusals · coverage median 0.998 | +0.517 [0.353…0.680], exact McNemar p = 7.8e-07 |
| Occamy 1.0 (24 GB `Q4_K_L`, 48 experts) | 26/60 = 0.433 (0.316–0.559) · `low_mass` 60/60 | **54/60 = 0.900** (0.799–0.953) · `low_mass` 0/60 | measured on the host, same instrument |

Both models are stock, frozen GGUFs on an 8 GB-VRAM box. The failure mode was never the weights:
the pre-v2 prompt shape let every cue row close the turn, so the engine correctly refused to read
an answer out of it (`W_CUE_REFUSED`). Policy v2 promotes the measured-good cell to the default;
the pre-v2 cell is one flag away (`--cue shipped --chat-format answer_sheet`) and still reproducible.

**Speed on the operator host (Vulkan, RTX 3060 Ti, pre-policy-v2 rows — `docs/BENCHMARKS.md` §3.8
marks them):** prefill 256 / 2 048 / 8 192 tokens at 2 436 / 2 786 / 2 505 tok/s, 92 / 157 / 234 ms
per question at 2 / 4 / 10 candidates, model load p50 1 094 ms, and a warm-cache request at 129 ms
of question time with `prefill_reused: true`.

## Install

```bash
git clone https://github.com/Rybens92/typed-gguf && cd typed-gguf
uv sync                # or: python3 -m venv .venv && .venv/bin/pip install .

uv run typed-gguf init                                        # pinned llama.cpp bundle, ~30 MB
uv run typed-gguf models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0 # 4.38 GB, SHA-256 verified
uv run typed-gguf ask --state "…" --choice "area=…:a|b|c"
```

Requires Python 3.11+ (no compiler, no CUDA toolkit, no build step); data lives in
`~/.local/share/typed-gguf` (`$TYPED_GGUF_HOME` / `$XDG_DATA_HOME` override it). Model licenses are
recorded in the registry as they are pulled, so the author's terms travel with the file.

**Without a clone**, the wheel carries its own pinned `runtime.lock` and works from any directory:

```bash
uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf version
uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf init          # pinned runtime
uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf doctor --json
```

The honest limit, measured rather than assumed: the install itself is verified out-of-tree — the
artifact gate installs the built wheel into a temp tool env and runs `version` / `init --dry-run` /
`doctor --json` from a neutral cwd that carries a *decoy* lock, and a real
`uvx --from . … typed-gguf init --backend cpu` installed build 11026 with `symbols_ok: True`. The
**git fetch** step of the `git+https://…` spelling needed the published repository — the
post-publish verification this paragraph always promised — and it now has one: the tag form of the
one-liner (`uvx --from git+https://github.com/Rybens92/typed-gguf@v0.1.0 typed-gguf version`)
fetched, built and installed this repository's tagged wheel and printed `typed-gguf 0.1.0`.
Receipt: `docs/evidence/v0_1_0_t_eff926f9_uvx_install.md`. What stays repository-root-only is the
development surface — the test suite and the oracle read `tests/`, `docs/evidence/` and `SPEC.md`,
which no wheel ships.

## What v0.2.0 does not include

- **HTTP and MCP serving.** `typed-gguf serve` and `typed-gguf mcp` are specified (SPEC §2.9:
  `/health`, `/v1/models`, `/v1/decide`, `/v1/systemone`; the `typed_gguf_*` tool set) but they are
  not implemented in this release — both commands exit 3 with a milestone pointer. The CLI is the
  interface that ships.
- **More than one resident model.** The warm host holds one model per data home: a request for a
  different model — or for the same model with placement-affecting options that differ — stops the
  old host *before* the new one loads, so a switch costs a full cold load. That is deliberate on an
  8 GB-VRAM box, where two resident models do not fit; `--keep-alive 0` turns the host off entirely
  and `keep stop` frees the device right now. Two honest edges: the window belongs to the call that
  spawned the host (a later call that merely reuses it does not change it), and nothing supervises
  a host that dies — the next call just pays a cold load. On a platform without unix sockets
  (Windows) `run`/`ask` answer inline with `W_KEEP_UNAVAILABLE`; no daemon is attempted.
- **Cross-question batching.** Questions in one request are decided sequentially (the parallelism is
  per candidate inside a question); a request's decode cost grows additively with question count.
- **Post-v2 re-measurement of everything else.** The latency, throughput, determinism, calibration
  and routing tables were measured under the pre-v2 prompt policy and stay published as that policy,
  each row marked, until the optional E2-v2 campaign re-measures them. The quality row, the Tiel row
  and the Occamy pair *are* v2.
- **A vendor comparison.** The dev set is ours and the agreement numbers are measurements with
  intervals, not a leaderboard entry; `docs/BENCHMARKS.md` §4 lists what the tables do not claim.
- **Big-model speed.** A 23–24 GB MoE on an 8 GB-VRAM host decides at ~0.3 tok/s decode; that is
  the box, measured, and the fit plan and routing recommendation say so instead of hiding it.

## License and credits

MIT (this project and the pinned llama.cpp release). `typed-gguf` builds on the work of others and
names it in `README.md` → *Credits and attribution*: llama.cpp (runtime), TypeSafe (the documented
wire shape this adapter mirrors — **no affiliation, no parity claim**), the "System One"
decision-readout line of work (`rorshopping/parallel-decisions`, `TheoLeeCJ/openjev`,
`bnsd55/openjev`), the RLCD prior art that showed decision behaviour in a small frozen model, and
the default model's authors (Apache-2.0).

## Verification

- **Offline suite, the shape CI runs** — `env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1
  TYPED_GGUF_BENCH_RUNTIME_DIR=<offline bundle> uv run --extra dev pytest -q -rs --timeout=120` →
  **1 609 passed, 57 skipped, 0 failed** in 43 s. The skips are the live cases that want a GPU, a
  model file or `--run-network`; nothing in the suite touches the network.
- **The two release gates.** `tests/test_public_docs.py` (17 tests) pins this file, the README and
  the packaged version together, including the numbers nobody may re-quote from memory. The second
  gate, `tests/test_release_publish.py` (15 tests), parses the publish workflow, executes its
  version gate against a fake `dist/`, and pins the four spellings of one version: `pyproject.toml`,
  `typed_gguf.__version__`, the name of these notes and the tag the release must carry. From this
  checkout, `uv run typed-gguf version` prints `typed-gguf 0.2.0`.
- **The live context receipts** (`docs/evidence/context-v2/`) — the default plan grown to 49 763
  tokens (`ctx_limit grown`), the shrunk one (`--fit-target 5200` → 4 096 @ `q4_0`,
  `W_CTX_BELOW_STANDARD`), a `--n-ctx 32768` pin that really plans 32 768, and the ~6 000-token
  request answered at `engine.n_ctx` 50 688 against a plan of 50 620 — that last one is the exact
  plan-versus-load equality, with the 256-cell pad written down rather than glossed over.
- **Mutation sweep over the module this build rewrote** (`src/typed_gguf/runtime/fit.py`, 2 267
  mutants, soft threshold): 1 335 killed / 805 survived = **62.4 %**, the card's whole changed
  surface scored. The sweep's one semantic gap — the equality boundary of the below-standard
  warning — was closed with a test that fails on that mutant and on nothing else. Receipt:
  `docs/evidence/context-v2/mutation.md`.
- **Independent review** — approve at the v2 head, on a reviewer's own device path and their own
  re-run of the gates; the findings that review raised are the three fixes above.

## Reproduce

Everything above is regenerated from this checkout: `uv run pytest -q` (offline suite + oracle),
`python3 docs/verify_runtime_contract.py`, one command per benchmark table
(`python3 tools/e2_reproduce.py --suite <name> --model <path.gguf>`), and the E3e decision tool
(`python3 tools/e3e_roles_decision.py --report <arm.json> …`, the command `.e3e/report.sh` drives).
The warm-host numbers come from the live gate on the real 4B:
`uv run pytest -q --run-network tests/test_keep_live.py -s` (~5 min).
`docs/BENCHMARKS.md`, `docs/TEMPLATES.md` and `SPEC.md` carry the full detail behind every number
here.
