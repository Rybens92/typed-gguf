# typed-gguf v0.3.0 — typed decisions on any GGUF

> Release notes for the v0.3.0 build. Two things a user can now *do* with the tool that they could
> not do before: **serve** a typed decision over HTTP to a TypeSafe/Jev client from the warm host
> the CLI already keeps, and **update the llama.cpp runtime `init` installed** — replace it in place,
> or roll back to the one it replaced. Those are the two headline sections below, each with its own
> live receipts. Around them: the platform matrix stopped being scaffolding and is now a measured
> thing (five jobs — Linux cpu/vulkan, Windows cpu, macOS metal/engine-smoke — all green on this head
> in run `36163763039`), CI runs the suite on 3.11, 3.12 and 3.13, and a batch of end-to-end
> findings is fixed rather than filed. Nothing the earlier releases *measured* was re-measured: the
> quality tables, the warm-host numbers, the latency/throughput/calibration tables and the install
> receipts are carried forward unchanged, each marked where it was measured. v0.2.3 is live on PyPI
> as `typed-gguf`.

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
  (shared libraries driven through `ctypes`) and verifies it. Users never build anything — and, as
  of this build, they can also *replace* it without a compiler (see the second headline below).
- **No text generation.** No sampler chain, no token loop: `decode_calls = 1 prefill + waves`, and
  the parallel readout is *exactly* the sequential one — the proof-of-concept measured
  `max |Δ| = 0.00e+00` against a fresh sequential decode on the same context (SPEC §2.4, A4).

## Headline 1 — the built-in server: `typed-gguf serve`, wire-compatible with Jev/TypeSafe

**You can now point a TypeSafe (Jev) client at this machine and get typed decisions back.**

`typed-gguf serve` is a stdlib HTTP server (no third-party dependency, like everything else here) on
`127.0.0.1:8088` that answers four routes — `GET /health`, `GET /v1/models`, `POST /v1/decide`
(native) and `POST /v1/systemone` (the TypeSafe wire) — and it answers them **from the same warm
`keep host` that `run`/`ask` use**: the same model, the same fit plan, the same calibration, the same
numbers. There is no second engine inside the server; the served callable *is* the CLI's own warm
decision path, and a gate asserts it (the served payload and `run`'s payload for the same state and
questions are equal, bar the `format` the CLI was asked for). A client only has to set
`TYPESAFE_BASE_URL` to the server and `TYPESAFE_API_KEY` to any non-empty string.

```bash
typed-gguf serve --host 127.0.0.1 --port 8088 --keep-alive 600      # ready for a TypeSafe client
curl -sS 127.0.0.1:8088/v1/models
curl -sS -X POST 127.0.0.1:8088/v1/systemone -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer local' -d @request.json
```

What was measured, and where the receipt is:

- **The wire is the SDK's, not ours.** The field lists the routes are checked against were
  *generated* from an installed `typesafe-sdk==0.7.1` (`tools/typesafe_fields_capture.py` →
  `tests/fixtures/typesafe_sdk_0_7_1_fields.json`), not typed in by hand, so a served key set that
  stops being the SDK's fails the suite. Answers map to the SDK's own shapes: `choice`
  `{type, choice, confidence, probabilities}`, `score` `{type, score, confidence, legend,
  probabilities}` (the score is the probability-weighted mean, `1.30 = 0×0.0 + 1×0.70 + 2×0.30`),
  and `noul` `{type, noul, probabilities}` — no `confidence` key on a noul, because the SDK has none.
- **A real client, a real model.** The end-to-end pass installs the built wheel into a fresh venv
  (the repo is not on `sys.path`) and drives the server with the official `typesafe-sdk==0.7.1`
  client over loopback, on the real 0.8B GGUF: **cold 6.76 s, warm 2.28 s**, both `200
  served_by=host`, and the gate's own `verify()` passes on 3 typed answers
  (`docs/evidence/t_559ed8c8/`, `logs/serve_long_story.log`). The same-engine claim is measured, not
  argued: the *served* `noul 0.574913` equals the *CLI's* `noul 0.574913` for the same body, on the
  same warm-host pid (6069) with the same usage counters (`128/2`).
- **The same warm host serves the SDK's route and the native route.** `/v1/systemone` rewrites the
  request to the registry *alias* and `/v1/decide` names the same model; the keep key is computed
  from placement fields only (no `format`), so both land on one host — which is why the second
  decision of the pass can be asserted warm (`model_load_ms == 0.0`).
- **It refuses the way the SPEC says.** A path or a `repo:quant` where an alias belongs is refused
  with the typed code; an unknown alias answers `422` with `E_MODEL_NOT_FOUND` naming the aliases;
  malformed JSON, a missing `model`, an extra key and a one-level `score` each answer `422` in the
  SDK's own `ValidationError` field shape; an unknown route answers `404`
  (`docs/evidence/t_559ed8c8/logs/negatives_serve.log`).
- **Two hard bounds, both from the `Content-Length` alone.** A body over 1 MiB is answered `413` in
  **0.000 s** (`E_BODY_TOO_LARGE` on the native route, `too_large` on the TypeSafe one), and a
  half-sent request is closed after **30.0 s** — so a caller can never park a connection and hold
  the single warm host. Sentinel credentials appear **0** times in the server log.
- **On Windows the server still serves, inline.** There is no unix socket there, so the warm host is
  not attempted: the answer is computed inline and the server says so by name
  (`W_KEEP_UNAVAILABLE` on its stderr, no `engine.keep` block in the body). That is asserted live —
  the Windows job's *Serve smoke — a real boot, a real HTTP decision, SPEC 2.12's inline fallback*
  step went green in run `36163763039`.
- **`mcp` is still not here.** The MCP surface is specified (SPEC §2.9) and the command still exits
  3; `serve` is the interface this release ships.

Receipts: `docs/evidence/t_f5d8b6c7_serve_gates.md` (the wire/mapping/same-engine/cold-warm gates,
the SDK 0.7.1 field fixture, the host gate and its in-container rehearsal),
`docs/evidence/t_559ed8c8/` (installed-wheel E2E: the real SDK, cold/warm, the negatives, the
no-leak teardown), `tests/test_serve.py` (67 gates, offline).

## Headline 2 — updating the runtime after `init`: `typed-gguf runtime update` / `runtime rollback`

**You can now replace the pinned llama.cpp build without a compiler, and undo it.**

`init` installs the bundle named in `runtime.lock` and stays there. `runtime update` moves the
installed runtime to the newest official release that carries *this host's* pinned asset name (the
lock's tag, re-tagged — never `releases/latest`, never a fork), staging the download, probing it in
a child process against the lock's required tools, symbols, minimum build and backend, and switching
`runtime.json` atomically after the warm host has stopped. The bundle it replaced stays on disk;
`runtime rollback` flips the record back to it without downloading anything. Nothing is ever
deleted, and no pin in `runtime.lock` is rewritten.

```bash
typed-gguf runtime update --check     # what would change (from, to, size) — touches nothing
typed-gguf runtime update             # download, verify, pre-flight, probe, atomic switch
typed-gguf runtime rollback           # back to the bundle the update replaced
```

What was measured, and where the receipt is:

- **The live leg ran against the real GitHub API and the real official bundle.** On this box:
  `init` → build 11026; `runtime update --check` planned build 11160 (17.00 MB, GitHub's own
  `digest`); `runtime update` finished in **1.601 s** wall and `version` reported the new bundle;
  `runtime rollback` returned to 11026; a second rollback refused with `E_UPDATE_UNAVAILABLE`
  (exit 2). A re-update *adopted* the kept bundle (1.230 s, `source: "already-downloaded"`) instead
  of downloading it again (`docs/evidence/t_d88b4be0_update_gates.md`).
- **A killed download costs nothing.** The E2E leg throttled the download and `kill -9`d it at
  **3 145 728 B** (exit 137): `runtime.json` was **byte-identical** before and after, no `.pending-*`
  staging and no half-installed bundle, and the retry completed normally. The resumable `.part` file
  stays by design (`docs/evidence/t_559ed8c8/logs/kill_leg.log`).
- **The refusals refuse before the download.** `--backend cuda` on a host that cannot load
  `libcudart.so.12` is refused by the pre-flight with `E_RUNTIME_SYMBOLS: … skipped the 168.81 MB
  download`, and the fixture's request log shows the release list only — **no asset GET**. Thirteen
  negative legs (no runtime installed, unreachable API, a tag with no matching asset, a 404, a
  truncated body, a digest mismatch, a release API 500, a rollback with no `previous`, and more)
  each leave `runtime.json` byte-identical, with the code and exit status the SPEC tables name.
- **The default follows what `init` actually installed.** The target variant is derived from the
  installed record rather than assumed, so a box that ran `init --backend cpu` does not get handed a
  Vulkan bundle by an update it did not ask for.
- **The record keeps the probe's facts across a rollback.** The E2E pass found that a rollback
  rebuilt a ten-key record and `version` then printed `backends unknown`; the retained record is now
  *merged*, so the probe's `backends`/`symbols_*`/asset keys survive while the keys that describe the
  active bundle are rewritten for the bundle the record now names
  (`docs/evidence/t_16067777/`).
- **The GitHub leg says GitHub.** An asset failure used to be worded as a HuggingFace failure and the
  asset request carried a frozen `typed-gguf/0.1` User-Agent; the transport wording now names the
  product and its host per leg, and every User-Agent carries the packaged `__version__`.

Receipts: `docs/evidence/t_d88b4be0_update_gates.md` (the live leg, the record, the RED→GREEN pins,
the Tier-M sweep), `docs/evidence/t_559ed8c8/` (installed-wheel E2E: update, rollback, the `kill -9`
leg, the negatives), `docs/evidence/t_16067777/` (the four findings this pass fixed, with a RED
transcript), `tests/test_runtime_update.py` (49 gates, offline).

## The rest of the wave, fixed rather than filed

**The default model after `models pull`.** A bare `ask` (or `run`, `fit`, `calibrate`, `doctor`) on a
registry that has aliases but no `current` used to die on `E_MODEL_NOT_FOUND: None is not a registry
alias …` — the word `None` where a model belongs. One resolver now answers it: `current` when it is
set and still resolves, else the registry's **sole** alias (one entry is unambiguous), else a refusal
that lists the aliases and names `models pull`, `models use` and `--model`. Two aliases with no
`current` still refuse — which model to run is the user's call, never a guess. A call that names a
model is untouched, and the resolution is reported on stderr
(`model: requested <default> -> resolved <alias> (why)`), never on the response wire. The registry's
`current` key is not rewritten by the fallback. Receipt:
`docs/evidence/t_a0fa2dc0_default_model.md` (17 gates, RED→GREEN).

**`keep stop` takes the ledger's log with it.** The record, the socket and the spec went; the host's
diagnostic `<key>.log` stayed behind. It is removed now. Receipt: `docs/evidence/t_16067777/`.

**The host gate measures one wire and refuses to measure another.** The driver that drives the real
SDK against a live server pins the SDK version it measures (`--expect-sdk`) and raises its client
timeout above the SDK's 10 s default — the first decision of the day on a cold shader cache measured
**32.1 s** on this box, and a gate that times out there reports the box, not the product.

**A py3.11 CI red, fixed at the class.** A test fixture's teardown killed its children without
waiting, so a losing child stayed a live `Popen` until the GC reached it — and this suite runs
`filterwarnings = ["error"]`, which turned that into an intermittent CI error attributed to whichever
test was setting up. The keep client now reaps an abandoned child (kill, wait, pop) and the fixture
waits; 41 acceptance runs on 3.11 (25 sequential, 16 concurrent) were green. Receipt:
`docs/evidence/t_ba767a2b/` (with the mechanism probe).

## The platform matrix comes alive

`.github/workflows/runtime-matrix.yml` is five jobs — `linux-cpu`, `linux-vulkan`, `windows-cpu`,
`macos-metal`, `macos-engine-smoke` — and this build is where they started doing the work they were
written for.

- **The oracle's known skips are a policy now.** `! grep -q SKIP` failed both Linux jobs on the two
  skips that are *inherent* on a runner (the 4.4 GB Spark GGUF is not there and must never be
  downloaded there). The step pipes the oracle's output through `tools/matrix_oracle_skips.py`,
  which allows exactly those two lines and fails on any other skip, including one that merely looks
  model-absent (RED receipt: the two real SKIP lines → exit 0; an injected third → exit 1).
- **Windows is a real smoke, not a zip download.** The job flattens the pinned bundle, runs the
  doctor, loads `llama.dll` and computes (`bench --suite latency` end to end), and then boots
  `serve` and judges a real HTTP decision with `--expect inline` (the Windows fallback above).
- **macOS serves.** The `macos-engine-smoke` job installs the pinned `typesafe-sdk==0.7.1`, registers
  the 0.5B smoke model as an alias, starts `serve`, drives it with the repo's own real-client gate,
  asserts the **second** decision is warm (`served_by=host`, `model_load_ms == 0.0`) and ends with a
  clean `keep stop`. It went **green end-to-end in 7m25s** in run `36159785190` (job
  `108153215494`).
- **The warm window is the idle unload itself.** `linux-cpu` ends by asking cold with
  `--keep-alive 20`, asking again inside the window (same pid, `model_load_ms == 0.0`), then waiting
  and asserting the host is gone, the pid with it and the socket unlinked — and **nothing in that
  step calls `keep stop`**, or the claim would be a tautology. Green in run `36163763039`.
- **The first live run found two real bugs, and they are fixed in the product.** Run
  `36159785190` reached steps no earlier run had: `linux-cpu`'s fake-OOM step died *before* the OOM
  path on a fixture that could not name its CPU device (a `cpu` row is CPU-pinned, so production
  resolves the bundle's CPU device first — and that row has **one** placement, while the three rungs
  belong to the ladder world, which now reports its own count); and `windows-cpu`'s engine smoke
  answered `E_MODEL_ARCH_UNSUPPORTED: the unknown build runtime … has no implementation for
  architecture 'qwen2'`, because every leg of the capability path was ELF-shaped. Both are fixed:
  the fixture names its CPU device and both worlds are judged as *data* by a tested tool, and the
  capability path is platform-aware — the build number comes from the platform's own CLI banner
  (the pinned `llama.dll` carries no build string at all: measured, `11026` occurs in no file of the
  zip), the architecture scan accepts both real ABI decorations (the NUL-tailed Itanium/ELF form and
  the MSVC `@@` type-descriptor form the DLL actually carries), no user-facing message hardcodes
  `libllama.so`, and the finder separates roles from file names, so a *complete* Windows bundle
  validates (`runtime.files` ok naming `llama.dll`/`ggml.dll`/`ggml-base.dll`, `runtime.build` =
  `b11026`). Run `36163763039` at this head is the re-run: **all five jobs green**, including the
  Windows doctor, the Windows engine smoke, the Windows serve smoke, the macOS serve step and the
  Linux warm window. Receipts: `docs/evidence/t_f96fed7f/README.md` (the matrix's shape and the AC1
  RED replays), `docs/evidence/t_8dab8b3a/README.md` (the two failures, the byte-level evidence, the
  fix, and the watch list).

**CI runs Python 3.11, 3.12 and 3.13.** A 3.13 certification of the wave found two red tests for one
reason: they spelled the `413` reason phrase out, and CPython 3.13 refreshed `http.HTTPStatus` per
RFC 9110 (`Content Too Large`). The server emits whatever the stdlib hands it, so the fix is on the
test side — the cap assertions read the **status code**, never a version's vocabulary — and `ci.yml`'s
offline-gate matrix gained `3.13`, so the next such refresh is CI's finding rather than a human's.
Receipt: `docs/evidence/t_b5872762/README.md` (the three-interpreter phrase probe, the red, the
vacancy probe that shows the code-only assertion is not vacuous, and the green runs on all three).

## Measured highlights

Nothing in this section was re-measured for v0.3.0: these are the numbers the earlier releases
published, carried forward unchanged, each marked where it was measured. What v0.3.0 measures is new
work (the served entry, the update cycle, the matrix, the interpreters), and those numbers live in
the two headline sections above and in the platform section.

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

## The warm engine host (the mechanism the server reuses; unchanged since v0.1.1)

`run`/`ask` no longer pay the load twice. The first call leaves a **keep host** behind — a detached
child process holding the loaded model and answering over a `0600` unix socket in the data home
(`$TYPED_GGUF_HOME/keep/`, never TCP) — and the next call to the same model is answered warm.
`typed-gguf serve` hands its callers the same host, which is why the served route is warm too.
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
  `30s`, `1h`) on `run`/`ask`/`serve` or with `TYPED_GGUF_KEEP_ALIVE`. Precedence: **flag > env >
  default**. `--keep-alive 0` is the pre-E4 path exactly — answer inline and unload, so the keep
  path never runs and the response carries no `engine.keep` block; it is also the escape hatch for
  an A/B workflow on a box that cannot hold two models.
- **One model at a time.** One host per data home, keyed by the resolved model plus its SHA and the
  placement-affecting options; a request for a different key gets a swap, never a wrong answer.
- **The control surface** is `typed-gguf keep status [--json]` / `keep stop [--json]`: `status`
  reports the pid, the model, the key it is holding, idle seconds left, and the device the engine's
  own log proved. `keep stop` now takes the host's log with it.
- **Who answered** travels in the response: `engine.keep.served_by` is `"host"` or `"inline"`, with
  the host's pid, its one-time `model_load_ms` and the named reason for a fallback. If a host cannot
  be reached, the caller answers inline on that same call — the CLI never wedges on a host — and on
  a platform without unix sockets (Windows) that is the standing behaviour, named
  `W_KEEP_UNAVAILABLE`.
- **An abandoned child is reaped, not left to the GC.** A spawn that loses a same-digest race is
  killed and waited for (this build's py3.11 CI fix).

Receipts: `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md` (the gate table, the two bugs the live
gates found, the Tier-M sweep over `src/typed_gguf/keep`); offline pins in `tests/test_keep*.py`;
the window itself is asserted live by the matrix's warm-window step (run `36163763039`).

## Context sizing v2 (carried forward from v0.2.0)

The context a call runs in decides what it can read, and it used to be sized from the request
(`prefix + question + margin`). It is now planned from the box:

- **The standard is 32 768 tokens.** With no `--n-ctx`, a plan aims at the standard whatever size
  the request itself is.
- **It grows when there is room.** The plan takes the largest context this box holds at the top KV
  rung that reaches it, and never above the model's own window. On the 8 GB-VRAM reference box the
  default plan is **49 763** tokens — `n_ctx: 49763 (standard 32768, grown from the box's free
  memory)`.
- **It shrinks gracefully, with a warning.** When the box cannot hold the standard, the KV rung
  steps down first (`f16 → q8_0 → q4_0`, each step naming `W_KV_TYPE_DOWNGRADE`), then the context
  itself goes below the standard and the plan carries `W_CTX_BELOW_STANDARD`.
- **`--n-ctx` is a pin.** A pinned window is answered as asked (`min(pin, plan)`), and `--no-fit`
  restores the request-sized behaviour for anyone who wants the old arithmetic.
- **Nothing is truncated, silently or otherwise.** A request that does not fit the loaded context
  fails with `E_CTX_TOO_SMALL`, naming the `--n-ctx` that fixes it.

`fit` prints the plan in one line and `fit --json` carries the machine-readable form
(`standard_n_ctx`, and `ctx_limit` = `standard` / `grown` / `shrunk` / `pinned` / `window`). The
policy is written down in `docs/SPEC-context-v2.md`; the live receipts are in
`docs/evidence/context-v2/README.md`. Those numbers were **not** re-measured for this build.

## The fixes carried in from v0.2.3 and v0.2.2 (unchanged here)

- **A model switch no longer throws away the answer in flight** (v0.2.3). Stopping the outgoing host
  is a drain, not a kill: the call in flight keeps its warm answer (**7 179 ms** against **31 019 ms**
  before) and the switch completes at exit 0 where it used to be exit 3. Receipt:
  `docs/evidence/e2e/t_176614c6-live/RECEIPTS.md`.
- **A host that cannot serve steps aside** (v0.2.3). A decision that cannot allocate stops that host
  instead of answering every later request with an instant `E_BACKEND_OOM`; a typed error about the
  *request* (`E_CTX_TOO_SMALL`) leaves the host serving. Pinned in `tests/test_keep_host.py`.
- **A busy card is not the box's permanent answer** (v0.2.3). A fit plan read while the box was busy
  is no longer written to the cache as *the* answer: the same `ask` took **25 756 ms** before and
  **15 940 ms** after, at 36 GPU layers instead of 0. Pinned in `tests/test_fit_free_vram.py`.
- **The cold-spawn race** (v0.2.2). Two simultaneous cold calls both answer; one wins the data home
  and starts the one shared host, the other is served by it; a losing host no longer publishes a
  `failed` record over the winner's `ready` one. Pinned in `tests/test_keep_client.py`.
- **The placement ladder** (v0.2.2). A context that cannot be allocated walks a real ladder — KV
  cache type, a smaller context, fewer GPU layers, CPU-only — and reports the rungs that failed
  instead of a message that promised rescues it never tried. Receipt:
  `docs/evidence/e2e/t_287e0d18-live/`; pinned rung by rung in `tests/test_fit_placement_ladder.py`.
- **The three fixes carried in from v0.2.0** (a cached plan can no longer cap a later call, the live
  KV check reads the plan's own warning list, `doctor` no longer dies on a fallback record): see the
  v0.2.3 notes for their full text. Unchanged here.

## Install

```bash
git clone https://github.com/Rybens92/typed-gguf && cd typed-gguf
uv sync                # or: python3 -m venv .venv && .venv/bin/pip install .

uv run typed-gguf init                                        # pinned llama.cpp bundle, ~30 MB
uv run typed-gguf models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0 # 4.38 GB, SHA-256 verified
uv run typed-gguf ask --state "…" --choice "area=…:a|b|c"
uv run typed-gguf serve --port 8088 --keep-alive 600          # point a TypeSafe client here
uv run typed-gguf runtime update                              # refresh the bundle `init` installed
```

Requires Python 3.11+ (no compiler, no CUDA toolkit, no build step) — and CI now exercises 3.11,
3.12 and 3.13 on every push. Data lives in `~/.local/share/typed-gguf` (`$TYPED_GGUF_HOME` /
`$XDG_DATA_HOME` override it). Model licenses are recorded in the registry as they are pulled, so
the author's terms travel with the file.

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
**git fetch** step of the `git+https://…` spelling was the post-publish verification this paragraph
always promised, and it has one: the tag form of the one-liner (`uvx --from
git+https://github.com/Rybens92/typed-gguf@v0.1.0 typed-gguf version`) fetched, built and installed
this repository's tagged wheel and printed `typed-gguf 0.1.0`. Receipt:
`docs/evidence/v0_1_0_t_eff926f9_uvx_install.md`. What stays repository-root-only is the development
surface — the test suite and the oracle read `tests/`, `docs/evidence/` and `SPEC.md`, which no wheel
ships.

## What this release does not include

- **MCP.** `typed-gguf mcp` is planned and not implemented in this release: the command exits 3
  with a pointer, and SPEC §2.9's `typed_gguf_*` tool set stays a specification. `serve` is the
  interface that ships here; the MCP surface is a separate one, not a rounding error on this one.
- **More than one resident model.** The warm host holds one model per data home: a request for a
  different model — or for the same model with placement-affecting options that differ — stops the
  old host *before* the new one loads, so a switch costs a full cold load. That is deliberate on an
  8 GB-VRAM box, where two resident models do not fit; `--keep-alive 0` turns the host off entirely
  and `keep stop` frees the device right now. Nothing supervises a host that dies: the next call
  just pays a cold load. Three surfaces now share that one host (CLI, server, and the SDK route), so
  a long decision in one of them is a wait in the others — the server serializes, and it will not
  let a caller park a connection to hold it (the 30 s idle bound).
- **TLS and remote binding.** The server is a `127.0.0.1` stdlib server: no TLS, no auth beyond a
  non-empty `Authorization` header being accepted (and never logged). It is a local socket for a
  local client, not an internet-facing service.
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
- **Exotic-platform wheels.** There is no published wheel for a platform the pinned runtime has no
  asset for: build a runtime yourself and point `TYPED_GGUF_RUNTIME_DIR` at it.

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
  green on each of 3.11, 3.12 and 3.13 (`ci.yml`'s matrix), run
  [`36163762253`](https://github.com/Rybens92/typed-gguf/actions/runs/36163762253) at this head.
  The skips are the live cases that want a GPU, a model file or `--run-network`; nothing in the
  suite touches the network.
- **The two release gates.** `tests/test_public_docs.py` pins this file, the README and the packaged
  version together; `tests/test_release_publish.py` parses the publish workflow, executes its version
  gate against a fake `dist/`, and pins the four spellings of one version: `pyproject.toml`,
  `typed_gguf.__version__`, the name of these notes and the tag the release must carry. From this
  checkout, `uv run typed-gguf version` prints `typed-gguf 0.3.0`, and the workflow's own gate script
  accepts the tag `v0.3.0` and refuses a neighbouring tag.
- **The served entry, verified end to end.** The installed-wheel pass described in Headline 1
  (`docs/evidence/t_559ed8c8/`), plus the live matrix runs below for the platform legs.
- **The update cycle, verified end to end.** The live leg against the real GitHub API
  (`docs/evidence/t_d88b4be0_update_gates.md`) and the installed-wheel pass with the `kill -9` leg
  and thirteen negatives (`docs/evidence/t_559ed8c8/`).
- **The platform matrix, live.** `runtime-matrix.yml` run
  [`36163763039`](https://github.com/Rybens92/typed-gguf/actions/runs/36163763039) at this head:
  all five jobs green — Windows doctor + engine smoke + serve smoke, macOS serve with the pinned SDK
  and a clean `keep stop`, Linux's warm window. The run before it (`36159785190`) is where the two
  bugs it caught are receipted (`docs/evidence/t_8dab8b3a/README.md`).
- **Mutation testing.** Tier M, as declared on each card. The sweeps that ran this wave:
  `runtime/update.py` 967 mutants / 830 killed = 85.8 % (`docs/evidence/t_16067777/`, scoped re-runs
  after the post-sweep pins), `api/http.py` 747 / 565 killed = 75.6 %
  (`docs/evidence/t_f5d8b6c7_serve_gates.md`), `registry/store.py` 317 / 231 = 72.9 % (with the new
  `find_default` surface fully killed, `docs/evidence/t_a0fa2dc0_default_model.md`),
  `runtime/capability.py` + `runtime/finder.py` 846 mutants → 419 killed after the triage's three new
  gates (`docs/evidence/t_8dab8b3a/README.md`). This release-prep card itself changes no `src/`
  statement — a version string, this file, the two version pins and the lock — so no sweep was run
  for it; the standing sweep over `src/typed_gguf/runtime/fit.py` (2 267 mutants, soft threshold)
  stays published at 1 335 killed / 805 survived = **62.4 %**,
  receipt `docs/evidence/context-v2/mutation.md`.

## Reproduce

Everything above is regenerated from this checkout: `uv run pytest -q` (offline suite + oracle),
`python3 docs/verify_runtime_contract.py`, one command per benchmark table
(`python3 tools/e2_reproduce.py --suite <name> --model <path.gguf>`), and the E3e decision tool
(`python3 tools/e3e_roles_decision.py --report <arm.json> …`, the command `docs/evidence/e3e/report.sh`
drives). The warm-host numbers come from the live gate on the real 4B:
`uv run pytest -q --run-network tests/test_keep_live.py -s` (~5 min). The served entry is driven by
`tools/host_gate_serve.sh` (a fresh venv, the pinned `typesafe-sdk==0.7.1`, a real `serve`), and the
update cycle by the `runtime update --check` / `update` / `rollback` sequence in
`docs/evidence/t_559ed8c8/README.md`. The platform matrix is a `workflow_dispatch` of
`.github/workflows/runtime-matrix.yml`. `docs/BENCHMARKS.md`, `docs/TEMPLATES.md` and `SPEC.md`
carry the full detail behind every number here.
