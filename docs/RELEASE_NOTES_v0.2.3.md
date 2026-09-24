# typed-gguf v0.2.3 — typed decisions on any GGUF

> Release notes for the v0.2.3 build. Three fixes to what happens while the warm host is working: a
> model switch no longer throws away the answer that was already being computed, a host that cannot
> serve steps aside instead of standing there answering every call with the same error, and a
> placement worked out while the card was busy no longer becomes the box's permanent answer. Two
> pieces of housekeeping ride along (the repository layout, and the CI actions moving onto a runtime
> GitHub still supports). Nothing was re-measured, and no number the tool *chooses* moved: the same
> request on the same box still reaches the same answer.
> Everything the earlier releases measured (the quality tables, the warm-host numbers, the install
> receipts) is carried forward unchanged; every number in it is already published inside this
> repository and marked where it was measured. v0.2.2 is live on PyPI as `typed-gguf`.

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

## The headline of this build: a model switch no longer throws away the answer in flight

`typed-gguf` keeps one warm host per data home, so asking for a different model means the old host
has to go before a new one can load (see *The warm engine host* below). The old host used to be
killed the moment the switch began, and a call that was being answered on it lost the work already
done: its client found the host gone, re-ran the whole question by itself, and answered inline
instead. On the operator box a decision that takes about 7 seconds warm came back after **31 019 ms**,
and the switch that caused it then failed on its own with `E_BACKEND_OOM` (**exit 3**) while it tried
to plan a replacement for a card the abandoned call was still holding.

What a caller gets now:

- **A switch waits for the call in flight.** Stopping the outgoing host is a drain, not a kill: it
  gets the client's own request ceiling to finish what it is already answering before it goes. The
  call in flight keeps its warm answer (**7 179 ms** on the box, against 31 019 ms before), and the
  switch completes as well, answering at **exit 0** where it used to be exit 3.
- **The replacement is planned against the card as it is when it starts.** Because the outgoing host
  is gone before the replacement is spawned, the new host plans its load against the memory that is
  really free at that moment. In the before-receipt the same request came out at 0 GPU layers and
  never answered; in the after-receipt it plans **36 layers at a 56 616-token context** and answers.
- **One model at a time, unchanged.** This is about who waits for whom, not about how many hosts may
  live: still exactly one host per data home, keyed by the model and the options that shape its
  placement.

Both runs happened in the same hour on the operator box before this release, and the receipts are in
`docs/evidence/e2e/t_176614c6-live/RECEIPTS.md`. The behaviour is pinned offline in
`tests/test_keep_client.py` and `tests/test_keep_host.py`.

## The second fix of this build: a host that cannot serve steps aside

A warm host publishes a record that says it is `ready`, and every later call trusts that record before
it decides to use the host at all. A host whose plan did not fit the box at load time used to keep
that record anyway, and answered **every** request with an instant (about 170 to 190 ms)
`E_BACKEND_OOM` until a human ran `keep stop`. The book said the box had a working host, and every
call that believed it paid for the mistake.

What a caller gets now:

- **The host leaves the ready list by itself.** A decision that cannot allocate stops that host. The
  caller gets the same typed error it would have got anyway, the process exits and its record goes
  with it, and `keep status` stops promising a host that cannot do the job it is there for.
- **Only the device cases step aside.** A typed error that is about the request itself
  (`E_CTX_TOO_SMALL`, for instance) leaves the host serving, because the next call with a workable
  request is still answered warm.

The recorded live case is the one that found this. The pressure that produced it could not be built on
demand in the release container (the receipts say what was tried and what the box refused), so it is
pinned offline instead, deterministically, by driving a host with a decision that raises the error:
`tests/test_keep_host.py::test_a_decision_that_cannot_allocate_leaves_the_ledger` (before: the typed
reply and the host stays up; after: the typed reply and the process exits), with the boundary beside
it in `test_a_typed_error_that_is_not_about_the_device_keeps_the_host_serving`.

## The third fix of this build: a busy card is not the box's permanent answer

The fit plan a box computes is cached per model and host, and that cache only ever shrank. A plan read
while the card was busy (the desktop holding memory, or our own host resident) came out at 0 GPU
layers, and that reading was written to the cache as *the* answer for the model and the box. Every
later call read it, so after the card was free the tool still planned CPU-only, and the only way back
was to delete the entry by hand. Measured on the operator box: the same `ask` took **25 756 ms**
before and **15 940 ms** after, at **36 GPU layers** instead of 0.

What a caller gets now:

- **The reading is still the reading.** A call that reads the box while it is busy gets the smaller
  plan and the warning that names it (`W_FIT_DOWNGRADE`). A busy box is a real answer for the moment
  it was measured; it is simply not stored as the box's own answer.
- **A free box goes back to its full placement.** After `keep stop`, one `fit` plans 36 layers again,
  and the next call uses them.
- **No flag required.** `--no-fit-cache` remains the escape hatch for anyone who wants the cache out
  of the picture entirely.

Receipts: `docs/evidence/e2e/t_176614c6-live/RECEIPTS.md` (the four-step table, with the cache entry
recorded at each step), offline pins in `tests/test_fit_free_vram.py`.

## The cold-spawn race (carried forward from v0.2.2)

The first call for a model has to start the warm **keep host** — the background process that holds
the loaded model and answers the next calls over a unix socket in the data home (see *The warm
engine host* below). When two of those first calls arrived at the same moment, they raced over the
same staging file in that data home: the loser's write removed the winner's, and the loser then died
on a path that was not supposed to fail — a raw internal error, exit 4, no inline fallback — and the
answer that caller had asked for was gone. Two people (or two scripts) asking at once is ordinary,
and an error where an answer belongs is the one outcome this project does not accept.

What a caller gets now:

- **Every caller answers.** Two simultaneous cold calls both come back with their answers. One of
  them wins the data home and starts the one shared host; the other joins that host and is served by
  it — the same loaded model, the same answer, no second copy of the model in memory.
- **A call that cannot use the host answers inline, on that call.** If the host belongs to another
  model, or another caller owns this data home, the call stops idling beside a host it may not use
  and answers inline instead; `engine.keep.served_by` says `"inline"` and the reason is named.
- **The loser writes nothing.** A host that lost the race no longer publishes a `failed` record over
  the winner's `ready` one — one way `keep status` could describe a perfectly healthy host as broken.
- **One model at a time — the invariant did not move.** This is about who writes what in the data
  home and who gets served by whom, not about how many hosts may live: still exactly one per data
  home, keyed as before.

Live-verified on the operator box before this release: three pairs of concurrent cold `ask`s, six
answers out of six, exactly one host per pair. Pinned offline, name by name, in `tests/test_keep.py`,
`tests/test_keep_client.py` and `tests/test_keep_host.py`.

## The placement ladder (carried forward from v0.2.2)

When the context could not be allocated on the device, the engine used to stop with a hard
`E_BACKEND_OOM` whose message listed the rungs it would have walked — a smaller KV cache, a smaller
context, fewer GPU layers, CPU-only — while it had only ever tried the KV types. The message promised
a rescue that never ran, so a call that could have been answered (slower, on fewer layers, or on the
CPU) failed instead.

The ladder is real now. It is walked in this order, each rung at most once, and never upwards:

1. **the KV cache type** — `f16` → `q8_0` → `q4_0`;
2. **a smaller context** (the compute buffer is sized from the context, so this is a real rung);
3. **fewer GPU layers** — which re-loads the model at the smaller placement;
4. **CPU-only** — the device is not touched at all.

What a caller sees:

- **The rungs that failed are published.** A call that walked past two KV rungs reports each one
  (`W_BACKEND_OOM` names the placement that failed) instead of printing only the rung that worked.
- **The message lists only what was attempted**, in order, and claims a CPU-only fallback only when a
  CPU-only placement was actually tried.
- **A driver that will not say how much memory is free is not handed all of it.** When the device
  driver reports no free-memory number at all, the plan is made against 75 % of the card's nominal
  size instead of against the whole card — a plan the box could not have honoured.
- **The walk starts from where the model really is.** A second session on the same loaded model walks
  *down* from that model's live placement, so a rung can never quietly map the weights back up to a
  placement that had already failed.
- **`calibrate --fit-target` and `--n-seq-max` reach the plan.** Both flags used to be accepted and
  then ignored: the plan `calibrate` printed and stored was the one you would have got with no flag
  at all. On this box `fit --fit-target 4500` is 15 layers, a 4 096-token context at `q4_0` and a
  2 134 MiB budget, where the un-flagged plan is 36 layers, 47 621 tokens at `q8_0` and 5 575 MiB.
- **Your chosen numbers never change — only the answer and the report get honest.** The arithmetic is
  untouched: the same request on the same box reaches the same decisions, the same probability
  distributions and the same reliability verdict. What changed is whether a call that *can* be
  answered is answered, and whether what the tool says about a placement is what the placement did.

Live on the operator box before this release (Vulkan bundle, 4B `Q8_0`, receipts in
`docs/evidence/e2e/t_287e0d18-live/`): one `ask` with 1 353 MiB of the card already held by the desktop answered in
**22.4 s** at 36 GPU layers with the KV rung really used (`q4_0`) and both failed rungs published;
two asks at once were re-placed to 18 layers; and with the device held by another call the planner
itself went CPU-only — `budget_bytes` 0, context shrunk to 4 096, `W_CTX_BELOW_STANDARD` — and said so
in its response instead of failing. The ladder's *new* rungs were not needed on this box in those
hours, so they are pinned offline instead, rung by rung, in `tests/test_fit_placement_ladder.py` (the
exact seven-rung walk, ending on a CPU-only placement that succeeds).

## Under the hood (no user-visible change)

Two pieces of housekeeping ride along with this build, and neither changes what a call does.

- **The repository layout.** The dev-run receipts and QA notes that used to sit in dot-folders at the
  repository root now live in `docs/evidence/` (per-run receipts) and `docs/qa/` (the QA notes), and a
  new gate, `tests/test_public_layout.py`, fails the suite the moment a fresh dot-entry appears at the
  root again. Nothing installed changes: the wheel ships the same package, with the same commands and
  the same entry points.
- **The CI actions.** The three workflows now use `actions/checkout@v7` and
  `astral-sh/setup-uv@v10`. GitHub had started warning on every run that the previous majors sit on a
  Node runtime it is retiring; these majors declare Node 24 and take the same inputs the jobs already
  pass. The release gate pins the two strings, so a workflow and its gate cannot drift apart.

The test hardening that arrived with the carried fixes below is unchanged here: the race gate waits,
bounded, for the losing host to leave the process table (5 s, 50 ms steps) and its inline double
carries a real answer body instead of an empty one, and the placement pins assert the seven-rung walk
through the production session rather than a hand-made list. Test-only, and no behaviour a user can
reach moved.

## Context sizing v2 (carried forward from v0.2.0)

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
  `ctx_limit pinned` when the pin is what got loaded and `ctx_limit shrunk` — with the note that
  names both numbers — when the box could not hold it (this build), and the pin is what re-keys a
  warm host. `--no-fit` restores the request-sized behaviour for anyone who wants the old
  arithmetic.
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

## The three fixes carried in from v0.2.0

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

## What this release does not include

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
  **1 650 passed, 57 skipped, 0 failed** in 51 s. The skips are the live cases that want a GPU, a
  model file or `--run-network`; nothing in the suite touches the network. The tests added since
  v0.2.2 are the offline pins of the three fixes above and the root-layout gate. The plain card
  command (the same line with no bundle stub) is **1 649 passed, 1 failed**: the one case that needs
  a *visible* bundle to choose CPU from, which CI satisfies with four empty library files in a
  temporary directory (its own step in `ci.yml`).
- **The two release gates.** `tests/test_public_docs.py` (17 tests) pins this file, the README and
  the packaged version together, including the numbers nobody may re-quote from memory. The second
  gate, `tests/test_release_publish.py` (15 tests), parses the publish workflow, executes its version
  gate against a fake `dist/`, and pins the four spellings of one version: `pyproject.toml`,
  `typed_gguf.__version__`, the name of these notes and the tag the release must carry. From this
  checkout, `uv run typed-gguf version` prints `typed-gguf 0.2.3`; the workflow's own gate script,
  run against this build's artifacts (`typed_gguf-0.2.3-py3-none-any.whl` +
  `typed_gguf-0.2.3.tar.gz`, 314 263 B + 4 044 675 B), accepts the tag `v0.2.3` (rc 0) and refuses
  `v0.2.2` and `v0.2.4` (rc 1 each, with the `::error::` line on the log). The same gate also pins
  the two action versions the workflows carry (`actions/checkout@v7`, `astral-sh/setup-uv@v10`), so
  a workflow and its gate cannot drift apart.
- **The three fixes, verified live before the release.** The runs receipted in
  `docs/evidence/e2e/t_176614c6-live/` (the harness, per-call JSON, exit codes and wall times, and
  the four-step plan-cache table): the switch that used to kill the call in flight now waits for it
  (the victim's answer 31 019 ms to 7 179 ms, and the switch exit 3 to 0), and the plan read on a
  busy card no longer sticks (25 756 ms to 15 940 ms, at 36 GPU layers). The one case that could not
  be rebuilt on demand in the release container, a host that says `ready` and cannot allocate, is
  pinned offline instead, by name, in `tests/test_keep_host.py`.
- **The cold-spawn race, verified live before the v0.2.2 release.** Three pairs of concurrent cold `ask`s on
  the operator box: 6/6 answers delivered, exactly one host per pair — and both callers of a pair
  report the same host, which is what "one shared host, one model copy" means in practice. The gate
  that keeps it is `tests/test_keep_client.py`: the racing pair, the bounded wait for the loser's
  exit, and the loser's own log quoted back to the caller it no longer serves.
- **The placement ladder, verified live before the v0.2.2 release.** The runs receipted in
  `docs/evidence/e2e/t_287e0d18-live/` (report, stdout and stderr per run, `nvidia-smi` before and after): the
  22.4 s GPU ask that published both failed KV rungs, the two-ask pair re-placed to 18 layers, the
  honest CPU-only plan taken against a held device, and the `--fit-target 4500` plan that `calibrate`
  now really feeds. The failed device allocation that used to be a hard error sits in `ask.stderr`,
  next to the compute buffer that was allocated instead.
- **A calibration receipt, not shipped data.** `docs/evidence/calibration-4b-2026-09-23.md` — the
  box's first stored 4B calibration (the per-type table, the refit hash, the store entry, and the
  choice the data refused). Nothing in this build reads it at runtime; it is the provenance for the
  numbers the project quotes.
- **Mutation testing.** Not run for this build: it moves a version string, these notes, the two
  version pins and the two action versions in three workflow files, and no product line at all, so a
  sweep would score the tree this build already reports on. The standing sweep over the module the
  last release reported on (`src/typed_gguf/runtime/fit.py`, 2 267 mutants, soft threshold) stays
  published: 1 335 killed / 805 survived = **62.4 %**, receipt `docs/evidence/context-v2/mutation.md`.

## Reproduce

Everything above is regenerated from this checkout: `uv run pytest -q` (offline suite + oracle),
`python3 docs/verify_runtime_contract.py`, one command per benchmark table
(`python3 tools/e2_reproduce.py --suite <name> --model <path.gguf>`), and the E3e decision tool
(`python3 tools/e3e_roles_decision.py --report <arm.json> …`, the command `docs/evidence/e3e/report.sh` drives).
The warm-host numbers come from the live gate on the real 4B:
`uv run pytest -q --run-network tests/test_keep_live.py -s` (~5 min).
`docs/BENCHMARKS.md`, `docs/TEMPLATES.md` and `SPEC.md` carry the full detail behind every number
here.
