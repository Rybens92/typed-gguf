# typed-gguf v0.1.1 — typed decisions on any GGUF

> Release notes for the v0.1.1 build. The measured content below is the v0.1.0 material, carried
> forward unchanged with the version bump, the README's positioning pass and the release workflow
> this repository now has (`.github/workflows/publish.yml`, PyPI Trusted Publishing). v0.1.0 is
> live on PyPI as `typed-gguf`. Every number below is already published inside this repository and
> marked where it was measured.

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

## The headline of this build: the warm engine host

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

## What v0.1.1 does not include

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

## Reproduce

Everything above is regenerated from this checkout: `uv run pytest -q` (offline suite + oracle),
`python3 docs/verify_runtime_contract.py`, one command per benchmark table
(`python3 tools/e2_reproduce.py --suite <name> --model <path.gguf>`), and the E3e decision tool
(`python3 tools/e3e_roles_decision.py --report <arm.json> …`, the command `.e3e/report.sh` drives).
The warm-host numbers come from the live gate on the real 4B:
`uv run pytest -q --run-network tests/test_keep_live.py -s` (~5 min).
`docs/BENCHMARKS.md`, `docs/TEMPLATES.md` and `SPEC.md` carry the full detail behind every number
here.
