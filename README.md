# typed-gguf

typed-gguf is a local alternative to Jev. You give it a `state` plus typed questions, and it gives
back typed answers with full probability distributions and confidence, computed locally on frozen
GGUF models. It can drive any GGUF model llama.cpp supports — not just one vendor's — and the
core is stdlib-only: nothing is generated, nothing is fine-tuned, no compiler is ever invoked.

The command line is the whole interface, and it works the same for a person at a terminal and for
an AI agent driving it from a script: one stable, scriptable tool whose answers are JSON, and a
whole request can come from a file instead of from flags.

It is inspired by the System-One-style typed-decision interface (Jev). The project has no
affiliation with it and makes no parity claim.

- One pass over a shared prefix. The state is prefilled once; every question starts from a copy of
  that decoded prefix (`llama_memory_seq_cp` forks the sequence state), so it pays only for its
  own short suffix.
- No text generation. Answers are read from the logits (the model's raw scores) over a fixed
  candidate set, and a restricted softmax turns them into probabilities. Nothing is sampled and
  nothing is parsed.
- No fine-tuning, ever. Any GGUF llama.cpp can load is a valid backend; the weights are never
  updated. The core has zero third-party runtime dependencies (stdlib only).
- No compiler, ever. `typed-gguf init` downloads a pinned official llama.cpp release bundle and
  drives its shared libraries through `ctypes`, so users never build anything.
- A separate process owns the runtime. The probes (`init`'s fallback chain, `doctor`, the warm-up)
  run in a disposable child (`typed_gguf.runtime.probe_child`) and report back as JSON, so
  third-party GPU libraries and drivers stay out of the command process and their teardown cannot
  take the command down with it.

The default model is
[`XHToken/Spark-X2.5-4B-GGUF`](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) `Q8_0`, pinned
by size and SHA-256. Its 4.38 GB of weights fit in 8 GB of VRAM, the budget most home PCs and
gaming laptops have, so the defaults work on first use for most people.

## Quickstart

This release is on PyPI and needs nothing but Python 3.11+. The core is stdlib-only and no compiler
is ever invoked:

```bash
uvx typed-gguf version          # run it without installing anything
uv tool install typed-gguf      # puts the command on your `PATH`
pip install typed-gguf          # or into a plain venv
```

`uv tool install` puts `typed-gguf` on your `PATH` from any directory, and `uvx typed-gguf …` runs a
single command without installing anything. Then the first run:

```bash
typed-gguf init                                      # pinned runtime, ~30 MB
typed-gguf models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0   # 4.38 GB, SHA-256 verified
```

Every command below is written `uv run typed-gguf …`: that is the checkout form, and `uv run` uses
the project's `.venv` (with a plain venv it is `.venv/bin/typed-gguf`). Installed by name, the same
commands read `typed-gguf …`. The repository clone is the development path:

```bash
git clone https://github.com/Rybens92/typed-gguf && cd typed-gguf
uv sync                # or: python3 -m venv .venv && .venv/bin/pip install .
```

`python -m typed_gguf …` runs the same CLI either way.

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

`uv run typed-gguf doctor` reports on both: exit `0` means ready, `2` means it works with
warnings, `1` means broken. A `2` is not a failed install. On this box `doctor` reports the
documented CUDA→Vulkan pre-flight fallback (an NVIDIA card, a Vulkan bundle) and `model.present`
until the pull lands.

### Install without a clone: `uvx`, `uv tool install`, pip

The wheel carries its own pinned `runtime.lock` (the build copies the repository's into the
package), so an installed `typed-gguf` reads its pins from itself and works from any directory, with
no checkout and no `cd` into one:

```bash
uvx typed-gguf version
uvx typed-gguf init          # pinned runtime
uvx typed-gguf doctor --json

uv tool install typed-gguf   # puts it on PATH
pip install typed-gguf       # or a plain venv
```

A revision that is not on PyPI yet installs straight from git — `uvx --from git+…`,
`uv tool install --from git+…`, `pip install "typed-gguf @ git+…"` — and from a checkout,
`uvx --from . …` builds the same wheel, lock included, and runs it from uv's cache instead of your
source tree:

```bash
uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf version
uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf init          # pinned runtime
uvx --from git+https://github.com/Rybens92/typed-gguf typed-gguf doctor --json

uv tool install --from git+https://github.com/Rybens92/typed-gguf typed-gguf   # puts it on PATH
pip install "typed-gguf @ git+https://github.com/Rybens92/typed-gguf"          # or a plain venv
```

`init` from such an install, run in an empty directory (measured 2026-09-20):

```
installed: True
variant: linux-x64-cpu
build: 11026
asset: llama-b11026-bin-ubuntu-x64.tar.gz
asset_verified: True
symbols_ok: True
rung: prebuilt
```

`$TYPED_GGUF_LOCK` still points a run at a different pin, and it is used *as-is*: a path that does
not exist is an error that lists every path that was searched, never a silent fallback. The test
suite and the oracle only run from a repository checkout: they read `tests/`, `docs/evidence/` and
`SPEC.md`, none of which a wheel ships.

### One state, three typed questions

```bash
uv run typed-gguf ask \
  --state "The billing page is blank for every user since 09:12. The incident channel is quiet, no deploy in the last 24h, and the API is healthy on synthetic traffic." \
  --choice "area=Which team owns this?:billing|technical|platform" \
  --score  "severity=How severe?:cosmetic|annoying|critical" \
  --noul   "page=Should we page the on-call engineer?"
```

The real response (this box, Vulkan, no policy flags), trimmed only where the `…` marks it:

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

Read it as the decision (`choice: billing`, `score: 0.68` on a 0–2 scale, `noul: 0.0047`), the full
distribution (`probabilities`, with `legend` naming the levels of a `score`), how concentrated it
is (`confidence`), and whether the model's answer really reads like an answer
(`reliability: "ok"`, `cue.verdict: "answered"`). `usage.decode_steps` counts the decode steps the
run spent; nothing is generated. `engine` carries the details for the rest:

- `engine.cue: "json_instructed"` and `engine.chat_format.kind: "role_split"` are the defaults the
  product ships (2026-09-20): `json_instructed` says the ask line is a JSON contract, `role_split`
  says the question is its own user turn rendered by the model's own chat template
  (`question_turn: "user"`). Both older switches still work:
  `--cue shipped --chat-format answer_sheet` (`docs/TEMPLATES.md` §4).
- `engine.template` says which template rendered the prompt. `kind: "gguf-renderer"` /
  `renderer: "internal"` is the model's own `tokenizer.chat_template` rendered by our renderer,
  `family: "spark2_5"`, `thinking: "suppressed"`, verified on the rendered bytes (see
  `docs/TEMPLATES.md` §3). Templates outside the internal Jinja subset fall through the chain to
  llama.cpp's built-in templates (`renderer: "builtin"`, `W_TEMPLATE_FALLBACK`) or to `--template`,
  and a template that nothing can resolve raises `E_TEMPLATE_UNRESOLVED`, whose message names the
  fix.

### The same request as a file

`run` takes a whole request (`--questions q.json`, with `--state`/`--state-json` to override the
state; the request schema is in `SPEC.md`):

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
`prefill_tokens: 0`, and everything else in the two responses is identical. The only differences
are the two prefill fields above and the free-memory budget the plan block records
(`engine.fit.budget_bytes`), which is re-read on each call.

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
dropped by the adapter on purpose, `usage` is reduced to its two documented counters, and an
unknown model name (`model: "jev-latest"` and friends) falls back to the configured default. The
adapter claims no parity: the confidence statistic is ours, and the one documented outlier in the
adapter target's docs is reproduced as-is in `docs/verify_runtime_contract.py`.

### Warm host: no cold start between calls

```bash
typed-gguf ask --state "Billing is down." --choice "area=Which?:billing|technical"   # cold: loads
typed-gguf ask --state "Billing is down." --choice "area=Which?:billing|technical"   # warm
typed-gguf keep status
```

The second call skips the model load: the first one left a keep host behind, a detached child
process holding the model and answering `run`/`ask` over a unix socket in the data home
(`$TYPED_GGUF_HOME/keep/`, mode 0600, never TCP). After `--keep-alive` seconds without a request it
exits itself and frees the device. Measured on the 4B with the pinned Vulkan bundle: cold 17.50 s
(load 2280 ms) against 2.58 s warm (load 0 ms) on the same host pid.

- One model at a time. One host per data home. Switching models stops the old host *before* the new
  one loads, so a swap never holds two models in RAM or VRAM.
- The window is 600 s (10 min) by default and configurable: `--keep-alive 10m` / `30s` / `1h` on
  `run`/`ask`, or `TYPED_GGUF_KEEP_ALIVE=10m` in the environment. Precedence: flag > env > default.
  `--keep-alive 0` is the old behaviour exactly: answer inline and unload.
- The key. A host serves one identity: the resolved model path plus its SHA (the registry's
  recorded sha256, else the file's own size+mtime) plus the placement-affecting options (`backend`,
  `n_ctx`, `kv_type`, `n_seq_max`, `threads`, fit flags). A request with a different key gets a
  swap, never a wrong answer; `keep status` prints the key it is holding.
- Who answered is in the response: `engine.keep.served_by` is `"host"` or `"inline"`, with the
  host's pid, its one-time `model_load_ms`, the idle time left, and, when a host could not be had,
  the named reason it fell back (`engine.keep.fallback`). The call that spawned the host reports
  that load in its own `timings.model_load_ms` (it waited for it); a warm answer reports `0.0`. With
  `--keep-alive 0` the keep path never runs, so the response is the pre-warm-host one verbatim,
  with no `engine.keep` block at all.
- `keep status` / `keep stop` are the control surface. `status` reports the pid, the model, the
  key, idle seconds left, the placement, and the device the *engine's own log* proved.
- Fallback policy. If the host cannot be reached, spawns but never becomes ready, or dies with a
  request, the client cleans up its ledger entry and answers inline on that same call; the CLI
  never wedges on a host. Crashes inside the host are typed errors on the wire, rebuilt as the
  product's own exception type. If a platform has no unix sockets (Windows), `run`/`ask` say so by
  name (`W_KEEP_UNAVAILABLE`) and answer inline; no daemon is attempted.

## Platforms

`init` picks the pinned official llama.cpp bundle for the host: Linux x86_64 (cpu, vulkan, cuda-12.8),
Windows x86_64 (cpu, vulkan, cuda-12.4) and macOS arm64/x64 (metal). The map lives in
`src/typed_gguf/runtime/pins.py`. Anywhere else there is no pinned asset to install: build a
runtime yourself, point `TYPED_GGUF_RUNTIME_DIR` at it, and `doctor` probes what you point it at
(the third rung of the runtime ladder in `SPEC.md`).

The Python package is stdlib-only and needs Python 3.11+, so installing works anywhere `uv`/`uvx`
does. The warm host needs unix sockets: on Windows `run`/`ask` answer inline with
`W_KEEP_UNAVAILABLE` and no daemon is attempted. CI runs on Ubuntu, so the Windows and macOS paths
have their own platform handling but are not exercised by that job.

## Interfaces

| command | what it does |
| --- | --- |
| `typed-gguf init [--backend auto\|cpu\|vulkan\|cuda\|metal] [--dry-run]` | downloads, verifies, extracts and probes the pinned llama.cpp bundle; `--dry-run` prints the plan |
| `typed-gguf doctor [--json]` | checks the bundle (files, symbols, build, `llama-fit-params`, backends, accelerator, recorded SHA) and the registry; exit 0 ok / 2 warnings / 1 broken |
| `typed-gguf models search <q>` / `pull <repo[:quant]>` / `use <alias>` / `ls [--json]` / `rm <alias>` / `verify [alias]` / `recommend-quant [--vram GiB]` | the model registry: resume + SHA-256 verified downloads, the model author's license recorded with the file, and a quant recommendation for a VRAM budget |
| `typed-gguf fit [<model>] [--json]` | the fit plan for this host (`n_gpu_layers`, `n_ctx`, `kv_type`, `n_seq_max`, `est_*` bytes), cached per (model SHA-256, host fingerprint) and applied on load unless `--no-fit` |
| `typed-gguf run --questions q.json [--state …] [--format native\|typesafe] [--out r.json] [--keep-alive <dur\|0>]` | a whole request from a file |
| `typed-gguf ask --state … --choice/--score/--noul "id=instruction:labels" [--keep-alive <dur\|0>]` | the same engine from the command line |
| `typed-gguf bench --suite latency\|throughput\|quality\|calibration\|determinism --model <path.gguf>` | reproduces the tables in `docs/BENCHMARKS.md`; never touches the registry and never opens a socket |
| `typed-gguf calibrate [--dry-run]` | fits the per-(model, question-type) temperature/scale on the committed dev set and keeps it only if a held-out split improves |
| `typed-gguf keep status [--json]` / `stop [--json]` | the warm host: one resident model per data home, answering `run`/`ask` over a 0600 unix socket and unloading itself after `--keep-alive` |
| `typed-gguf version [--json]` | versions, the pinned runtime tag, the installed runtime and the data home |
| `typed-gguf serve` / `typed-gguf mcp` | the HTTP (`/health`, `/v1/models`, `/v1/decide`, `/v1/systemone`) and MCP (`typed_gguf_decide`, `typed_gguf_models_list`, `typed_gguf_models_pull`, `typed_gguf_runtime_status`, `typed_gguf_fit`) surfaces are planned and not implemented in this release: both commands exit 3 today |

`python -m typed_gguf <command>` is the same CLI. Exit codes: `0` ok, `2` user error, `3`
runtime/model error, `4` internal (`doctor` adds `2` for "works, with warnings" and `1` for
"broken"). Every command accepts `--help`.

One interface, three shapes: `--format native` (default) is the full response above;
`--format typesafe` is the compatibility adapter; and the same request/response pair is what
`typed-gguf run --questions` and `typed-gguf ask` build internally, so anything the CLI can ask can
be driven from a file.

## Fit: what this host can actually hold

```bash
uv run typed-gguf fit                                  # the default model, human-readable
uv run typed-gguf fit Spark-X2.5-4B-Q8_0 --json        # {n_gpu_layers, n_ctx, kv_type, n_seq_max,
                                                      #  est_weights_bytes, est_kv_bytes,
                                                      #  est_total_bytes, backend, source}
```

`source` is `llama-fit-params` when the bundle's own tool produced the numbers, `estimate`
otherwise (with `W_FIT_ESTIMATED`). The plan is cached per `(model sha256, host fingerprint)` and
applied on load: `run`/`ask` honour it unless `--no-fit` is passed. Over budget, `kv_type` walks
`f16 → q8_0 → q4_0` (each step warns `W_KV_TYPE_DOWNGRADE`) before the context shrinks; the
estimate is cross-checked against the memory the process actually used at load, within ±20 % on
this box (`docs/TEMPLATES.md` §5).

## Limitations and known issues

- Questions inside one request are decided sequentially. The parallelism is *per candidate within a
  question*: the candidate answers for a question are batched into waves that fit `n_seq_max − 1`
  at a time, and there is no cross-question batched decode yet, so a request's decode cost grows
  additively with the number of questions. `usage.waves` and `usage.decode_steps` report what a
  request actually cost.
- The calibration, routing and latency/throughput/determinism tables were measured under the older
  default switches (the `shipped` cue with the `answer_sheet` chat format) and stay published as
  that policy until a later re-measurement campaign re-runs them. Each table's own report names
  the settings it was measured with, and `docs/BENCHMARKS.md` marks those tables rather than
  mixing them with current ones. Measured under the current defaults: the 4B quality table (§2.3),
  the Tiel table (§7.4.2) and the Occamy pair.
- `serve`/`mcp` are specified, not shipped, so the CLI is the only interface today.
- Exotic-platform wheels are future work. The pinned prebuilt llama.cpp bundle is the primary
  distribution and the only automated install path: on a platform with no official asset, this
  release has nothing to install automatically. On such a host, point `TYPED_GGUF_RUNTIME_DIR`
  at a runtime you built yourself and `typed-gguf doctor` probes it.
- Thinking suppression happens in the prompt. For the families with a real switch it is verified on
  the rendered bytes; for `k2-horizon` the `/no_think` marker is only advisory, because that
  family's thinking is a setting of the serving stack (`docs/TEMPLATES.md` §8).
- Quality numbers come from our own 60-item dev set (`src/typed_gguf/bench/devset.jsonl`, authored in
  this repo, provenance recorded). They are measurements with Wilson intervals (statistical
  confidence ranges), not a vendor comparison; `docs/BENCHMARKS.md` §4 lists what the tables
  deliberately do not claim.
- Big models on small boxes are limited by the box. A 23–24 GB mixture-of-experts model on an
  8 GB-VRAM host decides at ~0.3 tok/s decode (measured, §6.5) and its fit plan offloads only what
  free VRAM allows; a host that can keep the weights resident turns the same command into a
  compute-bound run. Read `docs/BENCHMARKS.md` §6/§7 before blaming the engine.
- A cached fit plan is never re-expanded. The plan is cached per (model SHA-256, host fingerprint)
  under `$TYPED_GGUF_HOME/fit/` and re-checked against free device memory on every load, but that
  check can keep or shrink the plan, never grow it back: a plan degraded for one busy run stays
  degraded after the device frees up (only its `budget_bytes` refreshes). Drop the cache with
  `--no-fit-cache` on a request, `typed-gguf fit --no-cache`, or by deleting
  `$TYPED_GGUF_HOME/fit/`, when free memory returns.
- No CUDA row exists in the published tables (no CUDA device was reachable when they were measured);
  each table says `measured: false` with the reason instead of omitting the backend.
- `typed-gguf` is not affiliated with TypeSafe and makes no parity claim; the `typesafe` output
  format is a compatibility adapter.
- The warm host keeps one model at a time. A request for a different model, or for the same model
  with placement-affecting options that differ, stops the resident host *before* the new one loads,
  so the swap costs a full cold load and there is no set of per-model hosts. That is deliberate on an
  8 GB-VRAM box, where two resident models do not fit; `--keep-alive 0` turns the host off entirely
  and `typed-gguf keep stop` frees the device right now. While a host is resident its model stays
  in RAM/VRAM, and a call for a different model stops that host first.
- The window belongs to the call that spawned the host. A later call that merely *reuses* the host
  does not change its `--keep-alive` (though every request does restart its countdown): the resident
  host answers for the window it was started with until it ages out, is stopped, or a different key
  evicts it. Nothing supervises a host that dies; the next call just pays a cold load.

## Verification

```bash
uv run pytest -q                                    # unit gate (offline: live tests are skipped)
uv run pytest -q --run-network                      # + real HF downloads / real GGUF headers
uv run pytest -q --run-network tests/test_engine_fork.py tests/test_cli.py   # fork equivalence,
                                                    # waves, determinism, state save/load, CLI e2e
uv run pytest -q --run-network tests/test_templates.py tests/test_fit_live.py  # the real templates
                                                    # + the real fit plan (measured process memory
                                                    #   within ±20 %)
uv run pytest -q --run-network tests/test_keep_live.py -s   # cold vs warm, idle unload, the
                                                    # A→B→A swap (real 4B, ~5 min)
uv run python tools/e1c_offline_gate.py             # the template and fit gates, network off
python3 docs/verify_runtime_contract.py             # oracle: pinned facts + formulas
TYPED_GGUF_RUNTIME_DIR=<runtime> python3 docs/verify_runtime_contract.py   # + live ctypes probes
```

The suite is offline by default: live gates are marked and skip by name when they cannot run, and
that is a normal green run. The one exception is the `@pytest.mark.needs_fork` gates: when the box
is starved for processes (`pids.max` shared with sibling sandboxes), they skip with that reason on
the summary line and the run refuses to exit 0, so a starved cgroup can never be read as a green
suite. Those gates say why they skipped instead of reporting "the backend does not load on this
host".

Deeper docs for contributors: `SPEC.md` is the contract, `docs/TEMPLATES.md` holds the template and
family contract, and `docs/BENCHMARKS.md` holds the measured tables.

## License

MIT. The bundled/used llama.cpp release is MIT as well; model licenses are the model authors'.
`typed-gguf` is not affiliated with TypeSafe; the `typesafe` output format is a compatibility adapter.

## Credits and attribution

`typed-gguf` stands on other people's work, and says so:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), the inference runtime and the C ABI this
  project drives through `ctypes` (`libllama.so` / `libggml*.so`, pinned release `b11026`).
  `typed-gguf init` downloads the official release bundle; nothing here is built from llama.cpp
  source.
- [TypeSafe](https://docs.typesafe.ai): the documented `state` + typed `questions` → typed `answers`
  wire shape that the `--format typesafe` adapter mirrors, and the interface whose "System One"-style
  typed decisions this project reimplements on frozen GGUFs. `typed-gguf` makes no parity claim with
  it: the confidence statistic is our own and the documented quickstart outlier is reproduced as-is
  in `docs/verify_runtime_contract.py`.
- [rorshopping/parallel-decisions](https://github.com/rorshopping/parallel-decisions) and
  [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) /
  [bnsd55/openjev](https://github.com/bnsd55/openjev): the "System One" decision-readout idea (score
  a fixed candidate set instead of generating text) that this project exists to reproduce on stock,
  frozen GGUFs. Their trained models are the reference point, not a dependency.
- [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD): RLCD-style
  training that demonstrated decision behaviour in a small model; cited as prior art for the *frozen*
  variant we build.
- [XHToken/Spark-X2.5-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) (Apache-2.0): the
  default model, used unmodified and pinned by size + SHA-256.

The `license` field of every pulled model is recorded in the registry and printed by
`typed-gguf models pull` / `models ls --json`, so the model author's terms travel with the file.
