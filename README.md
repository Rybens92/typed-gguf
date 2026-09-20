# typed-gguf

**Formerly `ggufone`** — the public name changed on 2026-09-20 (card t_5f9c15fe): the import package is `typed_gguf`, the command is `typed-gguf`, the default data home is `~/.local/share/typed-gguf`. Files under `docs/evidence/` keep the old spelling: they are records of runs, not names.

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

## Status

`SPEC.md` is the contract (milestones E1a → E3 with numbered acceptance criteria).

- **E1a (done)** — runtime + model registry: `typed-gguf init` (pinned llama.cpp bundle, no
  compiler on the path), `typed-gguf doctor`, `typed-gguf models {search,pull,use,ls,rm,verify,
  recommend-quant}`, the GGUF header reader, the conservative fit planner and the executed
  oracle green with the live runtime section.
- **E1b (done)** — the engine core: `typed-gguf run` / `typed-gguf ask`, `--format native|typesafe`,
  the fork readout (one prefill per state, waves bounded by `n_seq_max`, prefix states cached on
  disk), the schema/error catalog and the typesafe adapter. Measured numbers:
  `docs/evidence/e1b_perf.json`, gate-by-gate report `docs/evidence/e1b_t_34abf324_engine.md`.
- **E1c (done)** — the reasoning resolver + fit: the ordered template chain (the model's own
  `tokenizer.chat_template` through the internal renderer → `llama_chat_apply_template` built-ins
  → `--template` override → `E_TEMPLATE_UNRESOLVED` with the fix), provable thinking suppression,
  `typed-gguf fit` (the bundle's `llama-fit-params` or typed-gguf's own estimate, cached per
  `(model sha256, host fingerprint)`, applied on load unless `--no-fit`) and `docs/TEMPLATES.md`.
  Measured numbers: `docs/evidence/e1c_e2e.json` (four example question sets end to end),
  `docs/evidence/e1c_t_c8e36cad_*.md` (gate table + receipts).
- **E2 (done)** — benchmarks: `typed-gguf bench --suite latency|throughput|quality|calibration|
  determinism` over the pinned runtime, a committed 60-item labeled dev set
  (`src/typed_gguf/bench/devset.jsonl`, authored here, provenance recorded), and the published
  tables in `docs/BENCHMARKS.md` (raw reports `docs/evidence/e2_*.json`). A benchmark never reads
  the model registry and never opens a socket: `--model` is a path on disk. One command
  reproduces each table — `python3 tools/e2_reproduce.py --suite <name> --model <path.gguf>`.

## Thinking models are supported, thinking is off by default

The pinned `spark2_5` and `qwen35` templates both ship a reasoning block. typed-gguf renders through
the model's **own** template with `enable_thinking=false` and *checks the bytes*: the prompt that
is tokenized never leaves the model inside a thinking block (`docs/TEMPLATES.md` §3). `--thinking`
turns the block back on; `--template plain` switches to the model-agnostic framing.

## Quickstart (decide)

```bash
typed-gguf models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0      # once (4.4 GB, SHA-256 verified)
typed-gguf ask --state "The billing page is blank for every user since 09:12." \
            --choice "area=Which team owns this?:billing|technical|platform" \
            --score  "severity=How severe?:cosmetic|annoying|critical" \
            --noul   "page=Should we page the on-call engineer?"
```

```json
{"area": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.95, …},
          "confidence": 0.91, "coverage": 0.93, "reliability": "ok", "decode_steps": 3}}
```

`run` takes a whole request file (`--questions q.json`, with `--state`/`--state-json` to override
the state), `--format typesafe` emits the adapter's shape, `--out r.json` writes the response to
a file. Reuse a prefix across calls with `--state-id my-screen` (+ `--save-state`): the second
call reports `prefill_reused: true` and costs no prefill (the state file lives under
`$TYPED_GGUF_HOME/states/`).

The response says what **computed**, not what was requested: `engine.backend` is a claim whose
provenance is published next to it (`engine.backend_source`: `request`, `bundle`, `record` or
`default`), while `engine.devices` / `engine.device_buffers` / `engine.effective_backend` are read
back from the engine's own log (`llama_context`/`sched_reserve` compute buffers — `null` when that
log proves nothing: unverified, never claimed). A claim the log refutes, or cannot corroborate, is
named in `warnings` as `W_BACKEND_MISMATCH` instead of being published silently.

The same rule covers *why* an answer is unsure. `answers.<id>.reliability` is `ok`,
`low_confidence` or `low_mass`, and when the row the coverage was read from is a turn-closer the
response is named in `warnings` as `W_CUE_REFUSED` and carries the payload a flat list cannot —
the closer and its mass under `answers.<id>.cue`. "Not enough candidate mass" and "this family
closes the turn instead of answering" are different verdicts, and only one of them is a prompt
shape you can change (`docs/TEMPLATES.md` §4).

## Fit: what this host can actually hold

```bash
typed-gguf fit                                  # the default model, human-readable
typed-gguf fit Spark-X2.5-4B-Q8_0 --json        # {n_gpu_layers, n_ctx, kv_type, n_seq_max,
                                             #  est_weights_bytes, est_kv_bytes, est_total_bytes,
                                             #  backend, source}
```

`source` is `llama-fit-params` when the bundle's own tool produced the numbers, `estimate`
otherwise (with `W_FIT_ESTIMATED`). The plan is cached per `(model sha256, host fingerprint)` and
applied on load — `run`/`ask` honour it unless `--no-fit` is passed. Over budget, `kv_type` walks
`f16 → q8_0 → q4_0` (each step warns `W_KV_TYPE_DOWNGRADE`) before the context shrinks; the
estimate is cross-checked against measured load RSS within ±20 % on this box
(`docs/TEMPLATES.md` §5).

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

**Pid pressure (card t_a696ce02).** The worker container runs under a small, *shared* pid cgroup
(`pids.max = 256`, with the sibling sandboxes inside it), so a busy box answers `fork` with
`EAGAIN`. Two things follow, and the suite enforces both:

* the probe path (`typed_gguf.runtime.pressure.spawn`) retries a *transient* spawn failure for a
  bounded budget and names a *sustained* one — `E_PID_PRESSURE` plus the live reading
  (`pids.current=254/256 (2 free)`) — so a box that cannot fork is never reported as
  "the backend does not load on this host";
* tests that need a real child process are marked `@pytest.mark.needs_fork`. Every run's header
  prints the cgroup reading, and under a starved cgroup those gates **skip by name** while the run
  forces a non-zero exit: a run that could not measure them must not look green. Re-run when
  `pids.current` is lower; nothing here is a product finding.

`docs/evidence/e1a_baseline.json` records this box's measured numbers (prefill tok/s, warm
decode ms, KV footprint vs the conservative bound, the pull/resume/SHA transcript).

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
  typed `answers` wire shape that the `--format typesafe` adapter mirrors. `typed-gguf` makes **no
  parity claim** with it: the confidence statistic is our own and the documented quickstart
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

## Verifying a claim

Every number in `SPEC.md` is tagged: `[executed]` (reproduced in this repo right now),
`[recon]` (measured once, source recorded), `[target]` (to be measured later) or
`[UNVERIFIED]`. Anything in the E1a evidence file is `[executed]` unless it says otherwise.

## Quickstart (target shape, E1a+)

```bash
uv sync
uv run typed-gguf init                 # installs the pinned runtime, no compiler
uv run typed-gguf models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0
uv run typed-gguf run --questions q.json --state state.txt
```

```json
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days…",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this",
                   "criteria": {"billing": "Payment issues", "technical": "Bugs", "sales": "Pricing"}}
  }
}
```

## Verification

```bash
uv run pytest -q                                    # unit gate (offline)
python3 docs/verify_runtime_contract.py             # oracle: pinned facts + formulas
TYPED_GGUF_RUNTIME_DIR=<runtime> python3 docs/verify_runtime_contract.py   # + live ctypes probes
```

## License

MIT. The bundled/used llama.cpp release is MIT as well; model licenses are the model authors'.
`typed-gguf` is not affiliated with TypeSafe; the `typesafe` output format is a compatibility adapter.
