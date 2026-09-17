# ggufone

**GGUF-native typed decision engine** — `state` + typed questions → typed answers with full
probability distributions and confidence, computed locally on **frozen** GGUF models.

- **One pass over a shared prefix.** The state is prefilled once; every question forks the sequence
  state (`llama_memory_seq_cp`) and costs only its own short suffix decode.
- **No text generation.** Answers are read from logits over a fixed candidate set (restricted
  softmax), never sampled or parsed.
- **No fine-tuning. Ever.** Any GGUF llama.cpp can load is a valid backend; weights are never updated.
  The core has zero third-party runtime dependencies (stdlib only).
- **No compiler, ever.** `ggufone init` downloads a pinned official llama.cpp release bundle
  (shared libraries driven through `ctypes`). Users never build anything.

Default model: [`XHToken/Spark-X2.5-4B-GGUF`](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF)
`Q8_0`, pinned by size and SHA-256.

## Status

`SPEC.md` is the contract (milestones E1a → E3 with numbered acceptance criteria).

- **E1a (done)** — runtime + model registry: `ggufone init` (pinned llama.cpp bundle, no
  compiler on the path), `ggufone doctor`, `ggufone models {search,pull,use,ls,rm,verify,
  recommend-quant}`, the GGUF header reader, the conservative fit planner and the executed
  oracle green with the live runtime section.
- **E1b (next)** — the engine core: `run`/`ask`, fork readout, typesafe adapter.

## Verification

```bash
uv run pytest -q                                    # unit gate (offline: live tests are skipped)
uv run pytest -q --run-network                      # + real HF downloads / real GGUF headers
python3 docs/verify_runtime_contract.py             # oracle: pinned facts + formulas
GGUFONE_RUNTIME_DIR=<runtime> python3 docs/verify_runtime_contract.py   # + live ctypes probes
```

`docs/evidence/e1a_baseline.json` records this box's measured numbers (prefill tok/s, warm
decode ms, KV footprint vs the conservative bound, the pull/resume/SHA transcript).

## License

MIT. The bundled/used llama.cpp release is MIT as well; model licenses are the model authors'.
`ggufone` is not affiliated with TypeSafe; the `typesafe` output format is a compatibility adapter.

## Credits and attribution

`ggufone` stands on other people's work, and says so:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) — the inference runtime and the
  C ABI this project drives through `ctypes` (`libllama.so` / `libggml*.so`, pinned release
  `b11026`). `ggufone init` downloads the official release bundle; nothing here is built from
  llama.cpp source.
- **[TypeSafe](https://docs.typesafe.ai)** — the documented `state` + typed `questions` →
  typed `answers` wire shape that the `--format typesafe` adapter mirrors. `ggufone` makes **no
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
`ggufone models pull` / `models ls --json`, so the model author's terms travel with the file.

## Verifying a claim

Every number in `SPEC.md` is tagged: `[executed]` (reproduced in this repo right now),
`[recon]` (measured once, source recorded), `[target]` (to be measured later) or
`[UNVERIFIED]`. Anything in the E1a evidence file is `[executed]` unless it says otherwise.

## Quickstart (target shape, E1a+)

```bash
uv sync
uv run ggufone init                 # installs the pinned runtime, no compiler
uv run ggufone models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0
uv run ggufone run --questions q.json --state state.txt
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
GGUFONE_RUNTIME_DIR=<runtime> python3 docs/verify_runtime_contract.py   # + live ctypes probes
```

## License

MIT. The bundled/used llama.cpp release is MIT as well; model licenses are the model authors'.
`ggufone` is not affiliated with TypeSafe; the `typesafe` output format is a compatibility adapter.
