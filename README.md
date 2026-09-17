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
This repository currently contains the SPEC, the executed interface oracle
(`docs/verify_runtime_contract.py`), the captured evidence (`docs/evidence/`) and a
scaffold. Implementation starts at E1a.

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
