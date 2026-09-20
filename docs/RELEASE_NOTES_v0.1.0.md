# typed-gguf v0.1.0 — typed decisions on any GGUF (release-notes draft, not yet published)

> Draft for the v0.1.0 release. Nothing is tagged or published yet: no git tag, no GitHub release,
> no PyPI project (the name is reserved but unpublished), and the repository the install section
> points at is the artifact this text ships with. Every number below is already published inside
> this repository and marked where it was measured.

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

## What v0.1.0 does not include

- **HTTP and MCP serving.** `typed-gguf serve` and `typed-gguf mcp` are specified (SPEC §2.9:
  `/health`, `/v1/models`, `/v1/decide`, `/v1/systemone`; the `typed_gguf_*` tool set) but they are
  not implemented in this release — both commands exit 3 with a milestone pointer. The CLI is the
  interface that ships.
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
`docs/BENCHMARKS.md`, `docs/TEMPLATES.md` and `SPEC.md` carry the full detail behind every number
here.
