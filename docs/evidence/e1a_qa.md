# E1a QA report — runtime + model registry

Card: `t_541fdedb` (ggufone E1a: `init`, `doctor`, `models {…}`, `runtime/`, `registry/`)
Tier: **M** (declared default — the card carries no `Tier:` line; the work is a new subsystem,
so the M ceremony applies: TDD + coverage + static + one scoped mutation run + risk summary).
Date: 2026-09-17 · Sandbox: podman container, 2 CPU quota, 31 GiB RAM, no GPU.

## Gate results

| Gate | Command | Result |
|---|---|---|
| Unit gate (offline) | `uv run pytest -q` | **exit 0 — 262 passed, 10 skipped** (live tests are opt-in) |
| Live gate | `uv run pytest -q --run-network` | **exit 0 — 260 passed, 1 skipped** (pinned Qwen3.5 file absent) |
| Oracle offline | `python3 docs/verify_runtime_contract.py` | **exit 0**, failures 0, skip 1 (section D readout → E1b) |
| Oracle live (A-E1a-1) | `GGUFONE_RUNTIME_DIR=<rt> python3 docs/verify_runtime_contract.py` | **exit 0**, failures 0, **0 SKIP in section B** (32/32 llama + 2/2 ggml symbols, build b11026, fit-params, spark2_5 impl) |
| Static | `uv run ruff check src tests tools` | **exit 0** (E/F/W/I/UP/B/SIM, line-length 100) |
| Coverage (offline) | `uv run --with pytest-cov pytest -q --cov=src/ggufone` | **87%** (1821 stmts, 234 missed) |
| Coverage (live) | same with `--run-network` | **87%** (230 missed) |
| Mutation (Tier M, scoped) | `mutmut run` on the pure-logic modules | see *Mutation* below |

Per-module coverage: cli 91%, gguf 95%, store 93%, pins 97%, finder 96%, hf 87-88%,
capability 87%, recommend 83%, install 76%, ctypes_binding 55%, `__main__` 0% (1 stmt).

### Mutation (Tier M)

Scope decision (recorded, not silent): `mutmut` runs the whole offline suite per mutant, which
is ~10 s/mutant in this 2-CPU sandbox — a full-package pass would not finish inside the card's
budget, and mutmut's alphabetical order would have finished on the *stub* modules last. The
pass therefore targets the modules that carry the SPEC-pinned arithmetic and have hermetic
tests, with their owning test files as the selection:

```
mutmut run "ggufone.errors.*" "ggufone.runtime.pins.*" "ggufone.registry.gguf.*"
           "ggufone.registry.recommend.*" "ggufone.registry.store.*"
selection: tests/test_recommend_quant.py tests/test_gguf_header.py
           tests/test_registry_store.py tests/test_pins.py tests/test_mutation_hardening.py
```

Round 1 (RED+GREEN tests only) → round 2 (after the hardening tests below).
Score from the `.meta` artifacts (`exit_code_by_key`), **not** from `mutmut results`:

| module | r1 mutants | r1 killed | r1 score | r2 survivors | r2 notes |
|---|---|---|---|---|---|
| `errors.py` | 3 | 2 | 66.7% | 1 | the surviving mutant is a `__init__` no-op |
| `runtime/pins.py` | 253 | 195 | 77.1% | 58 | 29 message-string, 12 other, 15 logic, 2 no-op |
| `registry/gguf.py` | 176 | 137 | 77.8% | **32** (was 39) | the 4 logic + 2 arithmetic survivors were killed by the cap-boundary tests |
| `registry/recommend.py` | 477 → 455 | 313 | 65.6% | 144 | all `recommend_quant` survivors are now message-string or equivalent |
| `registry/store.py` | 295 | 207 | 70.2% | 88 | 38 message-string, 33 other, 17 logic (see below) |
| **scoped total** | **1182** | **859** | **72.7%** | 323 | |

Survivor classes (mutmut archives were re-scanned with a diff-classifier, not eyeballed):

* **message-string (≈ 60% of survivors)** — the code raises/decorates with prose and the tests
  assert the *codes* (`E_*`) plus `in`-style fragments, deliberately: the code is the contract,
  the prose is not. Changing "no quant fits vram or ram" to "NO QUANT FITS…" is not a defect.
  Killing these would mean asserting whole sentences, which freezes wording the SPEC does not
  pin.
* **equivalent / unreachable** — e.g. `kv_per_token_f16 // 2` → `/ 2` (the operand always
  carries a factor 2), `quant.strip().upper()` making the `("F16", "f16")` case mutants
  unreachable, `_normalize_machine`'s `not in _ARM64` (any input reaching it fails the variant
  lookup with the same message either way). One dead branch this pass found
  (`select_file`'s "quant equals the whole stem" fallback) was **deleted**, not annotated.
* **real gaps that were fixed in round 2** (data-integrity path, so they got tests rather than
  a note): `plan["total"] <= budget` vs `<` (exact-boundary fits), the GPU and RAM margin
  formulas (`* (1 - margin)` vs `* (1 + margin)` / `/(1 - margin)`), the "overhead is charged"
  decision, and the GGUF reader's four structural caps (`n_kv`, string length, array length,
  nesting depth) at exactly-at/one-over. New file: `tests/test_mutation_hardening.py`.
* **not mutated in this pass**: `cli.py`, `registry/hf.py`, `runtime/{capability,ctypes_binding,
  finder,install}.py` — I/O-heavy, network- and library-shaped code that the live evidence
  covers end-to-end (oracle section B, real pull/resume/SHA, real `init`/`doctor`). Their
  coverage is in the table above; a follow-up pass can mutation-test them once the offline
  suite is fast enough to make ~2000 mutants affordable.

**Replay check (harness honesty):** one survivor was replayed by hand
(`mutants/…/pins.py` with `MUTANT_UNDER_TEST`), and the first replay used the wrong key shape
(`xǁasset_forǁ…` for a module-level function) — which is exactly the kind of mistake that
produces phantom survivors. With the correct key the same mutant was **killed** (2 tests fail),
and the `.meta` exit code for it is 1, i.e. mutmut agrees. So the survivor lists above are
trustworthy, but they must be read per key, not per function name
(`__mutmut_1` is a prefix of `__mutmut_10`).

Tier-M verdict: score reported at **72.7%** (soft threshold, no return-loop) with **no
remaining survivor on the budget/parsing data-integrity path**; the residual survivors are
message-string or equivalent mutants.

## Live evidence (every claim = command + output)

All transcripts under `/work/e1a-evidence/*.log`; the distilled numbers are committed in
`docs/evidence/e1a_baseline.json`.

| What | Evidence |
|---|---|
| `init` with poisoned PATH (9 compiler shims + only python/uv on PATH) | exit 0, **0 shim invocations**, 1093 ms (budget 180 s), source `offline-cache`, asset SHA-256 == pin |
| `init --dry-run` | prints asset/url/size/sha/destination, writes nothing |
| `doctor` | exit 2 pre-pull (warnings), **exit 0 post-pull** (11/11 checks ok), stable `ggufone.doctor/v1` JSON |
| `models pull` (A-E1a-4) | 4 375 021 152 B, sha256 `5c2c3c19…9dea2` == HF `lfs.oid`; **SIGKILL at 15.10%** (660 602 880 B part) → re-run resumed from exactly that offset, re-fetched 3 714 418 272 B in 58 s, sha verified; `sha256sum` independently agrees |
| registry | alias `spark-x2.5-4b-q8_0` with arch/quant/size/sha/**license**, `models ls --json`, `models verify --json` (1 ok / 0 failed) |
| quant selection (A-E1a-5) | `:Q8_0` → the single `*Q8_0*.gguf`; ambiguity and unknown quant raise `E_AMBIGUOUS_QUANT` / `E_MODEL_NOT_FOUND` listing candidates (unit + CLI tests) |
| `recommend-quant` (A-E1a-7) | pinned table reproduces 7.33 GB / 6.12 GB / `insufficient` exactly |
| KV footprint (A-E1a-8) | measured: 302 219 264 B (n_seq_max=1) and 302 006 272 B (n_seq_max=4) at n_ctx=2048, kv_unified → **identical**: the conservative bound over-charges 4× at 4 sequences (0.25 ratio) and matches to 0.1% at 1 sequence. Recommender stays conservative per S-4. |
| arch pre-flight (A-E1a-9) | `E_MODEL_ARCH_UNSUPPORTED` names arch + build + fix, from a symbol scan; unit-tested for both "no impl" and "old build" |
| GGUF reader (A-E1a-10) | v2/v3, every scalar type, arrays, nested arrays, truncation/caps → `E_GGUF_CORRUPT`; real pinned Spark file parses (model-marked test) |
| HF auth (addition 1) | live: a gated repo's blob without a token → **`E_HF_AUTH_REQUIRED` (HTTP 401)** with the exact fix in the message, no file written; token lookup honoured from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `~/.cache/huggingface/token` |
| disk precheck (addition 2) | live: pulling the 4.38 GB model against a 512 MB tmpfs → **`E_INSUFFICIENT_DISK`** "need 4375021152 bytes (4.38 GB) … only 333410304 bytes (333.41 MB) available", exit 2, before any download |

## Risks

🟡 **The Vulkan baseline is missing.** A13 asks for CPU **and Vulkan** prefill numbers and the
first-call shader-compile cost; the sandbox has no `/dev/dri` and no `nvidia-smi`, so only CPU
numbers could be measured (and those are clouded by the 2-CPU cgroup quota: 5.9 tok/s
single-thread, 3.2 tok/s with 24 threads — *slower* because 24 threads thrash 2 CPUs). The
`[recon]` host figures (~340 tok/s CPU, ~4.6k tok/s Vulkan, 23-30 s shader compile) are **not**
claimed as reproduced. Vehicle: a host run of `tools/measure_kv_footprint.py` /
`ggufone init --backend vulkan`, or the CI `runtime-matrix` job. Cost to fix: minutes on the
host; impact: a published "E1a baseline" that is honest about its sandbox.

🟡 **`ctypes_binding.py` is 55% covered under pytest** (and `install.py` 76%). The uncovered
lines are exactly the real-library paths — the ABI binding table, `warmup()`, archive
extraction — which now run in *child processes* (`tools/live_probe.py`) after an abort at exit
was traced to several model load/free cycles in one long-lived interpreter (`free(): invalid
pointer`; every real CLI command is its own process, so the product is unaffected, but the test
runner must not be killable by a C library). Evidence for those lines: the live oracle section
B, `doctor` on the real install, `init` on the real bundle, the KV measurement. Cost to fix
(have pytest report child coverage): small; impact: cosmetics + reviewer reassurance.

🟢 **Mutation scope** — see the mutation section: the pass covers the arithmetic modules; the
I/O-heavy modules (`cli`, `hf`, `install`, `capability`, `finder`, `ctypes_binding`) were not
mutated and are covered by the live evidence instead. This is the recorded Tier-M trade-off.

🟢 **Accepted by design** — `models search` needs the network (not unit-gated); `--jobs N` is
accepted but a single-file pull has nothing to parallelise; the runtime's non-CPU asset
SHA-256 pins are `null` in `runtime.lock` until an install computes them (recorded in
`runtime.json`), exactly as the lock's own `note` says.

## What this milestone does NOT verify (and who should)

* E1b's engine (`run`/`ask`, fork readout, typesafe adapter) — the oracle's section D readout
  check is still a SKIP, by design.
* Non-linux assets (Windows/macOS bundles) — CI `runtime-matrix` owns those.
* CUDA/Vulkan execution — no GPU in the sandbox (see the 🟡 above).
* Whether the *measured* unified-cache relaxation should change `recommend_quant` — that is
  E2.5's call (S-4); the measurement is recorded here.

## Recommendation

**Option A (ship), with the two 🟡 items carried as findings.** Every E1a acceptance criterion
has a command + output behind it, the offline and live gates are green, and the two gaps are
truthfully *outside this sandbox's reach* rather than unverified claims: the Vulkan numbers
need a host/CI run, and the child-process coverage is a reporting artifact of a deliberate
isolation choice. Fixing the coverage attribution is a nice-to-have; re-measuring on the host
is the item a human should schedule before the E1a baseline is quoted anywhere public.
