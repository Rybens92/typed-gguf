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

---

# E1a FIX QA — t_eae35404: host-dependent detection + recorded backend fallback

Date: 2026-09-17 | Tier: M (default — no tier declared on the card) | Author: code-tdd
Commits: `f3ae67d` (probe purity), `6ef79d2` + this commit (fallback, doctor, host gate)

## Why (the coordinator's finding, reproduced)

On the operator's RTX 3060 Ti host the E1a suite had 7 failures: `detect_backend` read the
real machine *behind* injected probes, so `detect_backend({system: linux, dri_nodes: [...]})`
answered `cuda` (test wanted `vulkan`) and `host_variant("auto", linux, x86_64)` answered
`linux-x64-cuda-12.8` (test wanted `linux-x64-cpu`); 5 CLI tests use that same mapping.
The E1a worker had verified in a GPU-less podman sandbox, where the same code was green.

## What changed

1. **Probe purity** (`runtime/pins.py`). Detection is a pure function of a `HostProbes`
   object. `current_host()` is the single function in the package that reads the machine
   (`platform.system/machine`, `shutil.which("nvidia-smi")`, `DRI_DIR.glob("renderD*")`,
   `ICD_DIR`); supplying *any* probe argument (or `probes=`) switches to a synthetic world in
   which unset facts count as ABSENT, so no host read can leak behind injected probes. Only
   `detect_backend()` / `host_variant("auto")` with no probes at all read the real host, so
   production behaviour on a GPU box is unchanged.
2. **`InstallPlan.backend`** is the accelerator (`cuda`/`vulkan`/`metal`/`cpu`) via
   `pins.accelerator_of()`, not the version suffix (`"12.8"`); `--dry-run --json` also reports
   the probe facts it decided from (`host`).
3. **Recorded fallback** (`runtime/capability.py`, `runtime/install.py`). A deep probe dlopens
   every `libggml-<backend>` library it finds and keeps the raw error in
   `ProbeResult.backend_errors`. `install(backend="auto")` walks `cuda → vulkan → cpu`,
   records `backend_requested` / `backend_working` / `backend_errors` / `fallback_attempts` /
   `fallback_reason` in `runtime.json`, **removes** a bundle this host cannot drive (it would
   otherwise shadow the working tier in `find_runtime()`), and installs the first tier that
   loads. An explicit `--backend` is honoured as asked (no silent substitution).
4. **`doctor --json`** reports `runtime.working_backend`, per-backend load errors and a
   `runtime.fallback` check naming what was tried and why.
5. **`finder.find_runtime()`** prefers the variant `runtime.json` records.

## Tests added (each written first as RED)

| Test | What it pins |
|---|---|
| `test_pins.py::test_the_whole_mapping_in_a_fake_host_world[cpu/vulkan/cuda]` | detect → variant → pinned asset/size → install plan, GPU-absent **and** GPU worlds |
| `test_pins.py::test_probes_never_fall_back_to_the_real_host` | tripwires on `shutil.which`/`platform.*`/`/dev/dri`/ICD — any leak raises |
| `test_pins.py::test_current_host_is_the_only_reader_of_the_real_machine` | real vs synthetic probe object |
| `test_cli_e1a.py::test_init_dry_run_on_a_gpu_host_plans_the_pinned_cuda_bundle` | GPU-world CLI plan: variant, asset, 168 811 114 B, pin sha, probe facts |
| `test_runtime_fallback.py` (9 cases) | chain stops at vulkan, chain to cpu, no fallback when cuda loads, explicit backend honoured, rejected dir dropped, `find_runtime` preference, doctor on both paths, real dlopen seam |
| offline CLI suites | run in an explicit fake CPU machine (assertions unchanged) |

## Gate results — container, GPU simulated with a fake `nvidia-smi` on PATH

`tools/host_gate_e1a.sh /work/e1a/logs/gate-final`, all numbers from its logs (final code):

| step | exit | elapsed | result |
|---|---|---|---|
| `uv run pytest -q` (CPU-only container, no runtime) | 0 | 2.0 s | 283 passed, 11 skipped |
| `uv run pytest -q` (fake nvidia-smi on PATH) | 0 | 3.4 s | 283 passed, 11 skipped |
| `init --dry-run --json` | 0 | 1 s | `variant=linux-x64-cuda-12.8`, 168 811 114 B |
| `init --json` | 0 | 2 s | `variant=linux-x64-vulkan`; CUDA tier rejected + recorded (`fallback_attempts`) |
| `doctor --json` | 2 | 0 s | warnings; `backends: cpu, rpc, vulkan (driveable here: vulkan)`, `runtime.fallback` warn |
| `version --json` | 0 | 0 s | record read back |
| `uv run pytest -q` (runtime installed) | 0 | 17 s | 284 passed, 10 skipped (oracle section B live) |
| `python3 docs/verify_runtime_contract.py` | 0 | 5 s | failures: 0, **section B skips: 0** |
| `uv run pytest -q --run-network` | 0 | 28 s | 293 passed, 1 skipped (pinned Qwen3.5 GGUF absent) |
| poisoned PATH `init` (fresh home, offline cache) | 0 | 3 s | **0** compiler shims, budget 180 s |

The first full install of the CUDA tier in this sandbox (fresh home, real download) took 2 s wall
time end-to-end and produced the fallback below; the `already_installed` path is what the table
above shows for the re-run.

Fallback evidence, verbatim from `init.json` / `runtime.json`:

```
fallback_reason: cuda does not load on this host
  (libggml-cuda.so: libcudart.so.12: cannot open shared object file: No such file or directory)
backend_requested: cuda   backend_working: vulkan   backends: [cpu, rpc, vulkan]
```

Real assets measured in the sandbox (sha256 of the files on disk):

```
5b2d30d7a5e448fbe0aceda360c8f9ed2949aa1734e94db078e6d0b722521e2b  llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz (168 811 114 B == pin)
1b40310bf4d47c2c84853ebb4ccaf4dcbd992596cd1c2f610be6a0532a874708  llama-b11026-bin-ubuntu-vulkan-x64.tar.gz (30 294 625 B == pin)
5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2  Spark-X2.5-4B-Q8_0.gguf (4 375 021 152 B == HF lfs.oid)
ldd libggml-cuda.so → libcudart.so.12 => not found ; libcublas.so.12 => not found ; libcuda.so.1 => not found
```

## What this FIX does NOT verify (and why)

* **Real GPU execution.** The sandbox has no `/dev/dri`, no `/dev/nvidia*`, no GPU: the CUDA
  bundle's *dlopen* failure above is real, but device enumeration, Vulkan shader warm-up and
  CUDA kernels are not exercised. A-E1a-1's host half is a **pending operator run**:
  `tools/host_gate_e1a.sh` + `tools/host_gate_summary.py` (raw logs + `host_gate_e1a.json`).
* **Whether the host's CUDA bundle loads.** On the operator's box the driver is present; if
  libcudart/libcublas are not, the same fallback will fire there and the gate will record
  `variant=linux-x64-vulkan` + `fallback_reason` — that is a pass for requirement 4 and a
  finding for requirement 3's "per dry-run" expectation, not a silent downgrade.

## Mutation (Tier M, soft threshold)

mutmut 3.8, `source_paths = ["src/ggufone"]` with the test selection reduced to the tests that own
the touched modules (`tests/test_pins.py`, `tests/test_runtime_fallback.py`,
`tests/test_runtime_install.py`, `tests/test_cli_e1a.py`, `tests/test_cli_doctor_branches.py`),
scoped by name filter to the touched modules (`mutmut run 'ggufone.runtime.pins*'
'ggufone.runtime.capability*' 'ggufone.runtime.install*' 'ggufone.runtime.finder*' 'ggufone.cli*'`):

```
killed 2405 | survived 1759 | no tests 136 | timeout 3   ->  score 57.7%
(1803 mutants outside the filter: "not checked")
```

**Soft threshold, recorded not looped on.** Survivor triage (the method: `mutmut results`, then
`mutmut show <id>` on every survivor whose name touches the new decision path —
`detect_backend`/`resolve_host`/`fake_host`/`HostProbes`/`accelerator_of`/`usable`/`accelerator`/
`load_backend_library`/`install`/`_unusable_reason`/`_drop_rejected`/`find_runtime`):

* **Critical path: none.** No surviving mutant can silently accept an unloadable backend, install
  a rejected variant, skip a SHA-256 check, or break the atomic extract-then-move.
* **Real gaps found and closed in this commit** (tests added after the run, so the recorded score
  is the *pre-closure* one): `load_backend_library` dropping `RTLD_GLOBAL` (the pinned libs must
  load global — `test_load_backend_library_uses_rtld_global` spies on `CDLL`) and
  `ProbeResult.accelerator()` mis-handling the `("cpu","rpc","base")` tuple
  (`test_usable_and_accelerator_semantics` now pins `backends=("base","cpu","rpc")`).
* **Message-string / equivalent class (the bulk).** `ProbeResult.failures()`/`_unusable_reason`
  text, `find_runtime` error prose, docstring edits of `install()`, `system=None`/`machine=...`
  kwargs that are masked by an explicit `probes=` object.
* **Masked-by-design.** `_drop_rejected(ignore_errors=...)`, `runtime_dirs` ordering and
  `library_names(None)`: behaviour identical on every path the suite exercises.

The E1a baseline's own 72.7% was a different scope (`errors/gguf/recommend/store/pins`); modules
like `cli.py` (print-heavy) and `install.py` were never mutation-tuned before, which is where most
of the 57.7% comes from — `cli.py` alone contributes ~800 survivors, nearly all of them output
strings. Score is reported for the reviewer's call, not used as a gate here (Tier M).

## Risks

🔴 **None known.** The failing behaviour the coordinator found is reproduced by the new
fake-host tests and cannot recur without failing them.

🟡 **`runtime.fallback` warns whenever a GPU tier was rejected** — on a host where CUDA cannot
load, `init`/`doctor` deliberately stay at exit 2 (warnings) rather than pretending the GPU is
in use. Intended, but it means "doctor exit 0 on this host" needs the CUDA bundle to load.

🟡 **Fallback is probe-based, not device-based.** E1a proves a backend *loads*; that a GPU is
present and drives it is E1b's context init. Documented in `capability.load_backend_library`.

🟢 **Test edits were environment pins, not expectation changes.** The 5 CLI tests that failed
on the host assert the same plan as before; their autouse fixture now names the fake CPU
machine they always assumed. No assertion was weakened (verified by diff review).

## Recommendation

**Option A (ship the FIX), with the host gate as the single outstanding item.** Every
sandbox-observable criterion has a command + output behind it and the suite is green in both
worlds; the only acceptance criterion this worker cannot execute is the real-GPU run, which is
prepared as a one-command script with a machine-readable summary.
