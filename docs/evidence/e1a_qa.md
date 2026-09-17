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

---

# E1a FIX QA round 2 — t_eae35404: the host SIGABRT + the four coordinator findings

Card: `t_eae35404` (round 1 = probe purity + the recorded cuda→vulkan fallback).
Tier: **M** (the card declares no tier → default M: one scoped mutation run, soft threshold,
recorded — not looped on).
Date 2026-09-17 · same podman container: 2-CPU quota, no `/dev/dri`, no `/dev/nvidia*`, no
`nvidia-smi`, no `libcudart`/`libcublas`/`libcuda`.

## What the host run found

The coordinator ran `bash tools/host_gate_e1a.sh` on the operator's RTX 3060 Ti box: 10 of 11
steps green — the real `cuda → vulkan` fallback with its dlopen reason recorded, `doctor` 34/34
symbols / build b11026 / `llama-fit-params --help` exit 0, oracle section B with **0 SKIP**,
offline and `--run-network` suites green, poisoned PATH with **0** compiler shims — but:

```
init.exit  134   ← SIGABRT *after* printing the JSON: `double free or corruption (!prev)`
```

reproducible, including on the idempotent re-run. `init --dry-run` and the poisoned-PATH run
(fresh home, offline cache) did **not** crash — both are runs without the two-bundle probe.

## Root cause

A glibc heap-corruption abort at the very end of a command that had:

1. dlopened the CUDA bundle it then rejected (`probe_symbols` → that directory's
   `libggml-base/ggml/llama.so` copies become resident, `RTLD_GLOBAL`),
2. deleted that directory (`_drop_rejected`) while those copies stayed mapped,
3. dlopened the vulkan bundle on top — with a real driver behind it on that host, so the
   process also held a live GPU backend,
4. and then let its own shutdown run *other people's* destructors.

`doctor` (one directory, the same libraries) exits clean on that host; the **fallback chain** is
what makes `init` different. The repo already knew the shape of this problem — the live tests
probe every real bundle in a child process, "because shared-library teardown can abort at exit
(`free(): invalid pointer`)" (`tests/test_runtime_live.py`) — `init`/`doctor` were the place
that did not.

## Fix: one bundle per process

* `ggufone/runtime/probe_child.py` — a disposable child: one JSON request on stdin
  (`probe` | `warmup` | `libs`), one JSON object on stdout. Exit 0 means "the answer is data"
  (including "this bundle is broken"); a non-zero exit means the probe itself failed and the
  caller records that instead of guessing.
* `ggufone/runtime/isolated.py` — the client: `child_command()`, `run_child()` (a crash, a hang
  or garbage stdout all become `ChildFailure` carrying the exit code and the stderr tail),
  `scan_bundle()`, `warmup_in_child()`, `system_libs()`.
* `capability.DEFAULT_SCAN` (default `isolated.scan_bundle`) is the seam `probe_runtime` uses;
  `capability.scan_in_process` is the very code the child runs (and what tests inject through
  the `in_process_scan` fixture when they patch the dlopen seams).
* `install.DEFAULT_WARMUP` (default `isolated.warmup_in_child`) — the warm-up number is worth a
  process, not worth leaving a loaded model and a GPU driver behind in the command.
* No ggufone command dlopens a bundle any more: `probe_symbols` / `load_backend_library` /
  `load_system_lib` are called only inside the child (plus tests that opt in).

Measured on the real pinned bundles (`uv run python /work/e1a/live_isolation_check.py`):

```
[vulkan]      symbols_checked=True error=None  backends=('cpu','rpc','vulkan') accelerator=vulkan
mapped after the probe: <none>
[cuda]        backend_errors={'cuda': 'libggml-cuda.so: libcudart.so.12: cannot open shared object file: …'}
deleted:      …/b11026-linux-x64-cuda-12.8
[vulkan-copy] accelerator=vulkan
mapped after probe -> delete -> probe: <none>
OK: no bundle is mapped in this process; the probes still answered
```

i.e. probe → delete → probe (the sequence that aborted on the host) leaves this process with
nothing mapped while the probe keeps answering exactly what it answered before.

## The four secondary findings — all four taken

1. **`asset_verified: false` for the Vulkan bundle** → both shipped GPU assets are pinned in
   `runtime.lock` (vulkan `1b40310b…` — observed by the coordinator *on the host* and in the
   sandbox; cuda-12.8 `5b2d30d7…` — observed locally against the immutable release asset), and
   the gate's `init` JSON now reports `asset_verified: true`. The lock's `note` records the
   provenance rule: a sha is pinned only when it was observed.
2. **168.8 MB downloaded before falling back** → pre-flight: `runtime.lock` → `system_libs`
   (cuda-12.8 links `libcudart.so.12`/`libcublas.so.12`/`libcuda.so.1`, vulkan links
   `libvulkan.so.1`), dlopen-checked **in a child, before the download**. On the host-shaped
   world the CUDA archive never moves — the gate's fresh-home `init` downloaded only the
   30 294 625 B vulkan asset (`bytes_fetched: 30294625`, `asset_verified: true`) and recorded
   the reason verbatim:
   `pre-flight: the pinned linux-x64-cuda-12.8 bundle links libcudart.so.12, libcublas.so.12, libcuda.so.1, which this host cannot load (libcudart.so.12: cannot open shared object file: No such file or directory); skipped the 168.81 MB download and moved to the next tier`.
   An explicit `--backend cuda` deliberately skips the pre-flight: the user asked, the real
   probe then records the truth. A pre-flight that cannot run *fails closed* (skips the tier).
3. **`init.fallback` null in the summary** → the `already_installed` return carries
   `fallback_reason` (this run's attempts, else the record), the CLI prints it (plus the warning
   line) and `tools/host_gate_summary.py` falls back to `record.fallback_reason`. The gate
   summary now reads `"init.fallback": "pre-flight: …"`.
4. **Doctor suggested `init --backend cuda` although libcudart is missing** → when the install
   record names the reason for that backend, the `runtime.accelerator` check repeats it:
   `expected accelerator 'cuda' is not in this bundle (backends: cpu, rpc, vulkan): pre-flight:
   … libcudart.so.12 …; install the cuda runtime it needs and re-run \`ggufone init\`, or keep
   'vulkan' (driveable here)`. The retry hint stays for the case it was written for (a bundle
   that simply carries no such backend).

## Tests added (each written RED first) — `tests/test_probe_isolation.py`, 23 tests

| Test | What it pins |
|---|---|
| `test_deep_probe_does_not_dlopen_anything_in_this_process` | the in-process seams raise if touched: a probe that runs in the command is a test failure |
| `test_the_child_really_loads_the_bundle` | a real ELF in `libllama.so`: the child resolves (and misses) the ABI for real, `error is None` |
| `test_the_child_reports_symbols_and_backends_in_one_answer` | both halves of the scan in one answer (real ABI libs + an unloadable backend) |
| `test_the_probe_child_answers_the_documented_protocol` | `python -m ggufone.runtime.probe_child`, one JSON request/response, nothing else |
| `test_a_probe_child_that_crashes_is_recorded_not_swallowed` | exit 9 → `error` + every accelerator in `backend_errors` → the chain can still fall through |
| `test_garbage_from_the_probe_child_is_recorded` / `test_a_probe_child_that_hangs_times_out` | stdout that is not JSON is data; a hung probe is a timeout |
| `test_install_falls_all_the_way_to_cpu_when_the_probe_cannot_run` | a bundle nobody can verify never wins over a tier that loads |
| `test_preflight_skips_the_cuda_download_when_the_host_cannot_load_cudart` | the 168 MB archive is **not** in `<home>/downloads`; the reason is recorded |
| `test_preflight_proceeds_when_the_host_can_load_the_libraries` | no false skip |
| `test_a_dead_preflight_child_fails_closed` | an unrunnable pre-flight skips the tier instead of guessing |
| `test_an_explicit_backend_skips_the_preflight` | `--backend cuda` is an instruction, not a guess |
| `test_already_installed_reports_the_recorded_fallback_reason` | finding 3 at the `install()` level |
| `test_cli_init_on_an_installed_runtime_prints_the_fallback_reason` | finding 3 through `cli.main(["init","--json"])` |
| `test_host_gate_summary_reads_the_fallback_from_the_record` | finding 3 in the gate tool |
| `test_doctor_names_the_missing_runtime_instead_of_a_retry` / `…_keeps_the_retry_hint…` | finding 4 both ways |
| `test_runtime_lock_pins_the_gpu_assets_it_shipped` | finding 1: the pins and the `system_libs` table |
| `test_init_warmup_runs_outside_the_command_process` / `…_dies_is_recorded_not_raised` / `…_answers_the_documented_protocol` | the warm-up number is a child's answer; a dead warm-up is recorded, never raised |
| `test_system_lib_probe_reports_what_this_host_can_load`, `test_preflight_matches_the_pinned_lock…`, `test_probe_child_modes_are_callable_in_process` | the pre-flight surface and the child surface |

Plus, in `tests/test_runtime_live.py` (marked `model`, runs under `--run-network`):

* `test_the_command_process_maps_no_bundle_after_a_deep_probe` — deep-probes the **real**
  installed bundle and then asserts `/proc/self/maps` contains no `libggml`/`libllama` at all.
  That is the host crash, turned into an assertion.

The 4 fallback tests that patch `capability.load_backend_library` now request the
`in_process_scan` fixture: they test the implementation the child runs, on purpose; the
remaining fallback tests run the unpatched (child) path. No assertion was weakened.

## Gate results — this container, GPU world simulated with a fake `nvidia-smi` on PATH

| Gate | Command | Result |
|---|---|---|
| Unit gate (CPU-only world) | `uv run pytest -q` | **exit 0 — 307 passed, 12 skipped** (3.6 s) |
| Unit gate (GPU world) | `PATH=<fake-bin> uv run pytest -q` | **exit 0 — 307 passed, 12 skipped** (3.7 s) |
| Unit gate (GPU world + runtime installed) | `HOME=<gate home> uv run pytest -q` | **exit 0 — 308 passed, 11 skipped** (13.7 s; oracle section B live + the `/proc/self/maps` test) |
| Live gate | `HOME=<gate home> uv run pytest -q --run-network` | **exit 0 — 318 passed, 1 skipped** (29 s; pinned Qwen3.5 GGUF absent) |
| Oracle (offline) | `python3 docs/verify_runtime_contract.py` | **exit 0**, failures 0, skips 1 (section D readout → E1b) |
| Oracle (live, A-E1a-1) | `python3 docs/verify_runtime_contract.py` with the runtime installed | **exit 0**, failures 0, **0 SKIP in section B** |
| Static | `uv run ruff check src tests tools docs` | exit 0 |
| Coverage | `uv run --with pytest-cov pytest -q --cov=ggufone` | **88%** (2092 stmts, 249 missed) |
| Mutation (Tier M, scoped) | `mutmut run` on the changed modules | 66.0% on the five modules scored — see *Mutation* below |

Host gate rehearsal (the exact host command, `bash tools/host_gate_e1a.sh <logdir>`, fresh home,
real downloads, fake `nvidia-smi`, model symlinked for the oracle) —
raw JSON: `docs/evidence/host_gate_sandbox_rehearsal_round2.json`:

```
pytest_before        exit=0 elapsed_s=10   307 passed, 12 skipped
init_dry_run         exit=0 elapsed_s=0    backend=cuda variant=linux-x64-cuda-12.8
init                 exit=0 elapsed_s=2    variant=linux-x64-vulkan, bytes_fetched=30294625, asset_verified=true
doctor               exit=2 elapsed_s=0    34/34 symbols, build b11026, backends cpu, rpc, vulkan (working: vulkan)
version              exit=0 elapsed_s=0
pytest_after         exit=0 elapsed_s=17   308 passed, 11 skipped
oracle               exit=0 elapsed_s=4    failures: 0, section B skips: 0
pytest_network       exit=0 elapsed_s=30   318 passed, 1 skipped
init_poisoned_path   exit=0 elapsed_s=0    0 compiler shims (budget 180 s)
{
 "init.exit": 0,
 "init.variant": "linux-x64-vulkan",
 "init.fallback": "pre-flight: … skipped the 168.81 MB download and moved to the next tier",
 "doctor.exit": 2, "oracle.exit": 0, "oracle.section_b_skips": 0,
 "pytest_before.exit": 0, "pytest_after.exit": 0, "pytest_network.exit": 0,
 "poisoned_init.exit": 0, "poison_shims_invoked": 0
}
```

`init.exit` is the step that aborted with SIGABRT on the host; here it is 0, and the CUDA
archive was never downloaded.

## Coverage

`uv run --with pytest-cov pytest -q --cov=ggufone` → **88%** total (2092 stmts, 249 missed).

Per changed module: `pins.py` 97%, `cli.py` 91%, `capability.py` 90%, `isolated.py` 86%,
`install.py` 82%, `probe_child.py` 82%. The uncovered lines in the two new modules are the
child-side paths a parent coverage run cannot see (`warmup_in_child`'s error branches,
`probe_child.main()`), and they are unit-tested in-process (`probe_child.handle(...)`).

## Mutation (Tier M, soft threshold — recorded, triaged, not looped on)

`uv run --extra dev --with mutmut mutmut run 'ggufone.runtime.isolated*'
'ggufone.runtime.probe_child*' 'ggufone.runtime.capability*' 'ggufone.runtime.install*'
'ggufone.runtime.pins*'` with the new `tests/test_probe_isolation.py` in the selection:

| Module | killed | survived | score |
|---|---|---|---|
| `runtime/pins.py` | 295 | 49 | **85.8%** |
| `runtime/capability.py` | 215 | 116 | 65.0% |
| `runtime/isolated.py` (new) | 140 | 87 | 61.7% |
| `runtime/probe_child.py` (new) | 54 | 34 | 61.4% |
| `runtime/install.py` | 416 | 290 | 58.9% |
| **total scored** | **1120** | **576** | **66.0%** |

Triage: the survivors are dominated by **message strings** (every f-string in
`run_child`/`scan_bundle`/`system_libs`, the child's mode wrappers, install's pre-flight prose —
no assertion can see them and none changes a decision) and by `child_env`'s PYTHONPATH
assembly (indistinguishable from the one pytest already exports). **No survivor sits on a
decision path**: the decision seams are covered by kills — a dead/hung/garbage probe making
every accelerator unusable, the pre-flight skip, the explicit-backend bypass, the pins, the
`child_error` → `backend_errors` mapping. Two gaps the triage exposed were closed right after
the run (recorded, not hidden): the `backend_errors` mapping of a bundle whose ABI *does* load
(`test_the_child_reports_symbols_and_backends_in_one_answer`) and the fail-closed pre-flight
(`test_a_dead_preflight_child_fails_closed`).

`cli.py` was not re-scored in this cache: it has ~2400 mutants, is print-heavy, and the run was
stopped by the container's **256-pid cgroup** (a sibling card's Stryker run plus mutmut's
workers killed two attempts with `BlockingIOError: [Errno 11]` in `os.fork()`). Round 1 scored
it inside its 57.7% overall; its round-2 additions are covered by the two CLI/doctor tests in
the table above. Score reported for the reviewer's call, not used as a gate (Tier M).

## Risks and what this does NOT verify

🔴 **None known for the crash.** The mechanism (a command process that dlopens two bundles, one
of them deleted, and then exits) cannot recur without failing
`test_deep_probe_does_not_dlopen_anything_in_this_process` or the live `/proc/self/maps` test.

🟡 **The host crash itself was never reproduced here.** This container has no GPU driver, so a
teardown path that only executes with a live device is not reproducible in the sandbox. What is
proven here is that the *class* is gone (no third-party library in the command process at all,
measured) and that everything the command does before/after still works with the real bundles.
The host re-run is what proves the instance — stated plainly rather than claimed as reproduced.

🟡 **The warm-up number is now measured in a child** — same call, same bundle, one extra process
(~0.3 s). `test_init_records_the_warmup_number` (live, real 4.4 GB pinned model) still passes.

🟡 **The pre-flight is a host fact, not a promise.** `system_libs` is populated only for the two
variants whose dependencies were verified (`ldd` on the pinned archives: neither bundles
`libcudart`/`libcublas`); a variant absent from the map is not pre-flighted, and a passing
pre-flight changes nothing about the real probe that follows.

🟢 **No expectation was weakened.** Round 1's host-failing assertions are intact
(`test_detect_backend`, `test_host_variant_mapping`, the 5 CLI tests).

## Recommendation

**Option A (ship), with the host re-run as the single outstanding item.** Every
sandbox-observable criterion has a command and its output behind it, the suite is green in both
worlds, and the gate rehearsal is green *including `init`* — the step that aborted on the host.
The one acceptance criterion this worker still cannot execute is the live run on the RTX 3060 Ti
box; it is one command, and after this round the expected `init.exit` is **0** (either
`linux-x64-cuda-12.8` if that host can load cudart, or `linux-x64-vulkan` after the pre-flight
skip — with the reason recorded in both cases).
