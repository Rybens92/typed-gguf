# t_f3637b8e — cuda → vulkan → cpu fallback with a machine-readable reason, reported by doctor

Card: *Add cuda→vulkan→cpu backend fallback with reason + doctor reporting* (requirement 4 of the
E1a FIX decomposition of `t_eae35404`). Repo `/workspace/ggufone` (= the card's workspace), branch
`main`. The repo has **no remote** (`git remote -v` is empty), so `git push` cannot be satisfied —
flagged here and in the completion report instead of hidden (every sibling E1a card has the same
state).

The fallback chain itself landed with the parent FIX card (`6ef79d2` chain + record, `cd89f7e`
robustness, `77e40ee` one-bundle-per-process). This card closes what requirement 4 still left in
prose and pins it with tests **and** real-bundle evidence:

1. **machine-readable reason per step** — every `fallback_attempts[]` entry now carries a stable
   `code` from `ggufone.runtime.install.REASON_CODES` (`loader_error`, `system_libs_missing`,
   `no_asset`, `backend_absent`, `probe_failed`, `unusable`) next to the human `reason`; the code of
   the first fallback is repeated as `fallback_reason_code` in `runtime.json`, `init --json` and
   `doctor --json` (`runtime.fallback_reason_code`, and the `runtime.fallback` check text);
2. **`doctor --json` reports the WORKING backend and the probe-derived `backends` list** — top-level
   `backend` / `backends` (aliases of `runtime.working_backend` / `runtime.backends`), both taken
   from the probe of the *found* bundle, never from a table;
3. a small reporting fix found on the way: `init` on a runtime directory that has no `runtime.json`
   (copied in by hand) used to warn `fell back from None to …`; it now names the backend that *this
   run* detected and asked for.

Changed files: `src/ggufone/runtime/install.py`, `src/ggufone/runtime/capability.py`,
`src/ggufone/cli.py`, `tools/host_gate_summary.py`, `tests/test_runtime_fallback.py`,
`tests/test_probe_isolation.py`.

## 1. TDD: the new tests fail on HEAD and pass on the change

HEAD = `12c6393`; the tests were written first and dropped into a pristine worktree of that commit
(`/work/tf363/pristine`), unchanged:

```
$ cd /work/tf363/pristine && python3 -m pytest -q tests/test_runtime_fallback.py tests/test_probe_isolation.py
FAILED tests/test_runtime_fallback.py::test_install_falls_back_from_cuda_to_vulkan_and_records_why
FAILED tests/test_runtime_fallback.py::test_a_backend_that_does_not_load_here_is_coded_loader_error
FAILED tests/test_runtime_fallback.py::test_the_preflight_skip_is_coded_system_libs_missing
FAILED tests/test_runtime_fallback.py::test_a_tier_this_lock_does_not_pin_is_coded_no_asset
FAILED tests/test_runtime_fallback.py::test_a_bundle_that_carries_no_such_backend_is_coded_backend_absent
FAILED tests/test_runtime_fallback.py::test_a_dead_probe_child_is_coded_probe_failed
FAILED tests/test_runtime_fallback.py::test_reason_codes_are_a_closed_set
FAILED tests/test_runtime_fallback.py::test_doctor_reports_the_working_backend_the_list_and_the_fallback_code
FAILED tests/test_runtime_fallback.py::test_doctor_reads_the_backends_off_the_bundle_it_found
FAILED tests/test_probe_isolation.py::test_init_without_a_record_names_the_backend_this_run_asked_for
10 failed, 40 passed in 14.39s            EXIT=1
```

The same two files on the change (both files include the pre-existing fallback contract, so the
10th failure above is the one *existing* assertion extended from `{backend, variant, reason}` to the
4-key attempt shape — nothing was weakened, one key was added):

```
$ python3 -m pytest -q tests/test_runtime_fallback.py tests/test_probe_isolation.py
50 passed in 8.69s                        EXIT=0
```

`tests/test_runtime_fallback.py` gained 9 tests (reason codes, the closed code set, the two
doctor/probe tests, the dlopen-vs-file-list test); `tests/test_probe_isolation.py` gained 1 (the
`backend_requested` fix).

## 2. The world the live-shaped evidence runs in

This container has **no GPU**: no `nvidia-smi`, no `/dev/dri`, no NVIDIA driver — and, like the
operator's RTX 3060 Ti box, **no `libcudart`**. The GPU host is therefore shaped at the OS level
(a fake `nvidia-smi` first on `PATH`), and everything else is real: the real `runtime.lock`, the
real pinned assets over the network, the real loader (`ctypes`/`dlopen`), the real probe child.

```
$ cat /work/tf363/fakebin/nvidia-smi
#!/bin/sh
echo "NVIDIA-SMI 550.127.05   Driver Version: 550.127.05   CUDA Version: 12.4"
echo "GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-xxxx)"
```

The pinned CUDA bundle from an earlier run of this project
(`/work/e1a/home/runtime/b11026-linux-x64-cuda-12.8`, 207 MB, real `libggml-cuda.so`, 173 MB) is
used for the "the bundle really does not load here" observations.

## 3. Acceptance, claim by claim

### 3.1 The GPU-shaped world asks for CUDA (and would download 168.81 MB)

```
$ PATH=/work/tf363/fakebin:$PATH GGUFONE_HOME=/work/tf363/home-rtx uv run ggufone init --dry-run --json
  "backend": "cuda",
  "variant": "linux-x64-cuda-12.8",
  "asset": "llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz",
  "size": 168811114, "size_human": "168.81 MB",
  "sha256": "5b2d30d7a5e448fbe0aceda360c8f9ed2949aa1734e94db078e6d0b722521e2b",
  "host": {"system": "linux", "machine": "x86_64", "has_nvidia_smi": true, "backend": "cuda"}
                                                                             EXIT=0
```

### 3.2 `init` (auto) falls back cuda → vulkan with the reason recorded (real download)

```
$ PATH=/work/tf363/fakebin:$PATH GGUFONE_HOME=/work/tf363/home-rtx uv run ggufone init --json
  "installed": true, "variant": "linux-x64-vulkan", "backend": "vulkan", "working_backend": "vulkan",
  "asset": "llama-b11026-bin-ubuntu-vulkan-x64.tar.gz",
  "asset_sha256": "1b40310bf4d47c2c84853ebb4ccaf4dcbd992596cd1c2f610be6a0532a874708",
  "asset_verified": true, "bytes_fetched": 30294625, "build": 11026,
  "backends": ["cpu", "rpc", "vulkan"], "backend_errors": {},
  "fallback_attempts": [
    {"backend": "cuda", "variant": "linux-x64-cuda-12.8", "code": "system_libs_missing",
     "reason": "pre-flight: the pinned linux-x64-cuda-12.8 bundle links libcudart.so.12, libcublas.so.12,
                libcuda.so.1, which this host cannot load (libcudart.so.12: cannot open shared object file:
                No such file or directory); skipped the 168.81 MB download and moved to the next tier"}],
  "fallback_reason_code": "system_libs_missing"
                                                                             EXIT=0
```

`runtime.json` (the record) carries the same code and the per-step attempts:

```
$ python3 /work/tf363/record-fields.py
backend_requested = cuda
backend_working = vulkan
fallback_reason_code = system_libs_missing
fallback_attempts = [{'backend': 'cuda', 'code': 'system_libs_missing', 'reason': 'pre-flight: the pinned
  linux-x64-cuda-12.8 bundle links libcudart.so.12, …', 'variant': 'linux-x64-cuda-12.8'}]
```

### 3.3 The recorded reason *matches the actual failure observed* (re-observed live, not read back)

`/work/tf363/verify-claims.py` probes the pinned CUDA bundle again in a fresh probe child and
dlopens `libcudart.so.12` directly, then asserts the loader message appears in the recorded reason:

```
$ python3 /work/tf363/verify-claims.py
== 2. the SAME failure, re-observed here, now (not read back from a log)
   probe(pinned cuda bundle).backend_errors['cuda'] = 'libggml-cuda.so: libcudart.so.12: cannot open shared object file: No such file or directory'
   dlopen('libcudart.so.12')                        = 'libcudart.so.12: cannot open shared object file: No such file or directory'
   probe(pinned cuda bundle).accelerator()          = 'cpu' (usable(cuda)=False)
== 3. the recorded reason matches that observed failure
   ok: the recorded reason embeds the loader message verbatim, quoted in parentheses
== 4. doctor --json reports the WORKING backend and the probed backends list
   backend=vulkan backends=['cpu', 'rpc', 'vulkan'] runtime.working_backend=vulkan status=warnings exit=2
   runtime.fallback check: warn

ALL ASSERTIONS PASSED
```

### 3.4 The real-bundle `loader_error` path (a CUDA directory left behind by an earlier run)

With the real pinned CUDA bundle sitting in the home (`cp -al`, so the same bytes) and the vulkan
bundle beside it, the probe really dlopens the CUDA bundle, records the loader's own message and
drops the rejected directory:

```
$ PATH=/work/tf363/fakebin:$PATH GGUFONE_HOME=/work/tf363/home-stale2 uv run ggufone init --json
  "already_installed": true, "variant": "linux-x64-vulkan",
  "backend_requested": "cuda", "working_backend": "vulkan",
  "fallback_attempts": [
    {"backend": "cuda", "variant": "linux-x64-cuda-12.8", "code": "loader_error",
     "reason": "cuda does not load on this host (libggml-cuda.so: libcudart.so.12: cannot open shared
                object file: No such file or directory)"}],
  "fallback_reason_code": "loader_error"
$ # stderr
warning: fell back from cuda to linux-x64-vulkan: cuda does not load on this host (libggml-cuda.so:
libcudart.so.12: cannot open shared object file: No such file or directory)                          EXIT=0
```

Before this card that warning read `fell back from None to …` (no `runtime.json` in that home);
the new test `test_init_without_a_record_names_the_backend_this_run_asked_for` pins `cuda`.

### 3.5 `doctor --json` — the working backend, the probed list, the code

```
$ PATH=/work/tf363/fakebin:$PATH GGUFONE_HOME=/work/tf363/home-rtx uv run ggufone doctor --json
  "backend": "vulkan",
  "backends": ["cpu", "rpc", "vulkan"],
  "expected_backend": "cuda",
  "runtime": {"working_backend": "vulkan", "backends": ["cpu", "rpc", "vulkan"],
              "backend_errors": {}, "backend_requested": "cuda",
              "fallback_reason": "pre-flight: … libcudart.so.12: cannot open shared object file …",
              "fallback_reason_code": "system_libs_missing",
              "symbols_probed": true, "symbols_missing": [], "fit_params_help_exit": 0},
  "checks": [… {"id": "runtime.backends", "status": "ok",
                "detail": "backends: cpu, rpc, vulkan (driveable here: vulkan)"},
              {"id": "runtime.fallback", "status": "warn",
                "detail": "installed linux-x64-vulkan after cuda was unusable here [system_libs_missing]: …"}]
                                                                             EXIT=2  (warnings: fallback + no model)
```

`backends` is the probe's answer for the bundle that is actually installed (34/34 symbols resolved,
each `libggml-<backend>.so` dlopened in a child), and `working_backend` is what this host can really
drive — pinned by `test_doctor_reads_the_backends_off_the_bundle_it_found` (the same directory
gains `libggml-vulkan.so` → the list and the working backend both change) and
`test_the_working_backend_is_decided_by_dlopen_not_by_the_file_list` (a bundle that *carries* cuda
but cannot load it reports `backends=["cpu","cuda"]`, `accelerator()="cpu"`).

## 4. Gates

```
$ python3 -m pytest -q                                        328 passed, 12 skipped   EXIT=0   (/work/tf363/suite-plain.log)
$ PATH=<fake nvidia-smi> python3 -m pytest -q                 328 passed, 12 skipped   EXIT=0   (/work/tf363/suite-gpuworld.log)
$ ruff check src tests tools docs                             All checks passed!      EXIT=0   (/work/tf363/ruff.log)
$ python3 docs/verify_runtime_contract.py                     exit 0 · failures: 0 · skips: 3   (/work/tf363/oracle-offline.log)
$ GGUFONE_RUNTIME_DIR=<installed vulkan> python3 docs/verify_runtime_contract.py
                                                              exit 0 · failures: 0 · skips: 3   (/work/tf363/oracle-live.log)
      section B live: 32/32 llama symbols, 2/2 ggml symbols, build b11026 == pin, spark2_5 impl,
      llama-fit-params exit 0 — the only section-B SKIP is the 4.4 GB Spark model, which is not in
      this container (so `tests/test_runtime_contract.py::test_oracle_live_section_is_green_without_skips`
      reports that one missing-file SKIP; it is the same on pristine HEAD, see §5)
$ coverage (pytest --cov, offline suite)   cli 91% · capability 90% · install 83% · total 88%
```

Three "worlds" of the offline suite were checked: plain (above), GPU-shaped (above), and with a
runtime installed in `HOME` (`HOME=/work/tf363/home-inst`, 328 passed, 11 skipped, 1 failed — the
oracle-live test above, whose sole remaining SKIP is the absent 4.4 GB model; identical on pristine
HEAD).

## 5. Tier-M mutation testing (mutmut 3.8, whole-package sweep, soft threshold)

Run in a byte copy of the tree (`/work/tf363/mutrun`, made with `copy_function=shutil.copy` so
mutmut's own `copystat` cannot hit the container's SELinux xattr denial, plus the
`/work/mutmut-shim` sitecustomize):

```
$ python3 /work/tf363/uv-mutmut-run.py run --max-children 2
```

mutmut generates **every** mutant in the package (the name filter only selects which *results are
printed*), and the run orders by estimated test time, so this is a whole-package sweep that is only
read as per-function numbers. It was stopped at **5253/6704** after ~40 min: this container shares
a 256-pid cgroup with sibling cards (a Stryker + a vitest run were live), `os.fork()` inside
mutmut died with `BlockingIOError: [Errno 11] Resource temporarily unavailable`, and
`--max-children 2` only brought it back to ~2 mutations/s (the FIX card's run, on a quieter box,
was ~4x faster). **Partial, not a score** — numbers from
`/work/tf363/mutmut-score-partial.txt`, per function changed or touched here:

| function | killed | survived | score | triage |
|---|---|---|---|---|
| `install._attempt` | 9 | 0 | **100%** | — |
| `install._preflight_reason` | 12 | 3 | 80% | 2 equivalent (`system_libs.get(variant, ())` → `None` / implicit default: both falsy) + 1 message-string (`', '` join) |
| `install._unusable_reason` | 3 | 4 | 43% | 4 survivors are `or`→`and` / `'none'`→`'NONE'` / `', '`→`'XX, XX'` **inside the "(backends: …)" prose**; the branch *code* is asserted |
| `install._build_record` | 63 | 49 | 56% | pre-existing body (record assembly); the `fallback_reason_code` key this card added is asserted on both the fallback and the no-fallback path |
| `cli._cmd_init` | 216 | 152 | 59% | pre-existing body; the new `fallback_reason_code` / `backend_requested` payload keys are asserted |
| `cli._cmd_doctor` | 47 | 41 | 53% | pre-existing body; the new text-mode `backend=` / `fallback:` lines are covered by the CLI doctor tests |
| `cli.doctor_checks` | 370 | 317 | 54% | pre-existing body; the new `runtime.fallback_reason_code` + top-level `backend`/`backends` are asserted |
| `install.install` | — | — | not reached | the sweep was stopped before its mutants ran |

Survivor taxonomy is the E1a FIX card's (`docs/evidence/e1a_qa.md`): message-string and equivalent
mutants are **deliberately** not killed (asserting whole sentences freezes prose the SPEC does not
pin). **No survivor on the new decision path**: the codes, the closed `REASON_CODES` set, the
`fallback_reason_code` fields and the doctor aliases are all code-asserted, and the helper that
builds an attempt (`_attempt`) scores 100%.

## 6. Tier-M QA note (condensed: risk-weighted summary, decision, confidence)

🔴 **REQUIRES ATTENTION (blocks ship): none.**
All acceptance-relevant behaviour is asserted *and* re-observed on the real pinned bundles.

🟡 **WORTH CONSIDERING (your call):**
* *The live host gate is card `t_3831b7b3`, not this one.* This container cannot dlopen a driver
  (no `/dev/dri`, no `nvidia-smi`, no `/dev/nvidia*`). The GPU world here is shaped at the OS level
  with a fake `nvidia-smi`; everything downstream (detection, asset pins, SHA verify, the real
  loader in the probe child, the network) is real. → **Recommendation: that card confirms; no
  action here.**
* *Mutation coverage is partial.* The sweep was stopped by the shared pid cap (see §5). The
  functions this card changed are all scored except `install.install` itself, and every survivor
  seen is message-string or equivalent. → **Recommendation: accept for Tier M (soft threshold, no
  return-loop); a quieter box can finish the sweep.**
* *`REASON_UNUSABLE` is a defensive default.* `ProbeResult.usable()` is exactly
  `backend in backends and backend not in backend_errors`, so for an accelerator the third branch
  of `_unusable_reason` is unreachable today; it exists so a future `usable()` change cannot
  produce an unexplained skip. Its mutant can only survive. → **Recommendation: keep.**
* *Three pre-existing suite failures in the `GGUFONE_RUNTIME_DIR=<dir>` world* (`test_runtime_
  contract.py::test_oracle_live_section_is_green_without_skips` needs the 4.4 GB model that is not
  in this container; two `find_runtime` tests are bypassed by that env var). **Identical on pristine
  HEAD `12c6393`** — not a regression. → **Recommendation: ignore here.**

🟢 **ACCEPTABLE:** `pytest -q` 328 passed / 12 skipped in both the plain and the GPU-shaped worlds
(exit 0) · ruff clean · oracle exit 0, failures 0, section B live (32/32 symbols, build == pin) ·
coverage cli 91% / capability 90% / install 83% (total 88%) · the recorded reason *matches the
loader error observed live*, and doctor reports `backend=vulkan`, `backends=[cpu, rpc, vulkan]`
from the probe.

🤔 **DECISION — ship or fix?**
* **Option A — ship** (recommended): the whole change is additive keys + a code vocabulary, all
  asserted; the chain's behaviour is unchanged where it already worked (there is an explicit test
  that a loadable CUDA tier is *kept*). Residual risk: the host run is still pending on another
  card, and the mutation sweep is partial.
* **Option B — fix** (+~1 h): finish the mutation sweep on a quiet box and add message assertions
  for the surviving prose mutants. Buys little: prose is not the contract.

💡 **Recommendation: Option A.** No 🔴 item, no survivor on a decision path, and the acceptance
claims are backed by commands + real outputs (§3).

📈 **CONFIDENCE: 8/10**
  + 10 new tests, RED on HEAD (10 failed / 40 passed) → GREEN (50 passed); suite 328/12 in two worlds
  + real pinned bundles + real loader: the recorded code/reason was re-observed, not read back
  + doctor's `backend`/`backends`/`fallback_reason_code` are covered in both the "working" and the
    "nothing recorded" states
  − the live RTX 3060 Ti run belongs to `t_3831b7b3` (pending)
  − mutation is partial (pid cap), and `install.install` itself was not reached

## 7. Honest limits

* The **live gate on the operator's RTX 3060 Ti host is card `t_3831b7b3`** (code-e2e), not this
  one; this container cannot run it (no `/dev/dri`, no `/dev/nvidia*`, no driver). The coordinator's
  earlier host run (logdir `/home/rybens/.ggufone-host-gate-20260917T170441Z`, not visible from
  here) already recorded `init.variant=linux-x64-vulkan` with the real host fallback reason; the
  post-`77e40ee` re-run is that card's step 2.
* `nvidia-smi` is faked here; everything downstream of detection is real (assets, SHA pins, loader,
  probe child). A host with a working `libcudart` is expected to *keep* CUDA — the chain only
  leaves a tier the loader rejects (or the pre-flight proves impossible), which is exactly what the
  tests assert with an injected loader for the positive case.
* `git push` is impossible: `git remote -v` is empty in this repo (no `gh` either). Commits are on
  local `main`, as with every sibling E1a card.
