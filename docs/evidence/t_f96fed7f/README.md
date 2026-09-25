# t_f96fed7f — CI: the platform matrix comes alive (skip filter · Windows smoke · macOS serve/warm · warm window)

Head `ca3dbba` — the card's four commits (this receipt is the fifth; the card says NO push, so the
coordinator pushes and dispatches the matrix):

```
bee06b3  the oracle step's known-skip filter (AC1) + its 14 gates
20ba546  the serve probe — warm host vs the Windows inline fallback (AC2/AC3 core)
7e8a111  the model registration + the warm-window driver (AC2/AC3 wiring, AC4)
ca3dbba  the matrix itself — real Windows smoke, macOS serve/warm, the warm-window step
HEAD     this receipt (local, unpushed — `git log --oneline -5`)
```

Working tree clean afterwards, and **no `src/` change** at all (`git status --short src` → empty):
every gate below is a test/tool/workflow/docs change.

```
$ git diff --stat bee06b3~1..ca3dbba
 12 files changed, 2502 insertions(+), 13 deletions(-)

$ git show --stat ca3dbba
 .github/workflows/runtime-matrix.yml | 152 ++++++++++++++++++++++++---
 tests/test_matrix_windows_doctor.py  | 191 ++++++++++++++++++
 tests/test_runtime_matrix.py         | 246 +++++++++++++++++++++++++++++
 tools/matrix_windows_doctor.py       | 183 ++++++++++++++++++
 4 files changed, 759 insertions(+), 13 deletions(-)
```

Gates per new file (all green, `--collect-only` counts): the oracle filter 14,
the serve probe 16, the model registration 8, the warm window 12, the Windows doctor 10,
the workflow itself 22 — 82 collected, 80 passed + the 2 loopback legs skipped under the net block.

## GATES (host, all green)

```
$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-023 \
    uv run --extra dev pytest -q
1873 passed, 61 skipped in 56.39s

$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 uv run --extra dev pytest -q \
    tests/test_matrix_oracle_skips.py tests/test_matrix_serve_probe.py \
    tests/test_matrix_register_model.py tests/test_matrix_warm_window.py \
    tests/test_matrix_windows_doctor.py tests/test_runtime_matrix.py
80 passed, 2 skipped in 3.06s          # the two skips are the loopback legs, see AC3

$ uv run --extra dev ruff check src tests tools docs .github
All checks passed!

$ uv build
Successfully built dist/typed_gguf-0.2.3-py3-none-any.whl
```

**YAML sanity — what I used, stated as the card asks:**

1. **`actionlint` 1.7.12** (the Go binary from the PyPI package `actionlint-py`, no repo change):
   `actionlint .github/workflows/runtime-matrix.yml` → **exit 0**; `actionlint` (all workflows) →
   exit 0. That validates the job/step schema, `uses:` forms, expressions and `shell:` values.
2. **A per-step syntax pass** (`/tmp/wfcheck.py`, scratch): `yaml.safe_load` → every `run:` block is
   written to a file and checked with `bash -n` for the bash jobs, and with **PowerShell's own parser**
   (`[System.Management.Automation.Language.Parser]::ParseFile`, pwsh 7.4.6 tarball, `shell: pwsh`
   steps) for the Windows ones → **0 syntax failures** across all **28 run blocks** (5 of them pwsh).
3. A third check is the suite itself: `tests/test_runtime_matrix.py` parses the YAML with
   `yaml.safe_load` and reads it as data (steps by name), so a workflow that stops parsing fails 22
   tests, not one.

**QA note — mutation: not run, deliberately.** The card changes no `src/` statement at all
(`0` mutable statements under `src/typed_gguf/`), so a mutmut/stryker pass would mutate nothing this
card touched; its targets are exactly what the *tests* here pin (82 gates, each with a RED direction
driven in the suite — the injected skip, the host-claiming Windows body, the missing warning, the
never-found bundle, the drift-prone report, the warm window's five ways to fail).

## AC1 — the known-skip filter (linux-cpu + linux-vulkan)

`tools/matrix_oracle_skips.py` (committed earlier as `bee06b3` with its 14 gates). The old step was
`! grep -q "SKIP" <<< "$out"`, and the two skips that proved it wrong are *inherent* on a runner.

**RED receipt — the exact two lines from run `36151935399` through the new logic → allowed (exit 0):**

```
$ python3 tools/matrix_oracle_skips.py < /tmp/ac1_known_skips.txt
known model-absent skips: 2 (allowed — the 4.4 GB Spark GGUF cannot be on a runner)
  allowed: SKIP /home/runner/.hermes/models/Spark-X2.5-4B-Q8_0.gguf absent — sha256 download-verify evidence not re-run
           ^ the 4.4 GB Spark-X2.5-4B-Q8_0.gguf is not on a runner (and must never be downloaded here), so the sha256 download-verify pins are re-run on the box that holds the file
  allowed: SKIP local Spark GGUF absent — header-parse pins not re-run
           ^ the same file, the same reason: the GGUF header-parse pins need it
failures: 0  skips: 2
EXIT=0
```

**RED receipt — an injected third skip → non-zero exit (exit 1):**

```
$ python3 tools/matrix_oracle_skips.py < /tmp/ac1_injected.txt     # + "SKIP the live loader section did not run — no bundle in TYPED_GGUF_RUNTIME_DIR"
FAIL unexplained skip (a skip in this harness is a failure): SKIP the live loader section did not run — no bundle in TYPED_GGUF_RUNTIME_DIR
the oracle step is red: 1 problem(s)
… failures: 0  skips: 3
EXIT=1
```

**And the old logic on the same real lines** (why run 36151935399 was red):
```
$ grep -q "SKIP" /tmp/ac1_known_skips.txt ; echo $?
0        # `! grep -q "SKIP"` inverts this -> exit 1 -> both linux jobs fail
```

Both jobs now pipe the oracle's stdout through the filter (`printf '%s\n' "$out" | python3
tools/matrix_oracle_skips.py`); the file header, the step names and the comments say why. Matching is
on the two literal lines (stable substrings of the oracle's own messages), not on a prefix: an
injected skip of any shape is refused, including one that merely *looks* model-absent.

## AC2 — windows-cpu becomes a real smoke (and pins the TRUE Windows behaviour)

The job is no longer a zip download. Steps: flatten the pinned bundle (`llama.dll`, `ggml.dll`,
`ggml-base.dll` checked at the root, `TYPED_GGUF_RUNTIME_DIR` exported through `GITHUB_ENV`) →
`tools/matrix_windows_doctor.py` → 0.5B smoke GGUF → `bench --suite latency` end to end → a real
`serve` process + a real HTTP decision judged `--expect inline`.

**The Windows truth, from the product's own report.** `cli.doctor_checks()` with `platform.system()`
patched to `Windows` and no `nvidia-smi` (a CPU runner), bundle = the DLL names the pinned zip
actually carries (`/tmp/win_receipt.py` is the scratch driver; the same code path as the step, minus
`uv run`):

```
doctor exit 1 status 'failures'
  ok    runtime.present        /tmp/win-receipt-…/typed-gguf-rt
  fail  runtime.loadable       E_RUNTIME_MISSING: … is not a complete runtime bundle (missing libllama.so, libggml.so, libggml-base.so)
  fail  runtime.files          missing libllama.so, libggml.so, libggml-base.so
  warn  runtime.symbols        symbol probe skipped (TYPED_GGUF_DEEP_PROBE=0)
  fail  runtime.build          cannot determine the build number
  warn  runtime.fit_params     llama-fit-params not bundled (auto-fit limited)
  ok    runtime.backends       backends: cpu (driveable here: cpu)
  warn  runtime.accelerator    no accelerator in this bundle (expected 'vulkan'); CPU works but decoding is slower
  warn  runtime.sha_recorded   runtime.json has no libllama.so SHA-256 …
::warning::typed-gguf doctor cannot pass on Windows: runtime.lock's `required_files` are the Linux
SONAMEs (libllama.so, libggml.so, libggml-base.so), so `runtime.files`/`runtime.loadable` fail on a
bundle whose llama.dll is right there, and `runtime.build` cannot read a build from llama-cli.exe /
llama.dll. The bundle IS found and classified (backends: cpu (driveable here: cpu)). The loader is
platform-aware; the distribution check is not (SPEC R12). Card t_f96fed7f reports this instead of
hiding it.
judge: 0 problem(s) -> OK (the pinned truth holds)
```

So the step is **green while the mismatch is loud** (a `::warning::` annotation on every run), and it
turns **red** if any of it drifts: no bundle found, `runtime.files` not failing on the Linux SONAMEs,
`E_RUNTIME_MISSING` gone from `runtime.loadable`, `runtime.backends` not `cpu`, or an exit code other
than 1. Those are the 10 gates in `tests/test_matrix_windows_doctor.py` (including a "the gap is gone:
update this pin on purpose" direction).

**Facts about the pinned asset (verified on the real `llama-b11026-bin-win-cpu-x64.zip`, 18 439 911 B):**
```
   796672  ggml-base.dll      79872  ggml.dll
  3167232  llama.dll           9216  llama-cli.exe      9216  llama-fit-params.exe
51 entries at the archive root, no top-level directory (the job's flatten branch is for a re-pin)
$ grep -c "build 11026" llama.dll -> 0     # and "b11026" -> 0: the DLL carries no build string
```
The Windows serve leg is judged by the same `tools/matrix_serve_probe.py` as macOS, with
`--expect inline`: **no** `engine.keep` block in the served body, `W_KEEP_UNAVAILABLE` on the
server's own stderr (`cli.decide_payload_warm` prints it before the platform check's fallback), and
`served_by=host` must *not* appear — plus the model has to be a registry alias, which the new
`tools/matrix_register_model.py` provides (the served routes refuse paths, SPEC 2.9).
`Start-Process` + `taskkill /T /F` in a `try/finally`, with the probe's exit code carried out of the
`finally` (a pwsh step otherwise ends on the last command's code and would go green on a red probe).

## AC3 — macOS gains the serve story (real SDK 0.7.1, warm host, clean `keep stop`)

Added to `macos-engine-smoke` (AC3's own first option), **reusing the job's existing pinned bundle and
0.5B GGUF downloads — no new download of any kind**; the job is bounded at `timeout-minutes: 30`
(and all five jobs are now bounded, AC-constraints).

The step: fresh venv + `pip install typesafe-sdk==0.7.1` (asserted by `__version__`) → `matrix_register_model.py
--alias smoke-0.5b` → `serve --host 127.0.0.1 --port 8088 --keep-alive 600` in the background →
`TYPESAFE_API_KEY=… /tmp/sdk-venv/bin/python tools/host_gate_serve_client.py --base-url … --out …`
(the repo's own real-client gate, told nothing but the base URL) → `matrix_serve_probe.py --expect host`
(2 decisions; the **second** must be `engine.keep.served_by == "host"`, `timings.model_load_ms == 0.0`,
and `served_by=host` in the server's own log, because the TypeSafe projection drops `engine.keep`) →
`keep stop --json` **while the host is alive** → `keep status --json` must say `stopped`.

**Why one host answers both routes (checked locally, so the warm assertion is not a coin flip).**
`/v1/systemone` rewrites the body to the *registry alias* (`http.App._systemone`: `native = {"state",
"model": alias, …}`) and the probe names that same alias on `/v1/decide`; the two differ in `format`.
`keep.identity.KeepKey.of` reads only placement fields — no `format` — so both land on one key
(`/tmp/keycheck.py`, run against the repo):

```
probe  (/v1/decide, format=native) : {"backend":"auto","fit":true,…, "model_path":"/tmp/smoke.gguf","model_sha":"",…}
sdk    (/v1/systemone -> alias)    : {"backend":"auto","fit":true,…, "model_path":"/tmp/smoke.gguf","model_sha":"",…}
same host for the SDK's systemone call and the probe's native calls: yes
a different model path gets a different key: yes
```
(`model_sha` is `""` on both because `matrix_register_model.py` registers a local file — the same
registry entry serves both, so the key agrees.)

## AC4 — the warm window (linux-cpu)

New final step: `tools/matrix_warm_window.py --cli "uv run typed-gguf" --model /tmp/smoke.gguf
--keep-alive 20 --json /tmp/warm_window.json`. It drives the CLI as a subprocess:
`ask` (cold, with `--keep-alive 20`) → `keep status --json` (must say `running` with that pid and
`keep_alive_s == 20`) → `ask` again inside the window (must be `model_load_ms == 0.0`, same pid,
`served_by=host`) → sleep `20 + 10 s` → `keep status --json` must say `stopped`, the pid must be gone
and the socket file must be unlinked. **Nothing in that step calls `keep stop`** — a `keep stop` there
would make the claim a tautology, so the step is the idle unload itself (the mechanism behind the
600 s default). The 15–30 s window the card pins is asserted by `tests/test_runtime_matrix.py`
(a 10-minute window would burn runner for nothing). 12 gates in `tests/test_matrix_warm_window.py`
cover the judge in both directions plus the wiring (the four CLI invocations and the single sleep).

## AC5 — the constraints

* **No `src/` change** (`git status --short src` → empty). The only place a product change would have
  been needed is the Windows distribution gap, and that is a *finding*, not a test workaround: the
  job pins today's truth loudly instead of hiding it (AC2's decision slack).
* Existing pins stay green: `tests/test_release_publish.py` (the `checkout@v7` / `setup-uv@v7` mirror —
  which the new `tests/test_runtime_matrix.py` *imports* so the pins cannot drift apart),
  `tests/test_public_layout.py`, `tests/test_net_block_scope.py`, `tests/test_typed_gguf_surface.py`,
  `tests/test_pins.py` → 155 passed, 2 skipped with the six new files included.
* Offline vs live stays split: this workflow runs no `pytest` (gate: `"pytest" not in the workflow`),
  `ci.yml` downloads no bundle and has no `workflow_run` (gate: verified).
* Free-tier friendly: the only model anyone downloads here is the 0.5B smoke GGUF (gate: every
  `*.gguf` URL in the workflow ends in `qwen2.5-0.5b-instruct-q4_k_m.gguf`), no Spark URL anywhere;
  five standard runners; every job `timeout-minutes`-bounded (25/15/30/15/30).
* Receipts: this file. **No push** — the coordinator pushes, dispatches and reports the run.

## FINDINGS — docs vs reality (the card asks for these explicitly)

1. **`runtime.lock`'s `required_files` are Linux-only.** On Windows `doctor` fails `runtime.files` and
   `runtime.loadable` on a *complete* bundle (`libllama.so`, `libggml.so`, `libggml-base.so`), while
   the same report says `backends: cpu` — the loader is platform-aware, the distribution check is not.
   This is the SPEC R12 "CI smoke job per platform" gap, reported by the job on every run.
2. **`finder.TOOL_NAMES` has no `.exe`** (`("llama-cli", "llama-fit-params", "llama-tokenize")`) and
   the pinned `llama.dll` carries no build string, so `runtime.build` is "cannot determine the build
   number" on Windows and `runtime.fit_params` warns (auto-fit is limited there). Same family as (1).
3. **`runtime.accelerator` expects `vulkan` on a Windows host without `nvidia-smi`**, so a *CPU*
   bundle warns there too (warn, not fail). Not pinned: it depends on whether the runner has a GPU,
   so pinning it would over-fit the pin to one runner image.
4. **`serve`'s inline fallback on Windows is real and named** — `W_KEEP_UNAVAILABLE` is printed on
   **stderr** by `cli.decide_payload_warm` *before* the fallback, and the served body has no
   `engine.keep` block. Both halves are asserted (the job passes `--server-log` twice: stdout+stderr).
5. **The keep key ignores `format`** (AC3 above) — worth writing down: it is the reason the SDK route
   and the native route share one warm host, and it is the assumption the warm assertion rests on.
6. The stale Windows comment (`# … once E1a lands`) is gone, and the job no longer uses `/tmp/…` on
   Windows (which is `C:\tmp\…` there): `$env:RUNNER_TEMP` + `TYPED_GGUF_RUNTIME_DIR` via `GITHUB_ENV`.

## What only the dispatched run can prove

Everything above is the harness + the judge, exercised on this box: the tools are driven against real
loopback HTTP doubles (16/16 in `tests/test_matrix_serve_probe.py` with the net block off; the two
loopback legs skip under the block because `AF_INET` is forbidden there), the Windows report comes
from the product's own `doctor_checks()` under a patched platform, and the macOS step is judged from
the same code paths. **The live proof is the matrix run** (windows-cpu booting `llama.dll` and
answering a real decision; macos-engine-smoke's real SDK call; linux-cpu's 20 s window) — dispatched
by the coordinator from this branch, with the run id reported back. Two risks I would watch in that
first run: the Windows `--backend auto` placement on a CPU-only box, and the macOS serve step's
`nohup`+`pkill` teardown ordering.
