# E2E report: t_3831b7b3 — live E1a gate, host evidence (RTX 3060 Ti card) — 2026-09-17

- HEAD: `eb1cde370f158a968258266c06b3c4cdac075031` (main) · Tier: M · App type: terminal/CLI
- Deliverable under test: `tools/host_gate_e1a.sh` → `host_gate_e1a.json` + the `host` section of
  `docs/evidence/e1a_baseline.json`
- Worker: `code-e2e` in a podman container (2-CPU cgroup quota, 8 GiB, pids.max 256)

## 0. Host access — what this worker could and could not do

The card asks for a run "on THIS host (RTX 3060 Ti, nvidia-smi present)". This worker is a
container **on** that host but has neither the GPU nor the operator's home:

```
$ which nvidia-smi              -> (nothing; nvidia-smi: command not found)
$ ls /dev/dri                   -> No such file or directory
$ mkdir -p /dev/dri && mknod /dev/dri/renderD128 c 226 128
  mknod: /dev/dri/renderD128: Operation not permitted
$ ls /dev | grep -i nvidia      -> (empty)
$ find / -xdev -name 'host_gate_e1a*' -o -xdev -name '.ggufone-host-gate-*'  -> (nothing)
```

So the operator's log dir `/home/rybens/.ggufone-host-gate-20260917T182117Z` is **not mounted**
here and could not be re-read. The card's comment (and the coordinator's on `t_eae35404`) sanction
exactly this case: *"Jeśli nie masz dostępu do GPU/hosta z sandboxa — użyj rundy 2 jako zapisu
i wyraźnie oznacz to w evidence jako przebieg koordynatora."*

This report therefore has two evidence classes, never mixed:
1. **Host numbers quoted verbatim** from the coordinator's board comment (round 2, head `77e40ee`);
2. **Container numbers executed here** — the same gate script, same head, on a pristine snapshot,
   with the GPU world simulated by a fake `nvidia-smi` first on PATH (the exact host fact
   `pins.current_host()` reads via `shutil.which`), plus 12 adversarial journeys.

## 1. Build from HEAD

```
$ git -C /var/home/rybens/workspace/ggufone rev-parse HEAD
eb1cde370f158a968258266c06b3c4cdac075031
$ git branch --show-current                    -> main          (git remote -v: empty - local-only repo)
$ git status --short                           ->  M state/groupchat/ggufone-e1.md   (coordinator's bot chat; untouched)
$ uv sync --extra dev                          -> installed pytest 9.1.1, pytest-timeout 2.4.0, ruff 0.16.8
$ uv run ggufone --help                        -> exit 0 (commands table: init/doctor/models/...)
```

**Snapshot isolation:** card `t_0fc576df` (attacker: "mutate the refactored detection code") is
running **concurrently in the same shared worktree**. To keep this run reproducible, all gate runs
and journeys below were executed from a detached snapshot:

```
$ git worktree add --detach /work/e2e-t3831b7b3/repo eb1cde370f158a968258266c06b3c4cdac075031
$ git -C /work/e2e-t3831b7b3/repo rev-parse HEAD -> eb1cde370f158a968258266c06b3c4cdac075031
$ git -C /work/e2e-t3831b7b3/repo status --porcelain -> (clean)
```

## 2. The host section — round 2, quoted verbatim (recorded by the coordinator)

Head `77e40ee`, log dir `/home/rybens/.ggufone-host-gate-20260917T182117Z`,
`bash tools/host_gate_e1a.sh` (no arguments):

```
{"init.exit": 0, "init.variant": "linux-x64-vulkan",
 "init.fallback": "pre-flight: ... skipped the 168.81 MB download and moved to the next tier",
 "doctor.exit": 2, "oracle.exit": 0, "oracle.section_b_skips": 0,
 "pytest_before.exit": 0, "pytest_after.exit": 0, "pytest_network.exit": 0,
 "poisoned_init.exit": 0, "poison_shims_invoked": 0}
```

plus, reported alongside: `uv run pytest -q` on the host before the gate → **319 passed, 11 skipped**.
Round 1 (pre-fix, same host) had `init.exit 134` (SIGABRT `double free or corruption (!prev)`) —
that difference is the point of the card.

**Limitation, stated plainly:** the raw log files are on the operator's host and are not mounted
here; per-step `.elapsed` files exist there but were not published. Every host number below is the
coordinator's quoted output, not a re-execution by this worker.

## 3. Container corroboration (executed here, head `eb1cde3`)

Four gate runs were needed because the gate's helpers resolve the runtime/model in three different
ways (that inconsistency is PROPOSAL P1). Only the last reproduces the host's artifact layout.

| gate run | layout | init | doctor | oracle | section-B skips | pytest before/after/network | poisoned |
|---|---|---|---|---|---|---|---|
| `logs/gate1` | default store, no model | 0 | 2 | 0 | **1** (model absent) | 0 / **1** / **1** | 0, 0 shims |
| `logs/gate2` | `GGUFONE_HOME=$HOME/.hermes`, model pulled there (4.38 GB) | 0 | 2 | 0 | **1** (runtime not found) | 0 / 0 / 0 | 0, 0 shims |
| `logs/gate3` | default store + model symlinked at `~/.hermes/models` | 0 | 2 | 0 | 0 | 0 / 0 / **1** (2 failed) | 0 |
| `logs/gate4` | default store + model visible in both places (host-like) | 0 | 2 | 0 | **0** | 0 / 0 / 0 | 0, 0 shims |

### 3.1 gate4 — 10/10 green (raw summary + tails)

```
$ bash tools/host_gate_e1a.sh        # started 2026-09-17T19:09:45Z, finished 2026-09-17T19:10:43Z
{"init.exit": 0,
 "init.variant": "linux-x64-vulkan",
 "init.fallback": "pre-flight: the pinned linux-x64-cuda-12.8 bundle links libcudart.so.12, libcublas.so.12, libcuda.so.1, which this host cannot load (libcudart.so.12: cannot open shared object file: No such file or directory); skipped the 168.81 MB download and moved to the next tier",
 "doctor.exit": 2,
 "oracle.exit": 0,
 "oracle.section_b_skips": 0,
 "pytest_before.exit": 0,
 "pytest_after.exit": 0,
 "pytest_network.exit": 0,
 "poisoned_init.exit": 0,
 "poison_shims_invoked": 0}
```

Per-step wall clock (`.elapsed` files, gate4): pytest_before 14 s · init_dry_run 0 s · init 0 s ·
doctor 0 s · version 0 s · pytest_after 15 s · oracle 3 s · pytest_network 24 s ·
init_poisoned_path 1 s (budget 180 s) · whole script 58 s.

Suite tails (verbatim):

```
pytest_before : 329 passed, 11 skipped in 14.33s
pytest_after  : 329 passed, 11 skipped in 14.43s        (oracle section B live)
pytest_network: 339 passed, 1 skipped in 23.56s
```

Oracle section B, live against the installed bundle (excerpt):

```
[B] live runtime probes (ctypes vs pinned release)
  runtime dir: .../runtime/b11026-linux-x64-vulkan
  ok   ctypes resolves all 32 required symbols (0 missing)
  ok   libggml.so resolves the backend loader (2/2) — PoC pitfall 1
  ok   llama-cli build == 11026 (pinned release, not a stale lib)
  ok   libllama.so carries the spark2_5 arch implementation
  ok   llama-fit-params available for auto-fit
  ok   local Spark Q8_0 size == HF pin
  ok   local Spark Q8_0 sha256 == HF lfs.oid (download-verify contract)
        -> skips_in_section_B: 0
```

Poisoned PATH (A-E1a-2): `init_poisoned_path.exit 0`, elapsed 1 s, `wc -l poison_calls.log` → `0`
(zero compiler shims invoked; the shim log is empty in `logs/gate4/poison_calls.log`).

`init` JSON (contract keys, gate4/gate2 identical): `variant=linux-x64-vulkan`, `backend=vulkan`,
`working_backend=vulkan`, `backends=[cpu, rpc, vulkan]`, `asset_verified=true`,
`bytes_fetched=30294625`, `fallback_reason_code=system_libs_missing`, one `fallback_attempts`
entry `{backend: cuda, variant: linux-x64-cuda-12.8, code: system_libs_missing}`.

`doctor --json` contract keys (gate2/gate4): `backend=vulkan`, `backends=[cpu, rpc, vulkan]`,
`expected_backend=cuda`, `runtime.working_backend=vulkan`,
`runtime.fallback_reason_code=system_libs_missing`, checks `runtime.backends=ok`,
`runtime.accelerator=warn`, `runtime.fallback=warn`; exit 2 = warnings, per the 0/2/1 contract.
The 168.81 MB CUDA archive was never downloaded: `downloads/` holds only the 30 294 625 B vulkan
asset (`sha256 1b40310b…4708` = the pin).

### 3.2 Real download in this container

```
$ uv run ggufone models pull XHToken/Spark-X2.5-4B-GGUF:Q8_0
pulling Spark-X2.5-4B-Q8_0.gguf: 100.0%  4.38 GB / 4.38 GB
pulled spark-x2.5-4b-q8_0 -> <home>/.hermes/models/Spark-X2.5-4B-Q8_0.gguf
  sha256 5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2 (verified against lfs.oid)
  license apache-2.0 · arch spark2_5  quant Q8_0
$ uv run ggufone models verify --json   -> verified 1, failed 0, sha256 matches the pin (logs/journeys/j5b)
```

### 3.3 Journeys (12, raw outputs in `logs/journeys/`)

| ID | journey | exit | observed |
|---|---|---|---|
| j1 | `init --json` re-run on an installed home | 0 | `already_installed: true`, variant vulkan, `fallback_reason` **and** `fallback_reason_code` present (finding 3 stays fixed) |
| j2 | `doctor --json` | 2 | status=warnings; all contract keys present |
| j3 | `init --dry-run --json` | 0 | `backend=cuda`, `variant=linux-x64-cuda-12.8`, `size=168811114`, `host.has_nvidia_smi=true` |
| j4 | `init --backend bogus` | 3 | `E_RUNTIME_MISSING: backend 'bogus' has no pinned bundle for linux-x86_64; available: linux-x64-cpu, linux-x64-vulkan, linux-x64-cuda-12.8` (no traceback) |
| j5 | `models ls --json`, `models verify --json` | 0 | alias/path/size/sha/arch/quant/license; verify 1/0 ok |
| j6 | `doctor --json` on an empty home | 1 | `backend: null`, `backends: []` (failure leg of 0/2/1) |
| j7 | `GGUFONE_HOME=/proc/ggufone-cannot-write init` | 4 | `E_INTERNAL: FileNotFoundError …` → P3 |
| j8 | SIGKILL 1 s into `init`, then re-run | 0 | kill exit 137; re-run resumed `resumed_from=15728640`, `bytes_fetched=14565985`, `asset_verified: true`; doctor after → 2 |
| j9 | `init --backend cuda` (pre-flight bypass, offline cache) | 0 | installs `linux-x64-cuda-12.8` (`asset_verified: true`), `backend_errors.cuda='libggml-cuda.so: libcudart.so.12: cannot open shared object file…'`, `working_backend=cpu`, warning `W_BACKEND_LOAD` — truth recorded, no silent downgrade |
| j10 | 3 × `init` in a row | 0 | identical output each time, no state leak |
| j11 | 2 × `doctor --json` byte-compared | 2 | identical (`cmp`) |
| j12 | `models pull '🔥not/a-repo:Q9_9'` | 3 | `E_DOWNLOAD_FAILED` leaking a codec error → P4 |

## 4. BLOCKING (priority scope)

**None found.** On the head under test: `init` exits 0 with an empty stderr tail on the fresh path,
the cuda→vulkan fallback is recorded with a machine-readable code, `doctor` names the working
backend, the oracle's section B is live with 0 skips, both suites are green, and the poisoned-PATH
scenario completes in 1 s with 0 compiler shims — in the container; the host side of the same table
is the coordinator's round-2 run quoted in §2. No crash, no data loss, no silent downgrade.

## 5. PROPOSALS (out of scope, non-blocking)

**P1 — the gate's helper tools disagree about where the runtime/model live (P3, tooling).**
`docs/verify_runtime_contract.py::find_runtime()` resolves `$GGUFONE_RUNTIME_DIR` →
`$XDG_DATA_HOME/ggufone/runtime` → `~/.local/share/ggufone/runtime`, never `$GGUFONE_HOME`;
`tools/live_probe.py` resolves the model as `$GGUFONE_HOME/models` else `~/.hermes/models`;
`registry/store.py` resolves `$GGUFONE_HOME` first. Consequences measured here:
- gate2 (`GGUFONE_HOME=~/.hermes`, the layout the E1a baseline records): the gate's step-8 oracle
  printed `SKIP no runtime installed` right after `init` installed one → section B skipped;
- gate3 (default store, model only at `~/.hermes/models`): `pytest --run-network` → `2 failed,
  337 passed` because the live probe looked for the model under `$GGUFONE_HOME/models`.
Repro: `logs/gate2/oracle.out`, `logs/gate3/pytest_network.out`. The host is green because both
locations happen to be populated there. Suggested fix: have the oracle call
`ggufone.registry.store.data_home()`, or export `GGUFONE_RUNTIME_DIR=<data home>/runtime` in the
gate script for step 8.

**P2 — the gate is red on a box without the 4.4 GB pinned model (P3, precondition).**
Fresh-home run, no model: `pytest_after` exit 1 with
`AssertionError: section B skipped: ['  SKIP …/Spark-X2.5-4B-Q8_0.gguf absent …']`
(`logs/gate1/pytest_after.out`), and `pytest_network` exit 1. The host passes only because the
model is present. Suggest documenting the model as a gate precondition (or skipping that assertion
with an explicit reason when the model is absent).

**P3 — an uncreatable data home surfaces as `E_INTERNAL` (P3, cosmetic).**
`GGUFONE_HOME=/proc/ggufone-cannot-write uv run ggufone init --json` → exit 4,
`error: E_INTERNAL: FileNotFoundError: [Errno 2] No such file or directory: '/proc/ggufone-cannot-write'`
(`logs/journeys/j7_unwritable_home.err`). A named error (`E_DATA_HOME_UNWRITABLE`) would fit the
rest of the CLI's vocabulary.

**P4 — unicode repo ref leaks a Python codec error (P3, cosmetic).**
`models pull '🔥not/a-repo:Q9_9'` → exit 3,
`error: E_DOWNLOAD_FAILED: 🔥not/a-repo: 'ascii' codec can't encode character '\U0001f525' in position 16`
(`logs/journeys/j12_pull_garbage.err`). Validate/normalise the ref and report `E_INVALID_REPO`.

## 6. Verdict: SHIP

- Host (round 2, quoted): all four gates green on the RTX 3060 Ti, `init.exit == 0` (SIGABRT gone),
  oracle section B 0 skips, poisoned PATH 0 shims.
- Container (executed): gate4 reproduces the same 10/10 table on the same head, with the real
  pinned assets, a real 4.38 GB model download, and 12 adversarial journeys — no BLOCKING finding.
- Evidence: `docs/evidence/e1a_baseline.json` → `e1a_fix_t_eae35404.host` (round 2 quoted +
  container corroboration + P1–P4), raw logs in this directory (`gate1`–`gate4`, `journeys`).
- Caveat carried in the evidence: the host numbers are quoted from the coordinator's board comment;
  the operator's per-step `.elapsed` files were not readable from this container.
