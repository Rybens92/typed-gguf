# typed-gguf v0.1.0 — release gate, re-run at the post-fix head (card `t_0070415c`)

**Verdict: APPROVE — v0.1.0 is releasable at `fc4328c`.** The two blockers of the previous re-gate are
closed and re-verified by execution: the committed CI workflow's offline-suite step is **green** in a
clean clone (`1506 passed, 56 skipped`, exit 0, flag on) and `keep stop` no longer signals a pid the
record cannot prove is a host (a decoy process survives a debris record; the normal stop path still
kills the host). The E4 surface, the uvx artifact mechanics and the documents were re-run around them
and hold. What is left is two 🟡 MINORs and three ⚪ NITs — no blocker.

This file replaces the `ed48acb` BLOCKED re-gate (commits `d75795a` + `0bb9cb8`) and, before it, the
`cb79e92` APPROVE (`232a11b`): the repo's convention (`git log -- REVIEW.md`) is that each review
overwrites `REVIEW.md` and git keeps the predecessors.

## Certified head and its surroundings

- **Certified sha: `fc4328c`** — `fix(t_a4ebcd36): keep stop verifies the record before it signals a
  pid (M2)`, i.e. `18bd74e` (the net-block fix, B1) + `bbf0544` (the tests-only `sun_path` fix) on top
  of the previously certified `ed48acb`. Everything below was executed on a **clean clone of
  `fc4328c`** (`git clone --no-hardlinks`, `git checkout fc4328c`, `git status` empty; venv py3.11 +
  pytest 9.1.1 / ruff 0.16.8, the lock's own versions).
- **Product surface touched by the two fixes:** `src/typed_gguf/keep/client.py` only (`stop()`,
  `_verified_host()`); `18bd74e` changes `tests/conftest.py`, `tests/test_net_block_scope.py`,
  `.github/workflows/ci.yml`, `tools/e1c_offline_gate.py` — no product file. The gates that moved
  (1506 vs the receipt's 1485) are the new pins the fixes brought with them.
- **In flight while this ran, explicitly *not* certified:** `tests/test_probe_pressure.py` is
  modified **uncommitted** in the shared checkout (the flake follow-up `t_10247273`: it pins the
  cgroup reading's *source* instead of re-reading live). It is tests-only and does not touch the net
  block or the keep surface; my clone carries the committed file, and the CI shape passed with it.
- **Box recipe note (not a product finding):** the E4 live gates need the receipt's **EGL** ICD
  (`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`). With the other nvidia ICD
  (`/etc/vulkan/icd.d/nvidia_icd.x86_64.json`) `ggml_backend_load_all_from_path` loads **no Vulkan
  backend at all** (probe: registries `RPC, CPU`; devices `CPU` only) and the same 7 gates still pass
  — on **CPU**, with the ledger's own evidence saying so (`devices: ["CPU", "CPU_Mapped"],
  effective_backend: null` while the fit plan claims 36 offloaded layers). A re-runner who picks that
  path gets plausible-looking numbers and no warning. Suggest a follow-up card: make the live recipe
  carry the ICD, or make the gate fail/skip when a bundle is given and the placement is CPU.

## Re-gate at `fc4328c` — per-area verdicts

| # | area | verdict | what was executed (clean clone of `fc4328c`) |
|---|---|---|---|
| 1 | **CI-shape offline suite** (the public gate) | ✅ **GREEN** | `env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<empty stub> pytest -q -rs --timeout=120` → **1506 passed, 56 skipped, exit 0** (41.14 s); `0` FAILED/ERROR lines. `.github/workflows/ci.yml`'s step asserts exactly this shape. At `ed48acb` the same command answered 24 failed / 1466 passed / 11 errors — **B1 closed** |
| 2 | the keep surface under the flag (the B1 subject) | ✅ | the four keep files + `test_net_block_scope.py` + `test_keep_socket_path.py` under `TYPED_GGUF_TEST_BLOCK_NET=1` → **109 passed**; the four keep files alone, flag **off** → **97 passed** |
| 3 | **E4 live surface** | ✅ **re-derived by execution** | `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json TYPED_GGUF_RUNTIME_DIR=<b11026-vulkan> pytest -q --run-network tests/test_keep_live.py -s` → **7 passed in 170.25 s, exit 0**: `A-E4-1 cold 10.51 s (load 1647 ms) → warm 2.98 s (load 0 ms), pid 2405`; `A-E4-2 idled out 5.4 s after the 5 s window; device free 6753 → 2659 (loaded) → 6753 MiB (unloaded)`; `A-E4-3 pids: A 3068 → B 3205 → A 3529` |
| 4 | oracle, live shape | ✅ | `HOME=/var/home/rybens TYPED_GGUF_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<egl> python docs/verify_runtime_contract.py` → **`failures: 0  skips: 0`** |
| 5 | oracle, bundle-free shape | ✅ | `failures: 0  skips: 1` (the one skip is `no runtime installed`) |
| 6 | ruff | ✅ | `ruff check src tests tools docs .github` → **All checks passed!** |
| 7 | E3e / policy / parity gates | ✅ | E3e files **76 passed, 3 skipped**; `tests/test_policy_v2.py` + `tests/test_bench_prompt_parity.py` (with the stub) **15 passed, 1 skipped** |
| 8 | doc gates | ✅ | `pytest -q -k doc` → **96 passed**; `tests/test_public_docs.py` alone → **17 passed** |
| 9 | red path (no bundle, no model) | ✅ | `pytest -q -m "model or network"` with the runtime vars unset → **55 skipped, 1507 deselected, 0 failed** |
| 10 | committed offline tool | ✅ | `python tools/e1c_offline_gate.py` → **227 passed, 23 skipped**, `network-disabled run … (exit 0)` |
| 11 | **uvx / out-of-tree install** | ✅ artifact half | `pytest -q tests/test_wheel_install.py` → **8 passed**; plus an independent build: the wheel carries `typed_gguf/data/runtime.lock`, **byte-for-byte identical** to the root file (sha256 `4214efba…`) |
| 12 | citations ledger | ✅ no new dangling | own ledger over README + SPEC + release notes + `docs/evidence/**`: 1765 tracked files, **726 citations resolved, 0 cited-but-untracked**; the *missing* set is **identical to `ed48acb`** (byte-diff of the two sets is empty) — pre-existing pre-rename `src/ggufone/…` paths inside old receipts, `mutants/…` scratch globs and two MCP method names. The repo's own `.t07b5/check_citations.py` → the same two MCP names (tools/call, tools/list), nothing else |
| 13 | `--help` / exit-code surface | ✅ | re-probed every command: root `--help` exit 0 and the `serve`/`mcp` lines read *“(specified in SPEC §2.9, not implemented in v0.1.0; exits 3)”* (F1 closed), every `--help` exit 0, `serve`/`mcp` exit **3**, unknown command **2**, `keep` with no subcommand **2**, `version` **0**, `keep status --json` on an empty home **0 / `state=stopped`** (N3 below is the one oddity, pre-existing) |
| 14 | secrets | ✅ | `git grep` for AWS/`ghp_`/`github_pat_`/`hf_`/`sk-`/`xox*`/`AIza`/private-key shapes → **0 hits**; no `.env`/`.pem`/`.key`/`id_rsa` tracked |

**Not re-run, and why** (unchanged): the operator-host rows (RTX 3060 Ti Vulkan tables, the Tiel/Occamy
cells) need the operator host — their receipts are the evidence; the 20+ GB GGUFs do not fit this
container's 5 GiB cgroup; anything that downloads from HuggingFace is outside the offline shape.

## B1 — closed, re-verified in the shape the CI runs

`tests/conftest.py`'s hook is now `block_network()`: `_NoNetworkSocket` (a *subclass* of the real
`socket.socket`) refuses only `AF_INET`/`AF_INET6` (plus the default family, which is `AF_INET`) and
`create_connection`/`getaddrinfo`; local IPC is out of its scope. The re-run above is the whole
argument: **the committed CI step's exact command is green at this head**, where at `ed48acb` it was
`24 failed, 11 errors`, all of them the hook's own `AssertionError` in `tests/test_keep_host.py` +
`tests/test_keep_client.py`. The flag did not lose its teeth — `tests/test_net_block_scope.py` pins
the network families, the default family, the two helpers and the product's own pull probe
(`hf.model_info(offline=False)` still dies as `E_DOWNLOAD_FAILED`), and it re-runs the CI shape in a
child session; `tools/e1c_offline_gate.py`'s 227+23 above is the committed flag consumer, green.

*Residual (⚪, informational):* the block is a Python-level constructor patch; a path that already
holds a network fd could still wrap it (`socket.fromfd` is not patched). No code in the product or its
tests does that, and the flag's subject (a *decision path* reaching for the network) goes through
`create_connection`/`urlopen`, which are patched. Recorded, not a finding.

## M2 — closed, re-verified by execution; the sibling probes

`Client.stop()` now verifies the record first (`_verified_host`: this client's own live `Popen`
handle, or something answering on the record's socket) and treats a record nobody answers on as
**debris** — cleaned up, nothing signalled, the reason says which. Six probes against the real CLI
(`python -m typed_gguf keep stop …`, throwaway homes under `/tmp/keep-probe-0070415c`):

| probe | shape | result |
|---|---|---|
| A | debris (socket file, nobody listening) + decoy pid, `keep stop --json` | ✅ `stopped: false`, reason *“nothing is listening on … the record is debris … nothing was signalled”*, **decoy alive**, record + socket file removed — the exact case that killed a decoy at `ed48acb` |
| B | debris with no socket file at all, decoy pid, text form | ✅ *“no host was running (… nothing was signalled)”*, decoy alive, record cleared |
| E | a *real* host (the repo's `tests/fake_keep_host.py`, socket intact) | ✅ `stopped: true`, reason *“stopped (SIGTERM)”*, host dead, socket gone, ledger clear — the signal path still works |
| C | a **planted listener** at the record's path + decoy pid | ⚠ the pid is signalled — the socket *is* the identity; needs write access to the 0600 keep dir. ⚪ residual, same exposure class as before, now narrowed to "something answers" |
| D | a real host whose socket file was **unlinked by hand**, then `keep stop` | ⚠ reads as debris → nothing signalled, record dropped, the host ends only via its idle window. ⚪ residual / sibling of the fix's rule (outside contract: the socket lives in the 0600 keep dir) |
| F | a real listener whose **accept queue is saturated** (backlog 1, two held clients), then `keep stop` | ⚠ reads as debris → nothing signalled, record dropped, listener still up — see 🟡 **F** below |

The two E4 bugs the card says were found and fixed were re-checked as well: the *spawning call reports
the load it waited for* — the live run's cold call is exactly that (`timings.model_load_ms 1647 ms` on
the call that spawned, `0 ms` warm, same pid); and *`keep status` on a busy host* is pinned offline by
`tests/test_keep_client.py::test_status_of_a_host_that_cannot_answer_says_unresponsive` (in the 97/109
green above), with the live A-E4-4 half also green.

## E4-specific checks (card item 2)

**(a) The README/release-notes claims against the receipts** — claim by claim, receipt → document:

| claim as published (README *Warm host* / release notes) | receipt | re-checked here |
|---|---|---|
| cold **17.50 s** (load 2280 ms) → warm **2.58 s** (load 0 ms), same host pid 35915 | `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md` §1 A-E4-1 | ✅ mechanism re-derived live (10.51 s / **1647 ms** → 2.98 s / **0 ms**, same pid 2405); the *numbers* are the operator box's and the documents quote the receipt (`tests/test_public_docs.py` pins each of `17.50 / 2280 / 2.58 / 0 ms / 5.3 / 6384 / 2314 / 6409` to both) — numbers present in receipt/notes/README: 17.50 ✅/✅/✅, 2280 ✅/✅/✅, 2.58 ✅/✅/✅, 5.3 ✅/✅/✅, 2314 ✅/✅/✅, 6409 ✅/✅/✅, 6384 ✅/✅/— (the README quotes the free-memory delta `2314 → 6409`, not the pre-load baseline) |
| idle unload: host gone **5.3 s** after a 5 s window; device 6384 → 2314 → 6409 MiB | §1 A-E4-2 | ✅ re-derived: gone **5.4 s** after the window, device free **6753 → 2659 (loaded) → 6753 MiB (unloaded)** — returns to the pre-load level |
| one model at a time: a different key stops the old host *before* the new one loads | §1 A-E4-3/A-E4-5 | ✅ re-derived: pids `A 3068 → B 3205 → A 3529`, three distinct hosts |
| keep-alive precedence **flag > env > default**, 600 s default, `--keep-alive 0` = pre-E4 | §1 A-E4-6, SPEC §2.12 | ✅ code reads as documented (`keep/identity.py::resolve_keep_alive`, `DEFAULT_KEEP_ALIVE = 600.0`, `KEEP_ALIVE_ENV`); the live file's A-E4-6 (`--keep-alive 0`) ran green in the 7/7 |
| inline fallback + `W_KEEP_UNAVAILABLE` off-unix | §2, SPEC §2.12 | ✅ exercised by the offline pins and the live A-E4-4/A-E4-7 pair |
| `tests/test_keep_live.py` 7 gates in 297.82 s; offline keep pins 93; suite 1485/56 | §1 header/§1 tail | ✅ the 7-gate file is the gate (my run: 170.25 s — the box's speed, not the product's); the "93" and "1485" are that head's counts — this head has 97 keep pins and 1506 suite passes because the fixes added gates (records are point-in-time, as the receipt says) |

**(b) Warm claims re-derived by execution** — done twice: the warm call above reports
`timings.model_load_ms == 0.0` on the host that answered the cold call from pid 2405, and the
`--keep-alive 5s` host was watched **gone by pid** 5.4 s after its window with the device's free
memory back to 6753 MiB. Both of the card's suggested shapes pass.

**(c) The `keep` surface for footguns** — the two the card mentions (stale sockets, orphan pids) are
covered by the probes above and hold for the realistic shapes; the falsification attempt that *did*
land is **F** (a saturated accept queue reads as debris), reported below as 🟡 MINOR — it fails in the
safe direction (nothing is signalled) and needs ~9 queued clients (the product's own
`LISTEN_BACKLOG = 8`) while the host is inside one decision.

## uvx-specific check (card item 3)

- The README block (*Install without a clone*) names the packaged lock and the precedence rule, and
  the honored mechanics match it: the built wheel carries `typed_gguf/data/runtime.lock` **byte for
  byte** (sha256 `4214efba…`, verified independently here), the artifact gate (`8 passed`) installs
  the wheel into a temp tool env and drives it from a neutral cwd that plants a **decoy** lock, and
  `$TYPED_GGUF_LOCK` is honored as-is (a missing path is an error that lists every path searched). ✅
- The **post-publish-only half (the `git fetch` step) is declared as such — not claimed as
  verified**: the release notes say it in so many words (*“the **git fetch** step of the
  `git+https://…` spelling needs the published repository and is verified **post-publish**: this
  checkout has no remote configured and GitHub answers that URL with "Repository not found" today”*),
  cite the receipt (`docs/evidence/v0_1_0_t_eff926f9_uvx_install.md`, tracked), and the public-docs
  gate pins those words. The README's block carries the same one-liner with a locally-measured sample
  and the mitigation sentence (*“Before/without a published remote, the same thing works from a
  checkout — `uvx --from . …`”*) but not the explicit limit — 🟡 **M3**, one sentence, unchanged from
  the previous review.

## Docs consistency (card item 4)

- **README ↔ release notes ↔ code agree on the whole `keep` surface**: 600 s default,
  `--keep-alive <dur|0>`, `$TYPED_GGUF_KEEP_ALIVE`, *flag > env > default*, one host per data home,
  swap-on-different-key, `engine.keep.served_by`/`fallback`, `keep status`/`keep stop`, the Windows
  `W_KEEP_UNAVAILABLE` path. `tests/test_public_docs.py` (17 passed) asserts the same fact list
  against **both** documents and reads the field set out of `schema.Options()`, so a later edit to
  either side fails a gate instead of drifting. ✅
- **`schema.Options()` vs the documents:** `cue='json_instructed'`, `chat_format='role_split'`,
  `json_contract='question'` — the policy-v2 defaults the notes and README quote; the keep *window* is
  deliberately **not** an option field (read from `keep/identity.py`) and neither document names an
  `options.keep_alive` (spot-checked here against `dataclasses.fields(Options)`). ✅
- **SPEC §2.12 vs the code:** the section's claims (detached `setsid` child, 0600 unix socket in
  `$TYPED_GGUF_HOME/keep/`, never TCP, one host/one model/one data home, the key's composition, the
  countdown restarting per request, serialized requests, `os._exit` teardown, debris probed and
  replaced, the fallback policy) all match `src/typed_gguf/keep/*` line for line. ✅
- Carried from the previous review: `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md:11` still says
  “the 4B **Q4_K_M**” while the gate loads the pinned **Q8_0** (⚪ N1, numbers unaffected).

## Checklist (the release gate's own list, re-run at `fc4328c`)

| # | item | status | evidence |
|---|---|---|---|
| 1 | LICENSE (MIT) + credits | ✅ | `LICENSE` MIT; `pyproject.toml` MIT + classifier; README *Credits and attribution*; the notes repeat MIT + no-parity and a gate fails if they go |
| 2 | pyproject metadata | ✅ | `typed-gguf 0.1.0`, `requires-python >=3.11`, console script via `typed_gguf.cli:run`, `force-include` mapping `runtime.lock` into the wheel (verified byte-for-byte) |
| 3 | CI workflow sane (offline, no downloads) | ✅ | offline and downloads nothing; its "Offline suite" step is **green** at this head (area 1) — B1 closed |
| 4 | no secrets in tracked files | ✅ | 0 hits over the tracked tree (area 14) |
| 5 | hygiene applied | ✅ | clean clone of the certified sha is clean; only the in-flight tests-only change sits in the shared checkout (named above); `.gitignore` carries the live/scratch rules |
| 6 | fresh-install acceptance quoted | ✅ | unchanged (`.t07b5/logs/*` receipts + the acceptance report), plus the out-of-tree artifact gate (8 passed) and the byte-identical packaged lock |
| 7 | `--help` surface self-consistent | ✅ | F1/F2 closed; exit codes as documented (area 13) |
| 8 | docs ↔ code for the new surface | ✅ | areas 3/4 and the consistency sections above |

## Blocking issues

**None.** Both blockers of the previous re-gate are closed by execution:

- **B1** (the committed CI gate was red: the net block forbade `AF_UNIX`) → `18bd74e`; the CI's own
  command is now green in a clean clone (`1506 passed, 56 skipped`, exit 0) and the flag's scope is
  pinned by `tests/test_net_block_scope.py`.
- **M2** (`keep stop` could signal a recycled pid) → `fc4328c`; probes A/B/E above show debris is
  cleaned and never signalled while a real host is still stopped.

## Findings (non-blocking)

**🟡 M3 — the README's uvx block does not carry the “post-publish” limit its own notes do.**
The release notes state that the `git fetch` step of `uvx --from git+https://…` is verified
**post-publish**; the README (`README.md`, *Install without a clone*) shows the same one-liner with a
sample output under it and only implies the caveat (*“Before/without a published remote, the same
thing works from a checkout — `uvx --from . …`”*). The sample output block is captioned *“(measured
2026-09-20)”* without saying it came from the local-path build, which is exactly the misreading the
notes' sentence prevents. One sentence — the notes' own words — would make the two documents say the
same thing. No behaviour claim is affected; the artifact gate executes the local-path half. (Carried
from the previous review, unchanged at this head.)

**🟡 F — `keep stop` reads a *listening* host with a saturated accept queue as debris.**
`src/typed_gguf/keep/client.py::_listening()` collapses **every** `OSError` to `False` and
`_verified_host()` therefore calls the record debris — including the case where the connect failed
because the host's accept queue was full (`socket.timeout`/`EAGAIN`), not because nobody was there.
Reproduced (`/work/t_0070415c/probes/keep_probe.py f`): a listener with `listen(1)` at the record's
path, two held clients saturating the queue, a live pid → `keep stop --json` answers
`{"stopped": false, "reason": "nothing is listening on … the record is debris … nothing was
signalled", "cleaned": true}` and the listener is still up. With the product's own
`LISTEN_BACKLOG = 8` (`src/typed_gguf/keep/host.py:50`) the bar is ~9 queued clients while the host is
inside a single long decision — a monitoring loop plus a wedge, not a routine path. Impact: it fails
*safe* (nothing is signalled) but the ledger entry is dropped, so `keep status` afterwards says
`stopped` while the model stays resident until its window expires (600 s default) and `keep stop` can
no longer reach it. Direction (for a follow-up card, not this release): treat an *inconclusive* probe
(timeout / `EAGAIN`) differently from a refusal (`ECONNREFUSED`) or a missing file (`ENOENT`) — keep
the record (or treat the host as alive) when the probe could not be answered.

**⚪ C — a record whose socket is answered is signalled, whatever owns the pid.**
By design after `fc4328c`: the socket is the identity. Only reachable with write access to the 0600
keep dir (a planted listener) or in a millisecond-wide TOCTOU against a concurrent swap; recorded so
the boundary of the fix is explicit, not as a defect.

**⚪ D — a real host whose socket file was removed by hand is treated as debris.**
`keep stop` then cleans the record and signals nothing (probe D), leaving the process to end via its
idle window. It follows from the fix's own rule (“a record nobody answers on is debris”) and the
socket lives in the 0600 keep dir, so the trigger is outside the product's flows; the cost is
bounded by the keep-alive window.

**⚪ N1 — the E4 evidence doc names the wrong quant.**
`docs/evidence/v0_1_0_t_7e24cea4_warm_host.md:11` says “the 4B **Q4_K_M**”; the live gate loads the
pinned **Q8_0** (`tests/test_keep_live.py:40–41`), which is what the README, the notes and the card's
numbers are about. The receipt's numbers are the gate's own output, so nothing measured changes.

**⚪ N3 — `typed-gguf models` with no subcommand exits 0** and prints the usage line (every other
missing-argument path in the CLI is exit 2). Untouched by this card; noted only because the
help/exit-code surface was re-probed line by line.

## Verdict

**v0.1.0 is releasable at `fc4328c`.** The previous gate's one blocker is closed and re-verified in
the repository's own CI shape (clean clone, `TYPED_GGUF_TEST_BLOCK_NET=1`, `1506 passed / 56 skipped /
exit 0`), the `keep stop` hole is closed and re-verified by execution (debris never signals; a real
host still stops), and everything the release claims was re-run around them: the E4 live surface
(7/7 in 170 s — cold → warm with `model_load_ms 0.0` on the same pid, the 5 s idle unload by pid *and*
device memory, the A→B→A swap), the offline keep pins (97 flag-off / 109 under the flag), the oracle
(live `failures: 0 skips: 0`, bundle-free `failures: 0`), ruff, the E3e/policy/parity/doc gates, the
red path's all-skip, the committed offline gate (227+23), the wheel/uvx artifact gate (8 passed, lock
byte-identical), the citation ledger (0 cited-but-untracked, no new dangling), the `--help`/exit-code
surface and the secret scan. The three remaining notes (M3, F, N1/N3) are non-blocking; F and M3 are
named with their file and command for whoever picks them up next.
