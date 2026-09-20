# typed-gguf v0.1.0 — release gate, re-run at the post-E4 head (card `t_0070415c`)

**Verdict: BLOCKED — v0.1.0 is *not* releasable at `ed48acb`.** The E4 surface itself holds up under
execution (the warm host, the idle unload, the swap and the fallbacks were all re-derived on this box
and the documents match the code), but the **repository's own CI gate is red at the head**: the
offline suite under the flag the committed workflow sets fails 24 tests and errors on 11 more, all of
them the new `keep` gates. What ships today is a tree whose first push turns the CI red, so the tag
waits for one change. The details, the exact reproduction, and the rest of the battery are below.

This file replaces the `cb79e92` APPROVE (commit `232a11b`) — the repo's convention (`git log --
REVIEW.md`) is that each review overwrites `REVIEW.md` and git keeps the predecessors.

## Re-gate at `ed48acb` — per-area verdicts

| # | area | verdict | what was executed |
|---|---|---|---|
| 1 | **CI-shape offline suite** (the public gate) | 🔴 **RED** | clean clone of `ed48acb`, `TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<empty stub> pytest -q -rs --timeout=120` → **24 failed, 1466 passed, 56 skipped, 11 errors**; `.github/workflows/ci.yml` fails a step on that log. Same run at `dec1819`: 24 failed / 1461 passed / 11 errors. The 24 failures are *all* the net-block hook (`tests/conftest.py:193`), *all* in `tests/test_keep_host.py` + `tests/test_keep_client.py` — see the blocking issue |
| 2 | the same suite without the net-block flag (the shape the E4 receipt used) | ✅ | **1489 passed, 1 failed, 56 skipped** — the one failure is `tests/test_bench_prompt_parity.py::test_a_row_records_which_framing_it_measured`, which the CI's empty-file bundle stub exists for (run with `TYPED_GGUF_BENCH_RUNTIME_DIR` → 2 passed) |
| 3 | **E4 live surface** (the new feature) | ✅ **re-derived by execution** | `VK_DRIVER_FILES=<nvidia icd> TYPED_GGUF_RUNTIME_DIR=<pinned bundle> pytest -q --run-network tests/test_keep_live.py -s` → **7 passed in 410.19 s, exit 0**, on the real 4B + the pinned Vulkan bundle: `A-E4-1 cold 35.30s (load 2654 ms) → warm 3.11s (load 0 ms), pid 17839`; `A-E4-2 idled out 5.2s after the window; device free 6330 → 2255 (loaded) → 6345 MiB (unloaded)`; `A-E4-3 pids: A 19577 → B 20010 → A 22159` |
| 4 | E4 offline pins | ✅ | `pytest -q tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py` → **93 passed** (without the net-block flag); with it → 24 failed / 69 passed / 11 errors (area 1) |
| 5 | oracle, live shape | ✅ | `TYPED_GGUF_RUNTIME_DIR=<bundle> python docs/verify_runtime_contract.py` → `failures: 0  skips: 0` |
| 6 | oracle, bundle-free shape | ✅ | `failures: 0  skips: 1` (the one skip is `no runtime installed`; the pinned model is visible here) |
| 7 | ruff | ✅ | `ruff check src tests tools docs .github` → `All checks passed!` |
| 8 | E3e / policy / parity gates | ✅ | E3e files **76 passed, 3 skipped**; `tests/test_policy_v2.py` **13 passed, 1 skipped**; `tests/test_bench_prompt_parity.py` (with the stub) **2 passed** |
| 9 | doc gates | ✅ | `pytest -q -k doc` → **96 passed** (91 before the notes catch-up landed); `tests/test_public_docs.py` alone → 17 passed |
| 10 | red path (no bundle, no model) | ✅ | `pytest -q -m "model or network"` with the runtime vars unset → **55 skipped, 1491 deselected, 0 failed** (49 baseline + the 7 E4 live gates, which skip by name) |
| 11 | **uvx / out-of-tree install** | ✅ artifact half | `pytest -q tests/test_wheel_install.py` → **8 passed in 3.75 s** (real `uv build --wheel --offline` + `uv tool install` + three CLI runs from a neutral cwd carrying a decoy lock) |
| 12 | citations ledger | ✅ no new dangling | hygiene checker at `ed48acb`: 1763 tracked files, 262 citations, 245 resolved, **0 cited-but-untracked**, 17 missing = 16 pre-existing receipts + one mutation-scratch glob (`mutants/…`) the E4 evidence doc names |
| 13 | `--help` / exit-code surface | ✅ | re-probed every command: root `--help` exit 0 with `serve`/`mcp` reading *“(specified in SPEC §2.9, not implemented in v0.1.0; exits 3)”* (F1 closed), `models <sub> --help` no longer repeats the subcommand (F2 closed), `serve`/`mcp` exit **3** with the milestone pointer, an unknown command exits **2**, `keep` with no subcommand exits 2, `version` and every `--help` exit 0 |
| 14 | secrets | ✅ | `git grep` for AWS/`ghp_`/`github_pat_`/`hf_`/`sk-`/`xox*`/`AIza`/private-key shapes over the tracked tree → **0 hits**; no `.env`/`.pem`/`.key`/`id_rsa` tracked |
| 15 | hygiene at the certified sha | ✅ | the clean clone of `ed48acb` is clean; the *shared* checkout is dirty only with the in-flight tests-only card named at the bottom (not part of this certification) |

**Not re-run, and why** (unchanged from the first gate): the operator-host rows (RTX 3060 Ti Vulkan
tables, the Tiel/Occamy cells) need the operator host — their receipts are the evidence; the 20+ GB
GGUFs do not fit this container's 5 GiB cgroup; anything that downloads from HuggingFace is outside
the offline shape this gate runs in.

## E4-specific checks (card item 2)

**(a) The README/release-notes claims against the E4 receipt** — claim by claim, receipt → document:

| claim as published (README *Warm host* / release notes) | receipt | re-checked here |
|---|---|---|
| cold **17.50 s** (load 2280 ms) → warm **2.58 s** (load 0 ms), same host pid | `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md` §1 A-E4-1 | ✅ mechanism re-derived live (35.30 s / 2654 ms → 3.11 s / **0 ms**, same pid). The *numbers* are the operator box's and this box is slower under co-tenant load — the receipt itself says the cold delta is the box, and the receipt's numbers are what the documents quote (a new gate in `tests/test_public_docs.py` now pins each of `17.50 / 2280 / 2.58 / 0 ms / 5.3 / 6384 / 2314 / 6409` to both the receipt and the notes) |
| idle unload: host gone **5.3 s** after a 5 s window; device 6384 → 2314 → 6409 MiB | §1 A-E4-2 | ✅ re-derived: gone **5.2 s** after the window, device 6330 → 2255 (loaded) → 6345 MiB (unloaded), pid confirmed gone and the socket gone with it |
| one model at a time: a different key stops the old host *before* the new one loads | §1 A-E4-3/A-E4-5 | ✅ re-derived: `A → B → A` gave three distinct pids and the old pid was dead before the new model loaded; the `--threads` swap also swapped the host |
| keep-alive precedence **flag > env > default**, 600 s default, `--keep-alive 0` = pre-E4 | §1 A-E4-6, SPEC §2.12 | ✅ code reads as documented (`keep/identity.py::resolve_keep_alive`, `DEFAULT_KEEP_ALIVE = 600.0`) and the live gate re-ran `0` (flag and `$TYPED_GGUF_KEEP_ALIVE=0` → no `engine.keep` block, `keep status` stopped) plus `$TYPED_GGUF_KEEP_ALIVE=5s` → a host with a 5.0 s window |
| inline fallback (unreachable/dead spawn → answer inline, named reason) + `W_KEEP_UNAVAILABLE` off-unix | §2, SPEC §2.12 | ✅ exercised by the offline pins and the live A-E4-4/A-E4-7 pair; `keep status` on a socket nobody answers is `unresponsive` (bug 2.2's fix holds — re-probed offline, exit 0) |

**(b) One warm claim re-derived by execution** — done twice over: the warm call above reports
`timings.model_load_ms == 0.0` on a resident host that answered the previous call from the same pid,
and the `--keep-alive 5s` host was watched **gone by pid** 5.2 s after a 5 s window with the device
free memory back to its pre-load level. Both are the card's own suggestions, both pass.

**(c) The `keep` surface for footguns** — one found, and it is a real (if narrow) one: **`keep stop`
signals a pid it has not verified is a host** (🟠 M2 below). The two bugs the E4 card *did* fix were
re-probed and hold: `keep status` on a busy/dead host reports `unresponsive` instead of `E_INTERNAL`,
and a `kill -9`'d host's socket+record are cleaned up by the next call (the live A-E4-4/A-E4-7 gates
re-ran green).

## uvx-specific check (card item 3)

- The README block (`README.md` → *Install without a clone*) names the packaged lock and the
  precedence rule, and the artifact gate executes exactly that contract: the built wheel carries
  `typed_gguf/data/runtime.lock`, an installed tool reads it from a neutral cwd that plants a **decoy**
  lock, `$TYPED_GGUF_LOCK` stays authoritative and the `E_RUNTIME_MISSING` text lists every path
  searched. `pytest -q tests/test_wheel_install.py` → **8 passed** here. ✅
- The **post-publish-only half is declared, but in the release notes, not in the README**: the notes
  say the `git fetch` step of the `git+https://…` spelling is verified **post-publish** (the checkout
  has no remote and that URL answers *Repository not found* today), and a new gate pins those words.
  The README's own text only offers the alternative (“Before/without a published remote, the same
  thing works from a checkout — `uvx --from . …`”) next to a sample output measured from a local
  path. 🟡 **M3** — one sentence would make the two documents say the same thing.

## Docs consistency (card item 4)

- **README ↔ release notes ↔ code agree on the whole `keep` surface**: 600 s default,
  `--keep-alive <dur|0>`, `$TYPED_GGUF_KEEP_ALIVE`, *flag > env > default*, one host per data home,
  swap-on-different-key, `engine.keep.served_by`/`fallback`, `keep status`/`keep stop`, the Windows
  `W_KEEP_UNAVAILABLE` path. The new `tests/test_public_docs.py` block asserts the same fact list
  against **both** documents and reads the field set out of `schema.Options()`, so a later edit to
  either side fails the gate instead of drifting. ✅
- **`schema.Options()` defaults vs the warm host**: the window is deliberately *not* a request option
  — it is a CLI flag / env knob (`keep/identity.py::DEFAULT_KEEP_ALIVE`), and the new gate fails any
  public document that spells it `options.keep_alive` or names an `options.<field>` the schema lacks.
  The policy-v2 defaults (`cue=json_instructed`, `chat_format=role_split`, `json_contract=question`)
  are still read from `Options()` into SPEC §2.5 and the quickstart, unchanged by E4. ✅
- **SPEC §2.12 vs the code**: the section's claims (detached `setsid` child, 0600 unix socket in
  `$TYPED_GGUF_HOME/keep/`, never TCP, one host/one model/one data home, the key's composition, the
  countdown restarting on every request, serialized requests, `os._exit` teardown after a clean
  close, debris probed and replaced, the fallback policy) all match `src/typed_gguf/keep/*` line for
  line. ✅
- One stale line inside a **receipt**: `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md:11` calls the
  model “the 4B **Q4_K_M**” while the gate loads the pinned 4B **Q8_0** (`tests/test_keep_live.py:39`,
  and the release notes say `Q8_0`). ⚪ **N1** — the numbers are unaffected; the receipt names the
  wrong quant.

## Checklist (the release gate's own list, re-run at `ed48acb`)

| # | item | status | evidence |
|---|---|---|---|
| 1 | LICENSE (MIT) + credits | ✅ | `LICENSE` MIT; `pyproject.toml` MIT + classifier; README *Credits and attribution*; the notes repeat MIT + no-parity and a gate fails if they go |
| 2 | pyproject metadata | ✅ | `typed-gguf 0.1.0`, `requires-python >=3.11`, console script, `force-include` mapping `runtime.lock` into the wheel, `[tool.mutmut]` documenting the E4 pair |
| 3 | CI workflow sane (offline, no downloads) | 🔴 | the workflow *is* offline and downloads nothing — but its "Offline suite" step **fails at this head** (blocking issue B1) |
| 4 | no secrets in tracked files | ✅ | 0 hits over the tracked tree (area 14) |
| 5 | hygiene applied | ✅ | clean clone of the certified sha is clean; `.gitignore` carries the live/scratch rules; the 4.3 GB scratch home stays ignored |
| 6 | fresh-install acceptance quoted | ✅ | unchanged from the first gate (`.t07b5/logs/*` receipts + the acceptance report), plus the new out-of-tree artifact gate |
| 7 | `--help` surface self-consistent | ✅ | F1/F2 closed; exit codes as documented (area 13) |
| 8 | docs ↔ code for the new surface | ✅ | areas 3/4 and the two consistency sections above |

## Blocking issues

**B1 🔴 CRITICAL — the committed CI gate is red at `ed48acb` (and was at `dec1819`).**
`tests/conftest.py:180-200` (`_network_disabled_for_this_run`) replaces `socket.socket` with a raiser
for **every** address family when `TYPED_GGUF_TEST_BLOCK_NET=1`, including `AF_UNIX`. The E4 feature
is *built* on `AF_UNIX` sockets, and its gates open them in-process: `tests/test_keep_client.py` and
`tests/test_keep_host.py` fail with `AssertionError: network disabled for this run …: a decision path
must never need it` (the host's `bind()` raises inside its thread, which is also where the 11
teardown errors come from).

Reproduction on a clean clone of `ed48acb` (the exact shape `.github/workflows/ci.yml` runs):

```
TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<empty stub> \
  .venv/bin/pytest -q -rs --timeout=120
  → 24 failed, 1466 passed, 56 skipped, 11 errors      (log: /tmp/rg2/logs_e/suite_ci.txt)
TYPED_GGUF_TEST_BLOCK_NET=1 .venv/bin/pytest -q \
  tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py
  → 24 failed, 69 passed, 11 errors                    (log: /tmp/rg2/logs_e/keep_netblock.txt)
# same four files, flag off → 93 passed
```

`ci.yml`'s step ends with `! grep -qE "[0-9]+ (failed|error)" "$log"`, so the first push of this tree
is a red workflow. The E4 receipt's “1485 passed, 56 skipped”
(`.e2e/t_7e24cea4-warm-host/logs/suite_final.txt`) was measured **without** the flag, which is why the
card's own expectation (~1485/56 for the CI-shape suite) cannot be met at this head. The first gate at
`cb79e92` ran the CI shape green (1362/49) — the regression is E4's.

*Owner/direction (not fixed here, per the review's rules):* the hook in `tests/conftest.py` must stop
forbidding local IPC — scope it to the network families (`AF_INET`/`AF_INET6`, plus
`socket.create_connection` / `getaddrinfo`, which the same fixture already patches) or let the keep
gates opt out by name; the decision belongs to whoever owns that fixture. Either way the flag must
stay meaningful for the download paths it was written for.

## Findings (non-blocking)

**🟠 M2 — `keep stop` signals a pid the ledger never verified is a host.**
`src/typed_gguf/keep/state.py`'s own rule is “a record is a claim, not a fact — `alive()` checks the
pid *and* the socket; nothing is inferred from the file alone”, and `client.stop()`
(`src/typed_gguf/keep/client.py:362`) is the one verb that acts on the record: it reads
`record.pid`, and — after the `pid == os.getpid()` guard — calls `os.kill(pid, SIGTERM)` on
`state.pid_alive(pid)` alone. No socket probe, no identity check. The realistic path is the one the
module itself says is expected debris: a host killed with `kill -9` leaves its record **and** its
socket behind; its pid is recycled by any other process on the box; the next `keep stop`, a
`--keep-alive 0` call (which also calls `stop()`), or a swap to a different key then SIGTERM/SIGKILLs
whatever now owns that pid.

Confirmed by execution (clean clone, scratch home, a decoy `sleep 600`): with a record whose socket
nobody listens on and whose pid belongs to the decoy, `keep status --json` is honest (`unresponsive`,
exit 0, no harm) while `keep stop --json` answers `{"stopped": true, "pid": 20300, "reason":
"stopped (SIGTERM)", "cleaned": true}` and the decoy's `/proc/<pid>/stat` goes `R → Z`.
Suggested guard (two lines, and it is the same policy `_abandon(hard=True)` already applies):
`stop()` should treat a record whose socket is not *listening* as debris — clean it up, signal
nothing — and only signal a pid when the socket answers a probe (a busy host still listens).

**🟡 M3 — the README's uvx block does not carry the “post-publish” limit its own notes now do.**
The release notes (landed as `ed48acb`) state that the `git fetch` step of
`uvx --from git+https://…` is verified **post-publish**; the README's block shows the same one-liner
with the measured sample output beneath it and only implies the caveat (“Before/without a published
remote, the same thing works from a checkout”). One sentence — the notes' own words — would make the
two documents agree. No behaviour claim is affected: the artifact gate executes the local-path half.

**⚪ N1 — the E4 evidence doc names the wrong quant.** `docs/evidence/v0_1_0_t_7e24cea4_warm_host.md:11`
says “the 4B **Q4_K_M**”; the live gate loads the pinned **Q8_0** (`tests/test_keep_live.py:39–40`),
which is what the README, the notes and the card's numbers are about. The receipt's numbers are the
gate's own output, so nothing measured changes — it is a receipt line to correct on the next pass
into that file.

**⚪ N2 — one new citation only resolves while the sweep's scratch dir exists.**
The E4 evidence doc's census sentence names the meta-file glob the sweep writes into its own scratch
tree (`mutants/…`). That tree is dev scratch, never tracked, and consistent with two pre-existing
mutation receipts that cite the same shape; the hygiene ledger's *cited-but-untracked* count stays
**0** in a clean clone, which is the number the gate cares about.

**⚪ N3 — `typed-gguf models` with no subcommand exits 0** and prints the usage line (every other
missing-argument path in the CLI is exit 2). Untouched by E4 and not documented either way; noted
only because the help/exit-code surface was re-probed line by line this time.

## What landed around the certified head

- **Certified sha: `ed48acb`** (`docs(t_67bb0409): the notes catch up — the warm engine host and the
  uvx install path`), i.e. `dec1819` + the release-notes catch-up (+64 lines of notes, +63 lines of
  `tests/test_public_docs.py`). Everything reported above was run at `ed48acb` except the clean-clone
  CI-shape suite, which was run at **both** heads (24 failed / 1461 passed at `dec1819`, 24 failed /
  1466 passed at `ed48acb`) so the blocker is attributable to E4 and not to the notes.
- **The release-notes catch-up is folded in**, and it is what closes the one docs gap the first gate
  left open: the notes now headline the warm host, quote the E4 receipts' numbers, carry the uvx
  one-liner with its post-publish limit, and the new gates pin both documents to the same facts.
- **In flight while this review ran, explicitly *not* certified:** `t_c3195a5c` (tests-only, a
  `sun_path`/long-`TMPDIR` fix for the keep fixtures) is sitting **uncommitted** in the shared
  checkout (`tests/conftest.py`, `tests/test_keep*.py`, plus an untracked
  `tests/test_keep_socket_path.py`). It does not touch the net-block hook, so it cannot clear **B1** —
  and none of the 24 failures here were `sun_path`-shaped (0 occurrences of that path in the failure
  logs). Its own receipt must be judged at its own head.
- **This review's own commit lands after `ed48acb`** (`REVIEW.md` only), exactly as the previous
  gate's did at `232a11b`.

## Verdict

**v0.1.0 is not releasable at `ed48acb`.** One blocker: the committed CI workflow's offline-suite step
is red at this head — `TYPED_GGUF_TEST_BLOCK_NET=1` forbids `AF_UNIX`, so the 24 keep gates that
exercise the new socket surface fail (24 failed / 11 errors in a clean clone), and the published
receipt that says the suite is green was taken without that flag. Everything else the release claims
was re-executed and holds: the E4 live surface (7 gates, 410 s, cold → warm with `model_load_ms 0.0`
on the same pid, the 5 s idle unload by pid and by device memory, the one-host-at-a-time swap), the
offline keep pins, the oracle (live `failures: 0 skips: 0`, bundle-free `failures: 0`), ruff, the
E3e/policy/parity/doc gates, the red path's all-skip, the wheel/uvx artifact gate (8 passed), the
`--help`/exit-code surface (F1/F2 closed), the citation ledger (0 cited-but-untracked) and the secret
scan. Fix B1 (and, cheaply, M2) and this is a re-run of one command away: the same clean-clone suite
with the flag, which must come back `0 failed / 0 errors`.
