# E4 — the warm engine host (card t_7e24cea4)

The card adds `src/typed_gguf/keep/` (SPEC §2.12): a `run`/`ask` call can leave a resident host
behind, and the next call to the same model is answered warm — no second load, no second fit plan.
This is the receipt file: the gates, the numbers they produced on this box, the two product bugs
they found on the way, and the mutation sweep over the new module.

Everything below was run from `/var/home/rybens/workspace/ggufone` with
`TYPED_GGUF_HOME=/var/home/rybens/.local/share/ggufone`; the live gates additionally used
`TYPED_GGUF_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` and
`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` (the 4B Q4_K_M on the pinned Vulkan bundle).

## 1. The live gates and their numbers

`tests/test_keep_live.py` — 7 gates, **all passed, 297.82 s** for the file. Each gate drives the
*product* CLI in a child process against its own data home: nothing is stubbed, the model is real,
the host is a real process, and the device number is read from the driver.

| Gate | What it asserts | Measured on this box |
| --- | --- | --- |
| A-E4-1 | a second call to the same model is answered by the host | cold **17.50 s** (load 2280 ms) → warm **2.58 s** (load 0 ms), pid 35915 both times |
| A-E4-2 | the host ages out after `--keep-alive` and gives the device back | host gone 5.3 s after a 5 s window; device free 6384 → **2314** (loaded) → **6409 MiB** (unloaded) |
| A-E4-3 | a different model/quant stops the old host first | pids A 37279 → B 37591 → A 38083, one host at a time |
| A-E4-4 | a killed *client* leaves nothing wedged; a killed *host* and a stale socket are cleaned up | the surviving host answered the next call with load 0 ms; a stale socket is unlinked before the respawn |
| A-E4-5 | the placement key is part of the identity (threads change ⇒ different host) | a `--threads` change swapped the host; the old pid was gone before the new one loaded |
| A-E4-6 | `--keep-alive 0` (flag and `TYPED_GGUF_KEEP_ALIVE=0`) never enters the keep path | no host, no socket, `keep status` stays `stopped`, `engine.keep` absent |
| A-E4-7 | the host outlives nothing it should not: a clean exit removes socket + spec | receipts from A-E4-2/A-E4-3 (pid gone, files gone) |

An earlier full run of the same file measured cold 24.50 s (load 3187 ms) → warm 4.01 s (load
0 ms). The gap between the two runs is the box (page cache, driver init, a co-tenant's GPU work),
not the product: the *warm* number is the one the feature owns, and it moved 4.01 → 2.58 s between
runs because the cold one did.

Offline, the card's own gates are `tests/test_keep.py`, `tests/test_keep_host.py`,
`tests/test_keep_client.py`, `tests/test_keep_cli.py` — 93 gates, all green — and the whole suite is
**1485 passed, 56 skipped** (the 49-skip offline baseline plus these 7 live gates, which skip *by
name* and make a run exit non-zero unless the live flag asked for them), and the **oracle is
unaffected: `failures: 0 skips: 0`** with the box's models visible (without them its two
model-dependent sections skip by name — the container's `HOME` is `/root`, the models are mounted
under the operator's home). Receipts:
`.e2e/t_7e24cea4-warm-host/logs/{suite_final.txt,oracle_final.txt}`.

Note on the cold number: `timings.model_load_ms` (2280/3187 ms) is the session's model load; the
cold call's wall clock (17.50/24.50 s) is that load **plus** the one-time fit plan, the process
spawn and the first read of a 2.5 GB file. Only the warm call is decision-only.

## 2. Two product bugs the live gates found

Both were fixed RED-first: the pin was written and watched to fail before the fix landed.

### 2.1 The call that *paid* the load reported `model_load_ms: 0.0`

A host answers every request from the session it already opened, so `timings.model_load_ms` is
correctly 0.0 on every request it serves — including the first one, the call that paid for the
spawn and waited for it. The live gate read that number on the cold call and reported it.

Fix: the client credits itself with the host's one-time `model_load_ms` when *it* was the call that
spawned the host (and only then) — `keep/client.py::_credit_the_spawn`, pinned by
`tests/test_keep_client.py::test_the_spawning_call_reports_the_load_it_waited_for` and
`::test_an_inline_answer_keeps_its_own_load_number`.

### 2.2 `keep status` crashed on a busy host

After `kill -9` of a client mid-request, the host is still finishing the abandoned request. The
gate's next command was `keep status`, and it came back `E_INTERNAL` (exit 4) with
`TransportError: the host answered nothing usable (…the host closed the connection without an
answer…)`: `Client.status()` caught `KeepUnavailable, OSError, ValueError, JSONDecodeError` but
**not** `TransportError`, so the one verb whose job is to say *what state the host is in* died
instead of saying `unresponsive`.

Fix: `status()` reports `{"state": "unresponsive", "detail": "<TransportError…>"}` for that case
too, pinned offline by
`tests/test_keep_client.py::test_status_of_a_host_that_cannot_answer_says_unresponsive` (a real
listening socket that accepts and never answers, so the ping times out deterministically).

The live gate keeps both halves of the claim: a host busy with an abandoned request is
`unresponsive` (not broken), and the *next* call is still served by that same pid with load 0 ms.

## 3. The box is not the product

One gate of the first live run died with `libgomp: Thread creation failed` — the container's pid
cgroup was at its 256 cap (a sibling card's vitest/stryker campaign), and on this box the pid cap
*is* the thread cap. That is the same failure the wheel gates already classify, so
`tests/test_keep_live.py` reuses the discipline: `STARVATION_TOKENS` over the child's stderr and a
`pytest.skip` that names the cgroup, and every live gate carries `@pytest.mark.needs_fork` so
conftest's headroom probe skips it before it starts. A starved box must never read as a product
finding in either direction — the green run above was taken with `pids.current` under 150.

## 4. Mutation sweep (Tier M)

The pair in `pyproject.toml` is `source_paths = ["src/typed_gguf/keep"]` — the four modules the card
adds — with the card's four offline gate files as `pytest_add_cli_args_test_selection`:
**2176 mutants**. The driver is `.e2e/t_7e24cea4-warm-host/mutmut_sweep.sh` (mutmut 3.8 does not
read `max_children` from the table, and this box's pid cgroup kills an unthrottled run:
`--max-children 2`, retried around the shared cap). Two helper scripts keep the numbers checkable
without re-reading a 55 KB spinner log: `mutmut_census.py` (killed / survived / unrun straight out
of `mutants/**/*.py.meta`) and `mutmut_triage.py` (the same, grouped by the function the mutant
sits in).

| round | killed | survived | no tests | timeout | suspicious | total | killed / scored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 — before the pins | 1310 | 785 | 75 | 5 | 1 | 2176 | 62.5% |
| 2 — after six new pins | 1424 | 752 | 0 | 5 | 1 | 2176 | **65.4%** |

`no tests` is mutmut's own category — a mutant on a line *no selected test covers* — and it is not
a kill: a naive census that reads only the exit codes calls round 1 "1391 killed", which is 1310
real kills plus those 75 unexercised lines plus 5 timeouts and 1 suspicious. The table splits them,
because the difference is exactly what the round earned.

The sweep earned **six pins** (Tier M: survivors → new pins), all RED-first — the `_fail` pin was
watched to fail against the product (see below), the other five were confirmed by re-running their
mutants:

| the function | round 1 | round 2 | what the pin asserts |
| --- | --- | --- | --- |
| `host.Server._fail` | 0/45 (0%) | 36/45 (80%) | a host that cannot load leaves a readable `failed` record: code, message, exit code, no socket |
| `client.Client.stop` | 40/104 (38.5%) | 69/104 (66.3%) | stop of a host this client did **not** spawn (no `Popen`: the pid is waited for), the SIGKILL escalation, and the refuse-to-signal-itself guard |
| `identity.KeepKey.describe` | 4/32 (12.5%) | 25/32 (78.1%) | the exact line `keep status` prints, in both shapes (optionals set / unset) |
| `state.wait_pid_gone` | 0/15 (0%) | 11/15 (73.3%) | alive → False, exited → True (a zombie counts as gone), pid 0 → True |
| `client._listening` | 0/14 (0%) | 10/14 (71.4%) | the probe before every spawn: missing file, dead file, listening socket |
| **the five groups** | **44/210 (21.0%)** | **151/210 (71.9%)** | |

And one of those pins found a **product bug**. `Server._fail` called `self._write_record(state=
"failed", …)`, but that parameter is named `record_state` — a `TypeError`, swallowed by the
deliberate `contextlib.suppress(Exception)` around the write (a host that cannot start must still
exit with its code). So a host that failed to load left **no** diagnosis behind, and the
`record.state == "failed"` branch in `client.py` — the one that turns a dead spawn into a typed
`KeepUnavailable` with a reason — was unreachable: every failing spawn fell through to the log-tail
path instead. Fixed in `src/typed_gguf/keep/host.py` (`record_state="failed"`), pinned by
`tests/test_keep_host.py::test_a_host_that_cannot_load_leaves_a_readable_failed_record`. This is
exactly the class of bug a sweep is for: no test *covered* those 45 mutants, so no gate could see
the line was dead.

The 752 survivors split into two kinds, and neither is a hidden product hole:

* **Equivalent mutants on reporting paths** — the `round(…, 3)` precision in the four status
  payloads, the 19 `contextlib.suppress(...)` guards around best-effort teardown, the `timeout=2.0`
  / `interval=0.05` constants in the stop and wait loops, casing in the human line. Changing them
  changes nothing a caller can observe (or changes a *number* the spec explicitly leaves to the
  box).
* **Cross-process behaviour the offline selection cannot see** — the two biggest groups,
  `host.Server.handle_line` (74 survivors) and `host.Server.status` (47), are the wire protocol and
  the status reply: the offline gates drive them over a real socket but assert whole replies, while
  the mutants that survive touch fields only a *live* host's reader distinguishes (a real engine's
  placement, a real load time, a killed client mid-answer). `tests/test_keep_live.py` covers those
  paths end to end — and is deliberately not in this selection, because it needs the real 4B, the
  Vulkan bundle and five minutes.

Round 2 is not a second full sweep: it re-ran the five named groups (`mutmut run
'<name>*' …`, the mutant-name filter mutmut 3.8 accepts) and re-measured the tree mutmut had
already written. The verdicts of the untouched functions moved by a few mutants in the same
direction the `no tests` bucket explains — e.g. `host.py`'s census-killed count drops 431 → 422
because its 45 unexercised `_fail` mutants became 36 real kills and 9 honest survivors — so the
group table above, not the ±10-mutant package total, is the measurement of what the pins earned.

Receipts: `.e2e/t_7e24cea4-warm-host/logs/{mutmut.out,mutmut_round2.out,triage_round1.txt,
triage_round2.txt}` (the round-1 log holds `mutmut results`' per-mutant verdicts in full).

`tests/test_keep_cli.py` reaches the `cli.py` keep wiring (`keep_key_for`, `decide_payload_warm`,
`keep status|stop`) by injection, so the CLI surface is covered by the selection without making
`cli.py` (44 KB, the whole command surface) part of the mutated set.
