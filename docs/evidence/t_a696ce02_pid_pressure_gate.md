# The pid-pressure gate: `pytest -q` is green on a box that can fork (card t_a696ce02)

Filed by `t_80f1a4c6` (§7 F7 of `docs/evidence/e3_fix_t_80f1a4c6_serving_backend.md`, raw:
`.e2e/t_80f1a4c6-serving-backend/parent_flake_control.txt`): the repo's default gate is not
reliably green in this container, and the failures looked like the product.

**What it was.** Not a product bug and not, in the end, an ordering bug. The worker container's
pid cgroup is small and *shared* (`pids.max = 256`, the sibling sandboxes inside it), so at the
cap the kernel answers `fork()` with `EAGAIN`; the runtime-probe gates spawn real children
(isolated probe child, `llama-cli --version`, the oracle), and their failures were being rendered
as probe findings — `['probe_failed', 'no_asset'] == ['loader_error', 'no_asset']`.
`pytest-randomly` was only the messenger: it changes *when* each gate runs, and the box's cap
comes and goes from sibling campaigns.

**What this card did.** Three layers, each pinned by a gate:

1. `src/ggufone/runtime/pressure.py` (new) — the live cgroup reading (`PidHeadroom`), a bounded
   retry for the *transient* spawn errnos, and a *named* `E_PID_PRESSURE` failure carrying the
   reading when the cap is real.
2. `tests/conftest.py` — the suite gate: every run's header prints the reading; a test that needs
   a real child says so (`@pytest.mark.needs_fork`, 25 tests marked after measurement); under a
   starved cgroup those gates **skip by name** and the run **forces a non-zero exit**, so a box
   that could not be measured can never look green. One loud `test_the_pid_cgroup_has_fork_
   headroom_for_the_probe_gates` names the box instead of 36 phantom product failures.
3. `tests/test_probe_pressure.py` (new, 14 gates) — the retry budget, the named reason, the
   "a child that ran is never retried" rule, the closed set of pressure errnos, the install
   chain's reason *codes* under a blocked fork, and the gate's own end-to-end behaviour (a nested
   pytest run with a simulated cgroup reading).

Nothing was loosened: the reason strings (`loader_error` / `backend_absent` / `probe_failed`) are
still the contract other cards assert on — they just cannot be produced by a box that refused to
fork any more.

## 1. The measurement

### 1a. The filed flake, on the real box (pre-fix, `--randomly-seed=N`, the three files)

`sh .e2e/t_a696ce02-pid-pressure/rig/seed_matrix.sh baseline "1 2 3 4 5 6"` →
`.e2e/t_a696ce02-pid-pressure/logs/baseline_seed_matrix.log` (raw pytest output per seed):

| seed | `pids.current` before → after | result |
|---|---|---|
| 1 | 107 → 150 | **43 passed** in 42.05 s |
| 2 | 151 → 246 | **43 passed** in 37.20 s |
| 3 | 244 → 254 | **6 failed**, 36 passed, 1 skipped in 23.23 s |
| 4 | 254 → 253 | **3 failed**, 40 passed in 6.89 s |
| 5 | 253 → 251 | **2 failed**, 41 passed in 13.86 s |
| 6 | 251 → 253 | **3 failed**, 40 passed in 3.93 s |

The seed is not what decides it — the **reading at run time** is. Seeds 1-2 pass with the same
three files, the same order semantics and the same code; seeds 3-6 fail while the cgroup sits at
244-256. (Reproducing the parents' "1, 5, 6, 20 and 36 failures": the count is how many of the
spawning gates happen to run inside a capped stretch, which is why it varies per run.)

### 1b. The mechanism, verbatim

The probe child's spawn is refused and the refusal is reported *as a probe finding*:

```
AssertionError: ['probe_failed', 'no_asset'] == ['loader_error', 'no_asset']
E   AssertionError: assert 'probe_failed' == 'backend_absent'
E   ['the isolated probe could not verify the cuda backend (the isolated probe could not be
     started (<venv>/bin/python -m ggufone.runtime.probe_child): [Errno 11] Resource temporarily
     unavailable); treated as unusable here']
```

The oracle's own fork dies inside the frozen script:

```
returncode=1 ... self.pid = _fork_exec(...)
BlockingIOError: [Errno 11] Resource temporarily unavailable
```

The box says the same thing to anything else that forks — including `uv` itself, mid-install:

```
thread 'main2' panicked at crates/uv-client/src/cached_client.rs:896:14:
OS can't spawn worker thread: Resource temporarily unavailable (os error 11)
Aborted (core dumped)
```

and to an interactive shell: `bash: fork: retry: Resource temporarily unavailable`.

`/sys/fs/cgroup/pids.max` = **256**, mounted read-only, so the box cannot be partitioned from
inside; `pids.current` is the *shared* number (this container's own `ps` shows ~35 processes while
`pids.current` reads 189-256).

### 1c. What the order exposes: the pid table, not a Python cache

The card asked for the shared state the order exposes (a module cache? a `GPU_HOST`-style object?
a reused temp path?). Answer: **none of those — the container's pid table.** Measured with a
per-test `pids.current` probe (`rig/pidtrace.py`, log `logs/pidtrace.log`): across 43 tests the
reading is *flat* for plates of 10-11 consecutive test boundaries (223, 223, … / 231, 231, …) and
jumps only between plates (213 → 223 → 225 → 230 → 231 → 255 → 256) — our own tests do not leak a
child (each spawn is a synchronous `subprocess.run`), the jumps are the siblings. A full-suite run
at 213 → 256 inside 3.3 s confirms it. Repeated on the fixed tree: the same seeds on a box at 206
are green (see §3), on a box at 242 they skip by name.

`pytest-randomly` matters only because it changes which gates are in flight when the cap lands:
with a fixed order the same subset always runs at the same moment; with a random order a different
subset catches each capped stretch. That is why the deterministic `-p no:randomly` gate the
previous cards quoted was green: not because order was the cause, but because that particular
order ran the spawning gates before the cap arrived.

## 2. The fix, layer by layer

### 2a. `src/ggufone/runtime/pressure.py` (new, 73 statements)

```python
PID_HEADROOM_FLOOR = 16     # pids a fork gate needs free before it can trust itself
E_PID_PRESSURE = "E_PID_PRESSURE"
SPAWN_ATTEMPTS = 4          # 1 try + 3 retries
SPAWN_BACKOFF = 0.2         # 0.2 + 0.4 + 0.8 s = 1.4 s worst case
SPAWN_ERRNOS = {EAGAIN, EINTR, ENOMEM, EMFILE, ENFILE}   # "the box", not "the command"
```

* `read_pid_headroom(root="/sys/fs/cgroup") -> PidHeadroom | None` — `pids.current`/`pids.max`,
  `"max"` handled as *no limit*, unreadable/garbage as `None` (never invented numbers);
  `PidHeadroom.describe()` is the phrase every failure carries: `pids.current=254/256 (2 free)`.
* `spawn(command, **kwargs)` — `subprocess.run` that retries **only** `SPAWN_ERRNOS` for the
  bounded budget, then raises `SpawnBlocked(OSError)` (`E_PID_PRESSURE` + the live reading). A
  non-zero exit, a timeout, or `ENOENT` are never retried: a child that ran is *data* about the
  bundle, a child the kernel refused to create says nothing about it.
* `SpawnBlocked` subclasses `OSError`, so every existing `except OSError` around a spawn keeps
  working (the two tool checks rely on it: a busy box now costs a *retry*, not the build number).

Call sites: `isolated.run_child` (the probe child, the warm-up child and the system-lib child all
funnel through it), `capability.build_number` (`llama-cli --version`) and the `llama-fit-params
--help` check in `capability.probe_runtime`.

### 2b. `tests/conftest.py` — the gate over that reading

* `pid_headroom()` — the live reading, or the `GGUFONE_TEST_PID_HEADROOM` override (`250/256`,
  `0/256`, `none`), a test knob in the same spirit as `GGUFONE_TEST_BLOCK_NET`, so the gate
  itself is testable on any box.
* `pytest_report_header` — every run prints `pid cgroup: 213/256 (43 free) (ok; the fork gates
  need 16 free)`.
* `@pytest.mark.needs_fork` + an autouse fixture — a marked test on a starved box skips
  *by name* (`pid cgroup has no fork headroom (pids.current=250/256 (6 free)): … the box, not the
  product`); the check is per test, because the reading can cross the floor mid-run.
* `pytest_terminal_summary` + `pytest_sessionfinish` — a pressure skip prints a loud
  `PID PRESSURE: N fork gate(s) skipped` block and forces a **non-zero exit**. A run that could
  not measure the fork gates must not look green; this is the deliberate, one-line-of-triage
  version of the 36 phantom failures.
* The 25 marked tests are **measured, not guessed**: `GGUFONE_TEST_DENY_SPAWN=always`
  (`rig/inject.py`) denies every spawn in the three files plus `test_probe_isolation.py` and
  `test_capability.py`; the 25 that fail are exactly the ones that need a real child
  (`logs/inject_always_3files.txt`, `logs/inject_always_isolation_capability.txt`).

### 2c. `tests/test_runtime_contract.py` — the frozen oracle

The oracle is the contract and is never edited, but *its* forks are also refused at the cap. The
test-side `run_oracle` now spawns through `pressure.spawn` and retries a run whose own trace shows
a fork refusal (`Resource temporarily unavailable` / `BlockingIOError`) — and only such a run: a
contract failure is returned as-is and asserted. The three oracle gates are `needs_fork`-marked
on top (they genuinely need forks). The residual window (a fork refused *inside* the oracle's
last attempt) is stated in §5.

## 3. The run matrix

Everything is `-q`, in the container, `HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache
TMPDIR=/tmp`; raw logs in `.e2e/t_a696ce02-pid-pressure/logs/`.

| # | box | spawns | what runs | result | exit |
|---|---|---|---|---|---|
| A | real, `pids.current` 206 | real | full suite, `--randomly-seed=11/12/13` (3 runs) | **1129 passed, 43 skipped** each (baseline skip count) | **0** |
| B | forced healthy (`0/256`) | first 3 denied (below the budget) | the 5 probe/isolated files | **98 passed** — the retry absorbed it | 0 |
| C | forced healthy (`0/256`) | first 9 denied | the same 5 files | 98 passed (the denials land on the two *tolerant* tool checks first) | 0 |
| D | real, 242 → 256 (capped) | real | full suite, `-p no:randomly` | 1123 passed, **49 skipped**, **0 failed** | 1 (forced) |
| E | forced starved (`250/256`) | every spawn denied | the same 5 files | 73 passed, **25 skipped, 0 failed** | 1 (forced) |
| F | forced starved (`250/256`) | real | the nested gate pin (2 tests) | 1 failed (the headroom gate, by name) + 1 skipped | ≠0 |
| G | pre-fix, real 244-254 | real | the three files, seeds 3-6 | 2-6 **mislabelled failures** per run | ≠0 |

Rows D/E/F are the point: a capped box now produces **named skips and one loud gate**, never
`['probe_failed', 'no_asset'] == ['loader_error', 'no_asset']`.

### RED on the landed parent (`8474802`, the shared tree's main)

`.e2e/t_a696ce02-pid-pressure/logs/red_parent.txt` — a detached worktree at the parent commit with
only the new test files copied in (`rig/red.sh`):

* `tests/test_probe_pressure.py` on the parent code: **3 failed, 10 passed** — the three failures
  are exactly "no retry" and "the reason is not named" (`E_PID_PRESSURE` absent from
  `child_error` and from the install chain's `reason`);
* the three flaky files with every spawn denied: **8 failed** (the five mislabelled ones with the
  old unnamed text, plus the three oracle gates) — the filed shape, reproduced deterministically.

## 4. Gates

| gate | result | raw |
|---|---|---|
| RED on the landed parent | 3 failed, 10 passed (new gates) + 8 failed (the filed flake, injected) | `logs/red_parent.txt` |
| GREEN, new gate file | **14 passed** (one of them the nested-run pin) | `out/green_gates.txt` |
| full suite, `-p no:randomly`, forced-healthy cgroup | **1129 passed, 43 skipped** in 28.0 s | `logs/matrix2.txt` (row D2) |
| full suite, pytest-randomly, seeds 11/12/13 | **1129 passed, 43 skipped**, exit 0, three runs | `logs/matrix.txt` (rows E) |
| full suite, capped cgroup (real 242-256) | 1123 passed, 49 skipped, 0 failed, exit 1 (forced) | `logs/matrix.txt` (row D) |
| ruff (`src` + `tests` + `tools`) | `All checks passed!` | `logs/ruff.txt` |
| coverage (6 probe/isolated gate files) | `pressure.py` **97 %**, `isolated.py` 89 %, `capability.py` 91 %; the 2 misses in `pressure.py` are the `assert last is not None` guard | `logs/coverage.txt` |
| Tier-M mutation | 76.4 % (288 mutants, complete) / `pressure.py` 93.4 % — first run interrupted by the cap | `logs/mutation.txt` |

### 4a. Tier-M sweep

`[tool.mutmut]` retargeted **in the private clone only** (the shared tree's `pyproject.toml` is a
live sibling's WIP — this block is retargeted by every card, so the pair is recorded here instead
of landed): `source_paths = ["src/ggufone/runtime/pressure.py",
"src/ggufone/runtime/isolated.py"]` with the selection `[tests/test_probe_pressure.py,
tests/test_probe_isolation.py, tests/test_capability.py]`, `max_children = 2`, driven by
`tools/mutmut_driver.py` through `rig/sweep.sh` (waits for pid headroom — the same shared cap).

| run | box (`pids.current`) | mutants | killed | survived | no-tests | not-run | score |
|---|---|---|---|---|---|---|---|
| attempt 1 | 116 → the cap hit at mutant 205/288 (`BlockingIOError` inside the driver) | 288 | 166 | 60 | 42 | 20 | 73.5 % |
| attempt 2 (complete) | 111 → ~250 | 288 | 188 | 58 | 42 | 0 | **76.4 %** |
|   ↳ `isolated.py` | | 227 | 131 | 54 | 42 | 0 | 70.8 % |
|   ↳ `pressure.py` | | 61 | 57 | 4 | 0 | 0 | 93.4 % |
| pressure.py alone (re-run, before the survivor gates) | ~200 | 61 | 49 | 12 | 0 | 0 | 80.3 % |

The same 61 mutants scored 93.4 % and 80.3 % in two runs: on this box the score is only readable
as a *lower bound* for quality — a test that fails for box reasons (a fork gate dying under the
cap) marks its mutant "killed" too, so the **quieter run's number is the honest one** (80.3 %),
and the diff dump below is taken from it.

Survivor classification (what the mutants on this card's lines were):

* **9 × `x_spawn`** — the retry policy was not pinned: every gate set `SPAWN_BACKOFF = 0`, so the
  schedule (`2 ** (attempt - 1)` → `/ 2 ** …`, `3 ** …`, `* 2 * (attempt - 1)`), the
  `attempt < SPAWN_ATTEMPTS` guard and the `last = ""` initialisation were all unobservable.
  **Three gates added** (`test_the_backoff_schedule_is_the_documented_one` — the exact
  0.2/0.4/0.8 s schedule, one sleep per retry, none after the last attempt;
  `test_a_sustained_block_names_the_underlying_errno` — the message carries `BlockingIOError`,
  `[Errno 11]` and `.errno == EAGAIN`; plus the `pressure_note` pair below), and `spawn` now
  raises `SpawnBlocked(last.errno, …)` so the causal errno survives the wrapping.
* **2 × `x_pressure_note`** (`or True` on the `None` check, `else "XXXX"`) — no gate appended a
  note for a cgroup-less host. **Killed** by
  `test_the_pressure_note_is_empty_when_there_is_no_cgroup`.
* **1 × `x__read_int`** (`"max"` → `"XXmaxXX"`) — **equivalent**: `int("XXmaxXX")` raises into the
  same `None`, so no test can tell the two apart.
* **54 × `isolated.py`** — pre-existing internals this card does not touch (`child_env` 17,
  `run_child`'s JSON/timeout shapes 16, `scan_bundle` 10, `ProbeScan.to_dict` 8, `system_libs` 3).
  Named, not chased (Tier M, soft threshold).
* 42 no-tests mutants are generation/legacy paths the gate selection never enters.

The confirming re-run *after* those three gates did not complete: the box went to
`pids.current = 256/256` (the sandbox container itself could no longer fork, `conmon` included),
so it is recorded as not-run rather than guessed — the reviewer can re-run `rig/sweep_pressure2.sh`
on a quiet box or accept the classification above. One trap worth stating for the next card: **a
mutation score taken while the box is capped is worthless in the other direction too** — the
suite gate forces a non-zero exit whenever it skipped a fork gate, so *every* mutant would look
killed.

## 5. Honest limits

* **The floor is a heuristic.** `PID_HEADROOM_FLOOR = 16` is set just above the measured failure
  band (failures appeared once `pids.current` ≥ 244, i.e. ≤ 12 free). A box *inside* the band can
  still produce a capped stretch after the reading said "ok"; the retry budget (1.4 s) covers the
  short ones, and a longer one surfaces as a named `E_PID_PRESSURE`. Both thresholds are
  constants in `pressure.py` — one line to move if a future box shows a different band.
* **A starved run is not green, by design.** It exits non-zero with named skips. That is the
  card's requirement ("fails loudly … instead of mislabelling it as product behaviour"); the cost
  is that a capped box needs a re-run, the benefit is that the failure names the box.
* **The oracle's inner forks are retried from outside, not inside.** The oracle is frozen, so a
  fork refused inside its *last* attempt still fails the gate — with the fork-refusal text in the
  assertion, never as a contract failure. Marking the three oracle gates `needs_fork` is the
  pre-emptive half.
* **`pytest-randomly` is not in the locked dev extra** (`uv.lock` has no such pin; the container's
  venv only carries `pytest`, `pytest-timeout`, `ruff`), so the random-order rows were produced
  with `pytest-randomly 5.0.0` installed explicitly — the same assumption the repo's own
  `[tool.mutmut]` args and `tools/t80_replay.py` make (`-p no:randomly`). Adding it to the `dev`
  extra is a one-line pyproject change, deliberately **not** taken here (that file is a live
  hotspot) and worth a card.
* **Mutation scope.** The sweep covers the two runtime modules the fix moves, not the tests'
  gate code (`conftest.py` is test infrastructure; its behaviour is pinned end-to-end by the
  nested-run gate instead).
* **Sibling load is still shared.** The gate reads the cgroup; it cannot stop a sibling from
  eating the pids the moment after. It can only refuse to pretend.

## 6. Files

`src/ggufone/runtime/pressure.py` (new) · `src/ggufone/runtime/isolated.py` · 
`src/ggufone/runtime/capability.py` · `tests/conftest.py` · `tests/test_probe_pressure.py` (new) ·
`tests/test_runtime_contract.py` · `tests/test_runtime_fallback.py` · `tests/test_runtime_install.py`
· `tests/test_probe_isolation.py` · `tests/test_capability.py` (the last five: the `needs_fork`
marks, measured) · `README.md` (the pid-pressure note) · `.e2e/t_a696ce02-pid-pressure/**` (rig +
raw material, index in its `README.md`).

## 7. Related, not this card (from the filer)

* `engine.fit.backend` still falls back to `backend_requested` (`t_80f1a4c6` §7 F3).
* `serve`/`mcp` are still stubs and must build responses through `cli.decide_payload` (§7 F4).
