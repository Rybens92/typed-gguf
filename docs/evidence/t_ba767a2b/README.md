# t_ba767a2b — the batched fix card from review `t_c7d5e33d`

Two MAJORs (M1 the wave's py3.11 CI red, M2 `runtime update` skipping its own pre-flight), the
coordinator's M2b decision (option **(b)**: the default target is the variant `init` installed) and
the card's M3 minors (the `serve` HTTP bounds). No push, no tag: the worktree is the deliverable.

| | |
|---|---|
| Repo | `/workspace/ggufone` (host `~/workspace/ggufone`) |
| Head this card started from | `3fe31d8` (the review commit; pushed head `c85ae33`) |
| Reporters | `.venv` = **CPython 3.11.15** — the interpreter the CI red was reproduced on — and `/workspace/t_ba767a2b/venv312` = **CPython 3.12.13** |
| Every run below | `env -u TYPED_GGUF_HOME -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 …` (the CI shape; the offline suite adds `TYPED_GGUF_BENCH_RUNTIME_DIR=<stub bundle>`) |

## Files touched

| File | What |
|---|---|
| `tests/test_keep_client.py` | M1: `proc.wait(timeout=5.0)` after the teardown's blanket `kill()` (the card's one-line fix, line 78, mirrors `test_serve.py:714`); the RED pin for the class fix (bottom of the file) |
| `src/typed_gguf/keep/client.py` | M1: `Client._reap` (kill + wait + pop, **no** ledger write) and its call on `_wait_ready`'s "another caller's record won" path |
| `src/typed_gguf/runtime/update.py` | M2: `install._preflight_reason` asked after `plan_install` / before `_stage`, raised as the typed refusal; M2b: `installed_variant()` + the target-variant rule |
| `src/typed_gguf/api/http.py` | M3: `MAX_BODY_BYTES`/`HANDLER_TIMEOUT`/`BODY_TOO_LARGE`, `_too_large`, `Handler.timeout`, `_serve`/`_refuse`/`_answer`, `Server.handle_error` |
| `tests/test_runtime_update.py` | M2/M2b: `fake_lock(system_libs=…)`, `GPU_HOST`, `release_pair`, `gpu_box_home` and 8 new tests |
| `tests/test_serve.py` | M3: 5 new tests (cap on both wire shapes, a body at the cap, the idle drop, the RST) plus the three assertions the mutation sweep's survivors bought |
| `SPEC.md` | §2.8 step 2 (the target variant), §2.8 step 3 (the pre-flight), §2.9 (the HTTP bounds) |
| `README.md` | the `serve` command-table row (cap + idle drop) and the two paragraphs about the default target and the pre-flight |
| `pyproject.toml` | the Tier-M sweep retarget + its paragraph (repo convention) |
| `docs/evidence/t_ba767a2b/` | this receipt, the mechanism probe and the red/green logs |

## M1 — the py3.11 red: a `Popen` the fixture dropped with `returncode is None`

Root cause, exactly as the card proved it: the `make_client` teardown killed every child in
`client.children` **without a `wait()`**, so a child that lost a same-digest race (or was swapped
out) stayed a `Popen` with `returncode is None` until something dropped it — and `Popen.__del__`
warns (`ResourceWarning: subprocess N is still running`) whenever the GC reaches it first. This repo
runs `filterwarnings = ["error"]` with pytest's unraisable hook, which is what makes that a CI
`1 error` attributed to whichever test was setting up when the GC fired.

**The mechanism, reproducibly** (`probe_keep_fixture.py`, the reviewer's probe shape re-derived
here — the fixture's own order, plus an explicit `gc.collect()` in the window the fixture leaves
open):

```
$ .venv/bin/python -m pytest -q docs/evidence/t_ba767a2b/probe_keep_fixture.py     # _WAIT = False
ERROR … probe_keep_fixture.py::test_the_child_ran - pytest.PytestUnraisableExceptionWarning:
    Exception ignored in: <function Popen.__del__ at 0x7fbaf6f26980>
2 passed, 1 error in 0.11s                                    -> m1-probe-red.txt   (exit 1)

$ sed 's/^_WAIT = False/_WAIT = True/' … > /tmp/probe_fixed.py && .venv/bin/python -m pytest -q /tmp/probe_fixed.py
2 passed in 0.02s                                            -> m1-probe-green.txt (exit 0)
```

`2 passed, 1 error` is the CI's own `108 passed, 1 error` shape. The natural race was **not**
reproduced on this box (see *Limits*), so this probe is the reproduction and the acceptance runs
below are the after-state.

**The class fix's own pin** (RED before the change, GREEN after — the abandoned child is reaped
instead of handed to the GC):

```
$ .venv/bin/python -m pytest -q tests/test_keep_client.py::test_the_child_a_spawn_loses_the_ledger_with_is_reaped_not_left
E   assert {118900: <Popen: returncode: None args: ['.venv/bin/python', '-c', ...]>} == {}
E   AssertionError: the abandoned child must not be left to the GC          -> m1-red-pins.txt (exit 1)
    (after the fix: 1 passed)
```

**M1 acceptance** — the card's exact committed sub-gate on py3.11, `gate-m1.sh`:

```
$ sh gate-m1.sh m1 .venv/bin/python        # 25 sequential runs, then 4 rounds of 4-way concurrent
runs:   41
green:  41        (every one: 110 passed, exit 0)
red:    0
```

25 sequential runs take 8.97–9.45 s each; the 16 concurrent runs 9.32–10.46 s each (the pre-fix
timings on this same box, for the load comparison: `109 passed in 9.38–10.28 s` — the extra test is
the pin above, and the added `wait()` costs milliseconds). Exits in `m1-accept-311.txt`.

The count is **110**, not the card's 109: this card adds the pin above to that file. The card's
`109 passed` is the pre-fix count, and it is what the pre-fix stress recorded here as well
(`/workspace/t_ba767a2b/red-before/exits.txt`).

## M2 — `runtime update` refuses before the download it cannot use

`update()` now asks `install._preflight_reason(plan, target_lock)` after `plan_install` and before
`_stage`, and raises `RuntimeSymbolsError("E_RUNTIME_SYMBOLS: <the pre-flight reason>; nothing was
changed")` — the same code the staged probe refusal uses, in `init`'s own vocabulary, and a
**refusal**, never a move to another tier (SPEC 2.8's no-fallback-ladder rule is untouched).
`--check` returns above it, so the read-only plan report (SPEC A-E5-6) keeps its contract.

RED → GREEN (one test, `test_a_bundle_this_host_cannot_load_refuses_before_any_download`):

```
before:  E   Failed: DID NOT RAISE <class 'typed_gguf.errors.TypedGgufError'>
         (the update ran: a real file:// release, so an unrefused run downloads and switches)
         -> m1-red-pins.txt (exit 1)
after:   1 passed
```

The green run asserts what the reviewer's live box paid 169.49 MB for: no archive under
`downloads/`, nothing staged, `runtime.json` byte-identical, and `cli._fail(exc) == 3`.

*Wording note.* The card prescribes `message = the pre-flight reason + "nothing was changed"`, so
the reason is reused verbatim — it is `init`'s sentence and ends `… and moved to the next tier`,
which has no referent in `update` (there is no ladder to move to). The truthful part for this path
is the appended `nothing was changed`; the tier clause is left alone rather than editing `init`'s
user-facing text on a fix card.

## M2b — the default target is what `init` installed (the coordinator's option (b))

`installed_variant(current)` reads `runtime.json`'s `variant` **only when the record names
`find_runtime`'s own directory**; `--backend` still overrides, and a record about some other bundle
(or no record) leaves today's detection path exactly as it was. `--check` is unchanged.

RED → GREEN (three tests, one red each before the change):

```
before:  E   assert 'linux-x64-cuda-12.8' == 'linux-x64-cpu'
         E   AssertionError: detection must not re-decide what is installed
         -> m1-m2b-red.txt (exit 1)
after:   tests/test_runtime_update.py  … 7 new tests, all green
```

The three: the default target on a GPU-detecting box whose record says `cpu`; a real update that
stays on the installed variant (and switches the record it rewrites); and the edge the rule turns
into a clear error — an installed variant the lock no longer pins is `E_RUNTIME_MISSING`, not a
silent switch to what detection would pick. Two more tests guard what must *not* change: an explicit
`--backend cuda` still overrides the record, and a record that does not name the active runtime
leaves detection in charge.

## M3 — the `serve` HTTP bounds (the card's "fold in if cheap" minors)

`MAX_BODY_BYTES = 1 MiB`: a body over the cap is answered `413` **from `Content-Length` alone**
(`E_BODY_TOO_LARGE` on `/v1/decide`, the `too_large` TypeSafe detail on `/v1/systemone`), the body
is never read, the connection is closed, and the refusal gets its own line on the app's log sink.
`Handler.timeout = 30.0`: `ThreadingHTTPServer` is one thread per connection and `http.server` has
no timeout of its own, so an idle client used to hold a thread forever. `Server.handle_error`
swallows `ConnectionError`/`TimeoutError` (a client that vanished mid-request) and still lets
anything else reach the stdlib, so a bug of ours stays visible.

RED → GREEN (five tests, all red before the change):

```
before:  FAILED test_a_body_over_the_cap_is_refused_before_a_byte_of_it_is_read
         FAILED test_the_native_route_speaks_its_own_shape_for_the_cap
         FAILED test_a_body_at_the_cap_is_still_read_and_routed
         FAILED test_the_handler_drops_a_client_that_sends_nothing      (no Handler.timeout)
         FAILED test_a_client_that_left_mid_request_is_not_a_stdlib_traceback
         5 failed, 66 deselected in 0.27s                               -> m1-m3-red.txt (exit 1)
after:   tests/test_serve.py  … all green
```

The cap test is honest about "the body is never read": the client announces `MAX_BODY_BYTES + 1`
and **sends no body at all**, then reads the `413` — a server that tried to read the body first
would hang there instead of answering. A body of exactly the cap is still read and routed.

## Gates at this head

| Gate | Command | Result |
|---|---|---|
| CI-shape suite, py3.11 | `env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<stub> .venv/bin/python -m pytest -q -rs --timeout=120` | **1785 passed, 59 skipped, 0 failed** (53.68 s, `final-311-full.txt`) — the reviewer's 1771 + this card's 14 |
| CI-shape suite, py3.12 | same, `venv312/bin/python` | **1785 passed, 59 skipped, 0 failed** (55.45 s, `final-312-full.txt`) |
| keep sub-gate, py3.11 | the card's exact four files | **41/41** earlier in the run (25 sequential + 4×4 concurrent, 110 passed each) and `110 passed in 9.14 s` at this final head (`final-subgate-311.log`) |
| keep sub-gate, py3.12 | same four files, `venv312` | **10/10**, 110 passed each |
| ruff | `ruff check src tests tools docs .github` | clean (`final-ruff.txt`) |
| build | `uv build` | `dist/typed_gguf-0.2.3-py3-none-any.whl` (`final-uvbuild.txt`) |

## Mutation sweep (Tier M, soft threshold)

`pyproject.toml`'s `[tool.mutmut]` pair is retargeted for this card — one run per module the card's
logic adds, each with that module's own gate file (the repo's convention for a Tier-M sweep), and
the pyproject comment records which module each run covers.

**Run (A) — `src/typed_gguf/runtime/update.py` + `tests/test_runtime_update.py`** (mutmut 3.8,
`tools/mutmut_driver.py run --max-children 2`):

```
958 mutants, 819 killed = 85.5 %, 139 survived, 0 timeouts, 0 with no tests at all
-> after the falsy-variant pin below: 820 killed = 85.6 %, 138 survived   (mutmut-a.log)
```

`🫥 0` (mutmut's "no tests" bucket, i.e. lines no test reaches) is the useful half: every mutated
line of the module this card touches is covered. The survivor family split, from `mutmut results`
(`mutmut-a-survivors.txt`): `update` 30, `_require_probe` 25, `fetch_releases` 21, `rollback` 12,
`current_runtime` 12, `_stage` 12, `_stop_host` 7, `_same_dir` 5, `release_asset` 4, `pick_target`
2, `_updated_warnings` 2, `_probe_payload` 2, and one each in `variant_of_dir` / `parse_releases` /
`build_of_tag` / `_asset_payload` / **`installed_variant`**.

That last one was worth a pin: `x_installed_variant__mutmut_22` turned
`return str(variant) if variant else None` into an always-truthy answer (`str(None)` = `"None"`),
which would ask the lock for a variant named `None` and fail the update (`E_RUNTIME_MISSING`) where
it must fall back to detection. `test_a_record_with_no_variant_leaves_detection_alone` now pins
that clause, and a scoped re-run of the family
(`run 'typed_gguf.runtime.update.x_installed_variant*'`) kills all 23 of its mutants.

Two survivors are **equivalent mutants**, and the receipt says so rather than claiming them:
`x_update__mutmut_19` replaces the detection call's `system=system` with `system=None`, which on a
Linux box re-detects the same host (`platform.system()` answers the same thing); and the
`update`-family leftovers are the `run()` lifecycle / report-string mutants the *other* update gate
files assert. No claim is made that the sweep covers the two changed tests' whole surface beyond
the file selection quoted above.

**Run (B) — `src/typed_gguf/api/http.py` + `tests/test_serve.py`**, same driver:

```
827 mutants, 621 killed = 75.1 %, 172 survived, 30 timeouts, 4 with no tests at all
-> after the three M3 pins below: 626 killed = 75.7 %, 167 survived      (mutmut-b.log)
```

The 30 timeouts are the `Handler._serve` loop mutants that hang the socketpair gate — the same class
the serve card's own sweep recorded (`t_f5d8b6c7`), not new. The card's own surface gave up five
more kills to three assertions added to the existing tests (each one re-run and confirmed in
`mutmut-b-round3.log`):

| Survivor | What it was | The pin that killed it |
|---|---|---|
| `x__too_large__mutmut_3` | `_native_error(…, None, …)` — the native message dropped | `test_the_native_route_speaks_its_own_shape_for_the_cap` now asserts the refusal *names the announced size* |
| `x__too_large__mutmut_13`, `_16` | the `headers` argument dropped/`None` — the `413` losing its request id | `test_a_body_over_the_cap_…` now asserts `x-typesafe-request-id` is on the refusal like on any answer |
| `xǁHandlerǁ_serve__mutmut_12` | `length > MAX_BODY_BYTES` → `>=` — the cap refusing a body *at* the cap | `test_a_body_at_the_cap_is_still_read_and_routed` now builds a body of **exactly** `MAX_BODY_BYTES` |
| `xǁServerǁhandle_error__mutmut_4` | `client_address` → `None` | the RST test now asserts the printed report names the client |

Two survivors on this surface are equivalent, with the reason:
`xǁHandlerǁ_serve__mutmut_7/8` lowercase the `Content-Length` lookup, and
`email.message.Message.get` is case-insensitive by contract; `xǁServerǁhandle_error__mutmut_3` passes
`request=None` to the stdlib printer, which never reads that argument (only `client_address` is
printed — hence `_mutmut_4` above being killable and `_3` not). `xǁHandlerǁ_serve__mutmut_19`
(`if length > 0` → `or True`) is equivalent too: `BufferedReader.read(0)` returns `b""` either way.
The remaining `_refuse`/`_answer`/`App` survivors are the log-format and route-plumbing mutants the
other serve gates assert.


## Limits of this evidence

* **The natural M1 race did not reproduce here.** 17 pre-fix four-way-concurrent runs of the
  committed sub-gate were green on this 24-CPU box (the review box saw 2/32) — the reviewer's
  observation stands as the before-state, and `probe_keep_fixture.py` pins the mechanism instead of
  relying on luck. The acceptance runs are therefore an after-state under the shape the card asked
  for, not a red-to-green of the natural race.
* **M3's `handle_error` is asserted on the method, not on a real `RST`**: the offline gate forbids
  `AF_INET`, so the test drives `Server.handle_error` on an unbound instance (`Server.__new__`, no
  socket — the method needs no server state) and asserts the two branches: a transport failure
  silent, our own bug still printed. The wire behaviour itself is the stdlib's.
* The mutmut numbers are a single scoped run per module on a shared 2-CPU-quota box: they are
  quoted with their scope, never as a package-wide score.
