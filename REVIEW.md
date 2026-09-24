# typed-gguf — REVIEW: the serve wave (serve v1 + `runtime update` + UX default), card `t_c7d5e33d`

**Verdict: NOT GREEN — 2 🟠 MAJOR, 2 🟡 MINOR, 2 ⚪ NIT. One batched fix card: `t_dab7a690` (blocked).**
The wire itself holds: SPEC §2.8/§2.9/§2.12 conformance was checked behavior by behavior and the
adversarial pass on the new HTTP surface found no blocker. What blocks the wave is (M1) the **CI is red
on py3.11** at the pushed head — reproduced here twice, root-caused and mechanism-proven — and (M2) the
default `runtime update` **never runs the repo's own pre-flight**, so on a box whose detection picks a
backend whose system libs are missing it pays a full 169 MB download per attempt and can never succeed
without `--backend`.

This file replaces the v0.1.0 release gate (`t_0070415c`): the repo's convention
(`git log -- REVIEW.md`) is that each review overwrites `REVIEW.md` and git keeps the predecessors.

## Heads, workspace, method

- Reviewed: base **`6d68f55`** (SPEC card `t_a51b1205`) → local head **`335ca91`** = pushed **`c85ae33`**
  (serve `t_f5d8b6c7`, runtime `t_d88b4be0`) + the UX card's seven commits (`a94d773`…`335ca91`,
  `t_a0fa2dc0`). CI ran at `c85ae33`.
- Repo read-only from my side; every probe, log and scratch copy under `/tmp` (`/tmp/lw*.log`,
  `/tmp/probe*`, `/tmp/wttests` = a scratch copy of `tests/` for the control experiment). No push, no
  `src/` edit; `REVIEW.md` + the receipt dir are the only repo writes.
- Interpreter: the repo's own `.venv` = **CPython 3.11.15** (the interpreter the CI red is about);
  `TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/tg-offline-bundle` (four empty `.so` stubs) wherever the CI shape
  needs it.

## 1. Re-run gates (checklist item 1) — exact numbers

| gate | command | result |
|---|---|---|
| CI-shape suite | `env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<stub> uv run --extra dev pytest -q` | **1771 passed, 59 skipped, exit 0**, 56.57 s (log `/tmp/gate-ci-shape.log`) — matches the UX card's 1771/59/0 |
| targeted files | `tests/test_serve.py test_runtime_update.py test_default_model.py test_registry_store.py test_cli_e1a.py test_scaffold.py test_public_docs.py` | **226 passed, 2 skipped** |
| ruff | `env -u PYTHONPATH uv run --extra dev ruff check src tests tools docs .github` | **clean** |
| build | `env -u PYTHONPATH uv build` | tar.gz + wheel, OK |
| wire fixture provenance | regenerate `tests/fixtures/typesafe_sdk_0_7_1_fields.json` with `tools/typesafe_fields_capture.py` against a **real** `typesafe-sdk==0.7.1` (`/tmp/ts-venv`) | **byte-identical** — the fixture is not hand-typed |
| offline SDK rehearsal | `env -u PYTHONPATH uv run python tools/rehearse_serve_sdk_offline.py --sdk-python /tmp/ts-venv/bin/python` | **exit 0** — the real SDK client parsed cold+warm typed answers |
| the committed sub-gate, py3.11 | `TYPED_GGUF_TEST_BLOCK_NET=1 pytest -q tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py` | 46 sequential runs **all `109 passed`, exit 0**; ~68 four-way-concurrent runs: **4 red with the CI's exact shape** (M1) |

The container could run the offline SDK rehearsal because `typesafe-sdk==0.7.1` is on public PyPI — the
host gate's 4B half stays the coordinator's, and this review does not re-claim it.

## 2. SPEC conformance (checklist item 2) — behavior by behavior

**§2.9 `serve`** (stdlib server, `127.0.0.1:8088` default, `--host`/`--port`/`--format`/`--keep-alive`;
four routes; both wire formats always mounted; `/health`+`/v1/models` never take the decision lock):
every claim reads back from `src/typed_gguf/api/http.py` and is pinned in `tests/test_serve.py`
(`/v1/decide` never projected; body `format` beats `--format`; exit codes → 400/503/500 on
`/v1/decide`; 404 `{"detail": "Not Found"}`; `x-typesafe-request-id` echoed and repeated in the log
line `POST /v1/systemone 200 served_by=host 0.4ms req=<id>`). Probe-confirmed by hand: a 5 MiB body is
accepted, a path-shaped model ref answers 422 `E_MODEL_NOT_FOUND`, an unknown top-level key answers 422
`extra_forbidden`, an `Authorization` sentinel is accepted in all three forms (bearer/garbage/absent)
and appears neither in the response nor in the log.

**§2.8 `runtime update`/`rollback`** (resolve → target → stage → switch → report; atomic record write;
no-delete; refusals). Probed by hand against fake homes (`/tmp/probe_runtime_cli.sh`,
`/tmp/probe_runtime_update_live.sh`):

- empty home → `E_RUNTIME_MISSING`, exit 3, home untouched;
- `$TYPED_GGUF_RUNTIME_DIR` set → `E_UPDATE_UNAVAILABLE`, exit 2, for both `update` and `--check`;
- `runtime rollback` with no `previous` → `E_UPDATE_UNAVAILABLE`, exit 2, nothing changed;
- `--check` live → `current b11026 (linux-x64-cpu) / target b11160 (169.49 MB)` + the `--json` plan,
  "nothing changed", record byte-identical;
- `--check` with a dead proxy → `E_DOWNLOAD_FAILED` naming the URL, exit 3, no partial state;
- full default `update` → download → probe → `E_RUNTIME_SYMBOLS` refusal, exit 3, **`runtime.json`
  byte-identical**, staging removed, archive kept in `downloads/` (SPEC's stage-then-probe order,
  observed literally; M2 is about the *pre*-stage step this order does not require).

`write_runtime_record` is `tmp` + `fsync` + `os.replace` (finder.py:173) ✓ atomic; the switch-point
claim holds in every refusal I produced; no bundle is ever deleted ✓.

**§2.12 same engine by construction** — verified in the wiring, not just in the test: `cli.serve_decide`
resolves `home`/`keep_alive` once and calls `cli.decide_payload_warm`, the same function `run`/`ask`
call (`cli.py:1312`, `1411`, `2003`), and `/v1/decide` only adds the request's own `format`. The served
body is `schema.render_response(native, format=...)` — one projection, no second engine, no second
model handle; `decision_lock` serializes decisions.

## 3. Hostile input (checklist item 3)

| probe | result |
|---|---|
| unknown route (`GET /nope`, `POST /v1/nothing`, `GET /v1/decide`, `GET /`) | 404 `{"detail": "Not Found"}` |
| unknown top-level key in the SDK body | 422 `E_UNKNOWN_KEY` (typesafe shape), 400 on `/v1/decide` |
| wrong/missing JSON body | typed 4xx, `{error:{code,message}}` / `{detail:[…]}` per route |
| model refs cannot point at paths | 422 `E_MODEL_NOT_FOUND` |
| `Authorization` sentinel | accepted, never logged, never echoed |
| `--host 0.0.0.0` | warning printed, pinned by test |
| stdlib only | `[project] dependencies = []`; the diff adds no runtime dep |
| **huge bodies / slow clients** | 🟡 **MINOR-1** — see below |
| client RST mid-request | 🟡 MINOR-1 (stdlib traceback on stderr) |

## Findings

### 🟠 MAJOR-1 — py3.11 CI red: the keep fixture kills children it never reaps (this is the operator's CI finding #2)

The card's own comment thread carries the CI failure: push `c85ae33` red on py3.11 only, `108 passed,
1 error`, `ResourceWarning: subprocess N is still running` attributed to
`tests/test_keep_client.py::test_a_host_that_dies_mid_decision_falls_back_inline_once`.

**Reproduced here four times** (4 red of ~68 four-way-concurrent runs of the exact four-file subset;
0 red of 46 sequential runs): `par-2-c.log`, `lw6-3.log`, `lw9-3-c.log`, `lw9-4-a.log` (copied into
`docs/evidence/t_c7d5e33d/`, now that `/tmp` is gone) each read exactly
`ERROR at setup of test_a_host_that_dies_mid_decision_falls_back_inline_once` …
`ResourceWarning: subprocess <pid> is still running` … `108 passed, 1 error` — the CI's shape, four
different pids (7425, 12610, 17569, 17906). The leaked object's args are
`.venv/bin/python /wor…` — a **client-spawned `tests/fake_keep_host.py`**, i.e. a keep client's child,
not a test-side stand-in (those use `-c`).

**Root cause (mechanism proven, not inferred).** `Client.stop()` reaps only the pid the *record*
names; children the client does not own (a race loser that lost the ledger, a swapped-out host) stay in
`Client.children`. The `make_client` teardown (`tests/test_keep_client.py:74-77`) then does
`proc.kill()` with **no `wait()`**, so any such child stays `returncode is None` and is dropped;
CPython's `Popen.__del__` emits `ResourceWarning: subprocess N is still running` when the GC collects
it during a *later* test's phase, and this repo's `filterwarnings = ["error"]` + pytest's unraisable
hook turn that into the `1 error`. Proof: `/tmp/probe_fixture_wait2.py`, a synthetic replica of the
fixture shape, reproduces the identical `PytestUnraisableExceptionWarning: Exception ignored in:
Popen.__del__` with `kill()` alone and is green with `wait()` added (2 failed/1 passed vs 3 passed).
Instrumentation (`/tmp/leakwatch6.py`) shows the children that reach the blanket kill belong to the two
race tests and to `test_a_swap_over_debris…` / `test_the_spawning_call_reports_the_load_it_waited_for`.
**The witness pattern is in this same wave:** `tests/test_serve.py:712-716` does `stop()` → `kill()` →
`wait(timeout=2.0)` ("reaps the child it spawned"). Why py3.11-only and flaky: it depends on whether the
race leaves an unreaped child at all and on when the GC collects it relative to pytest's unraisable
collection points; 3.11/3.12 differ in that timing, and the CI is where the load exists.

**Fix (in `t_dab7a690`):** `tests/test_keep_client.py:76-77` → `proc.kill()` + `proc.wait(timeout=5.0)`
(one line, mirrors `test_serve.py:712-716`); optional class fix in `keep/client.py` — reap/pop the
children the client abandons. Nothing in the wave's diff caused it; it is pre-existing, surfaced by the
push's CI. **The wave's push cannot be green without it.**

### 🟠 MAJOR-2 — `runtime update` skips the pre-flight its own repo already has (host finding [1]/[3])

Live, at this head, on a box whose detection says cuda but which lacks `libcudart.so.12` (the
coordinator's box: detection says cuda, installed runtime vulkan):
`typed-gguf runtime update` → target `b11160 linux-x64-cuda-12.8` → **169.49 MB downloaded (11 s)** →
`E_RUNTIME_SYMBOLS: … libggml-cuda.so: libcudart.so.12: cannot open shared object file …; reinstall the
system libraries it needs or update with --backend for the one this host can drive`, exit 3,
`runtime.json` byte-identical. Safe and spec-ordered — but the repo knows better *before* the download:
`install._preflight_reason` (install.py:331) exists for exactly this and names the cost ("168.8 MB of
download buys nothing"), yet only `init`'s ladder calls it (install.py:518); `update.py` has no
reference to it. So the **default** `runtime update` on this box is a dead end that pays 169 MB per
attempt, and the target variant is raw detection (update.py:463) even when the *installed* runtime is
`linux-x64-cpu` (my probe).

**Fix (in `t_dab7a690`):** call `install._preflight_reason(plan, lock)` after `plan_install` and before
`_stage`; refuse with the same `E_RUNTIME_SYMBOLS` vocabulary, no ladder, no download. **Decision left
to the coordinator (M2b):** SPEC 2.8 step 2's "the host variant comes from `init`'s own detection" is
ambiguous — keep raw detection (+ one SPEC line + a README sentence naming `--backend`), or target the
installed record's variant when the record names the installed dir. Host finding [2]: `doctor`'s
expectation after a recorded init fallback repeats the *recorded* reason via
`cli._recorded_backend_reason` (cli.py:329) and advises installing the libraries or keeping the working
backend — pre-existing, untouched by this wave, honest wording; no action needed here.

### 🟡 MINOR-1 — `serve` has no body cap, no socket timeout, and leaks tracebacks on client errors

`api/http.py` uses `http.server`'s defaults: `rfile.read(length)` with an attacker-chosen
`Content-Length` (5 GiB → the handler blocks >4 s and allocates as bytes arrive; a 5 MiB body is
accepted), no `Handler.timeout` (an idle connection holds a thread forever — unbounded with
`--host 0.0.0.0`, which the CLI supports and only warns about), and a mid-request RST prints a stdlib
traceback to stderr. Loopback bind keeps this low-risk; ~10 lines (cap, timeout, `handle_error`)
closes it. Suggested for the same fix card, optional.

### 🟡 MINOR-2 — the keep gates can flake under concurrent load

One of ~68 four-way-concurrent subset runs went red on a *different* test:
`tests/test_keep_host.py::test_the_host_exits_by_itself_once_the_idle_window_passes` → "the host never
became ready" (`/tmp/par-5-c.log`). Not the CI failure and not the wave's code (keep untouched), but
the same class of CI risk; worth a bounded-timeout bump if the coordinator wants the CI-shape suite to
be load-proof.

### ⚪ NITs

- `runtime update`'s refusal leaves the 169 MB archive in `<home>/downloads/` — correct per SPEC
  (resume semantics), but the message could name the kept path;
- the `/v1/models` empty-registry shape and `release_date = mtime` are coded and pinned, but the
  "auto line" description is the one wire string not covered by the SDK fixture — fine as is.

## 4. Same-engine guarantee (checklist item 4)

By construction, verified in the wiring (`cli.serve_decide` → `cli.decide_payload_warm`;
`test_the_decision_callable_is_the_warm_cli_path` pins the call shape, including that `serve` hands over
the start-up-resolved `home`/`keep_alive` while `run` hands over the unresolved defaults with the same
effective values). `decode` numbers are untouched: the served body is the native body projected by
`schema.render_response`, and the projection is byte-equal in the gate. ✓

## 5. Docs honesty (checklist item 5)

`README.md`'s `serve` row names exactly what ships (four routes, `127.0.0.1:8088`,
`TYPESAFE_BASE_URL`, "from the same warm host"), the `mcp` row still says "planned and not implemented
in this release: the command exits 3 today", the root `--help` says `serve  - serve a decision API for
TypeSafe clients` and `mcp    - planned; not in this version`, and `serve --help` prints its four flags
plus "run `typed-gguf --help` for the command list". `api/mcp.py` untouched (no `mcp` file in the
diff). Nothing claims more than ships. ✓

## 6. The flipped pins (checklist item 6)

All four are coherent, and each keeps the claim and the behaviour apart:
`tests/test_scaffold.py` (`serve` exits 3 → `mcp` does, `serve --help` answers);
`tests/test_cli_e1a.py` (`test_frozen_commands_still_exit_3[mcp]` + `test_serve_is_no_longer_a_stub`
asserting `cli.NOT_IMPLEMENTED == ("mcp",)`);
`tests/test_public_docs.py` (the "not shipped" row now names `mcp` and asserts `cli.main(["mcp"]) == 3`
vs `serve --help == 0`; a new `test_the_readme_documents_the_shipped_serving_surface` pins the four
route paths + `127.0.0.1:8088` against `serve.*_PATH`/`DEFAULT_*` constants). No stale "not shipped"
claim for `serve` survives. ✓

## Reproduction recipes (for the fix card)

```bash
# M1 — the CI shape under load (4/68 red here, ~0/46 otherwise)
cd "$(git rev-parse --show-toplevel)"   # the checkout root; its directory name is not part of the contract
for round in 1 2 3 4; do for slot in a b c d; do
  ( TYPED_GGUF_TEST_BLOCK_NET=1 .venv/bin/python -m pytest -q \
      tests/test_keep.py tests/test_keep_host.py tests/test_keep_client.py tests/test_keep_cli.py \
      > /tmp/m1-$round-$slot.log 2>&1 ) & done; wait; done
grep -l "108 passed, 1 error" /tmp/m1-*.log        # -> the failing run; the error text is the finding
python -m pytest -q /tmp/probe_fixture_wait2.py     # mechanism proof (kill() only -> the same error)
# M2 — the pre-flight gap (no download should happen for a variant this host cannot load)
TYPED_GGUF_HOME=<fake home with a b11026-linux-x64-cpu runtime + record> \
  .venv/bin/typed-gguf runtime update               # today: 169 MB download, then E_RUNTIME_SYMBOLS
```

Raw logs kept by this review: `docs/evidence/t_c7d5e33d/` carries the four red-run logs, the mechanism
probe (`probe_fixture_wait2.py`), the instrumentation (`leakwatch6.py`), the §2.8 probe script + the
live `runtime update` output, and the re-run recipe. Also in `/tmp` (ephemeral): `gate-ci-shape.log`
(CI shape 1771/59/0), `netblock-py311-summary*.txt` (the 46 sequential runs), `/tmp/wttests` (the
scratch copy used for the control experiment).
