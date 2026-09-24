# `serve` — the TypeSafe HTTP surface on the warm host (card `t_f5d8b6c7`)

What landed, what was measured, and what only the host can prove. Card rules quoted where they
bind: SPEC §2.9 (the wire, as the parent card `t_a51b1205` settled it), §2.12 (the server has no
engine of its own), and the card's acceptance list 1–6.

## 1. Files touched

| file | what |
| --- | --- |
| `src/typed_gguf/api/http.py` | the server (491 lines): `Response`, `App` (`/health`, `/v1/models`, `/v1/systemone`, `/v1/decide`), `Handler`, `Server`, `make_server`, `run`, the error→status/`loc` maps |
| `src/typed_gguf/cli.py` | `serve` leaves `NOT_IMPLEMENTED` (now `("mcp",)`), gains `serve_decide` / `serve_app` / `_cmd_serve`, the real description, `--keep-alive` on its help line, and the `serve` dispatch |
| `tests/test_serve.py` | **62 tests** (61 + 1 loopback skipped under the net block) across wire / mapping / errors / same-engine / cold-warm / docs / host-gate |
| `tests/fixtures/typesafe_sdk_0_7_1_fields.json` | the SDK 0.7.1 field lists, generated (not typed in) |
| `tools/typesafe_fields_capture.py` | the generator for that fixture (runs in an SDK venv) |
| `tools/host_gate_serve.sh` | the host acceptance gate (card 6) |
| `tools/host_gate_serve_client.py` | its SDK client + the `verify()` judgements, a pure function |
| `tools/rehearse_serve_sdk_offline.py` | the in-container rehearsal of that gate |
| `tools/mutmut_driver.py` | `copytree(symlinks=True)`: the committed evidence venvs make mutmut's `also_copy` abort before one mutant exists |
| `README.md` | the `serve` row documents the shipped surface; `mcp` keeps the "not implemented in this release" wording and `cli.py`'s `PLANNED_NOTE` |
| `tests/test_scaffold.py`, `tests/test_cli_e1a.py`, `tests/test_public_docs.py` | the four "serve exits 3 / planned" pins flipped, RED→GREEN shown below |
| `pyproject.toml` | the Tier-M sweep's pair + `max_children` note (see §5) |

`docs/RELEASE_NOTES_v0.2.3.md` is **untouched** (card constraint 5: a later release card owns
notes). `src/typed_gguf/api/mcp.py` and every `mcp` mention are untouched (constraint).

## 2. RED → GREEN, receipted

RED (before the implementation, `tests/test_serve.py` committed as `bc87cde`):

```
$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 uv run pytest -q tests/test_serve.py
E   AttributeError: module 'typed_gguf.api.http' has no attribute 'SYSTEM_ONE_PATH'
1 error in 0.31s
```

The docs half went RED first too — the four pins that said `serve` is not shipped:

```
FAILED tests/test_scaffold.py::test_cli_version_and_unknown_command          assert 4 == 3
FAILED tests/test_cli_e1a.py::test_frozen_commands_still_exit_3[serve]        assert 4 == 3
FAILED tests/test_public_docs.py::test_the_readme_marks_the_serving_surface_as_not_shipped
FAILED tests/test_public_docs.py::test_the_root_help_marks_the_serving_surface_the_way_the_readme_does
4 failed, 175 passed
```

(The `4` is `E_INTERNAL`: with `TYPED_GGUF_TEST_BLOCK_NET=1` the stub refuses `AF_INET`, so a
`serve` that no longer exits 3 fails its own bind and reports an internal error — the honest
in-container answer, and the reason the flipped pins now gate `serve --help` instead of a bare
`serve`.)

GREEN (this card's commit):

```
$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 uv run pytest -q tests/test_serve.py
42 passed, 1 skipped          # round 1
62 passed, 1 skipped          # with the round-2 pins of §5
```

## 3. The gates the card asked for

CI shape — four-empty-libs bundle stub, network blocked:

```
$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 \
    TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/tg-offline-bundle uv run --extra dev pytest -q -rs --timeout=120
1707 passed, 58 skipped in 52.22s
```

Baseline at the parent's head (`6d68f55`, quoted in `t_a51b1205`'s handoff): 1650 passed /
57 skipped / 0 failed. The delta is exactly this card's gate file (+57 passing, +1 skip: the
loopback test, skipped because `TYPED_GGUF_TEST_BLOCK_NET=1` forbids `AF_INET`).

```
$ env -u PYTHONPATH uv run ruff check src tools tests docs .github
All checks passed!
$ env -u PYTHONPATH uv build
Successfully built dist/typed_gguf-0.2.3.tar.gz / dist/typed_gguf-0.2.3-py3-none-any.whl
$ env -u PYTHONPATH uv run typed-gguf serve --help
usage: typed-gguf serve --host IP --port N --format native|typesafe --keep-alive <dur|0>

serve a decision API for TypeSafe clients
run `typed-gguf --help` for the command list
$ env -u PYTHONPATH uv run typed-gguf version
typed-gguf 0.2.3   runtime not installed (`typed-gguf init`)   home /root/.local/share/typed-gguf
```

`typed-gguf version` is byte-identical to the baseline (card gate). `serve --help` never binds.

The four pins, as they now read: `serve` is out of `cli.NOT_IMPLEMENTED` (`== ("mcp",)`), its
`--help` exits 0, the root help's `serve` line ends with its own description
(`serve a decision API for TypeSafe clients`), the `mcp` line keeps `planned; not in this version`
and `mcp` still exits 3.

## 4. The SDL gate (acceptance 1, 2, 3, 4) — and where each part is gated

* **Wire conformance (1).** `tests/fixtures/typesafe_sdk_0_7_1_fields.json` is generated from the
  installed `typesafe-sdk==0.7.1` by `tools/typesafe_fields_capture.py` (`_fields` =
  `model.model_fields`; `Answer` is an `Annotated[Union[…], Field(discriminator="type")]`, so it
  records the discriminator and the member names in declaration order). The serve tests read that
  fixture and fail if a served key set stops being the SDK's — no field list is typed in by hand.
  Request bytes: the mixed noul/choice/score body, alias resolution (paths and `repo:quant` are
  refused with the typed code), `Authorization` present/absent accepted and never logged (a
  sentinel value is asserted absent from the log line), malformed bodies → `{detail:[…]}` with
  `type`/`loc`/`msg`/`input` inside the SDK's `ValidationError` field surface.
* **Answer mapping (2).** choice `{type, choice, confidence, probabilities}`; score
  `{type, score, confidence, legend, probabilities}` (legend = level numbers as string keys);
  noul `{type, noul, probabilities}` — **no `confidence`**; `model` echoes the *resolved* alias,
  never the requested `jev-latest`; `usage` = the native counters verbatim (`input_tokens` is the
  request's prompt tokens, `output_tokens` the decode steps; `prefill_tokens` is deliberately not
  reported as `input_tokens`). The served body is asserted byte-equal to
  `schema.render_response(native, format="typesafe")`.
* **Same engine (3).** `cli.serve_app`'s callable *is* `cli.decide_payload_warm` — the CLI's own
  warm path — so the fit plan, the calibration source and the ledger are the `run`/`ask` ones by
  construction, and the gate asserts it: the served payload and `run`'s payload for the same
  state/questions are equal (bar the `format` the CLI was told), the same data home, the same
  keep-alive window, and the answers byte-equal.
* **Cold/warm (4).** Over `tests/fake_keep_host.py` (the production `keep.client` + hosted server,
  fake engine): first request cold-loads, the second reuses the same ledger pid, both report
  `served_by == "host"`, the answers match, `keep stop` still works afterwards, and the record and
  the process are gone (`wait_pid_gone`). Serialization through the host is pinned in-process
  (two concurrent decisions arrive in order) and, on the real host, is the host's own one-model
  rule (§2.12).

## 5. Tier-M mutation sweep

Driver `tools/mutmut_driver.py`, `--max-children 2`:

```
$ env -u PYTHONPATH uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2
```

Round 1 — 747 mutants on `src/typed_gguf/api/http.py` against `tests/test_serve.py`:
**494 killed (66.1 %)**, 214 survived, 26 timeouts, 13 in mutmut's own "no tests" bucket, 0
suspicious. The survivors are concentrated in three families the round-1 gate only *sampled*
(a prefix assertion instead of the row): `_human_size` (33), `_release_date` (15),
`_description` (5) — 53 mutants whose whole job is the one line a caller reads — plus the
`Handler._serve` loop's 10 survivors and 9 "no tests" on `run()` (the blocking half no test
called, since the CLI gates stop at the seam).

Round 2 — four pins (exact `_human_size` values incl. the TiB tail; `_release_date` for a real
mtime / a missing file / an undated entry; `_location_for` for both maps and both miss paths;
`run()`'s lifecycle against a stand-in server), then a scoped re-run of the named survivor groups
(mutmut 3.8 re-runs what it is given: `run '<key>*' …`): **565 killed / 747 = 75.6 %**, 152
survived, 26 timeouts, 4 in the "no tests" bucket, 0 suspicious. The four groups moved as
intended: `_description` 15 → 0 survivors, `_location_for` 15 → 0, `_human_size` 33 → 3,
`_release_date` 15 → 3, `run()` 9 "no tests" → 0. The remaining survivors are the App's
error/route plumbing (37 in `App._decide`, 18 in `App._system_one`, 17 in `App._resolve`, 15 in
`App.handle`) — behaviour the *other* gate files assert and this selection deliberately does not
re-drive — plus 26 `Handler._serve` timeouts: mutants of the socket read/write loop that make the
socketpair gate hang instead of fail, which mutmut books as timeouts, never as kills.

A sweep caveat worth the next card's minute: mutmut 3.8 caches verdicts in
`mutants/mutmut-stats.json` and a second `run` of the *same* file reprints them without testing
(0.00 mutations/second is the tell). Round 2's total is the round-1 census updated by the scoped
re-run above, not a second full pass.

Two container notes, both in the driver/config so a later sweep does not rediscover them:
`also_copy` carries committed evidence venvs whose `bin/python*` are dangling symlinks (mutmut's
`copytree` aborts before one mutant; the driver now copies links as links), and `docs/` is dropped
from `also_copy` for this sweep — 4.3 GB of e2e artifacts the selection never reads.

## 6. The host gate (acceptance 6), rehearsed as far as the container allows

`tools/host_gate_serve.sh` builds a fresh venv, **pins `typesafe-sdk==0.7.1`** and reads the
installed version back (a different SDK aborts the run with exit 1 and a named reason), runs the
offline serve gates, then starts `typed-gguf serve`, drives it with the official SDK through
`TYPESAFE_BASE_URL`/`TYPESAFE_API_KEY` on a mixed 3-question body over a real state, prints the
typed answers cold *and* warm, requires the server's own log to say `served_by=host`, stops the
server, calls `keep stop`, and fails if any record survives.

Rehearsed in-container — everything except the model and the GPU
(`tools/rehearse_serve_sdk_offline.py`, which serves the in-process app on loopback with the real
decision path and runs the gate's own client inside the pinned SDK venv):

```
$ env -u PYTHONPATH uv run python tools/rehearse_serve_sdk_offline.py --sdk-python /tmp/tssdk-venv/bin/python
typesafe-sdk 0.7.1 -> http://127.0.0.1:59661
state: 732 chars; questions: renew_style(choice), risk(score), escalate(noul)

== [cold] client.system_one returned in 0.01s
{ "answers": { "escalate": {"noul": 0.5, "type": "noul"},
               "renew_style": {"choice": "save_with_discount", "confidence": 1.66533e-16,
                               "probabilities": {...}, "type": "choice"},
               "risk": {"confidence": 0.0, "legend": {"0": "healthy", …}, "score": 1.5, …}},
  "model": "spark-x2.5-4b-q8_0", "usage": {"input_tokens": 277, "output_tokens": 14}}
   ok: 3 typed answers, keys and ranges conform
PASS: the served wire answered both calls through the official SDK, typed.
driver exit: 0
```

Three things that rehearsal settled, each now a rule in the driver instead of a guess:

1. the SDK **parses** our wire — its pydantic models accept the response and `model_dump()` gives
   back the typed answers (the SDK ignores unknown keys, so the noul `probabilities` the SPEC sends
   does not appear in its typed view; `verify` checks the SDK's view and tolerates the wire's extra
   key);
2. `score` is the **probability-weighted mean** (SPEC §2.5: `1.30 = 0×0.0 + 1×0.70 + 2×0.30`), so
   the gate accepts a value in `[0, K-1]` and checks it against the legend — an integer-level
   assertion would have failed a correct host;
3. pydantic coerces the legend's string keys to the `int`s its `dict[int, str]` declares, so the
   gate reads both spellings; and probabilities sum to `1 ± K·5e-7` because the wire rounds to 6
   significant digits.

Also rehearsed: the refusal. Run with an interpreter that has no SDK, the driver exits **3** with
`REFUSING TO RUN: no typesafe-sdk in …` *before* opening a socket (a gate that measures one wire
must not report another), and with `--expect-sdk` set elsewhere it refuses on the version. That
in-container behaviour is gated by `tests/test_serve.py`.

**What only the host can prove** (recorded in the script's own `verdict.txt`, not in CI): that a
real `typesafe-sdk==0.7.1` client, told nothing but `TYPESAFE_BASE_URL`, gets typed answers *from
the 4B* through the resident keep host — the drop-in claim. The container has no model, no GPU and
no `AF_INET` for the tests; the numbers therefore come from `tests/fake_engine.py`'s deterministic
session everywhere above. The coordinator runs
`tools/host_gate_serve.sh` on the host after this card lands.
