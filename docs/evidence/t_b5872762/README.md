# t_b5872762 — py3.13 red: the `413` reason phrase belongs to the interpreter, not to the wire

The coordinator's 3.13 certification of `106e920` was red on exactly two tests, both new cap tests in
`tests/test_serve.py`, both for one reason: they spelled the reason phrase out
(`HTTP/1.1 413 Request Entity Too Large`) and CPython 3.13 refreshed `http.HTTPStatus` per RFC 9110
(`Content Too Large`). The server emits whatever the stdlib hands it (`Handler.send_response` ->
`BaseHTTPRequestHandler.responses`), so the fix is on the *test* side: pin the **code**, never a
version's vocabulary. `ci.yml`'s offline-gate matrix gains `3.13` so the next such refresh is CI's
finding instead of the coordinator's.

| | |
|---|---|
| Repo | `/workspace/ggufone` (host `~/workspace/ggufone`); local `main`, **no push** per the card |
| Head this card started from | `106e920` (the card's certification commit) |
| Interpreters | `3.11.15` (`/tmp/x311`), `3.12.13` (`/workspace/t_b5872762/venv312`), `3.13.5` (`/workspace/t_b5872762/venv313`), and the host `.venv` = **CPython 3.13.14** (`/home/rybens/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu`) |
| Command shape | `env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-023 uv run --extra dev pytest -q` (the stub bundle: the four empty `lib*.so` names from the CI step) |

## AC1 — the code, not the phrase

**The fact** (three interpreters, one probe, `probe-413-phrase.log` — `probe_413_phrase.py` prints
both the enum and the handler's own `responses` table, and they move together):

```
CPython 3.11.15   http.HTTPStatus(413).phrase = 'Request Entity Too Large'   responses[413] = same
CPython 3.12.13   http.HTTPStatus(413).phrase = 'Request Entity Too Large'   responses[413] = same
CPython 3.13.14   http.HTTPStatus(413).phrase = 'Content Too Large'          responses[413] = same
```

**The red, reproduced here on the pre-fix tree** (`red-313-host.log`, CPython 3.13.5, full suite):

```
E       AssertionError: b'HTTP/1.1 413 Content Too Large'
E       assert b'HTTP/1.1 41...ent Too Large' == b'HTTP/1.1 41...ity Too Large'
tests/test_serve.py:625: AssertionError   (and :641)
2 failed, 1783 passed, 59 skipped in 52.83s
```

**The change** — one parser plus the sweep AC1 asked for (every status assertion in the file, eight
sites; `tests/test_serve.py`):

```
-    assert status == b"HTTP/1.1 413 Request Entity Too Large", status          # -> 413 x2
+    assert _code_of(status) == 413, status

-    assert status != b"HTTP/1.1 413 Request Entity Too Large", "a legal body is not a refusal"
+    assert _code_of(status) == 200, "a legal body is not a refusal"            # stronger: the positive code

-    assert status == b"HTTP/1.1 200 OK", status                                # -> 200 x3
+    assert _code_of(status) == 200, status
-    assert first[0] == b"HTTP/1.1 200 OK" and json.loads(first[2])["status"] == "ok"
+    assert _code_of(first[0]) == 200, first[0]
+    assert json.loads(first[2])["status"] == "ok"
-    assert second[0] == b"HTTP/1.1 404 Not Found", second[0]
+    assert _code_of(second[0]) == 404, second[0]
```

```python
def _code_of(status: bytes) -> int:
    """`b"HTTP/1.1 413 Content Too Large"` -> `413`, whatever phrase the interpreter chose."""
    version, _space, rest = status.partition(b" ")
    assert version == b"HTTP/1.1", status
    code, _space, _phrase = rest.partition(b" ")
    assert code.isdigit() and len(code) == 3, status
    return int(code)
```

The helper's comment block names the 3.13 change and RFC 9110 §15 (clients SHOULD ignore the phrase),
so the next reader of a code-only assertion is not left guessing. The `200`/`404` phrases are stable
across 3.11–3.13; they were swept anyway, because the rule is what keeps the class of red out.

**Decision (the card's slack): tolerant assertion, not a pinned phrase.** Pinning one phrase in the
server would mean hand-rolling the status line (`send_response_only` + `send_header`) to override
`BaseHTTPRequestHandler.responses`, i.e. adding a non-contractual string to the wire and a second
code path in `Handler`, for a field RFC 9110 tells clients to ignore. Default recommendation taken.

## AC2 — the offline gate runs py3.13 too

`.github/workflows/ci.yml`, the `gate` job's matrix (`ci-matrix.txt`, read with the repo's own PyYAML):

```
-        python-version: ["3.11", "3.12"]
+        python-version: ["3.11", "3.12", "3.13"]
         (plus a five-line comment: 3.13 is here because the interpreter owns the reason phrase, and
          this list is to be widened before certifying on a new interpreter)
```

**Pins that name the matrix: there are none** — nothing had to move RED->GREEN. `tests/test_release_publish.py`
is the only gate that parses a workflow (with `yaml.safe_load`), and it reads `publish.yml` only (its
`WORKFLOWS` glob is used for the "exactly one publisher" check); no test or doc references `ci.yml`'s
matrix or its python versions (grep over `tests/`, `docs/*.md`, `tools/`, plus `pyproject.toml`'s zero
`Programming Language :: Python :: 3.1x` pins).

**`runtime-matrix.yml` is untouched**, and the reason is measurable rather than assumed: none of its
five jobs (`linux-cpu`, `linux-vulkan`, `windows-cpu`, `macos-metal`, `macos-engine-smoke`) carries a
`strategy.matrix` at all — its axes are the runner OS/backend, one job each, and its Python is the
runner's bare `python3` (`ci-matrix.txt`, `probe_ci_matrix.py`). There is no python-version list for
the same reasoning to cover.

## AC3 — gates (all after the change)

| Run | Interpreter | Result |
|---|---|---|
| the card's suite command | **3.13.14** (host `.venv`) | **1785 passed, 59 skipped, 0 failed** in 54.68s (`green-313-host.log`, the final tree incl. this receipt) |
| the same, py3.11 | 3.11.15 | 1785 passed, 59 skipped, 0 failed (`green-311.log`) |
| the same, py3.12 | 3.12.13 | 1785 passed, 59 skipped, 0 failed (`green-312.log`) |
| `pytest -q tests/test_serve.py` | 3.13.14 / 3.11.15 / 3.12.13 | 67 passed, 1 skipped each (`serve-313.log`, `serve-311.log`, `serve-312.log`) |
| the card's py3.11 spot: the **two touched tests** | 3.11.15 | `2 passed in 0.11s` (`two-cap-tests-311.log`) |
| `ruff check src tests tools docs .github` | 3.13.14 | `All checks passed!` (`ruff.log`) |
| `uv build` | — | wheel + sdist `0.2.3` (`build.log`) |

**The tolerant assertion is not vacuous** (the substitute for a Tier-M mutation sweep, since no `src`
line changed — see *Limits*): `_too_large` was temporarily mutated to answer `418` and the two cap
tests went RED with the code in the message (`vacuity-mutant-418.log` — mutant reverted, the tree
carries only the two intended files):

```
E       assert 418 == 413
E        +  where 418 = _code_of(b"HTTP/1.1 418 I'm a Teapot")
2 failed, 1 passed, 65 deselected in 0.18s
```

The phrase that produced this red (`I'm a Teapot`, i.e. the interpreter's text) is exactly what the
assertion ignores; the code is what it reads.

## Limits of this evidence

* **Mutation: not applicable, not skipped for convenience.** The diff is `tests/test_serve.py` and
  `.github/workflows/ci.yml`; mutmut mutates `src`, and the card forbids re-cutting the cap behaviour
  (review M3 accepted it), so there is no mutant on this surface to kill. The 418 probe above is the
  one behaviour these assertions pin, asserted directly.
* **The 3.12/3.13 interpreters are not the CI runners'**: `3.13.14` is the host `.venv` interpreter
  (the one the certification used, minus the patch-level difference the card's prose had as 3.13.15),
  `3.12.13` is uv-managed here, `3.13.5` is this container's `/usr/bin/python3.13`. A GitHub runner's
  patch release can in principle differ; the *phrase tables* were read on 3.11.15/3.12.13/3.13.14 and
  the CI job installs its own interpreter per matrix entry.
* **Environment note (transparency, not part of the deliverable):** a first `uv run` without an
  explicit `UV_PROJECT_ENVIRONMENT` re-created the repo's `.venv` as **CPython 3.12.13** — this
  container's uv default differs from the host's. It was moved aside and restored to the host shape
  (`uv venv --python /home/rybens/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13`
  + `uv sync --python … --frozen --extra dev` -> `.venv/bin/python -> …cpython-3.13…`, `Python 3.13.14`);
  every gate above passes `UV_PROJECT_ENVIRONMENT` explicitly. If the host's `.venv` metadata matters
  to a later gate, a host-side `uv sync --extra dev` re-asserts it.
* **Not taken here, flagged for the coordinator:** `pyproject.toml`'s classifiers still list
  `Programming Language :: Python :: 3.11` / `3.12` with no `3.13` (and no test pins that list). It is
  release metadata, not the CI matrix, so this card left it alone.
