# t_a51b1205 — SPEC: `serve` v1 (TypeSafe-compatible HTTP) + `runtime update` — receipts

Card: `t_a51b1205` (code-spec) | 2026-09-24 | Head at the gate run: `5609a11` (`SPEC.md` only;
this receipt file is the only later delta — the SPEC's own content is identical).

No `src/` change, no README/help change: the "specified, not shipped" flips for `serve` belong to
the implementation card `t_f5d8b6c7`, and `mcp` keeps its wording.

## Wire provenance [sdk-0.7.1] (what every §2.9 claim stands on)

The official SDK source read during the recon (a copy of `typesafe-sdk` 0.7.1 lives at
`/tmp/tssdk/src/typesafe_sdk`; the live handshake demo at `/tmp/ts_demo.py`):

```
_core/constants.py:5   SYSTEM_ONE_PATH = "/v1/systemone"
_core/constants.py:6   MODELS_PATH = "/v1/models"
constants.py:15        DEFAULT_BASE_URL = "https://api.typesafe.ai"
constants.py:18        DEFAULT_MODEL = "jev-latest"
constants.py:21        DEFAULT_TIMEOUT = 10.0
_core/retry.py:52      max_retries: int = 2            # 5xx/408/429 + connection/timeout retried
_core/retry.py:82      timeout: float | None = 30.0    # total retry budget per call
_core/errors.py        STATUS_ERROR_TYPES: 400/401/403/404/422/429, else 5xx → typed errors
_schemas/models.py     SystemOneRequest{state, model, questions}, SystemOneResponse{model, answers, usage}
                       ModelMetadata{name, description, release_date}, HTTPValidationError{detail:[…]}
```

## Live GitHub-API recon for `runtime update` (2026-09-24)

```
GET /repos/ggml-org/llama.cpp/releases/latest  -> v0.5.0 (2026-09-23T20:50:06Z), 1 asset: nightly-tag.txt
GET /repos/ggml-org/llama.cpp/releases?per_page=8
    b11160 prerelease=true assets=35  2026-09-24T13:50:41Z   <- per-build bundles live here
    b11159 prerelease=true assets=35  …
    … (b11153..b11160 in the first page)
b11160 asset list carries:  llama-b11160-bin-ubuntu-vulkan-x64.tar.gz  30943538
                            llama-b11160-bin-ubuntu-x64.tar.gz         17002550
                            llama-b11160-bin-macos-arm64.tar.gz       11190912
                            llama-b11160-bin-win-vulkan-x64.zip       32450504
                            … (35 total; cuda-13.3 from the lock has NO b11160 match — the honest refusal case)
HEAD https://github.com/ggml-org/llama.cpp/releases/download/b11160/llama-b11160-bin-ubuntu-vulkan-x64.tar.gz
    -> HTTP/2 302, then 200, content-length: 30943538     # the lock's url_template + retag rule resolves live
```

So `releases/latest` is **not** the update target (a milestone release with no bundles sits there);
the pinned name retagged (`llama-b11026-…` → `llama-b11160-…`) matches the first asset-bearing
release, and the lock's `url_template` builds the working download URL. Both facts are pinned in
§2.8 step 2 and R15.

## RED → GREEN: the living-surface rename gate (SPEC-reading)

The first draft of §2.8/§5 quoted the coordinator's reference filename (which keeps the pre-rename
spelling) and the checkout's own directory in the host-gate commands. The living-surface gate caught
all three lines — `tests/test_typed_gguf_surface.py::test_the_living_surface_carries_no_old_name`:

```
E       AssertionError: the old name survives in 3 living line(s) of the typed-gguf tree:
E         SPEC.md:19: `bot-fleet-dispatch/references/ggufone-project.md` § "Jev-compatibility"), §2.12 gains the
E         SPEC.md:1044: cd ~/workspace/ggufone
E         SPEC.md:1052: cd ~/workspace/ggufone
E       assert ['SPEC.md:19:...pace/ggufone'] == []
```

Fixed by naming the reference directory without the file, and by
`cd "$(git rev-parse --show-toplevel)"` in both host-gate blocks. After the fix:

```
$ uv run --extra dev pytest tests/test_typed_gguf_surface.py -q
10 passed in 34.79s
```

## Gates at the committed head

```
$ env -u PYTHONPATH uv run --extra dev ruff check src tests tools docs .github
All checks passed!
$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<four-empty-libs stub> \
    uv run --extra dev pytest -q -rs --timeout=120
1650 passed, 57 skipped in 81.52s (0:01:21)      # the same counts as the pre-change baseline (1650/57/0)
$ env -u PYTHONPATH uv run --extra dev pytest tests/test_public_docs.py -q
17 passed
```

Baseline before the change (same command, same stub): `1650 passed, 57 skipped` — so every
SPEC-reading gate is green with the new §2.5/§2.8/§2.9/§2.12/§5/§7/§8 text. The `--check` /
`E_UPDATE_UNAVAILABLE` / host-gate commands the two implementation cards must satisfy are in §2.8
and §5 (E5), not here.
