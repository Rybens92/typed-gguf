# t_d88b4be0 — `typed-gguf runtime update|rollback` (SPEC 2.8): gates & receipt

Card: *typed-gguf — FEAT: `runtime update` — refresh the llama.cpp bundle that `init` installed*
(owner: *"żeby dało się zaktualizować llama.cpp które się instaluje poprzez init"*). Serve wave,
Tier M. Local `main`, no push/tag/publish.

## 1. What shipped

| file | what changed |
| --- | --- |
| `src/typed_gguf/runtime/update.py` | **new** (269 statements): `current_runtime`, `releases_url`/`upstream_repo`, `parse_releases`/`fetch_releases`, `retag_asset_name`/`build_of_tag`/`pick_target`, `_stage`, `_require_probe`, `_stop_host`, `update()`, `rollback()` |
| `src/typed_gguf/runtime/pins.py` | `RuntimeLock.repo` — the API repo is the *lock's own* fact (`load_lock` reads `llama_cpp.repo`), never a hard-coded fork |
| `src/typed_gguf/cli.py` | `runtime` in `COMMANDS` + `_cmd_runtime` + `runtime_text` + `SUBCOMMANDS`/`RUNTIME_SUBCOMMAND_DESCRIPTIONS` + the `<cmd> <sub> --help` page (generalized from `models`) |
| `src/typed_gguf/errors.py` | `UpdateUnavailableError` (a user error, exit 2); `E_UPDATE_UNAVAILABLE` already in `ERROR_CODES` |
| `tests/test_runtime_update.py` | **new** (41 pass + 1 network skip): check / switch / record / rollback / every failure path |
| `tests/test_scaffold.py` | the frozen command set + `RUNTIME_SUBCOMMANDS` |
| `README.md` | the Interfaces row + `## Updating the runtime` (the one-command story + the rollback recipe) |
| `pyproject.toml` | the Tier-M sweep retargeted at this module (comment block + pair) |

`init` is untouched: it still installs the pinned tag. Updating is something the user asks for.

## 2. Gates (all in-container)

```
$ env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<4-empty-libs stub> \
    uv run --extra dev pytest -q -rs --timeout=300
1754 passed, 59 skipped in 56.56s          # baseline before this card: 1713 passed / 58 skipped / 0
$ env -u PYTHONPATH uv run ruff check src tests tools docs .github
All checks passed!
$ env -u PYTHONPATH uv build
Successfully built dist/typed_gguf-0.2.3.tar.gz + dist/typed_gguf-0.2.3-py3-none-any.whl
$ uv run --extra dev --with pytest-cov pytest --cov=typed_gguf.runtime.update tests/test_runtime_update.py
src/typed_gguf/runtime/update.py  269 stmts  7 miss  97%
```

Bundle-free container (the card's paste gate):

```
$ TYPED_GGUF_HOME=$(mktemp -d) typed-gguf runtime update --check
error: E_RUNTIME_MISSING: no llama.cpp runtime installed under /tmp/tmp.tThKlsWe8F/runtime; run `typed-gguf init` first (it installs the pinned bundle, no compiler needed)
EXIT:3
$ typed-gguf runtime update --help
usage: typed-gguf runtime update --check|--dry-run --tag TAG --backend auto|cpu|vulkan|cuda|metal --json

refresh the installed llama.cpp runtime
run `typed-gguf --help` for the command list
$ typed-gguf runtime
error: E_UNKNOWN_KEY: runtime needs update|rollback (SPEC 2.8), not no subcommand
EXIT:2
```

## 3. The live leg — a real update, not a rehearsal

The container has a real network and a linux-x86_64 host, so the whole chain ran against the real
GitHub API and the real official bundle (the offline suite never touches it; only the `network`
test does, and it skips by name):

```
$ TYPED_GGUF_HOME=$(mktemp -d) typed-gguf init --backend cpu
bytes_fetched: 16855810 / build: 11026 / symbols_ok: True / rung: prebuilt
$ typed-gguf runtime update --check --backend cpu
current b11026 (build 11026, linux-x64-cpu) at …/runtime/b11026-linux-x64-cpu
target  b11160 (build 11160, linux-x64-cpu) 17.00 MB -> …/runtime/b11160-linux-x64-cpu
nothing changed (--check prints the plan only)
   (--json: asset llama-b11160-bin-ubuntu-x64.tar.gz, size 17002550,
    sha256 48ece242…e435 — GitHub's own `digest` field, published_at 2026-09-24T13:50:41Z)
$ time typed-gguf runtime update --backend cpu
downloading runtime: 100.0%  17.00 MB / 17.00 MB
updated b11026 -> b11160 (build 11160, linux-x64-cpu) at …/runtime/b11160-linux-x64-cpu; the previous bundle is kept at …/runtime/b11026-linux-x64-cpu
real 0m1.601s
$ typed-gguf version --json      → runtime.build 11160, dir …/b11160-linux-x64-cpu
$ typed-gguf runtime rollback    → rolled back to b11026 … ; b11160 is kept on disk
$ typed-gguf runtime rollback    → error: E_UPDATE_UNAVAILABLE: … records no previous bundle …   EXIT:2
$ time typed-gguf runtime update --backend cpu        # re-update adopts the kept bundle
updated b11026 -> b11160 … real 0m1.230s              # record source: "already-downloaded"
$ grep asset_verified…  → true (download) / false + asset_sha256 from the digest (adopt)
$ typed-gguf doctor --json → runtime.tag b11160, build 11160, pinned_tag b11026,
                             "build b11160 != pinned b11026" (doctor's own warn), …
```

The live record after the download update (the switch point, in full):
`dir/tag/build` = the new bundle, `source: "url"`, `asset_verified: true`, `libllama_sha256`
re-hashed, `tools` pointing at `runtime/b11160-linux-x64-cpu/…` (**not** at the staging path),
`previous {dir,tag,build,installed_at}`, `update_from`, `updated_at`, `probe_warnings` =
*"this runtime was updated from b11026 to b11160; the committed oracle numbers were measured on
the pinned bundle (`typed-gguf runtime rollback` returns to it)"*.

## 4. RED→GREEN pairs (the logic pins)

| pin | RED (pasted) | GREEN |
| --- | --- | --- |
| the whole surface | collection died: `ImportError: cannot import name 'update' from 'typed_gguf.runtime'` (commit `79d37d0`) | 41 passed / 1 skipped |
| the record's `tools` must name the *installed* dir, not the staging dir | live find: `probe.tools` = `…/.pending-llama-b11160-bin-ubuntu-x64.tar.gz/llama-cli` while `record.dir` was the final one | `_retarget_tools`; pinned by `record["tools"] == payload["probe"]["tools"]` |
| **no `init --force` advice on a runtime the user moved** | `AssertionError: ['runtime build b11160 differs from the pinned b11026 (…re-run `typed-gguf init --force`)', …]` | `_updated_warnings` answers the one warning an update itself answers |
| `build_of_tag("v0.5.0")` | `assert 50 is None` (digits joined out of a milestone tag) | strict `b(\d+)` full-match |
| fixture `state["record_bytes"]`, `probe["tools"]`, `--backend cpu` on the CLI-level check | 4 failed / 30 passed mid-round | 41 passed / 1 skipped |

## 5. Mutation sweep (Tier M, `--max-children 2`, `tools/mutmut_driver.py`)

`source_paths = ["src/typed_gguf/runtime/update.py"]`,
`pytest_add_cli_args_test_selection = ["tests/test_runtime_update.py"]`.

```
round 1 (before the pins):  921 mutants — 731 killed 🎉 / 190 survived 🙁 → 79.4 %, 0 ⏰ / 0 🔇 / 0 🫥
round 2 (the 7 pin hunks):  921 mutants — 785 killed 🎉 / 136 survived 🙁 → 85.2 %, 8.88 mut/s
results: two verdicts, source unchanged between them (`mutants/mutmut-stats.json` rewritten;
"0.00 mutations/second" is mutmut 3.8 reprinting a cache — a scoped re-run of the named keys is
the only real re-run).
```

The pins took survivors from 190 → 136, and the families they closed were: the release-list
parser (a tagless/non-object entry must not stop the walk), a missing/non-numeric asset `size`,
a `runtime.json` that names *another* directory (and one that names this one without a `build`),
`--tag` narrowing the walk, the fetch's own kwargs (tag/limit/timeout), the record's `tools` and
`updated_at`, rollback's `host_stopped` + `rolled_back_at`, and what the update asks the
production probe for (`lock`, `expect_backend`, `run_tools`, `system`).

Remaining 136 survivors are the triaged families the offline selection cannot see: message-string
prose (`str(exc)[:200]` → `[:201]`, `report.get("reason")` casing), kwargs passed *into* the
injected seams the gates replace (`lock=None`, `progress=None`, `run_tools=False` — the fake probe
ignores them; the real ones are driven by `test_runtime_install.py`), the unreachable
`except OSError` arm of `_same_dir`, `digest.split(":", 1)` variants (GitHub digests carry one
colon), and the `rmtree(ignore_errors=True/False)` cleanup arms where the tree is already gone.

## 6. Decisions the card left open (simple, reversible)

* **Staging location**: `<data home>/runtime/.pending-<asset>` — the data home, not a cache, and
  `init`'s own dot-name so a half-extracted bundle sits exactly where the next run would clear it.
* **An interrupted download resumes** (`.part` + HTTP range, `hf.download_url`) and the switch is
  `os.replace` — the same semantics `init` already had, not a second set of rules.
* **The record write is the switch point**: every earlier exit leaves `runtime.json`
  byte-identical (all failure gates assert exactly that), and nothing is ever deleted.
* **A target directory already on disk is adopted** (probed + recorded, `source:
  "already-downloaded"`), so `rollback` + `update` never re-downloads; `--check`/`--dry-run` are
  the same flag pair.
* **The update does not warm up** (no model load): the record keeps the old `warmup_model`, so
  the next `run`/`ask` warm-up still has its hint.
* **`E_UPDATE_UNAVAILABLE` is a user error (exit 2)**, not an internal one; a `$TYPED_GGUF_RUNTIME_DIR`
  rung and an empty data home refuse *before* any socket is opened.

## 7. What only the host can prove (for the coordinator)

* A real update on the **4B/Vulkan** rung: `typed-gguf init` + `runtime update` with the real
  Vulkan bundle and the warm host resident (the container has no GPU; the cpu leg above did run
  for real, against the live release list).
* The dpkg/glibc side of a *real* GPU bundle's `dlopen` (the offline gates inject the backend
  errors instead).
* The rollback recipe the README documents, end to end on the host's own data home.
