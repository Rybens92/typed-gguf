# t_bde46896 — RELEASE PREP 0.3.0: the version, the notes, and the version-pinned gates

The repository is release-ready for **v0.3.0**: one version everywhere live, the notes written in the
established format, the two version-pinned gates shown RED then GREEN, and the build producing
`typed_gguf-0.3.0*`. **No tag, no push, no GitHub Release** — the coordinator certifies, pushes, tags
and releases after this card.

```
1c6a555  docs(evidence): t_8dab8b3a — the final-head gate logs and the commit list   (the card's base)
b531423  tdd(t_bde46896): RED — the v0.3.0 notes land, and the two version pins move to 0.3.0
f64dc7c  tdd(t_bde46896): GREEN — the packaged version is 0.3.0 (pyproject + __init__ + classifier
         3.13 + uv.lock); the two version pins now pass
HEAD     this receipt (local only)
```

Working tree clean after the receipt commit; `git status --short` is empty, `dist/` is gitignored and
not committed, and nothing was pushed (`git log origin/main -1` → `1c6a555`, the base).

## 1. Files changed

| file | what |
| --- | --- |
| `docs/RELEASE_NOTES_v0.3.0.md` | **new** (476 lines, 35 666 B — v0.2.3's notes are 454 lines / 32 909 B): the release notes, both headline sections cited with receipts |
| `tests/test_public_docs.py` | the version pins move with the version: the notes file name (×2) and `assert version == "0.3.0"` |
| `pyproject.toml` | `version = "0.3.0"`; classifier `Programming Language :: Python :: 3.13` added |
| `src/typed_gguf/__init__.py` | `__version__ = "0.3.0"` |
| `uv.lock` | `uv lock` → the project pin moves `0.2.3 → 0.3.0` (line 462) |
| `docs/evidence/t_bde46896/*` | this receipt, the gate scripts and the logs |

No `src/` statement changed other than the version literal; `.github/workflows/*` untouched (in
particular `publish.yml`); `dist/` not committed.

## 2. The version, everywhere live (AC1, AC2, AC3)

```
$ uv lock
Resolved 18 packages in 0.81ms          # "Updated typed-gguf v0.2.3 -> v0.3.0"
$ uv sync --frozen --extra dev
Checked 9 packages in 0.23ms            EXIT=0
$ uv run python -c "import tomllib,pathlib,typed_gguf; pyv=tomllib.load(open('pyproject.toml','rb'))['project']['version']; print(pyv, typed_gguf.__version__); assert pyv=='0.3.0'==typed_gguf.__version__"
0.3.0 0.3.0                             EXIT=0
$ uv run typed-gguf version             # isolated TYPED_GGUF_HOME
typed-gguf 0.3.0
```

`version_pins_sweep.txt` is the sweep AC1 asks for (`$ git grep -n '0\.2\.3'`). Outside the receipt
trees (`docs/evidence/`, `docs/qa/`) and the old notes (`RELEASE_NOTES_v0.2.2/v0.2.3.md`), the
mention survives in exactly **three** places, all of them historical prose in a test docstring that
names the version at which a past observation was made — never a pin, never read by a gate:

```
tests/test_default_model.py:3    Card t_a0fa2dc0 (UX, 0.2.3): after `typed-gguf models pull` …
tests/test_hf.py:307             It shipped as `typed-gguf/0.1` while the release API's UA said `typed-gguf/0.2.3` …
tests/test_runtime_update.py:688 `typed-gguf/0.2.3`: `registry/hf.py` — the module the GitHub asset download borrows its …
```

None of the three is a version *reference* to keep in step: they record what was true then (`0.2.3`
is the release in which that UA was observed), and rewriting them would falsify history for no gate's
benefit. They are reported rather than swept — say the word and a later card can reword them.
`pyproject.toml`, `src/typed_gguf/__init__.py` and `uv.lock` carry `0.3.0` (`version_pins_sweep.txt`
§3/§4), and the classifier list now reads 3.11 / 3.12 / 3.13.

## 3. The version-pinned gates: RED → GREEN (AC5)

The sweep (`version_pins_sweep.txt` §1/§5) found two gates that pin the version or the notes file:

* `tests/test_public_docs.py` — the notes file name (lines 37, 47) and `assert version == "0.3.0"`
  (line 83, whose own message is the instruction to rename the file with the version);
* `tests/test_release_publish.py` — derives `NOTES` from `pyproject.toml` and asserts that the docs
  gate carries **that** version's file name and literal (lines 254/256), plus that the notes exist and
  are titled for the packaged version.

**RED** (`red_version_pins.log`, on commit `b531423`: the notes are v0.3.0, the packaged version is
still 0.2.3):

```
$ env -u PYTHONPATH uv run --extra dev pytest -q tests/test_public_docs.py tests/test_release_publish.py
E       AssertionError: this file pins the v0.3.0 notes: rename it with the version
E       assert '0.2.3' == '0.3.0'
E       AssertionError: the docs gate still reads another release-notes file than v0.2.3
FAILED tests/test_public_docs.py::test_the_release_notes_are_pinned_to_the_packaged_version
FAILED tests/test_release_publish.py::test_the_docs_gate_is_renamed_with_the_version_it_pins
2 failed, 31 passed in 0.18s
```

**GREEN** (`green_gates.log`, on the bumped tree):

```
$ env -u PYTHONPATH uv run --extra dev pytest -q tests/test_public_docs.py tests/test_release_publish.py
33 passed in 0.15s
```

The two gates only tell the truth once both spellings move, which is exactly the drift they exist to
catch: `test_release_publish` reads the *docs gate's own text* to prove the two pins cannot drift
apart.

## 4. The gates (AC1, AC3, AC6) — verbatim

`gates.sh` (the card's commands, plus the two pinned files and `version`; `UV_PYTHON` is the one
container prefix, §6):

```
$ env -u PYTHONPATH uv run --extra dev pytest -q
1 failed, 1905 passed, 58 skipped in 58.16s            EXIT=1
```

The one failure is `tests/test_bench_prompt_parity.py::test_a_row_records_which_framing_it_measured`
— `BenchError: none of the requested backends (cpu) has a local llama.cpp bundle`. This container has
no llama.cpp bundle and the card's constraint is metadata + docs, so it is **not this card's** and not
new: the parent card proved the identical failure at the parent revision in a clean worktree
(`docs/evidence/t_8dab8b3a/logs/parent_rev_parity_failure.txt`, `7c72c4b`), and both earlier cards of
this wave saw it. The 0-failed shape the card asks for is the CI shape — the four empty `lib*.so`
names `ci.yml` creates in a stub bundle dir:

```
$ TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-030 env -u PYTHONPATH uv run --extra dev pytest -q
1906 passed, 58 skipped in 58.54s                      EXIT=0     # green_gates_bundle_dir.log
```

```
$ uv run python -c "import tomllib,pathlib,typed_gguf; … assert pyv=='0.3.0'==typed_gguf.__version__"
0.3.0 0.3.0                                            EXIT=0
$ env -u PYTHONPATH uv run --extra dev ruff check src tests tools docs .github
All checks passed!                                     EXIT=0
$ env -u PYTHONPATH uv run --extra dev ruff check …   (green_gates.log)   # docs included
$ env -u PYTHONPATH uv build
Successfully built dist/typed_gguf-0.3.0.tar.gz
Successfully built dist/typed_gguf-0.3.0-py3-none-any.whl          EXIT=0
$ ls dist/
typed_gguf-0.3.0-py3-none-any.whl
typed_gguf-0.3.0.tar.gz
```

`dist/` is gitignored (`git check-ignore -v dist` → `.gitignore:13:dist/`); the two stale `0.2.3`
artifacts an earlier card had left there were deleted before this build, because `publish.yml`'s gate
**refuses an ambiguous `dist/`** ("two wheels in dist/ make the gate ambiguous") and a release build
must not inherit one. `dist/` is untracked either way.

## 5. The notes' outline, and what feeds each section (AC4)

Structure and phrasing are mine (the card's DECISION SLACK): the two things a user can now *do* are
the front page — the operator's explicit ask — and everything older keeps the v0.2.3 wording it
already had, so the carried-forward sections are recognisably the same claims rather than new ones.

| section | fed by (one line) |
| --- | --- |
| title + blockquote | this card's bump + the wave's headline cards + run `36163763039` + PyPI `latest = 0.2.3` (`github_run_36163763039_jobs.txt`) |
| *What this is* | carried verbatim from v0.2.3 (SPEC §2.4/A4, `tests/test_no_finetune.py`) |
| **Headline 1 — `typed-gguf serve`** | `3fd16e8` + `a3f8f94` + `7c1e8c6` (the server, its wire gates, the mutation round-2 pins); receipts `docs/evidence/t_f5d8b6c7_serve_gates.md` and `docs/evidence/t_559ed8c8/` (cold 6.76 s / warm 2.28 s, `noul 0.574913`, the negatives, the bounds); live legs: `macos-engine-smoke` step 11 and `windows-cpu` step 9 of run `36163763039` |
| **Headline 2 — `runtime update` / `runtime rollback`** | `c85ae33` + `79d37d0` (the feature + its RED gates), `cc44aad` (P1/P2/P3); receipts `docs/evidence/t_d88b4be0_update_gates.md` (live b11026 → b11160, 1.601 s), `docs/evidence/t_559ed8c8/` (the `kill -9` leg at 3 145 728 B, 13 negatives), `docs/evidence/t_16067777/` |
| *The rest of the wave* | `49a8746`/`250955f`/`a94d773` (the default model) + `docs/evidence/t_a0fa2dc0_default_model.md`; `106e920` (the py3.11 reap) + `docs/evidence/t_ba767a2b/`; `cc44aad` (P3/P4) + `docs/evidence/t_16067777/` |
| *The platform matrix comes alive* | `bee06b3`, `20ba546`, `7e8a111`, `ca3dbba` + `docs/evidence/t_f96fed7f/README.md`; `519adb6`…`c39a5f3` + `docs/evidence/t_8dab8b3a/README.md` (the two live failures); runs `36151935399`, `36159785190`, `36163763039`; `f23a352` + `docs/evidence/t_b5872762/` for 3.11/3.12/3.13 |
| *Measured highlights* | carried unchanged from v0.2.3 (`docs/BENCHMARKS.md`, `docs/evidence/e3e_role_split_t_9bcbecff.md`) — explicitly marked as **not** re-measured here |
| *The warm engine host* | carried from v0.2.3 (`docs/evidence/v0_1_0_t_7e24cea4_warm_host.md`), plus this wave's three changes to it (serve shares it, `keep stop` takes its log, an abandoned child is reaped) |
| *Context sizing v2* | carried from v0.2.3/v0.2.0 (`docs/SPEC-context-v2.md`, `docs/evidence/context-v2/README.md`) |
| *The fixes carried in from v0.2.3 and v0.2.2* | carried (`docs/evidence/e2e/t_176614c6-live/RECEIPTS.md`, `docs/evidence/e2e/t_287e0d18-live/`) |
| *Install* | carried + `f23a352`'s 3.11/3.12/3.13 matrix + `docs/evidence/v0_1_0_t_eff926f9_uvx_install.md` |
| *What this release does not include* | carried + this wave's real bounds (the 30 s idle drop, one host for three surfaces) and `mcp` still a stub |
| *License and credits* | carried verbatim |
| *Verification* | this card's gates (§4) + the wave's receipts + the two live runs + the Tier-M sweeps the wave ran |
| *Reproduce* | carried + the served entry's host gate and the update recipe |

The content gates in `tests/test_public_docs.py` were the specification for the carried sections
(the measured rows `50/60`/`53/60`/`54/60`/`22/60`/`26/60`, the warm-host numbers, the uvx one-liner
and its receipts, the pre-v2 switch mentions, the MIT/credits line) — they are the reason the notes
cannot silently drop what an earlier release measured, and they pass on the v0.3.0 file (33 passed).
One RED of that file was *my* drafting error, not a product one: the phrase "not implemented in this
release" was split across a line break and the gate reads it as a substring; the fix is in `b531423`.

## 6. Environment notes (transparency, not the deliverable)

* **`UV_PYTHON` is the one prefix this container needs.** The sandbox's `uv` defaults to CPython
  3.12.13 and re-creates `UV_PROJECT_ENVIRONMENT`'s interpreter on a bare `uv run --extra dev`; the
  host's `.venv` is 3.13.14 (`/home/rybens/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu`).
  The first gate run here did rebuild it as 3.12.13, so it was moved aside and restored
  (`uv venv --python <3.13> .venv` + `uv sync --frozen --extra dev --python <3.13>` →
  `Python 3.13.14`), and every logged gate sets `UV_PYTHON=<3.13>` **inline**. The gate text is
  otherwise the card's, `TMPDIR` is `/tmp` (the card's long-`TMPDIR` note does not apply in-container),
  and `TYPED_GGUF_HOME` / `TYPED_GGUF_BENCH_RUNTIME_DIR` are unset for the bare run.
* The first full-suite run in this session used the sandbox's 3.12.13 venv and the final two (bare
  and stub-bundle) ran on 3.13.14; both shapes were green on the same failure count.
* `.venv` is untracked and gitignored — no tracked file was touched by any of this.

## 7. Findings (for the coordinator / the next card)

1. **The full suite is 1-failed in a container without a llama.cpp bundle** (the bench-prompt-parity
   gate). Pre-existing, receipted by the parent card; the card's 0-failed shape needs either the CI
   stub bundle (`TYPED_GGUF_BENCH_RUNTIME_DIR`, §4) or a host with a bundle.
2. **The default data home is not clean in this container** (`/root/.local/share/typed-gguf` holds a
   `runtime.json` from 2026-09-24 pointing at a pytest tmp dir, plus six `.runtime.json.tmp-*`
   leftovers). Some earlier card's test wrote into the *default* home instead of an isolated one. It
   does not affect this card — the receipt sets an isolated `TYPED_GGUF_HOME` for `version` — but a
   `typed-gguf version` run with no override reports a foreign runtime, which will confuse the next
   reader.
3. **Three test docstrings still name `0.2.3` as history** (§2). Reported, not swept.
4. **`dist/` carried stale `0.2.3` artifacts** from an earlier card; deleted before this build so
   `publish.yml`'s ambiguity refusal cannot fire on a genuinely fresh build (§4).

## 8. Reproduce

```
cd /workspace/ggufone                      # host: ~/workspace/ggufone
bash docs/evidence/t_bde46896/version_pins_sweep.sh
bash docs/evidence/t_bde46896/gates.sh
bash docs/evidence/t_bde46896/lock_sync.sh
TYPED_GGUF_BENCH_RUNTIME_DIR=<four-empty-libs dir> \
  UV_PYTHON=<host 3.13> env -u PYTHONPATH uv run --extra dev pytest -q
```

The RED half is reproduced by moving the version back and re-running the two pinned files:
`git show b531423 --stat` is the tree it was measured on.
