# typed-gguf 0.2.3 — release prep: version bump + release notes + gate pins + Node-24 action bumps — QA Report

Card: `t_ace98219` | Date: 2026-09-24 | Tier: **M** (card declares none; the change is
docs/metadata/workflow-only, so the default stands) | HEAD: `d311803` on `main`,
**not pushed, not tagged, not published**

## Summary (risk-weighted)

🟢 **ACCEPTABLE — every gate green at 0.2.3, and no product line moved.**

- `git diff 8bb4b38..HEAD`: 9 files, +477 / -23 — the new notes file (451 lines), the three
  workflow files (14 `uses:` lines), `pyproject.toml` (1), `src/typed_gguf/__init__.py` (1: the
  `__version__` string), `uv.lock` (1), and the two gate files that pin the moved values
  (`tests/test_public_docs.py` 3 pins, `tests/test_release_publish.py` 2 constants + comment). The
  only `src/` line in the diff is the version string.
- The three worked pins moved RED→GREEN, each shown below with its own before/after:
  1. **the action pin** — `5044f51` (RED on `tests/test_release_publish.py`:
     `test_the_steps_are_checkout_uv_build_gate_and_publish`, 1 failed / 34 passed) → `fcbf3a5`
     (GREEN, 35 passed);
  2. **the version pins** — `6d4b7d3` (RED: `test_the_release_notes_are_pinned_to_the_packaged_version`
     + `test_the_docs_gate_is_renamed_with_the_version_it_pins`, 2 failed / 33 passed) → `c36dc20`
     (GREEN, 35 passed);
  3. `d311803` — the notes' Verification section moved onto this build's own numbers.
- docs gate `tests/test_public_docs.py` **17/17**, release gate `tests/test_release_publish.py`
  **15/15**, layout gate `tests/test_public_layout.py` **3/3** → **35 passed** together.
- offline suite, the CI shape (`env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1
  TYPED_GGUF_BENCH_RUNTIME_DIR=<four empty libs> uv run --extra dev pytest -q -rs --timeout=120`) →
  **1 650 passed, 57 skipped, 0 failed** in 51.66 s; the same run before the last notes edit was
  1 650 / 57 / 0 in 50.85 s (`/work/t_ace98219-ev/suite_green.log`). The card's expected shape
  (~1 650 / ~57 / 0) is met exactly.
- engine oracle `docs/verify_runtime_contract.py` → `failures: 0  skips: 2` (the two expected
  bundle-free skips).
- `ruff check .` (the card's command) and `ruff check src tests tools docs .github` (CI's shape) →
  **All checks passed!** No type checker in this repo.
- `uv lock --check` → rc 0, one-line `uv.lock` diff (the editable package's own version);
  `uv run typed-gguf version` → `typed-gguf 0.2.3`, `--json` → `"version": "0.2.3"`.
- `uv build` → `typed_gguf-0.2.3-py3-none-any.whl` (314 263 B) + `typed_gguf-0.2.3.tar.gz`
  (4 044 675 B); `publish.yml`'s own gate script executed **verbatim** against those artifacts:
  `v0.2.3` → **rc 0** ("version gate: v0.2.3 is the built 0.2.3"), `v0.2.2` → **rc 1** and
  `v0.2.4` → **rc 1** (`::error::built 0.2.3 is not the release tag …`), the `workflow_dispatch`
  shape → rc 0 (prints and continues), a stale wheel beside the fresh one → rc 1
  (`/work/t_ace98219-ev/gate_probe.out`).
- Version sweep: `grep -rn "0\.2\.2" tests/ src/ pyproject.toml | grep -v __pycache__` → **no
  hits**; the remaining `0.2.2` mentions in the checkout are the historical notes
  (`docs/RELEASE_NOTES_v0.2.1.md`), the coordinator's journal (`state/groupchat/…`), and the
  git-ignored `mutants/` working copies — all named, none load-bearing.
- Root hygiene: the root listing is unchanged (no new entry), and the layout gate that enforces it
  is green. The QA note lives here, under `docs/qa/`.

🟡 **WORTH CONSIDERING (reviewer's call, no fix required):**

1. *The card's verbatim suite command is red in a bundle-free container.* `env -u PYTHONPATH uv run
   --extra dev pytest -q` (no bundle stub) → **1 649 passed, 1 failed, 57 skipped**, the failure
   being `tests/test_bench_prompt_parity.py::test_a_row_records_which_framing_it_measured`, which
   needs a *visible* bundle to select `cpu` from. `ci.yml` already carries that stub as its own step
   (four empty library files), and the notes state both numbers. Risk: a reviewer running the bare
   command sees one red and reads it as this card's. Cost to fix: none (the stub is CI's, and the
   card's GATES section expects the stub shape). Detectability: HIGH.
2. *The carried-forward prose keeps its em dashes and its `t_`-named receipt paths.* The card asks
   for no dev jargon, no `t_` ids, no em dashes. That rule was applied to everything **new**: the
   headline, the two fix sections, "Under the hood" and the Verification edits carry **zero** em
   dashes (audited by line range), and no prose names a card id. The byte-identical carried sections
   (the `What this is` / v0.2.2 sections the docs gate pins by name) keep the v0.2.2 wording, and
   the receipts themselves are directories named by card (`docs/evidence/e2e/t_176614c6-live/`),
   exactly as the previous releases cite theirs (`docs/evidence/e2e/t_287e0d18-live/`). Rewriting
   the carried sections would break the pinned strings the docs gate asserts. Blast radius: the
   notes only.
3. *Mutation testing not run.* Tier M asks for one sweep over the changed files; they are a version
   string, a lock line, three workflow files, the notes and five gate lines. There is no product
   line to mutate, and the configured sweep targets `src/typed_gguf/runtime/fit.py`, which this card
   does not touch (its published score stands: 2 267 mutants, 1 335 killed / 805 survived = 62.4 %,
   `docs/evidence/context-v2/mutation.md`). The notes say so in their own words.
4. *The notes quote "51 s", the receipts measured 51.66 s and 50.85 s at the same head.* Wall clock
   varies run to run and no assertion reads it.
5. *Untracked files this card did not stage.* `docs/evidence/e2e/t_b67f9c49-calibrate/*` (15 files)
   were already untracked before this card and are left exactly as they were; nothing in this card's
   commit set touches them.

🔴 **REQUIRES ATTENTION: none.**

## Addendum, same day: the CI-red and its fix (card `t_70dc92b5`)

The push of `f2dfc3b` turned CI red in **Set up job** in both jobs of `ci.yml`: GitHub could not
resolve the `astral-sh/setup-uv` step at all (the log's `unable to find version v10`, and the run
stopped before any step executed), so nothing this report measured had run on GitHub. `setup-uv`
ships no floating major above `v7` (`v8` and later exist as exact versions only) and `v7` declares
`runs.using: node24`, so the seven `uses:` references moved to `astral-sh/setup-uv@v7`: the
release-gate constant and its comment first (RED:
`test_the_steps_are_checkout_uv_build_gate_and_publish`, 1 failed / 31 passed), then the three
workflow files, then green (32 passed across the release and docs gates, 3/3 layout). The two prose
spots in the release notes that named the old pin now name `@v7`. Everything else this report
verified stands as recorded; `actions/checkout@v7` was already resolvable and did not move.

## Decision

🤔 **Ship (coordinator pushes `main` → tag `v0.2.3` → GitHub Release), after the owner's OK.**

- **Option A — ship as committed.** Verified: every gate above at `d311803`; the tag↔version
  contract (`v0.2.3`, accepting `v0.2.3`, refusing `v0.2.2` / `v0.2.4`) proven by executing the
  workflow's own gate script, not by reading it; the workflow files keep their names, and
  `publish.yml` keeps its `environment: pypi` block, so the four values PyPI's trusted publisher
  matches on are untouched; the notes carry every fact the docs gate pins because the
  carried-forward sections are byte-identical to the v0.2.2 file's. Accepted risks: the five 🟡
  notes above. Overall risk: **LOW**.
- **Option B — rename the notes' carried sections to drop their em dashes (+~20 min).** Would break
  the docs gate's pinned strings (`WARM_FACTS`, the receipt paths, `no parity claim`) unless the gate
  moved with it, which is a rewrite of a document the previous release deliberately froze. It does
  not change what the release ships.
- 💡 **Recommendation: Option A.** This card stages only; the coordinator pushes and the owner
  decides on the tag. Nothing here is a release risk.

## Confidence

📈 **9/10**

- Increasing: three RED→GREEN pairs, each with the failing test named and logged; 17/17 + 15/15 + 3/3
  at HEAD; full suite 1 650 / 57 / **0**; oracle failures 0; ruff clean in both shapes; the version
  the two spellings of one number; `uv lock --check` clean; the publish gate proven by execution
  against the built artifacts; the docs gate and the release gate re-run *after* the last notes edit.
- Decreasing: the bare card command is red without CI's bundle stub (🟡1, environment, documented);
  the carried sections keep the v0.2.2 typography against the card's no-em-dash wording (🟡2);
  mutation deliberately not re-run (🟡3); the quoted suite timing is one of two measurements of the
  same tree (🟡4).

## Evidence

`/work/t_ace98219-ev/` — `red_workflows.log` (1 failed / 34 passed), `red_version.log` (2 failed /
33 passed), `green_workflows.log` + `green_version.log` (35 passed each), `suite_green.log`
(1 650 / 57 / 0, 50.85 s), `suite_final.log` (1 650 / 57 / 0, 51.66 s), `suite_barestub.log`
(1 649 / 1 failed / 57), `gate_probe.out` + `gate_probe.py` (the five tag cases), `dist/`
(typed_gguf-0.2.3 wheel + sdist), `bundle/` (the four empty library files CI's step builds).
Commits on `main`, local: `5044f51` → `fcbf3a5` → `6d4b7d3` → `c36dc20` → `d311803`.
