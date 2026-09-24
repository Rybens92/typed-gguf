# typed-gguf 0.2.2 — release: version bump + release notes — QA Report

Card: `t_63439bd8` | Date: 2026-09-23 | Tier: **M** (card declares none; the change is
docs/metadata-only, so the default stands) | HEAD: `59935db` on `main`, **not pushed, not tagged**

## Summary (risk-weighted)

🟢 **ACCEPTABLE — every gate green, and nothing product-shaped moved.**

- `git diff 8d8eff5..HEAD`: 5 files, +357 / -6 — the notes (351 new lines), `pyproject.toml` (1),
  `src/typed_gguf/__init__.py` (1: the `__version__` string), `uv.lock` (1), the docs gate's three
  pins (3). The only `src/` line in the diff is the version string; no product line moved.
- docs gate `tests/test_public_docs.py` → **17/17**; release gate `tests/test_release_publish.py`
  → **15/15**.
- offline suite, the CI shape → **1 640 passed, 57 skipped, 0 failed** (45.09 s at HEAD;
  43.45 s in `/work/t63439-ev/suite_green.log`, the run the notes quote). Baseline before the
  change: 1 640 / 57 / 0 — no test was added, removed or skipped differently by this card.
- engine oracle `docs/verify_runtime_contract.py` → `failures: 0  skips: 2` (the two expected
  bundle-free skips).
- `ruff check src tests tools docs .github` → All checks passed. No type checker in this repo
  (ruff is the static gate; CI runs ruff + pytest + oracle).
- `uv run typed-gguf version` → `typed-gguf 0.2.2`.
- `uv build` → `typed_gguf-0.2.2-py3-none-any.whl` (311 778 B) + `typed_gguf-0.2.2.tar.gz`
  (3 886 710 B); `publish.yml`'s own gate script executed verbatim against those artifacts:
  `v0.2.2` → rc 0 (“v0.2.2 is the built 0.2.2”), `v0.2.1` / `v0.2.3` → rc 1 with `::error::`,
  the dispatch shape → rc 0, continues (`/work/t63439-ev/gate_probe.out`).
- RED→GREEN as the two prior releases did it: `9d678ea` (bump + notes; measured RED 2 failed /
  30 passed, full suite 1 638 / 57 / 2) → `59935db` (the three gate pins; 17/17 + 15/15 + 1 640 / 0).

🟡 **WORTH CONSIDERING (reviewer's call, no fix required):**

1. *The notes quote “43 s” for the suite.* That is the receipted run (43.45 s,
   `/work/t63439-ev/suite_green.log`); a re-run at the same head measured 45.09 s — wall clock
   varies run to run, and no assertion reads it. Cost to fix: a third commit re-wording a timing
   that would drift again on the next run. Recommend: leave it, the number is receipted.
2. *The race section has no in-repo receipt path.* The placement section points at the committed
   `.e2e/t_287e0d18-live/`; the race's live gate (3 pairs, 6/6, one host per pair) is the fix
   card's own (`t_9249bb0c`) and no log for it exists in this checkout, so the notes state the
   result and cite the offline pins (`tests/test_keep*.py`) instead of a path that is not there.
   If a repo path is wanted, the race card has to commit the log first. Risk: a reader cannot
   re-open the receipt from the notes — a provenance gap, not a correctness one. Detectability:
   HIGH (a reviewer reading the section). Blast radius: the notes only.
3. *Mutation testing not run.* Tier M asks for one sweep over the changed files; the changed
   files are a version string, a lock line, the notes and three gate pins — no product line to
   mutate. The configured sweep (`[tool.mutmut] source_paths = ["src/typed_gguf/runtime/fit.py"]`,
   `tests/test_context_v2.py` + `tests/test_fit.py`) targets a module this release does **not**
   touch, and would re-score the tree this build already reports on (its published score stands:
   2 267 mutants, 1 335 killed / 805 survived = 62.4 %, `docs/evidence/context-v2/mutation.md`).
   Same call the 0.2.1 release card made, and the notes say so in their own words.

🔴 **REQUIRES ATTENTION: none.**

## Decision

🤔 **Ship (push → tag `v0.2.2` → GitHub Release), after the owner's OK.**

- **Option A — ship as committed.** Verified: every gate above at `59935db`; the tag↔version
  contract (`v0.2.2`) proven by executing the workflow's own gate; the notes carry every fact the
  docs gate pins (measured rows, warm-host numbers, uvx one-liner, the honest limits) because the
  carried-forward sections are byte-identical to the v0.2.1 file's. Accepted risks: the two 🟡
  provenance/timing notes above. Overall risk: **LOW**.
- **Option B — commit the race receipt first (+~5 min, owned by another card).** Removes 🟡2 and
  would need a re-run of the gates. It does not change what the release ships.
- 💡 **Recommendation: Option A.** The card's own scope (bump + notes, coordinator pushes and tags)
  is complete; the 🟡 items are provenance wording and wall-clock noise, not release risks.

## Confidence

📈 **9/10**

- Increasing: 17/17 + 15/15 at HEAD; full suite 1 640 / 57 / **0**; oracle failures 0; ruff clean;
  version the two spellings of one number; the publish gate proven by execution, not inspection;
  the docs gate re-run *after* the last edits; carried-forward sections byte-identical.
- Decreasing: the race section's receipt lives outside the repo (🟡2); the quoted suite timing is
  one of two measurements of the same tree (🟡1); mutation deliberately not re-run (🟡3).

## Evidence

`/work/t63439-ev/` — `red.log` (2 failed / 30 passed), `suite_redhead.log` (1 638 / 57 / 2),
`suite_green.log` (1 640 / 57 / 0, 43.45 s), `suite_final.log` (45.09 s at HEAD), `gate_probe.out`,
`dist-v022/` (the built artifacts), `build_notes.py` (the notes' generator, with its assertions).
Commits: `9d678ea` (RED) → `59935db` (GREEN), both on `main`, local.
