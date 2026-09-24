# typed-gguf — README install-by-name (PyPI leads) — QA Report

Card: `t_f711a7b1` · Date: 2026-09-22 · Tier: default **M** (card declares none)
Artifact: commit `6864ae4` on `main`, **README.md only** (+35/−11), local-only — the coordinator
pushes after verification, per the card.
Recommendation: **Option A — verify and push.** One 🟡 follow-up belongs on the board, not in this
commit.

## Gates (all re-run in this sandbox, repo `/workspace/ggufone`)

| # | Gate (card) | Result |
| --- | --- | --- |
| 1 | `env -u PYTHONPATH uv run --extra dev pytest -q tests/test_public_docs.py` | **17 passed** (pre-commit and post-commit) |
| 2 | dev-token sweep over README.md | **empty** (grep exit 1: no `t_…`, `milestone`, `E1a`/`E1b`/`E1c`/`E2.5`/`E4`, `[executed]`, `[recon]`, `SPEC §`) |
| 3 | full offline suite, CI shape (`TYPED_GGUF_TEST_BLOCK_NET=1`, bench runtime dir) | **1552 passed, 56 skipped, 0 failed**, pytest exit **0** (41.5 s, `--timeout=120`) |
| 4 | card acceptance harness (16 checks, `/tmp/t_f711a7b1/install_story_check.py`) | **RED first (7 failing → the card's premises reproduced), GREEN after (16/16, exit 0)** |
| 5 | byte-stability receipt (`/tmp/t_f711a7b1/byte_stability_check.py`) | **STABLE, 0 failures** |

Gate 5 detail (this is the card's "everything else stays byte-stable" rule, measured against
`93d83c6`): every top-level section except `## Quickstart` is **byte-identical** (intro, Platforms,
Interfaces, Fit, Limitations, Verification, License, Credits); whole-file numbers **115 → 115**;
URLs **9 → 9**; inline-code tokens **182 → 187** (5 added, none lost); all **102** command lines of
the old Quickstart's fenced blocks survive — verbatim, or as the name-based form the card mandates
(`uv run typed-gguf init` → `typed-gguf init`).

Mutation testing: **not run — nothing to mutate.** The change touches no executable line (README.md
prose only; no Python/test file changed), so a mutmut/Stryker run would have no subject. Coverage:
n/a for the same reason.

## Risk-weighted summary

🟡 WORTH CONSIDERING (coordinator's call — follow-up, not a blocker)

  **`docs/RELEASE_NOTES_v0.1.0.md` still carries the pre-release claims this card retires.**
  Line 3–5: *"Nothing is tagged or published yet: no git tag, no GitHub release, no PyPI project
  (the name is reserved but unpublished)…"* — all three are now false; and its `## Install` section
  (line 93+) still leads with the `git+` spellings.
  Risk: a reader who arrives from PyPI lands on notes that deny the release they just installed
  (the same class of drift this card just fixed in README).
  Cost to fix: ~15 min docs card, **deliberately not folded in here** — the card scopes to the
  README, forbids touching other sections, demands a docs-only single commit, and the notes have
  their own pins in `tests/test_public_docs.py`; half-editing one clause would leave the rest of the
  draft banner inconsistent. Recommendation: **separate follow-up card**.

🟢 ACCEPTABLE (no action)

- Quickstart prose now says "v0.1.0 is on PyPI" and leads with exactly the three name-based forms
  (`uvx typed-gguf …`, `uv tool install typed-gguf`, `pip install typed-gguf`), no `git+` in the
  lead block — machine-checked, not eyeballed.
- "needs nothing but Python 3.11+", the stdlib-only claim, the clone/`uv sync` dev path (now
  labelled "The repository clone is the development path"), the `python -m typed_gguf …` mention,
  the wheel/`runtime.lock` explanation, the measured `init` receipt and `$TYPED_GGUF_LOCK` paragraph
  all survive — substring-checked.
- The `git+` forms stay as the fallback for unreleased revisions, after the name-based block, and
  the `UVX_ONE_LINER` pin in `tests/test_public_docs.py` is still satisfied.
- Both stale caveat phrasings are gone: "installs from this repository (there is no PyPI release
  yet)" and "while the repository is not published".

## Decision matrix

Option A — **verify and push** (recommended)
  ✅ all 3 card gates green; the change is docs-only, mechanically receipted, and nothing else in
     the file moved; risk of reader-facing breakage: none found.
  ⚠️ residual: the new install story is pinned by no *in-repo* test (the harness lives in `/tmp` to
     keep the commit docs-only), so a later paste could regress the wording silently.
  → Overall risk: **LOW**.

Option B — fix the release notes here too (+~15 min)
  ✅ removes the last stale pre-release claim in the public pair.
  ⚠️ breaks the card's docs-only/one-commit scope and the "do not touch other sections" rule;
     needs a re-read of the notes' own test pins.
  → Risk **LOW**, but scope-wise wrong for this card.

💡 Recommendation: **Option A**, plus a follow-up card for the release-notes banner (the 🟡 above).

## Confidence

📈 **9/10**

Increasing: 17/17 public-docs gate (pre and post commit) · 1552 passed / 0 failed offline suite
with exit 0 · RED-before/GREEN-after harness over the card's own rules · mechanical byte-stability
receipt for "everything else stays" · tree clean at `6864ae4`, commit touches README.md only.

Decreasing: the acceptance harness is not committed (docs-only constraint), so the new wording has
no permanent red/green guard in-repo · the notes' staleness (🟡) remains until the follow-up lands.
