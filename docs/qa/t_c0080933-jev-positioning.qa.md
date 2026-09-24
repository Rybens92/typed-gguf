# typed-gguf — README & package positioning: the Jev-alternative story — QA Report

Date: 2026-09-22 | Card: `t_c0080933` | Tier: M (card declares none) — report in M shape
Commit: `fc91778` on `main` (local, **NOT pushed** — the coordinator pushes after verification)
Parent: `6864ae4`

## What changed

| file | +/- | what |
| --- | --- | --- |
| `README.md` | +14/−9 | intro leads with the differentiator (+1 paragraph); the four current-scope `v0.1.0` spots made version-agnostic |
| `pyproject.toml` | +1/−1 | `description` only: one sentence, 182 chars |
| `tests/test_public_docs.py` | +3/−1 | one gate pin moved (see 🟡 below) |

Region receipt (`/tmp/t_c0080933/readme_region_receipt.py`, HEAD~1 → HEAD):
sections byte-identical — `## Platforms`, `## Fit…`, `## Verification`, `## License`,
`## Credits and attribution`; changed regions — `<intro>`, one sentence in `## Quickstart`,
one table row in `## Interfaces`, three lines in `## Limitations`. Whole-file invariants:
urls 14 → 14, inline-code tokens 251 → 251, fenced blocks 17 → 17, table rows 13 → 13
(no command, flag, path or link lost). Numeric tokens 297 → 293: exactly the four `0.1.0`
removals (the one remaining `N.N.N` is the `§7.4.2` section number, not a version).

## Risk-weighted summary

🟢 **ACCEPTABLE — every card gate is green on the committed tree**
- `env -u PYTHONPATH uv run --extra dev pytest -q tests/test_public_docs.py` → **17 passed**
  (pre-commit and post-commit identical).
- dev-token sweep over `README.md` → **empty** (0 matches), and `grep -n v0\.1\.0 README.md`
  → **empty** (all four current-scope references gone).
- Full offline suite, CI shape (`TYPED_GGUF_TEST_BLOCK_NET=1`, bench bundle stub,
  `-q -rs --timeout=120`) → **1552 passed, 56 skipped, 0 failed** (41.9 s).
- CI's live-set check (`pytest -q -m "model or network"`, runtime vars unset) → **55 skipped,
  0 failed**; engine oracle (`docs/verify_runtime_contract.py`) → **failures: 0, skips: 2**
  (both the expected bundle-free skips).
- `ruff check src tests tools docs .github` → **clean**.
- `pyproject.toml` diff is exactly the one `description` line; every other field unchanged
  (name/version/readme/requires-python/deps verified by parse).

🟡 **WORTH CONSIDERING — your call: the commit is docs+metadata *plus one gate pin***
- What: the card asked to make README line ~337 version-agnostic, and
  `tests/test_public_docs.py::test_the_readme_marks_the_serving_surface_as_not_shipped`
  pinned the literal `"not implemented in v0.1.0"` — the exact sentence being rewritten.
  The two card requirements (version-agnostic row **and** 17/17 green) are mutually
  exclusive unless the pin moves, so the pin moved (1 assertion line + 2 comment lines).
- Risk: a coordinator check that reads "docs-only" as "README.md + pyproject.toml only" sees
  a third file. No behavior, no gate strength changed: the assertion is still a literal, still
  the same node, and the file's own docstring instructs "update this file together with the
  public docs".
- Cost to fix if you disagree: revert the pin and the README row together (2 min) — but then
  one of the four named spots keeps its version number, i.e. the card's rule stays unmet.
- Recommendation: **accept as committed.**

🟢 **ACCEPTABLE — mutation testing: not run, nothing to mutate.** No file under `src/` moved
(no executable line touched), so the Tier-M sweep has no subject. Coverage: n/a (no new code;
the docs claims are covered by the 17-node public-docs gate).

## Decision matrix

```
Option A — ship as committed (fc91778)
  ✅ all card gates green; four version spots gone; description carries the differentiator
  ✅ region receipt proves nothing else in the README moved; invariants (urls/code/flags) intact
  ⚠️ the commit carries one 3-line gate-pin diff (documented above)
  → Overall risk: LOW

Option B — reject the pin move, keep the README row's "v0.1.0"
  ✅ commit truly docs-only (2 files)
  ⚠️ the card's "version-agnostic" rule fails on 1 of its 4 named spots; the gate would have to be
     run with that one node red, i.e. the "17/17 green" gate fails
  → Overall risk: MEDIUM (a stated gate goes red)

Option C — revert pin + row, keep 3 of 4 spots changed
  ⚠️ same red gate as B, plus partial compliance
  → Overall risk: MEDIUM
```

💡 Recommendation: **Option A** — every stated gate holds and the only deviation is one
literal moved in the gate that quotes the sentence the card rewrote.

## What could go wrong (the 🟡, made concrete)

⚡ SCENARIO: a verifier diffs `git show --stat` and stops at `tests/test_public_docs.py`.
  Probability: medium (it is a one-line flag in an otherwise docs-only series).
  Trigger: a strict reading of "one docs-only commit".
  Impact: one round-trip of rework at worst; the README itself is correct either way.
  Detectability: HIGH (visible in the first diff). Blast radius: this card only.

## Confidence

📈 CONFIDENCE: 8/10
- + `pytest -q tests/test_public_docs.py` 17/17 on the committed tree (pre- and post-commit).
- + full offline suite 0 failures; ruff clean; oracle failures: 0.
- + region receipt: unchanged sections byte-identical, no command/flag/link lost, no new numbers.
- + the RED step is on record: with the pin moved first, the gate said `1 failed, 16 passed`
  and named the exact README sentence — so the pin move is evidence-driven, not convenience.
- − the "docs-only" definition is the one judgment call the card left to the coordinator.

## Decision

Human/coordinator chose: _(pending — this report is the input to that decision)_
Agent recommendation: Option A.
