# E3d F2 — the §5 prose patch, applied verbatim (card `t_ebe3648c`)

Card `t_ebe3648c` (code-tdd) · 2026-09-19 · shared tree on `main` (no git remote is configured there,
so nothing was pushed) · one commit: the generator (`tools/e3d_cue_decision.py`), the regenerated
`docs/evidence/e3d_cue_decision_4b.{md}` and this document.

The E3d audit (`t_fc037544`, `state/fights/e3d-ci/scorecard.md`) endorsed the decision (default
frozen, mechanism behind `--cue`, `json_field` excluded) with one prose-only disagreement: §5 spoke
of *"what the numbers recommend"* and *"the flip, if the second opinion agrees"*, which reads as a
promotion the card's own rule (paired agreement CI excludes zero) does not license. The coordinator
approved **F2** and asked for it verbatim, with zero numbers moving. This is that patch.

## 1. What was applied

Two paragraphs replaced in `decision_section()` (§5 of the generated doc), plus the NICE item F4.
The text was **extracted from the scorecard's §7.1 block by script** (`ast.literal_eval` over the
block, `state/fights/e3d-ci/scorecard.md` lines 191-236), never retyped:

| id | where | what |
|---|---|---|
| F2a | `**What the numbers recommend.** …` | → `**What the data supports — and what it licenses.** …` (127 words, unchanged wording of the auditor) |
| F2b | `**The flip, if the second opinion agrees.** …` | → `**What a flip would need (not licensed by this data).** …` (90 words; also consumes the stray `+` → `(plus `Options.cue`)`) |
| F4 | after the reproduce commands | one line: `Byte-identity of the JSON holds under CPython 3.11; 3.12+ changes the last ulp of `mean` fields — diff with tolerance.` |

Verbatim check (`/tmp/e3df2/check_verbatim.py`, run against the regenerated doc):

    paragraph 1: 127 words, found at char 5009
    paragraph 2: 90 words, found at char 6275
    check exit=0

**One deviation, formatting only:** two lines of the auditor's proposal are 101 characters, and the
repo's `[tool.ruff] line-length = 100` (E501) rejects them. The paragraphs are therefore re-wrapped
to the same words at different break points; the word-sequence assertion above is the guarantee
that the text did not move. `ruff check tools/e3d_cue_decision.py` → `All checks passed!` (the
proposal as written would have failed the repo's lint).

## 2. The patch (diff)

    docs/evidence/e3d_cue_decision_4b.md | 29 +++++++++++++++++++----------
     tools/e3d_cue_decision.py            | 29 +++++++++++++++++++----------
     2 files changed, 38 insertions(+), 20 deletions(-)

Full md diff is three hunks, all inside §5 (two paragraphs + F4); the diff tail:

    @@ -107,6 +113,9 @@ refusal stop, the bench knob, the CLI flag, the payload key) are already gated b
         python3 tools/e3d_engine_check.py --record .e3d/full.json --items 6
         bash .e3d/run_bench_arms.sh               # the bench arms (## 6)

    +Byte-identity of the JSON holds under CPython 3.11; 3.12+ changes the last ulp of
    +`mean` fields — diff with tolerance.
    +
     ## 6. The bench arms — and the instrument split they expose

     Same box, same model, same 60 committed items, same context; only `--cue` moves.

## 3. Zero numbers move — the regeneration check

Baseline *before* the edit (the committed doc was a fixed point of the committed generator):

    $ python3 tools/e3d_cue_decision.py report --run .e3d/full.json --out /tmp/e3df2/base.md --json /tmp/e3df2/base.json
    $ diff docs/evidence/e3d_cue_decision_4b.md /tmp/e3df2/base.md   → empty
    $ diff docs/evidence/e3d_cue_decision_4b.json /tmp/e3df2/base.json → empty
    md   107b8be79e9a1007c4c36b116454c8884c4f742aab044eb307c2806586eb60e8   (= HEAD blob)
    json 5986ed4208e9e510e17dd27fecb5264a81dd1ea83173b116fdfe202beeba4a00   (= HEAD blob)

After the edit (CPython 3.11.15 — the version the artifacts are pinned to):

    $ python3 tools/e3d_cue_decision.py report --run .e3d/full.json \
        --out docs/evidence/e3d_cue_decision_4b.md --json docs/evidence/e3d_cue_decision_4b.json
    $ git diff --stat docs/evidence/e3d_cue_decision_4b.md
     docs/evidence/e3d_cue_decision_4b.md | 29 +++++++++++++++++++----------
     1 file changed, 19 insertions(+), 10 deletions(-)          # the three §5 hunks only
    $ git show HEAD:docs/evidence/e3d_cue_decision_4b.json | sha256sum
    5986ed4208e9e510e17dd27fecb5264a81dd1ea83173b116fdfe202beeba4a00  -
    $ sha256sum docs/evidence/e3d_cue_decision_4b.json
    5986ed4208e9e510e17dd27fecb5264a81dd1ea83173b116fdfe202beeba4a00  docs/evidence/e3d_cue_decision_4b.json

So the JSON is **byte-identical to HEAD** — no number moved — and the markdown changed in §5 alone.
Fixed point re-checked after the write (a second regeneration into `/tmp` diffs empty against the
working tree):

    md   084b8a343751062ed7908c9a880c963572f3152e1f7b490ade9a8b66d9322a97
    json 5986ed4208e9e510e17dd27fecb5264a81dd1ea83173b116fdfe202beeba4a00
    $ diff docs/evidence/e3d_cue_decision_4b.md /tmp/e3df2/round2/a.md  → empty   (and the json)

## 4. F3, F4 — one line each

* **F4 applied** (above): the caveat sits next to the reproduce commands — and was **verified here,
  not assumed**: `/usr/bin/python3.13` (3.13.5) regenerating the same record against the same
  patched generator leaves the markdown *byte-identical* and changes exactly **6** of the JSON's
  `mean` fields (12 diff lines, last ulp: `0.08833688777584976` → `…978`, `0.9031220458161828` →
  `…826`), which is the compensated-`sum()` difference the line names. Only the opening letter was
  capitalised to start the sentence.
* **F3 left**: the scorecard proposes no wording for its parenthetical, and its fix column points at
  `docs/BENCHMARKS.md` §8, which is *not* marker-generated (the `@@E3C_*@@` blocks stop at §7) — so
  it is a hand edit of a published section, not "the same regeneration"; @auditor's text first.

## 5. Gates

| gate | command | result |
|---|---|---|
| lint | `ruff check tools/e3d_cue_decision.py` (0.16.3, repo config) | `All checks passed!` |
| compile | `python3 -m py_compile tools/e3d_cue_decision.py` | ok |
| E3d gates | `python3 -m pytest tests/test_e3d_cue_decision.py tests/test_e3d_cue_switch.py -q` | **37 passed** (the scorecard's count) |
| full suite | `python3 -m pytest -q` | **1242 passed, 44 skipped**, exit 0 (30 s) |

Provenance note: the gates ran on the container's system CPython 3.11.15 with a container-local
`pytest` (9.1.1) install, not through `uv run --frozen` — the shared tree's `.venv` points at the
host's uv-managed interpreter, and a container-side `uv run` would rebuild it under the sibling card
running in the same tree (`t_6de5fc53`). Byte-identity of the JSON needs 3.11 anyway (§4).

**Tier M note:** the change is prose-only — string literals in a report generator, no new branch or
expression — and `tools/` is outside mutmut's `source_paths` (the Python mutation targets live under
`src/`), so there is no new mutatable line and no sweep was run for this card.
