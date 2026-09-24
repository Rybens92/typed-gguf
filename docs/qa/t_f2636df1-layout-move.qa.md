# t_f2636df1 — LAYOUT: root dot-dirs → `docs/evidence/<name>/` + `docs/qa/` (QA note)

Card: *typed-gguf — LAYOUT: root dot-dirs → `docs/evidence/<name>/` + `docs/qa/` + a gate that keeps
new ones out*. Tier **M**; **mutation not run** — move-only change-set, no `src/` logic touched (the
two `src/` edits are path strings inside docstrings). Version stays 0.2.2, no tag, no bump.

## 1. What moved

`git mv` of the thirteen tracked dev-run dirs (exact mapping from the card), 1499 files:

| old | new | files | | old | new | files |
| --- | --- | --- | --- | --- | --- | --- |
| `.e2e` | `docs/evidence/e2e` | 1138 | | `.e3e` | `docs/evidence/e3e` | 40 |
| `.e3b` | `docs/evidence/e3b` | 24 | | `.t07b5` | `docs/evidence/t07b5` | 26 |
| `.e3c` | `docs/evidence/e3c` | 37 | | `.t5b75` | `docs/evidence/t5b75` | 9 |
| `.e3c_scratch` | `docs/evidence/e3c_scratch` | 9 | | `.t5f9` | `docs/evidence/t5f9` | 10 |
| `.e3c_specials` | `docs/evidence/e3c_specials` | 26 | | `.t7c9` | `docs/evidence/t7c9` | 47 |
| `.e3c_tiel` | `docs/evidence/e3c_tiel` | 46 | | `.t9bcb` | `docs/evidence/t9bcb` | 70 |
| `.e3d` | `docs/evidence/e3d` | 17 | | | | |

History survives (`git log --follow --oneline -2 -- <new path>` after the move — the rename commit,
then the pre-move commit the file was written in):

```
docs/evidence/e2e/t_287e0d18-live/report.md   29e967e (the move)  4b03847 docs(evidence): the live Vulkan placement attempts … (t_287e0d18)
docs/evidence/e3b/sweep.json                  29e967e (the move)  953434c evidence(E3b t_6952f0dd): the campaign's runs, drivers and logs …
docs/evidence/t9bcb/stats.json                29e967e (the move)  5766954 qa(t_9bcbecff): the landing on 901ea6f … (oldest: 8521461)
```

`.gauntlet/` (19 QA/spec notes, untracked + gitignored) is now tracked `docs/qa/`: `/.gauntlet/` was
removed from `.gitignore`, the notes are `git add`ed under the same filenames, and the four files
that cited them were re-pointed at `docs/qa/<note>` (`.gitignore`, `pyproject.toml`,
`tests/test_typed_gguf_surface.py`, and the two receipt lines below).

## 2. Decisions (the card's slack, taken)

1. **Receipts stay byte-frozen.** `docs/evidence/**` and `docs/qa/**` keep the paths they were
   produced with. Three in-repo rules say so (the frozen-receipt docstring and `FROZEN_DOC_DIRS`
   in `tests/test_typed_gguf_surface.py`, `.t5f9/sweep.py`'s "Frozen (never touched)"), the card's
   own gate excludes those two trees, and the card's gate design only makes sense if mentions are
   kept: rewriting captured logs numerically is history-editing. Every kept mention is listed, with
   its pre-move spelling, in `docs/qa/t_f2636df1-kept-receipt-mentions.txt` — **1101 lines in 236
   files**, nothing silent.
   * Consequence, stated rather than hidden: **one** living literal was narrowed instead of
     rewriting one receipt line — `tests/test_e3e_docs.py` pinned
     `".e3d/bench_templated_shipped.json"` *inside* `docs/evidence/e3e_role_split_t_4c48f40a.md`;
     it now pins the artifact's name (`“bench_templated_shipped.json”`) with a comment naming the
     retired prefix. The receipt is untouched; the assertion still fails if the provenance line
     leaves the document.
2. **`docs/qa` joins the frozen set.** `tests/test_typed_gguf_surface.py`'s old-name sweep skipped
   `.gauntlet/**` (gitignored); tracked `docs/qa/**` would have red-lit it on eight notes that carry
   the pre-rename data-home path. `RECEIPT_DIRS`/`FROZEN_DOC_DIRS` now spell the two receipt
   subtrees of `docs/` — same semantics as before, no weakening (those files were never scanned).
3. **The regex needed four segment-shaped fixes it cannot see.** `ROOT / ".e3e" / …`
   (`tests/test_policy_v2.py`, the published v2 arm), `root / ".e3d" / name`
   (`tools/e3d_cue_decision.py::default_arms`), `tmp_path / ".e3b"` (`tests/test_e3b_evidence.py`),
   and one docstring `.e2e`: all four are real paths, all four now spell `docs/evidence/<name>/`.
   `tools/t80_splice_doc.py` pointed at a foreign `/work/t80serve` copy; it now works on this
   repo's own moved receipt dir (`ROOT / "docs/evidence/…"`), which is where the raw material is.
4. `.gitignore`: the per-dir rules follow the dirs (`docs/evidence/e3b/logs/*`,
   `…/e3c/logs/*`, `…/e3c_tiel/*.log`, `…/t07b5/{logs,home,venv}`, …), every `!` negation kept for
   the still-tracked receipts, `/.gauntlet/` deleted, the scratch comment block re-worded and no
   unrelated rule weakened. Proof: the 63 files that were ignored before the move and briefly
   surfaced as untracked after it are ignored again (untracked count back to the 15 that were
   untracked before, all of them in `docs/evidence/e2e/t_b67f9c49-calibrate/` — that card never
   committed them; they moved with the directory and stay untracked, exactly as before).

## 3. Gates (verbatim tails)

| gate | result |
| --- | --- |
| `env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<ci stub> uv run --extra dev pytest -q -rs --timeout=120` | `1650 passed, 57 skipped in 51.85s`, exit 0 |
| the same, **before** this card (same stub) | `1647 passed, 57 skipped` — the +3 are `tests/test_public_layout.py` |
| `uv run pytest -q tests/test_public_docs.py tests/test_public_layout.py` | `20 passed` |
| old-path sweep (the card's regex, `:!docs/evidence :!docs/qa :!tests/test_public_layout.py`) | prints **nothing** (`git grep` exit 1) |
| `uv run ruff check src tests` / `ruff check src tests tools docs .github` | `All checks passed!` |
| path citations (AC 7): a one-off ledger over the living surface — every `docs/evidence/…` / `docs/qa/…` token in the 96 living files that carry one | `507 checked: 491 resolve, 16 flagged` = **12 tokenizer artefacts** (brace/`..`/`\n`/placeholder forms my regex cannot read — `{before,after}_mixed.raw`, `devset_00{1..6}.jsonl`, `e2_t_858c54d1_bench.md.\`, `…_4b.{md`, `{path.name}` — each verified present by hand) + **4 real misses, all pre-existing**: `.e3e/bench_shipped.json` / `.e3e/bench_role_split_shipped.json` in `tools/e3e_roles_decision.py`'s usage block and `docs/evidence/{e2p5_route,e1a_kv_footprint}.json` in two tools this card never touched — all four verified absent at `HEAD~2` too, so the move introduced none |
| the required bare-checkout shape, for real: `git worktree add /tmp/wt-layout HEAD` (`.git` as a *file*, no `.venv`, no caches) → `pytest -q tests/test_public_layout.py` | `3 passed in 0.07s` (the worktree was removed again) |
| mutation | **not run** (Tier M, move-only: receipts + path strings; no `src/` logic in the diff) |

The stub matters: the card's own command uses an empty `mktemp -d`, and `test_bench_prompt_parity`
fails on an empty `TYPED_GGUF_BENCH_RUNTIME_DIR` (the CI job creates the four empty `lib*.so` of
"Offline bench bundle stub"). Baseline and final run both used it.

## 4. Scratch deleted (AC 6)

`du -sh` before → after, per dir (measured, then removed by a script that refuses anything outside
the seven names and never touches `mutants/`):

```
7.9M mutants-e1c-backup        19M mutants.stale-0919-1505     19M mutants.stale-0919-1659
 36M mutants.stale-t5f9        44M mutants.stale-t_dd15582e    26M mutants_e3e_r1
 26M mutants_e3e_r2           --- 177.9 MB total, repo root 4.6G -> 4.5G
```

All seven deleted; `mutants/` (44M, the live workspace) untouched. No receipt names one as its
**only** evidence source: they are mutmut *working copies*, named only as the label of the tree a
committed census ran in (`docs/evidence/e3e/mutmut_summary.py` censused `mutants_e3e_r1/r2` and its
output `mutmut_summary.txt` is committed; `.e3b/logs/mutmut_final.out` names
`mutants.stale-0919-1505`). The numbers they produced are recorded; the trees are regenerable.

## 5. The new gate

`tests/test_public_layout.py` (written **RED first** — it failed on the 14 root dot-dirs before the
move, and its own second and third tests pin the bare-checkout shape and that the gate *finds*
clutter). Allowlist exactly as the card spells it: dirs `.git`, `.github`, `.venv`,
`.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.hermes`; files `.gitignore`, `.coverage`;
pattern `.mutmut-cache*`; root = `Path(__file__).resolve().parents[1]`; `.git` tolerated as a file
so a linked worktree (the `t_2b89cce2` shape) does not read as a violation.

## 6. Artifacts

* `tests/test_public_layout.py` — the gate.
* `docs/qa/t_f2636df1-kept-receipt-mentions.txt` — the complete list of kept pre-move mentions.
* this note.

Commits (on `main`, no push):

* `553464e` `test(layout): RED — a gate for the public root layout` (the failing gate, before the move)
* `29e967e` `refactor(layout): root dot-dirs -> docs/evidence/<name>/ + .gauntlet -> docs/qa/`
