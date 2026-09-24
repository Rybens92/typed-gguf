# Landing receipts — card `t_22544707` (the `t9bcb-e3e-tiel` branch onto `901ea6f`)

The card's two evidence commits (`d170db0`, `5486f8e`) were based on `4ef74f6`; main had moved to
`901ea6f` — the E3e audit's F1–F4 (the freeze field set, the decision wording, the Wilson note, the
`--runs` line, and the regenerated E3e documents). This directory holds the receipts of the rebase,
the re-render and the gates; the commit that carries it is `qa(t_9bcbecff)` on `t9bcb-e3e-tiel`.

## 1. The rebase — `git rebase 901ea6f`, **no conflict**

| before | after | subject |
|---|---|---|
| `d170db0` | `8521461` | the `[host]` E3e policy row on Tiel — `role_split` + `json_instructed`, 53/60 vs 22/60 |
| `5486f8e` | `7c02bf9` | the optional Occamy second pass — both E3e cells, 54/60 vs 26/60 |

The two commits replayed cleanly because the regions are disjoint: this branch writes
`docs/BENCHMARKS.md` **§7.4.2** (immediately after §7.4.1's end anchor) and adds
`docs/evidence/e3e_role_split_t_9bcbecff.md`, while F1–F4 edit **§9** and the E3e evidence files
(`e3e_role_split_t_4c48f40a.md`, `e3e_roles_decision.{json,md}`). A clean textual replay is not the
same as a correct one — hence the re-render below, which is the acceptance the card actually names
("regenerate rather than hand-merge if the doc-render gate can do it").

## 2. The re-render on the post-F1–F4 tree — what it changed, and what it did not

`python3 .t9bcb/analyse.py && python3 .t9bcb/render_doc.py`, run on the rebased tree (so the stats
are recomputed through the **post-F1–F4** `tools/e3e_roles_decision.py`, imported at
`.t9bcb/analyse.py:37`):

* **`stats.json` — byte-identical.** `diff` of the pre-regeneration copy against the regenerated
  file: 0 lines (`stats_regen_diff.txt`; the pre-copy's `sha256sum` is in `pre_rebase_sha.txt`).
* **The evidence document — byte-identical** on that regeneration: `ef58d30c…` before and after
  (`render1_sha.txt` vs `pre_rebase_sha.txt`).
* **`docs/BENCHMARKS.md` — the block itself byte-identical**; the only delta was the splicer's own
  duplicated end-marker run, 24 → 1 (`render1_sha.txt`, and the 23-line deletion in the commit).

Why no line moved (the card's requirement 2, checked rather than assumed — `f1f4_strings_check.txt`):
the F1–F4 strings do not occur anywhere this card reproduces. The stats are built from
`e3e.cell_stats` / `e3e.pair_stats` / `e3e.mcnemar_exact` / `e3e.cell_label` only — the branch never
calls `decide()`, so F2's re-worded verdicts cannot reach the block; it never calls `freeze_check()`
(the F1 field set) or `build_caveats()` (F3's note), and `pair_stats`'s own `caveat` text is
untouched by F1–F4. The one `--runs 1` in the evidence document is a **verbatim quote of the
committed report's own `reproduce` line**, and this card's arm driver passes `--runs 1` explicitly
(`.t9bcb/tiel_e3e_arm.py:86`), so F4's finding has no counterpart here.

### The splicer fix (`splice_benchmarks` in `.t9bcb/render_doc.py`)

Two defects, both in the one function:

* the block's **own** end marker was left inside `body` and then re-appended, so every render added
  one marker — 12 at `d170db0`, 24 at `5486f8e`);
* the replace path kept whatever marker run the previous render had left in the tail.

Fixed: the body is cut at the end marker (`split(end, 1)[0]`), and the tail's leading marker run is
eaten before the tail is written back. Receipt: `render1_sha.txt` ("replaced (23 duplicate end
marker(s) eaten)"), then `render2_sha.txt` / `idempotency_sha.txt` — a second and third render with
`0 duplicate end marker(s) eaten` and **the same `sha256sum` for both documents**:

```
8cbdb5f14a0c94a52a19de8bd83a2a7bc9b32657353634252f8fd5965e07e428  docs/BENCHMARKS.md
4b91b91ec345efac99d43a10649daaee366f645fbebf598fd93a27a6e4c82e20  docs/evidence/e3e_role_split_t_9bcbecff.md
```

## 3. The gates on the rebased tree

| gate | command | result | receipt |
|---|---|---|---|
| oracle | `python3 docs/verify_runtime_contract.py` | `failures: 0  skips: 0`, exit 0 | `.t9bcb/oracle.txt` |
| the four E3e gate files | `uv run --frozen --offline --extra dev python -m pytest tests/test_e3e_roles.py tests/test_e3e_role_tool.py tests/test_e3e_roles_decision.py tests/test_e3e_docs.py -q` | **76 passed, 3 skipped**, exit 0 | `land/e3e_gate_files.txt` |
| the docs tests | `uv run --frozen --offline --extra dev python -m pytest -q -k doc` | **77 passed**, exit 0 | `land/docs_tests.txt` |
| the full suite | `uv run --frozen --offline --extra dev pytest -q -rs -p no:cacheprovider` | **1331 passed, 48 skipped**, exit 0 | `.t9bcb/gates.txt`, `.t9bcb/logs/final_suite.txt` |
| lint | `uv run --frozen --offline --extra dev ruff check src tests tools .t9bcb` | `All checks passed!`, exit 0 | `land/ruff_full.txt` |

Two honest notes on the numbers:

* the branch's commit message claimed "the docs tests 75 passed" — that is the **same selection**
  (`-k doc`) *before* F1–F4 added its two doc gates (`test_the_freeze_claim_counts_the_cue_refusal_verdict`,
  `test_the_instrument_line_names_the_command_that_actually_ran`); on the rebased tree the same
  selection collects and passes 77, which is 75 + those two;
* `901ea6f`'s own message quotes the suite as `1332 passed / 47 skipped`. That is **not** what this
  box's host run of the same tree returns: `901ea6f` measured here (main, before the rebase) is
  also **1331 passed / 48 skipped**, with a skip list identical line for line to the rebased
  branch's (`main_skips.txt` vs `branch_skips.txt` in the run's scratch; the extra skip against the
  container figure is `test_probe_pressure.py:312`, no readable pid cgroup on the host — the skip
  the baseline card recorded here too).

## 4. Landing

Fast-forward only (`git merge --ff-only t9bcb-e3e-tiel` from the main worktree); the branch is kept.
`git log --oneline -3` after the merge and the post-merge re-run of the gates are in the card's
completion; the branch is **not** deleted.
