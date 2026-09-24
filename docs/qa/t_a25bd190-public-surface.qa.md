# The v0.1.0 public surface (card `t_a25bd190`) — QA Report

Date: 2026-09-20 | Tier: **M** (the card declares none; the profile default). Report format: M
(Risk-Weighted Summary + Decision Matrix + Confidence).

## What changed

Six release-review items applied to the surface a reader meets first: the root help stops claiming
`serve`/`mcp` ship (F1) and `models <sub> --help` names the subcommand once (F2); the README
limitations carry the fit-cache and exotic-wheel findings (F3/N2); SPEC §2.5 lists the policy options
and the whole error catalog the code can emit (F4). Housekeeping: the dispatch-only wheel stub is
gone (N2), the 2.8 MB agent-session export is out of the public tree (N3), the operator e-mail in the
E2 receipt's commit helper is a placeholder (N4). Five commits, `4df6310` → `a7cd2e7`, on the shared
tree. No `docs/BENCHMARKS.md` line, no published row, no default moved.

## 🟢 ACCEPTABLE (no action needed)

* **The help surface now says what the documents say.** Probes before/after
  (`/work/t_a25bd190-ev/{before,after}-{root,models}-help.txt`): the root listing marks `serve`/`mcp`
  `(specified in SPEC §2.9, not implemented in v0.1.0; exits 3)` and every shipped command keeps its
  own milestone; `typed-gguf models search --help` prints `search` once. Both stubs exit 3; an
  unknown command still exits 2.
* **Suite**: at `a7cd2e7` **1374 passed, 48 skipped** (CI shape: empty bundle stub, network blocked,
  `--timeout=120`), vs **1362 passed + 1 failed**, 48 skipped at the pre-card head `232a11b`
  (`suite-final.txt` / `suite-baseline-232a11b.txt`). +11 tests, the one failure fixed (see 🟡),
  skips unchanged; `-k doc` **90 passed** (+4 new doc gates); the red path (`-m "model or network"`,
  both runtime vars unset) **48 skipped, 0 failed, 1374 deselected**.
* **Oracle + pinned gates green**: bundle-free `python3 docs/verify_runtime_contract.py` → `pass 216,
  failures 0, skips 0`; the E3e/policy/parity selection **91 passed, 4 skipped**; `ruff check src
  tests tools docs .github` → *All checks passed*.
* **The tree got smaller and the citation ledger did not move.** Tracked: 1737 files / 26.82 MiB →
  1735 / 24.17 MiB (**−2.65 MiB, −9.9 %**). Hygiene ledger: 255 citations / 239 resolved / 3
  exists-but-untracked (ignored scratch) / 13 missing before → 254 / 238 / 3 / 13 after — deleting the
  export and the stub introduced **zero** dangling citations (the review's N3 pointer was repointed
  at the scorecard, which resolves).
* **Tier-M sweep, the changed production file only** (mutmut 3.8; `source_paths=[src/typed_gguf/cli.py]`,
  selection = the 7 CLI/doc gate files, `--max-children 4`, isolated worktree, one attempt, exit 0):
  **4675 mutants in 484 s (9.74/s) → 1781 killed, 2161 survived, 730 "no tests", 3 timeout.**
  Raw receipt: `mutmut-cli.log`, per-mutant verdicts: `mutmut-results.txt`.
* **The changed surface is triaged to the last mutant.** 30 non-killed mutants sit in the two
  functions this card touched (`_usage`, `_command_usage`); 12 of them on the branches it rewrote:
  * 3 die to the gate this card added — verified by replaying exactly those mutants by hand
    (`x__usage__mutmut_9`, the `XX…XX`-wrapped note literal; `…_12`, `note = None`;
    `…_13`, `MILESTONES.get(None, 'E1')`) and watching `test_the_root_help_marks…` fail on each
    (`(XXspecified…`, `run (None)`, `run (implemented in E1)`); the same pin kills the `.get('E1')`
    sibling. The gate now asserts the exact tails, so padding cannot survive it.
  * 8 are **equivalent mutants**: they mutate the `'E1'` default of `MILESTONES.get(…)`, which no
    reachable call can hit — all 11 `COMMANDS` have a `MILESTONES` entry, and `models <sub> --help`
    prints its own usage line (`cli.py:1638`) instead of calling `_command_usage`. Recorded, not
    chased.
  * The rest of the 30 are prose literals in `_command_usage` this card did not touch (`usage:`,
    `commands:`, the join separator, the notes trailer) — the repo's documented prose-mutant class.

## 🟡 WORTH CONSIDERING (your call)

* **The sweep's 38 % is a *selection* number, not the file's.** 4675 mutants cover all of `cli.py`,
  including the command bodies whose gates live in the other CLI test files; this sweep drives the
  help surface, so those mutants land in mutmut's `no tests` bucket (730) rather than counting as
  survivors. The card's own surface is fully triaged above. Recommendation: accept.
* **The release review's own commit left the rename gate red.** `tests/test_typed_gguf_surface.py::
  test_the_living_surface_carries_no_old_name` failed at `232a11b` (`REVIEW.md:145` spelled the old
  product name inside a backticked grep) — measured, not inferred: the baseline suite run shows
  exactly that one failure, and `git show 232a11b:REVIEW.md` carries the line unchanged. The tag
  would have shipped red; the line is reworded here (`git grep -in` *for* the old name) and the gate
  is green. Flagged because it is a review-file edit, not something the card listed.
* **N4's residual.** The operator e-mail is scrubbed in the file the review named
  (`.e2e/t_a696ce02-pid-pressure/rig/git.sh`). The same public address still appears *inside two
  frozen receipts* under `state/fights/e2-provenance/logs/` — they are captured transcripts, not
  prose, and this repo's rule is that receipts keep their history; editing them would falsify
  evidence. Left alone deliberately, recorded for the coordinator.
* **`models <sub> --help` prints no milestone line.** It bypasses `_command_usage` (pre-existing
  behaviour, unchanged here); the root help carries the milestone for the subcommand family. Harmless,
  but it is the reason the `'E1'` defaults are dead code (above).

## 📈 CONFIDENCE: 9/10

Increasing: every card criterion has a receipt from this session (probes, suite logs at both heads,
oracle, ledger, per-mutant sweep verdicts + hand replays); the two doc gates this card added fail
before the code fix and pass after; the whole-file sweep found no survivor a reachable input can
distinguish; the tree is smaller with no new dangling citation.
Decreasing: the sweep's selection leaves `cli.py`'s command bodies in `no tests` (their gates are
other files, not re-run here); the `.t07b5/check_citations.py` public-docs checker still reports its
two pre-existing false positives (`tools/call`, `tools/list` in SPEC §2.9 read as repo paths — present
at `232a11b` too, untouched by this card).

Evidence directory: `/work/t_a25bd190-ev/` (help probes before/after, suite logs for both heads,
citation ledgers, the sweep log + results + per-mutant diffs, driver scripts).
