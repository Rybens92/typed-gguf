# t_5f9c15fe — `ggufone` -> `typed-gguf`: what was run, what it said

> Receipt dir for the rename card. `receipts.md` is the write-up; the scripts beside it are the
> machinery, kept verbatim (their absolute `/workspace/ggufone` + `/work/t5f9` paths are the
> container they ran in): `sweep.py` (the ordered rules + the frozen-path list), `repair.py` (the
> 15 post-sweep module-name repairs), `rewrap.py` (the 23 re-wraps), `codecheck.py` (the
> "hyphen used as code" guard), `equivalence.py` + `rename_equivalence.txt` (103/154 byte-identical
> after replaying the rules), `compare_meta.py` (the mutant-for-mutant mutation comparison),
> `ci_dryrun.sh` (the CI gate in a fresh clone), `mutmut_sweep.sh` (the Tier-M driver).
> Raw logs (~20 MB: RED receipt, suite runs, oracle, mutmut TUI) deliberately stay out of the repo;
> they live at `/work/t5f9/logs/` in the worker container. This dir is a receipt: the old name
> appears in it on purpose, and — like every other receipt dir (`.e3e/`, `.e2e/`, `.e3c_tiel/`,
> which carry 86 ruff findings between them) — it sits outside the lint gate
> (`ruff check src tests tools docs .github`), so these scripts are not lint-clean.

Commit: `3358c0f` on `main` (154 files changed, +1569 / -1371). Repo: `/workspace/ggufone`
(host: `/var/home/rybens/workspace/ggufone`). Test env for every line below:
`HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache GIT_CONFIG_GLOBAL=/root/.gitconfig`,
`uv run --extra dev` (Python 3.11).

## 1. TDD — RED first

* `tests/test_typed_gguf_surface.py` written *before* the sweep; RED receipt
  `/work/t5f9/logs/red_surface.txt`: **8 failed / 1 passed** (the one that passed is the
  frozen-receipts gate, which holds before and after).
* Same file after the sweep: 9 passed (inside the full-suite numbers below).

## 2. The rename itself

* `git mv src/ggufone src/typed_gguf` (43 modules), then a mechanical sweep of the living surface
  (`/work/t5f9/sweep.py`, ordered rules: `GGUFONE_` -> `TYPED_GGUF_`, `Ggufone*` -> `TypedGguf*`,
  `src/ggufone`, `share/ggufone`, `import/from/-m ggufone`, `ggufone.`/`_`/`/`/`-`, backticked and
  bare `ggufone` -> `typed-gguf`) and a repair pass (`/work/t5f9/repair.py`, 15 edits) that
  restored the *module* spelling where the literal is code or a path component.
* **Rename-equivalence audit** (`/work/t5f9/equivalence.py` -> `/work/t5f9/logs/equivalence.txt`):
  replaying the sweep rules on the HEAD blobs reproduces **103 of 154 files byte-for-byte**; the
  other 51 are listed there with their real edits — module-name repairs (`-m typed_gguf`,
  `src/typed_gguf`, the oracle's `getattr(typed_gguf, "__version__")`), ruff's import re-ordering,
  four re-wrapped >100-char lines, the README/SPEC "formerly" notes, the E3 region gate's
  normalization, and the new `tests/test_typed_gguf_surface.py`.

## 3. Receipts are records, not names

* `git diff --stat` over `docs/evidence .e2e .e3b .e3c* .e3d .e3e .gauntlet .t5b75 .t7c9 .t9bcb
  state mutants*`: **0 lines** — every dev-run dir and evidence file is byte-identical to `HEAD`
  (the ones that live in the repo; `/work/...` dev dirs are outside it).
* `tests/fixtures/typesafe_doc_captures.json:12` still reads "…ggufone makes no parity claim
  (SPEC 2.4) — our statistic is 0.760".

## 4. Grep proof (requirement 6)

`git grep -in 'ggufone'` over the living surface (excluding `docs/evidence`, `.e2e`, `.e3*`,
`.gauntlet`, `.t5b75`, `.t7c9`, `.t9bcb`, `state`, `mutants*`) returns exactly **3 lines**, each an
intentional "formerly" note: `README.md:3`, `SPEC.md:3`, `SPEC.md:6`. Nothing else in the living
surface carries the old name.

## 5. Gates

| gate | result |
| --- | --- |
| `ruff check src tests tools docs .github` | **clean** (was 53 errors right after the sweep: 29 import-order fixes by ruff itself, 24 re-wraps) |
| full suite (tree, migrated data home visible) | **1354 passed / 48 skipped** (pre-rename baseline 1345 / 48 + the 9 new pins) |
| fresh clone, CI shape (empty `HOME`, empty bundle stub) | ruff clean; **1353 passed / 49 skipped**; oracle `failures: 0  skips: 2`; red path `48 skipped, 1354 deselected` (live gates skip, never fail) |
| oracle, bundle visible | `failures: 0  skips: 0` |
| the four E3e gates | **76 passed / 3 skipped** |
| the doc gates (`test_e3`, `test_e3b_evidence`, `test_e3e_docs`, `test_templates`) | **80 passed / 4 skipped** |

## 6. Live smoke on the pinned 4B (requirement 5)

`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`, data home = the migrated
`/work/agent-home/.local/share/typed-gguf`:

* `typed-gguf --help` and `typed-gguf version` — new name, data home `…/.local/share/typed-gguf`.
* `typed-gguf init --backend vulkan --json` — `already_installed: true`, dir under the new home,
  **0.47 s, no download** (the extracted bundle was adopted, not re-fetched).
* `typed-gguf doctor` — 34/34 symbols, backend vulkan driveable, warnings are the honest ones
  (no model in this registry, no sha record — the record was never written in this home).
* `typed-gguf ask --model …/Spark-X2.5-4B-Q8_0.gguf --backend vulkan --threads 4` with the **v2
  defaults** (no policy flags): source `cue: json_instructed`, `chat_format.kind: role_split`,
  answers identical to the published row, fit-cache schema `typed_gguf.fit/v1`.

## 7. External references (outside the repo)

Living, still carrying the old name — the sandbox mounts this tree read-only, so these are handed
to the operator:

* `/root/.hermes/skills/software-development/ggufone-bench-runs/SKILL.md` (host:
  `/home/rybens/.hermes/profiles/code-tdd/skills/…`): frontmatter `name:`/`description:` (lines
  2-3), header line 6, `GGUFONE_RUNTIME_DIR=` line 48, repo path line 52,
  `ggufone.runtime.pressure.spawn` line 199, `src/ggufone/bench/harness.py` line 264.
* `/root/.hermes/skills/tdd-gauntlet/references/shared-workspace-pipeline.md:531`:
  `ggufone.__file__` -> `typed_gguf.__file__`.

History, leave alone: `/root/.hermes/skills/.curator_ledger.jsonl` (19 hits),
`.usage.json` (1).

Not modified, and cannot be from here (read-only mount) — the two skill files above.

## 8. Host data home (the one external mutation)

The container cannot write `/var/home/rybens/.local/share/` (read-only mount). Operator commands:

```sh
mv /var/home/rybens/.local/share/ggufone /var/home/rybens/.local/share/typed-gguf
# old shells keep `GGUFONE_*`; the CLI now reads `TYPED_GGUF_*`:
grep -rn 'GGUFONE_' ~/.bashrc ~/.zshrc ~/.profile 2>/dev/null   # rename each to TYPED_GGUF_*
```

The in-container equivalent was done (`/work/agent-home/.local/share/ggufone` ->
`…/typed-gguf`) and is what section 6 exercised.

## 9. Tier-M mutation

`sh /work/t5f9/mutmut_sweep.sh 6` — mutmut 3.8, `--max-children 2`, on the pair the tree was
already configured for (`tools/e3e_roles_decision.py` + `tests/test_e3e_roles_decision.py`), so
the score is comparable with the pre-rename sweep the E3e F1/F4 card recorded (its receipt
`.e3e/logs/f1_f4_patches.txt` says "1941 mutants … 1412 killed / 529 survived = 72.7 %", the
killed column there folding in the no-tests bucket).

This run: **1941 mutants, 1133 killed, 279 no tests, 529 survived, 0 pending** — 3.3 min
(13:06:17 → 13:09:46), single attempt, pids peaked at 69/256, no `BlockingIOError`.

**The decisive comparison** (`/work/t5f9/compare_meta.py`, both `…e3e_roles_decision.py.meta`
files): the mutant **key sets are identical (1941 vs 1941, 0 only-in-old, 0 only-in-new)** and the
verdict histograms are identical — `{1: 1133, 0: 529, 33: 279}` before *and* after the rename,
with **zero** key-by-key verdict differences. So the rename produced the same mutant set and the
same verdict for every mutant; the score is unchanged, not merely reported.

