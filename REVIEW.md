# typed-gguf v0.1.0 — release-gate review (card `t_41c4fdda`)

**Verdict: APPROVE — v0.1.0 is releasable.** Every gate the release names was re-run by execution
at the reviewed head, the document set says the same thing as the code that ships, and nothing was
found that must be fixed before the tag. The findings below are 🟡 minor / ⚪ nit; none of them
changes a shipped behaviour, a published number, or a claim the README makes.

- **Reviewed head: `cb79e92`** (`main`, shared checkout; `cb79e92887415ca868e154315eca7b8291fd6c3f`,
  2026-09-20 13:40:45 +0000). **The tree moved during this review**: `296c498` — the head the public
  docs pass published — was the checkout when I started; card `t_2b89cce2` landed `cb79e92` while I
  was running the first gate battery. `git diff --stat 296c498 cb79e92` is
  `tests/test_typed_gguf_surface.py | 81 ++++…`, "1 file changed, 66 insertions(+), 15 deletions(-)",
  and **no `src/` line**: the rename
  gate's worktree/`TYPED_GGUF_HOME` false-failures, fixed test-side. At `296c498` I measured the
  suite 1361 passed / 49 skipped, oracle `failures: 0 skips: 0` (live shape), ruff clean; every
  gate below was then **re-run at `cb79e92`**, and the live-engine evidence carries because the
  delta touches no engine file.
- **Deliverable path.** The repo's convention (`git log -- REVIEW.md`) is that each review overwrites
  `REVIEW.md` and git keeps the predecessors, so this release review is filed there. The review it
  replaces is the E3 completion review at `d0f4932` (its round 1 at `9ec0c0e`, the E1a review at
  `05f3ee4`).
- **Environment.** podman container (1 CPU-second/s, 5 GiB memory cap — the box the receipts
  describe), scratch env `/tmp/rvgg/venv` built with `uv sync --frozen --extra dev` on CPython 3.11,
  pytest 9.1.1, ruff 0.16.8. The live half ran against the pinned bundle and the pinned 4B from the
  docs pass's isolated data home (`.t07b5/home`), with the pinned `Spark-X2.5-4B-Q8_0.gguf`
  reachable at `~/.hermes/models/` so the oracle's model pins execute. `/dev/dri` is present in
  this container, but the Vulkan backend reports *no devices* here (the pinned bundle loads and the
  engine falls back to CPU with `W_BACKEND_MISMATCH`), so live numbers here are CPU-computed — the
  published Vulkan rows are the operator-host ones, as the tables say.

## The checklist (release gate item 3)

| # | item | status | evidence |
|---|---|---|---|
| 1 | LICENSE (MIT) + credits | ✅ | `LICENSE:1` "MIT License", `Copyright (c) 2026 typed-gguf contributors`; `pyproject.toml` `license = {text = "MIT"}` + the MIT classifier; credits in `README.md` → *Credits and attribution* (llama.cpp MIT, pinned `b11026`; TypeSafe **no parity claim** + the documented outlier reproduced in the oracle; the "System One" prior art; the default model's Apache-2.0, pinned by size + SHA-256). The release notes repeat the MIT + no-parity + not-shipped statements, and `tests/test_public_docs.py` fails if they go |
| 2 | pyproject metadata | ✅ | `name = "typed-gguf"`, `version = "0.1.0"`, description, `readme = "README.md"`, `requires-python = ">=3.11"`, 5 classifiers (MIT, 3-only, 3.11, 3.12, SciEng-AI), console script `typed-gguf = typed_gguf.cli:run`, `Homepage = https://github.com/Rybens92/typed-gguf` |
| 3 | CI workflow sane (offline, no model downloads) | ✅ | `.github/workflows/ci.yml` is the public gate: `uv sync --extra dev`, `ruff check src tests tools docs .github`, an **empty-file** bundle stub (no download), the offline suite (`TYPED_GGUF_TEST_BLOCK_NET=1`, network replaced at the Python level), the red path (`-m "model or network"` must skip, never fail), the oracle's bundle-free shape; 3.11 + 3.12, `permissions: contents: read`, 15-min timeout. `runtime-matrix.yml` (the live acceptance harness, pinned bundle + one 0.5 GB GGUF) is `workflow_dispatch` + weekly and is deliberately not part of the CI trigger surface. `wheels-fallback.yml` is dispatch-only scaffolding — see N2 |
| 4 | no secrets in tracked files | ✅ | git-grep for AWS keys, private keys, `ghp_`/`github_pat_`/`hf_`/`sk-`/`xox*`/`AIza` tokens over the whole tracked tree: **0 hits**; no `.env`, `.pem`, `.key` or `id_rsa` tracked |
| 5 | hygiene applied | ✅ | `git status --porcelain -uall` empty at the head (before this file's own commit); ignored-only entries 6946. `.gitignore` carries the dev-run scratch rules, the live/`.t07b5` scratch (`.t07b5/home/`, `.t07b5/venv/`), the local coordination thread (`state/groupchat`, never staged) and the quick-bench default report. The acceptance card's 4.3 GB scratch concern is **closed**: it is `.t07b5/` (4.3 GB — the docs pass's isolated data home with the pulled model + pinned bundle, plus its throwaway venv), and `.gitignore:82–83` ignore `.t07b5/home/` and `.t07b5/venv/` while the small receipts stay pinned; `git check-ignore -v .t07b5/home/models/Spark-X2.5-4B-Q8_0.gguf` → `.gitignore:82:.t07b5/home/`. Cited-but-untracked is **0 in a clean clone** (239 resolved / 16 missing, all pre-existing or the one intentional frozen-evidence pointer) |
| 6 | fresh-install acceptance quoted | ✅ | The acceptance is card `t_ea16db9e` + `/tmp/tg-acceptance/ACCEPTANCE_REPORT.md` (outside the repo): pristine clone → fresh venv → project-only install → `init` with 15 compiler shims (**0 shim invocations**) → `doctor` → `ask` on the 4B with no policy flags, 8 runs one prompt digest, the quick bench 7.1 s. The repo-side equivalent is committed: `.t07b5/logs/{init,pull,ask_clean,run_1,run_2_reuse,run_typesafe,pip_install,suite_after}.{txt,json}` and the README quotes the run it ships |
| 7 | `--help` surface self-consistent | ⚠️ two cosmetic defects | every command and subcommand answers `--help` with exit 0, `serve`/`mcp` exit 3 with the milestone pointer, unknown commands exit 2 (probed). Two cosmetic drifts: the root help calls `serve`/`mcp` *"(implemented in E1b)"* (F1) and `models <sub> --help` duplicates the subcommand name (F2) |

## Gates, executed (release gate item 1)

| gate | command (as the repo documents it) | result at `cb79e92` |
|---|---|---|
| full offline suite (CI shape) | `TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_BENCH_RUNTIME_DIR=<empty stub> pytest -q -rs --timeout=120` | **1362 passed, 49 skipped, exit 0** in the working tree (5:57, contended) and in a **clean clone** of the head (29.4 s) |
| the same, ambient-runtime shape | as above, with a bundle discoverable | 1361/49 at `296c498`; at `cb79e92` the tree run is 1362/49 and the fix card's 1363/48 is the same tree **plus** the unmarked env-conditional oracle test (`tests/test_runtime_contract.py`), which runs only when a runtime is discoverable — same reconciliation the E3 review recorded |
| oracle, live shape | `TYPED_GGUF_RUNTIME_DIR=<b11026 bundle> python docs/verify_runtime_contract.py` | **failures: 0  skips: 0** — the reference table executed end to end (the release/asset pins, the `llama.h` header pins, the model's local header pins, the no-training-dependency list, the readout mirrors `restricted_softmax` / `confidence_normalized_peak` / `score_weighted_mean`, the math table incl. `recommend_quant`) |
| oracle, bundle-free shape | `python docs/verify_runtime_contract.py` (system python, stdlib-only — the README's own line) | **failures: 0  skips: 1** with the pinned model reachable (the skip is `no runtime installed`); **2 skips** when the model is absent too — the CI's documented shape |
| ruff | `ruff check src tests tools docs .github` | **All checks passed!** (the CI scope; a bare `ruff check .` still reads the committed dev-run receipts, which the CI comment explains) |
| E3e gate files | `pytest -q tests/test_e3e_roles.py tests/test_e3e_role_tool.py tests/test_e3e_docs.py tests/test_e3e_roles_decision.py` | **76 passed, 3 skipped** (all three skips are `test_e3e_roles.py:814`'s live gate) |
| policy v2 / parity pins | `pytest -q tests/test_policy_v2.py` | **13 passed, 1 skipped** (the live 4B half), incl. the **default-parity byte pin** through both assemblies and the pre-v2 byte freeze |
| bench↔product prompt parity | `pytest -q tests/test_bench_prompt_parity.py` (with the CI's empty-file bundle stub) | **2 passed**; without the stub it reproduces the CI's documented red path (1 failed) — the stub exists for exactly that test |
| doc gates | `pytest -q -k doc` | **86 passed** |
| red path | `pytest -q -m "model or network"` with `TYPED_GGUF_RUNTIME_DIR` / `TYPED_GGUF_BENCH_RUNTIME_DIR` unset | **48 skipped, 1362 deselected, 0 failed** — the live gates skip, never fail |
| citations | the hygiene card's checker over a clean clone of the head this review is committed on | **239 resolved / 0 cited-but-untracked / 16 missing** (15 pre-existing receipts + the one intentional dangling pointer, below; in the shared working tree the same ledger shows 3 extra *exists-but-untracked* hits — the ignored local scratch dirs, mutmut trees and gauntlet QA notes, which never ship) |

**The three handoff probes, all re-derived by execution:**

1. **The no-flags request *is* the flagged v2 cell.** Live on the pinned 4B: `pytest --run-network
   tests/test_policy_v2.py -k test_the_defaults_measure_the_published_v2_cell_on_the_4b`
   → **1 passed in 134.81 s** (prefix_tokens, decision, correctness and cue verdict identical item
   for item against the published `.e3e/bench_json_instructed_role_split.json` arm). I also ran the
   quickstart's own `ask` (its state and question set) twice — once with no policy flag, once with
   `--cue json_instructed --chat-format role_split --json-contract question`: the two responses are
   **byte-identical apart from `timings` and one `engine.fit.budget_bytes`** (5 907 677 184 vs
   5 904 531 456 — the plan re-reads free device memory, exactly as the README's reuse note says).
   The `usage` block is identical to the README's quoted block field for field (`input_tokens 255 ·
   output_tokens 10 · questions 3 · forks 8 · prefill_tokens 104 · decode_steps 10 · waves 4`),
   `engine.chat_format` reproduces the quoted `{role_split, user, question, prefix_chars 507,
   dropped "\n"}`, and the per-answer `decode_steps`/cue tokens line up too (79 / 29-30 / 1643). The
   only divergence from the quoted numbers is in the third decimal of the answers themselves
   (`area` 0.7338/0.1758/0.0904 vs 0.73758/0.171766/0.0906544): this container's Vulkan backend
   reports **no devices**, so the run is CPU-computed (`effective_backend: "cpu"`,
   `W_BACKEND_MISMATCH`) where the quoted block is the [host] Vulkan one (`effective_backend:
   "vulkan"`, `warnings: []`).
2. **The pre-v2 cell still collapses on the same state.** The same `ask`, once with the pre-v2 cell's
   flags (`--cue shipped --chat-format answer_sheet`): exit 0, but `area` reads `coverage 0.00415`
   → `reliability low_mass`, `page` answers `noul 0.8617` (`low_mass`, and the wrong direction — the
   v2 run says 0.0045), warnings `[W_BACKEND_MISMATCH, W_LOW_MASS]`, and `input_tokens` 200 (the
   prompt bytes differ). The readout is on a tail, which is the §2.2/§7.4.1 collapse the release
   notes quote at full size (Tiel 22/60, Occamy 26/60).
3. **The fit-plan cache only ever shrinks — confirmed** (see F3). Replaying the real cached plan
   through the real `replan_for_host`: cached `ngl 36 / budget 5634 MiB` → busy device (1500 MiB
   free) degrades it to `ngl 0` → the *same free* device (7000 MiB) leaves it at **`ngl 0`** (budget
   refreshed to 5976 MiB) while the original plan at that budget stays `ngl 36`. The cache never
   walks back up.

**Not re-run, and why** (named so the gap is visible): the [host] rows (RTX 3060 Ti Vulkan tables,
the Tiel/Occamy E3e cells) need the operator host — their receipts and the [host] tags are the
evidence; the two 35B GGUFs (20.8 and 23 GB) do not fit this container's 5 GiB cgroup; anything that
downloads from HuggingFace is outside the offline shape the release gate runs in.

## Document-set consistency (release gate item 2)

- **The v2 defaults are stated the same way everywhere they are stated.** `README.md` (quickstart
  prose + the quoted response + the interfaces table), `docs/BENCHMARKS.md` (the policy blockquote,
  §2.3, §7.4.2, §9), `docs/TEMPLATES.md` (§4/§4.3 + the E3e section) and
  `docs/RELEASE_NOTES_v0.1.0.md` all name `cue=json_instructed · chat_format=role_split ·
  json_contract=question`, and two pinned tests enforce it (`test_public_docs.py` reads the defaults
  out of `schema.Options()` instead of a literal; `test_policy_v2.py` checks the two documents by
  name).
- **The comparability rule holds.** Every pre-v2 row sits under a marker: §2.2 (pre-fix *and*
  pre-v2), §3 whole, §5 whole, §6 whole, §7 whole, §8 whole, and the pre-v2 switches appear in the
  public pair only as flags/cells/policy mentions (the docs gate's grep is clean). The v2 rows (§2.3,
  §7.4.2, the Occamy pair) each say which cell they are and refuse the comparison with the pre-v2
  neighbour. The one frozen-block sentence that reads "the default stays `shipped`" (§7.4.1) is not an
  exception: §8's marker quotes it and explains it (N1).
- **Every README claim is measured or tagged.** Spot-checked against the repo: the quickstart's
  `init`/`pull` transcripts (`.t07b5/logs/init.txt`, `pull.txt`), the response block (re-derived
  live above, usage identical), the reuse note (`run_1.json` / `run_2_reuse.json`), the TypeSafe
  reduction (`run_typesafe.json`), the family/template matrix (TEMPLATES §4 + the E3e role-render
  table), the fit ±20 % cross-check (TEMPLATES §5, `test_fit_live.py`), the limitations (§below),
  the credits. `SPEC.md`'s tag legend is intact and `[UNVERIFIED]` is genuinely unused (the only two
  occurrences are the legend lines — the README's claim about it is true).

## Known limitations (release gate item 4)

The README's *Limitations and known issues* matches reality on every line I could execute: sequential
questions (measured here: `usage.waves`/`decode_steps` are per request, the tables' own accounting),
the pre-v2 latency/throughput/determinism/calibration tables (marked, never mixed), `serve`/`mcp`
shipped as stubs (`cli.main(["serve"]) == 3`), prompt-level thinking suppression, the own-dev-set
provenance, box physics for big models, no CUDA row, no TypeSafe parity. Two additions it does not
carry today: the shrinking fit-plan cache (F3) and — if the coordinator wants it public — the
`--help` annotation drift (F1/F2). The one `[UNVERIFIED]`-class gap is not in the README at all:
`SPEC.md` §2.5/§2.8 predate the `cue`/`chat_format`/`json_contract` options (F4).

## Leftovers (release gate item 5) — listed, not fixed

- **Receipt density.** 1737 tracked files: 1058 under the E-run dirs (`.e2e/`, `.e3*/`, `.t*`), 246 in
  `docs/evidence/`, 61 in `state/` — **~79 % of the tree is the verification story**. The hygiene
  card's decision (keep the cited receipts, drop the scratch) is what makes the citations resolve;
  the slim-down is a coordinator call, not a release defect.
- **`REVIEW.md` lineage.** This file replaces the E3 completion review (`d0f4932`); its predecessors
  live in git. `REVIEW.md` is referenced by one frozen evidence doc (`docs/evidence/e1a_t_83ee1eed_survivor_pins.md`).
- **the `.t07b5/`, `.t5b75/`, `.t9bcb/` tool scripts** are committed on purpose (receipt tooling the
  cards cite); they are outside the lint scope by the CI's own rule.
- **TODO/FIXME audit:** exactly one hit in the whole living surface —
  `.github/workflows/wheels-fallback.yml:35` (`TODO(E1a): checkout the pinned llama.cpp commit…`,
  N2). No dead code found in `src/typed_gguf/` (ruff + the suite + the oracle pins all pass; the
  unused-import/no-op gates are green).
- **Stale references:** the old product name survives in exactly the three deliberate lines
  (`README.md:3`, `SPEC.md:3`, `SPEC.md:6`) and inside frozen receipts by design
  (`git grep -in` for the old name over the living surface). The `SPEC.md → state/groupchat/…` dangling
  pointer the hygiene card handed over is **closed**: only the frozen
  `docs/evidence/t_a696ce02_pid_pressure_gate.md` still names it, which the hygiene card classified
  as the one intentional miss.
- **The publish step.** `git remote -v` is empty; `https://github.com/Rybens92/typed-gguf` answers
  **404** and PyPI has no `typed-gguf` project (JSON API 404) — so the README/notes install lines
  (`git clone …`) are the *post-publish* form, exactly as the release-notes draft's header says. This
  is a publish-step dependency, not a code defect; the tag/release/PyPI steps remain the operator's.

## Findings

**🟡 F1 — the root `--help` marks the two unimplemented commands as implemented.**
`typed-gguf --help` prints `serve (implemented in E1b)` and `mcp (implemented in E1b)`
(`src/typed_gguf/cli.py:118`, fed by `MILESTONES`, `cli.py:37–39`, whose comment reads
"command → milestone that implements it"). Both commands exit 3 and their own message says
`'serve' is not implemented yet (milestone E1b)`; `README.md:224` and the release notes say they are
*specified, not shipped*. Suggested fix: drop "implemented" from that line (e.g. name the milestone
without the verb, or add "(not yet implemented)"), or move `serve`/`mcp` to a "specified, not
implemented" line of the usage block. Cheap, and it is the one public surface where the
honesty claim is currently contradicted by the tool itself.

**🟡 F2 — `models … --help` duplicates the subcommand name.** `typed-gguf models search --help`
prints `usage: typed-gguf models search search <query>` (same for pull/use/ls/rm/verify/
recommend-quant): `cli.py:1625–1626` prefixes `models <sub>` *and* prints the `COMMAND_HELP["models"]`
entry, which already begins with the subcommand name. Suggested fix: strip the leading `wanted`
token from the matched entry before printing. Cosmetic; exit code and behaviour are right.

**🟡 F3 — the fit-plan cache only ever shrinks (verified above).** `runtime/fit.py::replan_for_host`
re-checks a *cached* plan against free device memory and walks it down the ladder; nothing walks it
back up, so a plan degraded for one busy run stays degraded after the device frees up (only
`budget_bytes` refreshes). Impact: placement/perf only — a user who first ran on a busy GPU keeps
fewer offloaded layers until `fit --no-cache` / `--no-fit-cache` (or deleting `$TYPED_GGUF_HOME/fit/`)
— the decisions are unaffected, and it is why the docs cards' runs were conservative. Recommendation:
one line in the README's limitations ("a cached plan is never re-expanded; drop the cache with
`--no-fit-cache` when free memory returns") **or** a small follow-up card that lets a cached plan be
re-derived upward. Not a release blocker either way, but the limitations section is the checklist's
own lens for exactly this class.

**🟡 F4 — `SPEC.md` §2.5/§2.8 predate the prompt-policy options.** The frozen request schema's
`options` block and the CLI-surface block do not list `cue`, `chat_format`, `json_contract` (or
`thinking`), and its error catalog predates `E_ROLE_SPLIT_UNSUPPORTED` / `E_BENCH_*`; the code and
the public docs do carry them, and no gate reads SPEC's option list (`tests/test_scaffold.py` asserts
SPEC's *existence* plus the module imports, the frozen command sets and two error-catalog entries —
never the option list). A reader who goes to SPEC for the wire schema —
which the README calls "the contract" — cannot find the switches the defaults are made of.
Recommendation: a docs-only SPEC touch-up (add the three options with their defaults + the missing
error names) in whichever card owns the next docs pass; it changes no behaviour.

**⚪ N1 — `docs/BENCHMARKS.md:1145` (frozen §7.4.1 block) still says "The default stays `shipped`".**
Checked and **reconciled, not stale**: §8's pre-v2 marker (`docs/BENCHMARKS.md:1276`) quotes that
exact sentence and explains it as the state when the row was written, and §7's own header marks the
section pre-v2 — so the one surviving "default stays `shipped`" string in the docs set is already
declared. No action; recorded so the next reader of §7.4.1 does not have to re-derive that.

**⚪ N2 — `.github/workflows/wheels-fallback.yml` is a stub.** Dispatch-only; its build step is
`echo "TODO(E1a): …"` plus a `pip install` wrapped in `|| true`, and it uploads `dist/*.whl` which
nothing ever produces (`if-no-files-found: warn`, so a dispatch silently "succeeds"). Either
implement it or drop it before the repo goes public — a workflow named *wheels-fallback* that cannot
produce a wheel invites the wrong trust.

**⚪ N3 — the largest tracked file was an agent-session transcript.**
The E2 provenance fight's session export (2 787 035 B — the biggest single item in the 27 MB tracked
tree, embedding worker prompts rather than measurements) is **gone from the public tree**: the
slim-down card `t_a25bd190` removed it before the tag and recorded the removal where it was cited —
`state/fights/e2-provenance/scorecard.md` — where the measurements it accompanied stay.

**⚪ N4 — one receipt carries the operator's e-mail.** `.e2e/t_a696ce02-pid-pressure/rig/git.sh:11`
sets `user.email=rybens92@gmail.com`. The identity is already public as the repo author's commit
e-mail; noted only because a public-repo sweep may want it scrubbed.

## Blocking issues

**None.** No 🔴 CRITICAL and no 🟠 MAJOR finding. The only checklist item with a defect is #7, the
`--help` self-consistency check (F1/F2 — cosmetic strings); F3/F4 are a limitations line and a SPEC
section, each a one-to-few line change, and none of them touches an answer, a published number, or
the install path.

## Verdict

**v0.1.0 is releasable.** At `cb79e92`: the offline suite is green in a clean clone (1362 passed /
49 skipped), the oracle is `failures: 0 skips: 0` against the pinned live bundle and model and
`failures: 0` in its bundle-free shape, ruff is clean on the repo's gate, the E3e/policy/parity/doc
gates are green, the citation ledger has zero cited-but-untracked entries, the live 4B gate
reproduces the published v2 arm item for item while the pre-v2 cell still collapses on the same
state, the defaults/limitations/credits/install story in the README matches what the code does, and
nothing in the tree is a secret. The findings above are worth a follow-up docs/help pass before or
right after the tag — the tag itself is not blocked by any of them.
