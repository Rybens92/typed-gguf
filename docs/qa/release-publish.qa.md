# Release mechanics — `publish.yml` (Trusted Publishing) + version 0.1.1 (card t_6027fb77) — QA Report

Date: 2026-09-22 · Tier **M** (the card declares none) · Decision: **ship** — every card gate is
measured on the committed tree; the only step left is the one the card reserves for
@bots-coordinator (push + the first Trusted-Publishing run as the v0.1.1 GitHub Release).

Commits on `main` (local; **not pushed** — the coordinator pushes):
`e0f1ed2` RED gate → `22af20c` publish.yml → `6061cb1` the 0.1.1 bump + the two gate-driven fixes.
Working tree clean at `6061cb1`, `main` = `origin/main` + 3.

## Summary — risk-weighted

🔴 REQUIRES ATTENTION — **none.** No engine line moved; the changed surface is one workflow, one
version string, one lock line, one notes file (renamed) and two gate files.

🟡 WORTH CONSIDERING (3, all recorded, none blocks)

* **The PyPI-side publisher configuration cannot be verified from here.** The workflow pins the four
  values PyPI matches on (file `publish.yml`, environment `pypi`, owner/repository via the project
  URL), but whether the owner's PyPI form carries the same four is a fact only the first real run
  proves. That run is the coordinator's step, and its failure mode is loud and harmless: a mismatch
  is a 403 with *nothing* published, and the version gate inside the job already refuses a tag that
  disagrees with the artifact. Detectability: HIGH.
* **A manual dispatch publishes whatever the chosen ref builds** (by design — there is no tag to
  compare against). A dispatch on a stale branch whose `pyproject` still says 0.1.0 gets a PyPI
  rejection (the file exists), not a wrong release. Accepted; the alternative (refusing all
  dispatches) would remove the re-run path the card asks for.
* **The `dev` extra gained PyYAML** (the card's own YAML gate is unrunnable without it: `uv run
  --extra dev python -c "import yaml"` answered `ModuleNotFoundError`). It is the *gate's* parser,
  never the tool's — `dependencies` stays `[]` — and it is reported here because it is a fourth
  edit beyond the three the card listed. Revert is one line in `pyproject.toml` (and the gate file
  would then fail to import, which is the signal that the gate needs a parser).

🟢 ACCEPTABLE (measured, no action)

* Card gates, verbatim: the YAML parses (`uv run --extra dev python -c "import yaml, pathlib; …"` →
  ok); `env -u PYTHONPATH uv run --extra dev pytest -q tests/test_public_docs.py` → **17 passed**
  with the new pin; the full offline suite in CI shape (net-blocked + bench bundle stub) →
  **1567 passed / 56 skipped / 0 failed**; `uv build` → `dist/typed_gguf-0.1.1-py3-none-any.whl`
  (298 790 B) + `dist/typed_gguf-0.1.1.tar.gz` (3 784 267 B), left in the ignored `dist/`.
* The workflow's own gate script, extracted from the committed YAML and run with `bash` against
  the **real** 0.1.1 artifacts in a clean `dist/`: tag `v0.1.1` → `version gate: v0.1.1 is the
  built 0.1.1` (rc 0); tag `v0.1.0` → `::error::built 0.1.1 is not the release tag v0.1.0 — refusing
  to publish` (rc 1); `refs/heads/main` → prints and continues (rc 0).
* Live selection `-m "model or network"` → 55 skipped / 0 failed (the live set never fails on a
  bundle-free box); oracle → `failures: 0 skips: 2` (both expected: no runtime installed, local
  Spark GGUF absent); `ruff check src tests tools docs .github` → clean. No type checker is
  configured in this repo (no mypy/pyright section in `pyproject.toml`).
* Typos/naming: `git show --name-status` detects the notes move as `R092` — a rename, so history
  follows the file; the `v0_1_0_t_*` receipt names inside the notes are historical and stay.

## Mutation note (Tier M, soft threshold) — the config artifact was probed instead

No test-killable production surface moved: `src/typed_gguf/__init__.py` changed one string
constant, and the rest is workflow/config/docs/gates — a mutmut sweep of `src/typed_gguf/keep`
(what `pyproject.toml` currently targets) would mutate nothing this card touched. So the artifact
*was* mutated by hand: five real regressions a later card could introduce, each run through the new
gate — drop `id-token: write`, rename the environment to `production`, add `--no-attestations`, add
a `UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}` env, break the tag case in the gate script.
**5/5 KILLED, 0 survivors, and `publish.yml` restored byte-identical** afterwards
(`/tmp/gate_bite.py`, run in this session).

## What could go wrong — the first run

⚡ SCENARIO: the first release run reaches PyPI and is refused.
  Probability: low-moderate — it depends on the owner-side form, not on this tree.
  Trigger: any of the four values differing (a differently-named environment, a typo'd workflow name).
  Impact: the GitHub Release exists with no artifact on PyPI; nothing is published, no version burned.
  Detectability: HIGH (the job fails with PyPI's own message; the log names the mismatch).
  Blast radius: one release run; re-dispatching after fixing the form is the intended path.

⚡ SCENARIO: a stale `dist/` in the runner makes "the built version" ambiguous.
  Probability: ~0 in CI (a fresh checkout has no `dist/`; it is git-ignored) — this is exactly what
  the hardening closes, and the local checkout that *did* carry the published 0.1.0 wheel is the
  proof the case is real.
  Impact if it slipped: the gate could compare the wrong name and let a wrong artifact through;
  now it exits 1 with `expected exactly one built wheel`.
  Detectability: HIGH. Blast radius: none (no upload happens).

## Confidence

📈 CONFIDENCE: **9/10**

Increasing: every card gate measured on the final commit, not on an intermediate tree; the version
gate's behaviour executed rather than read (4 python-level cases + 3 real-artifact cases + 5 killed
config regressions); the version pinned in all four places the tree carries it, with a gate between
each pair; the notes' one false claim found, measured and rewritten (`uvx --from
git+https://github.com/Rybens92/typed-gguf@v0.1.0 typed-gguf version` → `typed-gguf 0.1.0`, rc 0).

Decreasing: −1 — no OIDC publish has actually happened yet (that is the coordinator's first run, and
PyPI's side is outside this sandbox's reach); the `dev`-extra PyYAML is a scope addition, however
small and reported.

## Decision

Recommended: **Option A — ship.** Push `6061cb1`, then drive the first run (GitHub Release `v0.1.1`
on the tag; the workflow does the OIDC exchange). Accepted risks: the three 🟡 items above, all
loud-failing or one-line-revertible.
