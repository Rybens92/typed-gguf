# t_a696ce02 — pid-pressure gate — QA report (Tier M, condensed)

Date: 2026-09-19 · Tree: the shared `ggufone` main (`8474802` + this card) · Verdict: **ship**
(Option A), with two named follow-ups.

## Risk-weighted summary

🟢 ACCEPTABLE
* **The filed flake is explained and fixed at the right layer.** It is the container's *shared*
  pid cgroup (`pids.max=256`), not ordering, not a module cache: the probe child / `llama-cli` /
  oracle spawns were refused with `EAGAIN` and the refusal was rendered as a probe finding.
  Measured: green at `pids.current` 107-151, red at 244-254; per-test tracing shows our tests leak
  no children (the reading moves in sibling-sized jumps).
* **Transient pressure is absorbed** (`spawn` retry, 4 attempts / ≤1.4 s) and **sustained
  pressure is named** (`E_PID_PRESSURE` + `pids.current=254/256 (2 free)`); the contract reason
  codes (`loader_error` / `backend_absent` / `probe_failed`) are unchanged and now unreachable
  from a box that refused to fork.
* **A starved run cannot look green.** 25 measured `needs_fork` gates skip by name, a loud
  `PID PRESSURE` block prints, and the exit status is forced non-zero; one named
  `test_the_pid_cgroup_has_fork_headroom_for_the_probe_gates` replaces the 36 phantom failures.
* Gates: RED on the parent (3 failed / 10 passed; plus the filed flake reproduced under
  injection), 14/14 new gates green, full suite 1129 passed / 43 skipped on three random seeds,
  ruff clean, `pressure.py` 97 % covered, `isolated.py` 89 %, `capability.py` 91 %.
  Mutation: see below.

🟡 WORTH CONSIDERING
* **`PID_HEADROOM_FLOOR = 16` is a measured-band heuristic** (failures appear at ≤12 free pids).
  A cap that arrives *after* the reading can still bite — absorbed by the retry budget if short,
  named `E_PID_PRESSURE` if not. One constant to move; no code change.
* **The oracle's inner forks** are retried from the test side (the oracle is frozen and never
  edited). A fork refused inside its last attempt fails that gate with the fork-refusal text in
  the assertion — never as a contract failure, but it *is* a red gate on a capped box.
* **Mutation scope is the two runtime modules** (see the sweep table); `conftest.py` is test
  infrastructure and is pinned end-to-end by the nested-run gate instead.

🟢 FOLLOW-UPS (not this card)
* `pytest-randomly` is not in the locked `dev` extra (`uv.lock`), yet the repo's own
  `[tool.mutmut]` args and `tools/t80_replay.py` assume it — one pyproject line, deliberately not
  taken (`pyproject.toml` is a live sibling hotspot).
* F3/F4 from `t_80f1a4c6` (the `engine.fit.backend` fallback, the `serve`/`mcp` stubs).

## Decision matrix

**Option A — ship now.** ✅ The flake's mechanism is measured, the fix is layered (bounded retry →
named reason → suite gate), nothing was loosened, and the starved case is honest instead of
silent. ⚠️ Accepted: the floor heuristic above; a capped box needs a re-run (by design).
→ Risk: **LOW**.

**Option B — lower the floor / retry harder first (+~30 min).** ✅ Fewer skips on a merely busy
box. ⚠️ A lower floor buys skips back at the price of running fork gates inside the measured
failure band (more red, still named). → Not recommended: the current band sits *above* every
failure the card measured.

**Option C — also pin the order for those files (+~1 h).** ✅ Belt and braces. ⚠️ Fights
`pytest-randomly` globally and hides genuine order leaks. → Not recommended: the measurement
shows order is not the cause, only the messenger.

💡 **Recommendation: Option A.** The evidence separates "the box" from "the product" in one look
(the header line, the named skips, the one loud gate), and the retry budget removes the transient
half of the flake entirely.

## Confidence: 8/10

+ Reproduced on the real box (6 seeds, `pids.current` logged per run) *and* deterministically
  (injected `EAGAIN`, the exact kernel error).
+ RED demonstrated on the landed parent before the fix; GREEN on the fix; the `needs_fork` set is
  measured, not guessed.
+ New gates kill 14/14 in the gate file; the changed modules are 89-97 % covered.
+ Full suite green on three random seeds with the baseline skip count.
- The floor is a heuristic (the box can cap mid-run after a healthy reading).
- The frozen oracle's last-mile fork is retried but not made impossible.

## Metrics

| metric | before | after |
|---|---|---|
| the three probe files under pytest-randomly at a capped box | 2-6 mislabelled failures per run | 0 failures; named skips + a forced non-zero exit |
| reason strings under spawn refusal | `probe_failed` text that reads like a bundle finding | `E_PID_PRESSURE` + the live cgroup reading |
| full suite (deterministic) at a healthy reading | 1116 passed / 43 skipped (the unmarked parent) | 1129 passed / 43 skipped |
| full suite (random order, seeds 11-13) | not green (mislabelled) | 1129 passed / 43 skipped, exit 0 ×3 |
| `pressure.py` coverage | — (new) | 97 % |
| Tier-M sweep (`pressure.py` + `isolated.py`, gate selection, `max_children 2`) | — | 288 mutants, **76.4 %** killed/scored (188/246); `pressure.py` 93.4 % in the busy run / **80.3 %** in the quiet re-run — the box *inflates* kills, so the quiet number is the honest one |
| survivors on this card's lines | — | 9 × `spawn` (the retry schedule / the `last` init), 2 × `pressure_note`, 1 × `_read_int` (**equivalent** mutant) → **three gates added** to pin them; the other 54 are pre-existing `isolated.py` internals, named not chased |
| the confirming re-run after those gates | — | **not completed**: the box went to `pids.current=256/256` (no fork at all, `conmon` included). Recorded as not-run, never guessed — `rig/sweep_pressure2.sh` re-runs it on a quiet box |

## Landed

`f848872` → `9cc941b` → `e0070dd` → `e30b173` on the shared `main` (rebased in a private clone onto
`37fc224` first, fast-forwarded; the sibling WIP hashed byte-identical across the merge; nothing in
the card touches `pyproject.toml`, whose `[tool.mutmut]` block is a live sibling retarget).

Verified **on the landed tree**: 72 passed across the three gate files; starved simulation
`GGUFONE_TEST_PID_HEADROOM=250/256` → 5 named skips + the one named headroom failure + exit
non-zero; healthy simulation → 17 passed. The full suite is 1182 passed / 43 skipped with **5
failures in a sibling's in-flight file** (`tests/test_e3c_cue_refused.py`, E3c `t_6c119626`: its
`docs/TEMPLATES.md` content is not landed — content assertions, no fork involved, **not this
card**).
