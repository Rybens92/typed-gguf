# `.e2e/t_83ee1eed-survivor-pins/` — raw logs for the four pinned survivors

Card `t_83ee1eed` (defender side of duel `t_0fc576df`). Report:
`docs/evidence/e1a_t_83ee1eed_survivor_pins.md`.

Fight inputs (read-only, not copied into the repo):
`/home/rybens/workspace/state/fights/e1a-t0fc576df/` — `patches/m08..m11.patch`,
`witnesses/w08..w11.py`, `MUTANTS.md`, `logs/`.

Fresh copies (container path `/workspace/tmp/e1a-fix-t83ee1eed/<copy>`, built by
`scripts/setup_mutant_copies.sh` from `git archive HEAD` = `1c20ad4`, patch applied per mutant):

| copy | patch sha256 | mutated module | first-16 sha256 | == MUTANTS.md |
|---|---|---|---|---|
| base | — (pristine HEAD) | — | — | — |
| m08 | 39a56a702590da7ef72e04be4fe7051a8e679c6d1869365e866bdb806e3df6de | `runtime/pins.py` | aea6c1c5c3aa63c8 | yes |
| m09 | 70330fe0c950ed8a65d65a2609fb798c29ddca81f6627b15aa2cedf652787daa | `runtime/pins.py` | 8729be0fee3423c6 | yes |
| m10 | cb99717b2cfaf4f13bac819af8ce30ffd54d7dadafdfc2d7afabe1cf72041a85 | `registry/recommend.py` | a2f45d53898dde9f | yes |
| m11 | 33664f93eddaca82571a2d851763035f32ea95dc5523805444c3508176458f3c | `runtime/capability.py` | 04d555f57161afd4 | yes |

Logs (runner: `uv run --project /workspace/ggufone pytest -q -p no:cacheprovider`, cwd = the copy):

| file | what |
|---|---|
| `<copy>.whoami.log` | the copy's own `src` was the import source (`tests/_whoami_test.py`) |
| `<copy>.gateA.log` | the named gate — `tests/test_host_purity.py` (updated), 16 tests |
| `<copy>.suite.log` | the full offline suite in the copy |
| `base.new_nodes.log` | the five new node ids on unmutated HEAD (`5 passed`) |
| `w11.witness.log` | the duel's `w11.py` on all five copies (base `[]` … m11 `['cpu']`) |
| `w10.witness.log` | the duel's `w10.py` on base/m10 — no delta in a GPU-less world (see report §3d) |
