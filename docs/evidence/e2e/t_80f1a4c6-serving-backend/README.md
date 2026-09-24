# `.e2e/t_80f1a4c6-serving-backend/` — raw material for the E3 FIX card

Card: `t_80f1a4c6` — *the serving path's `engine.backend` says `cpu` while the Vulkan device
computed*. Published document: `docs/evidence/e3_fix_t_80f1a4c6_serving_backend.md`.

Everything here was produced on the operator host's container (GPU passed through; every GPU run
needs `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json`, and `GGUFONE_RUNTIME_DIR` points at
`/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` unless the entry says
otherwise). The work happened in the card's private clone `/work/t80serve` (scratch; the shared tree
is `/var/home/rybens/workspace/ggufone`), rebased twice onto the shared `main` and then landed there
as a fast-forward from `f9d9a08` to `f1b9272` (see the document's commit table); the box was busy
with sibling cards throughout (§7 of the document).

## Gates and tables (rebased tree — what the document quotes)

| file | what it is |
|---|---|
| `red_gates_landed.txt` | the same RED check on the **landed** parent tree (`f9d9a08`, the shared `main` the card fast-forwarded onto): **20 failed**, 1 skipped |
| `red_gates_rebased.txt` | `pytest -q tests/test_serving_attribution.py` in a detached worktree at the **parent** commit `a475090` with this card's gate file copied in: **20 failed**, 1 skipped |
| `green_gates_rebased.txt` | the same file on this card's tree: **20 passed**, 1 skipped (the live gate is `model`-marked) |
| `baseline_full_suite.txt` | the offline suite *before* this card's change (measured for the E3 card at `15e89a9`): 993 passed, 41 skipped |
| `green_full_suite_rebased.txt` | the offline suite on the rebased tree (clean env, no `GGUFONE_*`/`VK_*` exported): **1089 passed, 42 skipped** in 2 min 45 s |
| `ruff.txt` | `ruff check src tests tools` — `All checks passed!` |
| `coverage_modules_rebased.txt` | `coverage report` after the full offline suite (rebased tree) |
| `coverage_changed_statements_rebased.txt` | coverage of the *changed statements* only (`changed_lines_coverage.py`, base `a475090`): 60/60 = 100 % |
| `live_gate_rebased.txt` | the `model`-marked live gate (`-k live --run-network`, 4B model, pinned bundle, `VK_DRIVER_FILES`): **3 passed** in 2 min 33 s |
| `replay_survivors.txt` | the first pass's `device_evidence` survivors replayed against the gates, **control row included** (`tools/t80_replay.py`) |
| `replay_session_init.txt` | the second pass's `ModelSession.__init__` survivors replayed the same way (7/18/20/21 by the explicit-source gate, 47 by the sink gate) |
| `survivor_lines_session.txt` / `survivor_lines_decide.txt` | every survivor of the module diffed against the original, with the ones that mutate *this card's* lines named (`tools/t80_survivor_lines.py`) |
| `mutation_score.txt` / `mutation_survivors.txt` | the Tier-M mutmut sweep over `engine/session.py` + `engine/decide.py` (`mutants/` is gitignored) |
| `mutmut.out` / `mutmut.out.round1` / `sweep_driver.log` / `mutmut_sweep.sh` | the sweep's terminal output (the completed first pass is `.round1`), the driver and its outer retry wrapper (pid-cap waiting, `--max-children 2`) |
| `parent_flake_control.txt` | the **pre-existing** random-order flake: four full-suite runs on the parent tree (`a475090`, this card's gate file ignored) — one with 36 failures in the runtime-probe files, three that could not fork at the pid cap; plus the quiet-box control (the flaky pair passes 40/40 on both trees) |
| `t80_meta_none.py` / `t80_spans_debug.py` | the two diagnostics that found the span-key mismatch in `tools/mutation_span_check.py` (span keys are unqualified, meta keys are qualified — the repo tool reads every mutant as unrun) |

## The before/after pair (§4 of the document)

| file | what it is |
|---|---|
| `before_e3_batch_response.json` | the published E3 batch response (byte-identical copy of `docs/evidence/e3_batch.json`, sha256 `7b70fb1c…`) — the BEFORE row |
| `before_e3_batch_stderr.log` | the E3 batch's own stderr tail — the BEFORE `Vulkan0 compute buffer size` line |
| `after_occamy_row.json` | this card's AFTER row: `ggufone run`, one dev item (`c01`), Occamy 1.0, the pinned Vulkan bundle |
| `after_occamy_stderr.log` | that run's stderr (`time` output included) — the AFTER `Vulkan0 compute buffer size` line |
| `smoke_4b_request_claim.json` / `.err` | live 4B row, `--backend vulkan` (claim source `request`) |
| `smoke_4b_bundle_claim.json` / `.err` | live 4B row, no `--backend` (claim source `bundle`) |
| `smoke_4b_refuted_cpu_claim.json` / `.err` | live 4B row, `--backend cpu` → `W_BACKEND_MISMATCH` |

## Tools (committed under `tools/`)

| file | what it is |
|---|---|
| `before_after.py` (here) | generates `before_after.txt`, the side-by-side table the document quotes (`before_after.txt.prev` is the first attempt's copy, kept to show the regeneration is identical) |
| `show_fields.py` (here) | prints the card's fields out of one response JSON |
| `compare_rows.py` (here) | checks the AFTER row is the same item (`c01`) as a row of the BEFORE batch |
| `changed_lines_coverage.py` (here) | changed-statement coverage from `coverage json` + `git diff` |
| `tools/t80_census.py` | per-module killed/survived/no-tests/pending census of a mutmut run dir |
| `tools/t80_replay.py` | replays the sweep's survivors one by one, with the control row |
| `tools/t80_survivors.py` | dumps one symbol's survivors out of a run dir, diffed against the original |

## Kept from the first attempt (pre-rebase tree, `15e89a9`)

`red_gates.txt`, `green_gates.txt`, `green_full_suite.txt`, `coverage_modules.txt`,
`coverage_changed_statements.txt`, `live_gate.txt`, `baseline_full_suite.txt` — measured before
this card's clone was rebased onto `a475090`. They describe the same change on an older base; the
document quotes the `*_rebased` rows.

## Reproduce (from this card's clone)

```bash
cd /work/t80serve
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
.venv/bin/python -m pytest -q tests/test_serving_attribution.py                    # 18 offline gates
env GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan \
    VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json \
    .venv/bin/python -m pytest -q --run-network tests/test_serving_attribution.py -k live
sh .e2e/t_80f1a4c6-serving-backend/mutmut_sweep.sh 6                               # Tier M (retrying)
.venv/bin/python tools/mutation_score.py mutants                                   # the score
.venv/bin/python tools/t80_replay.py                                               # survivor replay
```
