# t_8cb0a05e — E1c FIX: `--fit` on a busy desktop (raw logs)

The readable version of all of this lives in `docs/evidence/e1c_t_8cb0a05e_fit_oom.md` (and the QA
note in `.gauntlet/e1c-fit-oom.qa.md`). This directory is the raw material, kept because every
claim in those files is checkable from here.

| file | what it is |
|---|---|
| `logs/round2.log` | the sandbox rehearsal of `tools/host_gate_e1c_fit.sh` (7 steps, exit code + wall time per step; the fake driver stands in for the GPU) |
| `logs/host_gate_e1c_fit.json` | the same run, folded into the boolean checks of the card's requirements (`is_host_run: false` — the rehearsal can never pass for a host run) |
| `logs/red_prefix.json` | `tools/fit_oom_red_probe.py` against a byte copy of the PRE-fix tree (`c095c51`): 7516192768 B budget, `E_MODEL_ARCH_UNSUPPORTED` on a failed allocation |
| `logs/fit_plan.out`, `logs/fit_target_bounded.out` | the `fit --json` tails for `--fit-target 1024` and `5200` (the free budget and the 0 MiB consequence) |
| `logs/run_busy_desktop.json`, `logs/run_no_fit.json` | the two real engine runs (pinned bundle + pinned model) on the busy-desktop plan and with `--no-fit` |
| `logs/fake_oom_degrade.json`, `logs/fake_oom_all_rungs.json` | the fake-allocation-failure fixture: degraded-to-CPU plus the `E_BACKEND_OOM` message |
| `logs/live_fix_tests.log` | the two live tests (`--run-network`): the log-capture ABI and the busy-desktop load |
| `logs/coverage.txt` | `pytest --cov=…` over the E1c selection (214 passed / 8 skipped; `fit.py` 96 %) |
| `logs/mutmut.out` | the Tier-M sweep (`fit.py`, 1639 mutants, 69.3 % killed/ran) |
| `logs/fit.py.meta`, `logs/session.py.meta` | mutmut's per-module `exit_code_by_key` artifacts — the scores were read from these, never from the tool's screen |
| `logs/after.txt` | the driver's free-memory reading after the runs |

The host run (the operator's box, the real Vulkan allocation failure) is still PENDING: this
container has no `/dev/dri`, no `/dev/nvidia*` and no driver, so the three GPU readings cannot be
produced here. `tools/host_gate_e1c_fit.sh` without `GGUFONE_GATE_FAKE_DRIVER` produces exactly
this directory with `is_host_run: true`.
