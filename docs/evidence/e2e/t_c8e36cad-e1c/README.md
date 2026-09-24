# E1c raw evidence — card `t_c8e36cad` (resolver + fit)

Head under test: the E1c commits on `main` (this repo has no remote; the tree is shared). Pre-E1c
baseline for A/B checks: `9f53315`. Final head of this card: `d3dd0d2` (implementation `324324c`
+ mutation pins `d3dd0d2`). Reproduce with:

```bash
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu
uv run pytest -q                                              # canonical offline gate
uv run pytest -q --run-network tests/test_fit_live.py tests/test_templates.py \
    tests/test_engine_fork.py tests/test_ctypes_binding.py tests/test_cli.py
uv run python tools/e1c_offline_gate.py                       # A-E1c-10 (network disabled)
uv run python tools/e1c_offline_gate.py --run-network         # A-E1c-10 with the live assets
uv run python tools/e1c_e2e.py                                # A-E1c-8 → docs/evidence/e1c_e2e.json
# line coverage of the E1c surface with the selection + the pins:
COVERAGE_FILE=/tmp/.cov_e1c uv run --with coverage coverage run -m pytest -q \
    tests/test_templates.py tests/test_fit.py tests/test_cli_e1c.py \
    tests/test_engine_fork.py tests/test_cli.py tests/test_e1c_mutation_pins.py
COVERAGE_FILE=/tmp/.cov_e1c uv run --with coverage coverage report \
    --include='*/engine/template.py,*/runtime/fit.py,*/engine/decide.py,*/engine/prompt.py,*/schema.py'
# mutation sweep (r2 = the tree with the pins; resumable: see logs/mutation_retry.sh):
bash .e2e/t_c8e36cad-e1c/logs/mutation_retry.sh 3
uv run python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py status
uv run python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py kinds
uv run python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py digests <function-substring> 5
# per-key replay of a verdict (harness proof; the key must be copied from a .meta, never retyped):
bash .e2e/t_c8e36cad-e1c/logs/replay_all.sh
```

| file | what it is |
|---|---|
| `logs/e2e_run1.log` | A-E1c-8 stdout: 4 example sets end to end on the pinned model (answers, timings, waves/forks); machine-readable twin: `docs/evidence/e1c_e2e.json` |
| `logs/live_all_1.log` | the `--run-network` suite at `324324c`: 105 passed in 302.86 s (fit/templates/engine fork/ctypes/CLI) |
| `logs/live_all_2.log` | the same suite re-run at the final head `d3dd0d2` (Phase-5 fresh verify) |
| `logs/offline_gate_run1.log` | A-E1c-10, network disabled, before the pins: 142 passed, 21 skipped |
| `logs/offline_gate_run2.log` | A-E1c-10 at the final head (pins included): 218 passed, 21 skipped, exit 0 |
| `logs/offline_gate_live.log` | A-E1c-10 with `--run-network` (live assets, network still disabled) |
| `logs/cov_run1.log` | the old pre-pin selection run (`template.py` 81 %, `fit.py` 90 % line); the pins ran it up to 84 % / 97 % (see the evidence file §6.3) |
| `logs/tpl_run2.log` | `tests/test_templates.py` offline (30 passed) at the resolver's final shape |
| `logs/fit_run2.log` | `tests/test_fit.py` offline (32 passed) |
| `logs/mutation_r2.log` | the r2 sweep at `d3dd0d2`: 3674 mutants, mutmut's own summary line + the retry wrapper's bookkeeping (attempt 1/8 hit `BlockingIOError` on `os.fork()`; the retry finished, 4.98 mutations/s) |
| `logs/mutation_retry.sh` | the resumable driver (retries only on `BlockingIOError`; the pids cap is shared with sibling workloads) |
| `logs/mutation_report_template.txt`, `logs/mutation_report_fit.txt` | the *r1* per-file reports (the survivor list the pins were written from) — kept, because that is the state the numbers in the evidence file refer to |
| `logs/triage_kinds.txt`, `logs/triage_kinds_r2.txt` | survivor classes (mutation operator families) for r1 / r2 |
| `logs/triage_classify.txt`, `logs/triage_classify_r2.txt` | survivors grouped by enclosing function, r1 / r2 |
| `logs/mutation_triage.py` | the triage tool: `status` (per-file killed/survived/no-tests/timeout), `tests <fn>` (the mutmut test→function map), `kinds`, `classify`, `digests <fn>`, `source <fn>`, `affected <test>`, `verdicts <file>` |
| `logs/replay_all.sh`, `logs/replay_keys_all.txt` | the per-key replay harness (`MUTANT_UNDER_TEST=<key>` against the final tests) |
| `logs/replay_all_r2.txt` | its result: 32 r1-survivor keys → 27 killed, 5 survived; agrees with the r2 `.meta` on all 32 |
| `logs/replay_pins.sh`, `logs/replay_keys.txt` | the first replay pass (pin file only); its two false kills on the RSS keys are what exposed the flaky assertion — see the evidence file §6.2 |

Report with the gate table and the receipts: `docs/evidence/e1c_t_c8e36cad_resolver_fit.md`.
QA / ship-or-fix decision: `.gauntlet/e1c-resolver-fit.qa.md`.
