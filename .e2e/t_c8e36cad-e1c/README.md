# E1c raw evidence — card `t_c8e36cad` (resolver + fit)

Head under test: the E1c commits on `main` (this repo has no remote; the tree is shared). Pre-E1c
baseline for A/B checks: `9f53315`. Reproduce with:

```bash
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu
uv run pytest -q                                              # canonical offline gate
uv run pytest -q --run-network tests/test_fit_live.py tests/test_templates.py \
    tests/test_engine_fork.py tests/test_ctypes_binding.py tests/test_cli.py
uv run python tools/e1c_offline_gate.py                       # A-E1c-10 (network disabled)
uv run python tools/e1c_offline_gate.py --run-network         # A-E1c-10 with the live assets
uv run python tools/e1c_e2e.py                                # A-E1c-8 → docs/evidence/e1c_e2e.json
uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 3
uv run python .e2e/t_c8e36cad-e1c/logs/mutation-report.py src/ggufone/engine/template.py
uv run python .e2e/t_c8e36cad-e1c/logs/mutation-report.py src/ggufone/runtime/fit.py
```

| file | what it is |
|---|---|
| `logs/e2e_run1.log` | A-E1c-8 stdout: 4 example sets end to end on the pinned model (answers, timings, waves/forks); machine-readable twin: `docs/evidence/e1c_e2e.json` |
| `logs/live_all_1.log` | the `--run-network` suite at this head: 105 passed in 302.86 s (fit/templates/engine fork/ctypes/CLI) |
| `logs/offline_gate_run1.log` | A-E1c-10, network disabled: 142 passed, 21 skipped |
| `logs/offline_gate_live.log` | A-E1c-10 with `--run-network` (live assets, network still disabled) |
| `logs/cov_run1.log` | coverage of the E1c test selection (`template.py` 81 %, `fit.py` 90 % line) |
| `logs/tpl_run2.log` | `tests/test_templates.py` offline (30 passed) at the resolver's final shape |
| `logs/fit_run2.log` | `tests/test_fit.py` offline (32 passed) |
| `logs/mutation_report_template.txt`, `logs/mutation_report_fit.txt` | mutmut per-file report (killed/survived + survivor list) |
| `logs/mutation-report.py` | the script that derives those numbers from `mutants/**/*.meta` (E1c copy of the E1b one; pass the repo root as argv[2] when not running from the repo) |

Report with the gate table and the receipts: `docs/evidence/e1c_t_c8e36cad_resolver_fit.md`.
QA / ship-or-fix decision: `.gauntlet/e1c-resolver-fit.qa.md`.
