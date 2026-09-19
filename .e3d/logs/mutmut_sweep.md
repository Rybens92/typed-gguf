# E3d — the Tier-M sweep of `engine/prompt.py` (card t_d90404ac)

Written **after** the fact from the run's own output: the driver's first version truncated
`.e3d/logs/mutmut.out` on every invocation, so the raw log of the run these numbers come from was
overwritten by the follow-up attempts (the truncation is fixed — the driver appends now). The
numbers below are the run's final progress line, quoted from the session record:

```
149/149  🎉 74 🫥 25 ⏰ 0 🤔 0 🙁 50  🔇 0  🧙 0        (17.87 mutations/second)
```

* **74 killed**, **50 survived**, **25 `no tests`**, 0 timeouts, 0 errors.
* The pair in `pyproject.toml` at that moment: `source_paths = ["src/ggufone/engine/prompt.py"]`
  with `pytest_add_cli_args_test_selection = [tests/test_e3d_cue_switch.py, tests/test_templates.py,
  tests/test_e1c_mutation_pins.py, tests/test_engine_fork.py]`.
* The `no tests` block is `empty_candidate_code` (its gates are the CLI suites); the survivors are
  the module's other branches (the `descriptions`/`readout` paths of `build_question`), which this
  selection does not drive.

## The widening that did not run

Adding `tests/test_e3b_labels.py` + `tests/test_scaffold.py` to reach those branches produced **no
verdicts at all**: 149 mutants, 0 killed, 50 survived, 0 `not checked` — i.e. the mutation phase
reported passing-without-failing instead of aborting, which means the tests no longer saw the
mutations. The trigger was environmental, not the selection:

1. `uv run --with mutmut` re-syncs the *project* venv. Without `--extra dev` it drops `pytest`, so
   mutmut's baseline run dies and every mutant reads `not checked` (this is what the first widening
   attempt produced).
2. After `uv sync --frozen --extra dev` restored `pytest`, the same 149 mutants all read `survived`.

The driver carries `--extra dev` now, and the numbers above are the only ones this card stands
behind. A later sweep should rebuild the venv the way the first run had it before widening the
selection (recorded in `pyproject.toml` next to the pair).

Reproduce: `sh .e3d/mutmut_sweep.sh 3` (logs to `.e3d/logs/mutmut.out`, appending).
