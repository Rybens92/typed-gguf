# t_603a35a0 — E2 FIX: bench backend labels must follow the effective compute path (raw material)

The readable version lives in `docs/evidence/e2_fix_t_603a35a0_backend_attribution.md`. Every
claim there is checkable from here.

| file | what it is |
|---|---|
| `logs/red_pretest.txt` | the new gate file run against the **parent tree** (`1205c0b`) with the new parser module present and no row/report change: **8 failed, 4 passed** (the RED of card requirement 3) |
| `logs/red_test_file.py` | that gate file, verbatim from the RED commit (`086cd1e`) — the version whose failure is quoted |
| `logs/before_mixed.raw` | operator box, parent tree, CPU bundle + Vulkan bundle visible, `--backend all`: the `vulkan` row is 12.29 tok/s prefill (CPU-class, `runtime_dir` = the Vulkan bundle, no `Vulkan*` line anywhere in the process log), report `ok: true`, then `double free or corruption (!prev)` at teardown (exit 134) |
| `logs/after_mixed.raw` | the same command on this card's tree: the `vulkan` row carries `effective_backend: null`, `devices: []` and `warnings: ["W_BACKEND_MISMATCH"]`, the report is `ok: false` and the note says why (still the pre-existing teardown abort) |
| `logs/before_opoffload.raw` | operator box, parent tree, only the Vulkan bundle installed (`cpu` resolves to its directory): the row *labelled* `cpu` (`n_gpu_layers=0`, note "CPU only…") runs its graph on the device — the `~llama_context: Vulkan0 compute buffer size` lines prove it; report `ok: true`, exit 0 |
| `logs/after_opoffload.raw` | the same command on this card's tree: `effective_backend: "vulkan"`, `device_buffers {"Vulkan0": 3, "Vulkan_Host": 3}`, `W_BACKEND_MISMATCH`, report `ok: false`, **exit 1** |
| `logs/rows_before_after.txt` | the four rows side by side, straight from the reports (`compare_rows.py`) |
| `logs/rendered_after_mixed.md`, `logs/rendered_after_opoffload.md` | the markdown `render_report` produces for the two "after" runs (`render_report_from_raw.py`) |
| `logs/runs.md` | the exact commands, bundles, box caveats and exit codes of the four runs |
| `logs/t603-run.sh` | the driver that produced the two "before" runs (and the shape of the "after" ones) |
| `logs/green_attribution_tests.txt` | `pytest -q tests/test_bench_attribution.py` → 14 passed |
| `logs/green_full_suite.txt` | `uv run pytest -q` → 923 passed, 40 skipped |
| `logs/ruff.txt` | `ruff check src/ggufone tests` → `All checks passed!` |
| `logs/prefix_environmental_failure_control.txt` | the one full-suite failure seen on a **VRAM-starved** box (`tests/test_fit.py::test_the_cli_fit_command_returns_the_documented_json`), reproduced on the parent tree — environment-dependent, not this card's |
| `logs/coverage_changed.txt` | `coverage run -m pytest` + report over the changed modules (devices.py 100 %, harness 93 %, suites 99 %, errors 100 %) |
| `logs/mutmut.out`, `logs/devices.py.meta`, `logs/mutation_score.txt` | the Tier-M sweep of `runtime/devices.py` (81 mutants, 73 killed → **90.1 %**; the 8 survivors are equivalent mutants — inspected with `mutmut show`) |
| `logs/mutation_score.py` | how the score was read (from the meta, not from the terminal) |
| `logs/compare_rows.py`, `logs/render_report_from_raw.py` | the two read-only helpers used above |
