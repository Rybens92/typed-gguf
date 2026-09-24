# t_dd62ec29 — E2 FIX: `bench --backend all` on a two-bundle host (raw material)

The readable version lives in `docs/evidence/e2_fix_t_dd62ec29_mixed_bundle_isolation.md`. Every
claim there is checkable from here. All runs use the same venv and the same environment; a RED run
differs only in `PYTHONPATH` (`/work/t603-dd62-red/src` = the parent commit `1b7192f`).

Box: the operator host's container, `cgroup quota 2.0`, GPU passed through, the E3 campaign on the
device throughout; `VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` in every command. Bundles:
`GGUFONE_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu` (the pinned `linux-x64-cpu` asset) and
`GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime` (the installed Vulkan
bundle). Models: `Qwen3.5-4B-Q4_0.gguf` (4B) and `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (small).

| file | what it is |
|---|---|
| `logs/runs.md` | the exact commands, exit codes, VRAM snapshots and caveats of every run below |
| `logs/red_small.raw` | RED: the parent tree, the operator's command on the small model — report `ok: false`, then `double free or corruption (!prev)` / `Aborted (core dumped)`, **exit 134** |
| `logs/vulkan_alone.raw` | control: the Vulkan bundle **alone**, 4B — `effective_backend: "vulkan"`, `Vulkan0`/`Vulkan_Host` compute buffers, **exit 0** (one bundle, one process) |
| `logs/after_mixed.raw` | this tree, operator's command, 4B under VRAM pressure: `cpu` row measured, `vulkan` row **withheld** (child exit `-11` after a complete report), `ISOLATED_CHILD_FAILED` note, **exit 1** — no glibc line |
| `logs/after_mixed_2.raw` | this tree, same command again: `cpu` measured, `vulkan` reported `E_BACKEND_OOM` by the child itself, **exit 0** |
| `logs/after_mixed_small.raw` | this tree, same command, small model: **both rows measured in their own children**, `vulkan` on `Vulkan0`, **exit 0** |
| `logs/rendered_after_mixed_small.md`, `logs/rendered_after_mixed_2.md` | what `render_report` prints for those two reports (`render_from_raw.py`) |
| `logs/red_pretest.txt` | the 45-gate file against the **parent tree**: **43 failed, 2 passed** |
| `logs/red_test_file.py` | that gate file, verbatim from the RED commit |
| `logs/red_live_gate.txt` | the live gate against the **parent tree**: `--backend all` exited `-6` (SIGABRT) where the gate requires 0/1 |
| `logs/live_gate.txt` | the live gate on this tree, two real bundles, real model: **1 passed in 131.03 s** |
| `logs/green_full_suite.txt` | `pytest -q` (offline, fresh) → **968 passed, 41 skipped** |
| `logs/ruff.txt` | `ruff check src tests` → `All checks passed!` |
| `logs/coverage_changed.txt` | `coverage run -m pytest` over the six bench/CLI gate files + report: `bench/isolation.py` **100 %**, `bench/suites.py` **99 %** |
| `logs/mutmut.out` + `logs/mutmut_r1_survivors.txt` + `logs/mutation_score.txt` + `logs/survivor_triage.txt` | round 1 of the Tier-M sweep of `bench/isolation.py` with `tests/test_bench_isolation.py`: 533 mutants, 387 killed, **72.6 %**, 146 survivors triaged per mutant |
| `logs/mutmut_r2.out` + `logs/mutmut_r2_survivors.txt` + `logs/mutation_score_r2.txt` + `logs/survivor_edits_r2.txt` | round 2 after the triage's first pins: 417 killed, **78.2 %**, 116 survivors |
| `logs/mutmut_r3.out` + `logs/mutmut_r3_survivors.txt` + `logs/mutation_score_r3.txt` + `logs/survivor_edits_r3.txt` | round 3, the committed gate file: 429 killed, **80.5 %**, 104 survivors (all triaged as message / diagnostics pass-through / encoding equivalents) |
| `logs/survivor_replays.txt` | the harness proof: six r2 survivors spliced one by one into `src/` (gate run, source restored) and **KILLED** by the pins, plus one equivalent mutant that must survive and does |
| `logs/vram_before_controls.txt` | `nvidia-smi` free/total device memory at the start and the end of the control round |
| `controls.sh`, `red_control.sh` | the two drivers behind the raws above (bundle paths, env, `PYTHONPATH`) |
| `raw_report.py`, `summarise_report.py`, `render_from_raw.py` | the read-only helpers used to quote rows out of a `.raw` |
| `measure_child_overhead.py` | the per-child cost of an isolated row (`-m ggufone bench --help`, 5 runs) |

Card-side helpers, run from the repository root (they resolve `mutants/`, `src/`, `tests/` relative
to it): `triage_survivors.py` + `list_survivor_edits.py` + `classify_survivors.py` (the survivor
triage), `find_survivor_key.py` + `meta_verdict.py` + `list_survivors.py` (locating and scoring
keys), `replay_survivor.py` + `annotate_replays.py` (the splice-and-run proof above),
`summarise_report.py` + `raw_report.py` + `render_from_raw.py` (quoting rows out of a `.raw`),
`measure_child_overhead.py`, and the two drivers `controls.sh` / `red_control.sh`.
