# t_b67f9c49 — CALIBRATE: the first stored calibration for the default 4B (Spark-X2.5-4B-Q8_0)

Verdict: **SHIP** (the card's outcome is met: the box's default store now carries the 4B's entry, and
a live `ask` reports `calibrated: true` with the applied parameters visible). Two non-blocking
product findings are recorded below.

## What ran (all evidence files are in this directory)

| step | command | result |
|---|---|---|
| preflight | `git rev-parse HEAD`, `keep status`, `nvidia-smi` | `d9e41d0` (main, clean), keep `stopped`, no model resident |
| baseline ask | `ask --state state.txt --choice … --score … --noul … --keep-alive 0` (pre-store) | exit 0, `"calibrated": false`, `calibration.applied: false` |
| **calibrate (live, CPU)** | `calibrate --model ~/.hermes/models/Spark-X2.5-4B-Q8_0.gguf --threads 2 --json --out docs/evidence/calibration-4b-2026-09-23_rows.json` | **exit 0**, 3 087 734 ms, table stored |
| refit from rows | `calibrate --from-report …rows.json --model … --dry-run --json` | exit 0, 183 ms, **identical `params_hash`** |
| live asks (post-store) | 3 × `ask … --keep-alive 10m` (choice+score / noul+score / choice+noul) | exit 0, all `"calibrated": true` |

Store: `/var/home/rybens/.local/share/typed-gguf/calibration.json` (snapshot:
`calibration-store-snapshot.json`), one entry `file:Spark-X2.5-4B-Q8_0.gguf:4375021152`,
`accepted_types: ["noul","score"]`, `params_hash sha256:f09fe20e…3fc81`.

Accepted/rejected: `noul` T=0.0500 (`normalized_peak`, held-out ECE 0.0355 → 0.0000), `score`
T=1.6475 (`margin`, held-out ECE 0.2158 → 0.2026), `choice` **no calibration applied** (held-out
split did not improve). Repo receipt: `docs/evidence/calibration-4b-2026-09-23.md` (39 lines,
commit `612741a`, `docs/evidence/` only).

## Raw files here

`calibrate-cpu.stdout|stderr|exit`, `calibrate-4b-refit.stdout`, `a1.json`–`a3.json`, `asks.log`,
`runs.log`, `pre-ask.stdout|stderr`, `calibration-store-snapshot.json`, `store-print.txt`,
`vulkan-oom-*.stderr` (the failed GPU attempts), `sha256-model.txt`.

## BLOCKING
None for this card's outcome.

## PROPOSALS (non-blocking, with repro)

**P1 — the Vulkan placement of the default 4B cannot be planned/placed on this 8 GB box.**
`TYPED_GGUF_RUNTIME_DIR=<…>-vulkan calibrate --model …Spark-X2.5-4B-Q8_0.gguf` fails 3/3 with
`E_BACKEND_OOM` (weights 4.5 GB + KV + compute vs ~6.3 GB free next to the desktop; the driver
reports *unknown* free, so the fit's ~5.5 GB budget over-counts the card).
Repro: `vulkan-oom-1/2/3.stderr`. Note the same model *did* answer on Vulkan minutes earlier
(`pre-ask`, degraded placement), so the failure is a planning/headroom gap, not a hard "does not
fit". The deploy ladder named in the message only varies `kv_type` (`ctx kv_type=f16|q8_0|q4_0 ->
oom`) and never shrinks `n_ctx` or `n_gpu_layers`, although the model *loaded* fine — the phrase
"tried 3 placement(s) down to CPU-only" describes rungs that were not walked. A `--fit-target`
bigger than the desktop's resident ~1.5 GB would have given a smaller context; the CPU bundle is
the working route today.

**P2 — `calibrate` accepts `--fit-target` / `--n-seq-max` and silently ignores them.**
`CALIBRATE_VALUE_FLAGS` (cli.py:1540) lists both, but the live path builds its plan with
`fit_plan_for(model_path, use_cache=…, kv_type=…)` only (cli.py:~1588, `calibration_rows`), so the
documented workaround for an OOM placement (`--fit-target <MiB>`) has no effect on `calibrate`
while `fit`/`run`/`ask` honour it. Repro: run `calibrate --fit-target 4500 --json --dry-run` and
compare `types`/fit plan with a run without the flag — identical plan. Same shape as the earlier
card's "a pin that had to shrink is mislabeled" finding: an accepted flag that silently does
nothing.

## Notes for the next worker

* The run is CPU (`E_BACKEND_OOM` on Vulkan): the stored parameters describe the CPU serving path,
  which is what every live 4B run on this box has used; the GPU route is P1.
* `noul` was accepted at the **bottom edge** of the 61-point temperature grid (0.05) on a 6-item
  held-out split — the fit does not flag a boundary pick; `score` improved by +0.0132 on 6 items.
* The repo `.venv` was disturbed by `uv run` (this container's uv 0.12.0 removes a venv whose
  `python3` links to a host-only uv-managed CPython `/home/rybens/…`); the CLI was driven via
  `PYTHONPATH=src python3` (launcher `tg.py`) exactly as the sandbox's earlier live runs do. See
  the kanban report for the state left behind.
