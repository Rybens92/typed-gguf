# The runs behind card t_dd62ec29 (exact commands, exit codes, box state)

Every command below was run **in this container** (the operator host's container), from the working
tree `/work/t603-dd62-1451` (this card) or `/work/t603-dd62-red` (the parent commit `1b7192f`,
reached by `git worktree add`), with the same `.venv`:

```
CPU=/work/t603-runtime/b11026-linux-x64-cpu          # the pinned linux-x64-cpu asset, unpacked
HOST_RT=/var/home/rybens/.local/share/ggufone/runtime # the installed Vulkan b11026 bundle
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
BIG=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf
SMALL=/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json   # the container's Vulkan ICD
```

GPU state at the start of the control round (`vram_before_controls.txt`, `nvidia-smi
--query-gpu=memory.total,memory.free`): `8192 MiB, 3272 MiB`, and `8192 MiB, 3212 MiB` at the end —
the E3 campaign was on the device throughout, so free device memory moved between runs (and the
`vulkan` child of `after_mixed.raw` died at teardown during a low-memory window).

| # | run | command | exit |
|---|---|---|---|
| 1 | RED control (parent tree, small model) | `GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT PYTHONPATH=/work/t603-dd62-red/src python -m ggufone bench --suite throughput --model $SMALL --backend all --runs 1 --threads 4 --sizes 64 --json` | **134** |
| 2 | the same command on this tree | `GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT python -m ggufone bench … $SMALL …` | **0** |
| 3 | the operator's command, 4B, this tree | `… --model $BIG …` | **1** (a child died) |
| 4 | the same command again | `… --model $BIG …` | **0** (child OOM, reported) |
| 5 | control: Vulkan bundle alone, 4B | `GGUFONE_RUNTIME_DIR=$VULK python -m ggufone bench … --backend vulkan … $BIG …` | **0** |
| 6 | the live gate, two bundles, small model | `GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT GGUFONE_BENCH_MODEL=$SMALL pytest -q --run-network tests/test_bench_isolation_live.py -s` | **0** (1 passed, 131.03 s) |
| 7 | the same live gate, parent tree | `cd /work/t603-dd62-red && … pytest -q --run-network tests/test_bench_isolation_live.py -s` | **1** (the CLI exited -6) |
| 8 | the offline gate file, parent tree | `cd /work/t603-dd62-red && python -m pytest -q tests/test_bench_isolation.py` | 38 failed, 2 passed |
| 9 | a 4B run *inside* the parent tree, in-process (the raw abort, before this card's tree existed) | `repro_mixed.py` — `suites.run_suite(config, factory=suites.live_factory)` with both bundles visible | **134**, `wall_ms 246194` |

Notes:

* Runs 1–5 are driven by `controls.sh` / `red_control.sh` in this directory; the raws in `logs/`
  are the verbatim stdout+stderr of those commands (`exit` in the matching `.exit` file). The
  engines print their logs to stderr, so a `.raw` holds the report JSON *and* the log around it;
  `raw_report.py` pulls the report out.
* Run 9 is the same shape as the neighbour card's `.e2e/t_603a35a0-backend-attribution/logs/
  {before,after}_mixed.raw` (4B, `--sizes 64 --runs 1 --threads 4`) and exists to show the abort is
  not a property of this card's driver: it is the two bundles in one process.
* Absolute tok/s and ms in the raws are "this box, this hour" (a shared 2-CPU quota and a busy
  device). The rows' `devices`/`device_buffers`/`effective_backend` are what these runs are about.
* The per-child cost of an isolated row is measured by `measure_child_overhead.py`
  (`python -m ggufone bench --help`, i.e. interpreter start + the whole CLI import, 5 runs).
