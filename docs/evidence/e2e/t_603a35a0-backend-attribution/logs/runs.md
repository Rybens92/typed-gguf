# The four operator-box runs behind this card (raw commands and exit codes)

Box: the operator host's container (`localhost-live.home`), GPU passed through as
`/dev/nvidia0` + `/dev/dri/renderD128`; the Vulkan ICD for that GPU is not on the default
loader path inside the container, so every run sets
`VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json` (NVIDIA EGL vendor library as ICD) —
without it `llama-bench --list-devices` prints `(none)` and no row can reach the device.

CPU/memory are bounded by the cgroup the reports themselves quote: `cgroup quota 2.0`
(cpu.max) on a 24-CPU box; the host ran the E3 campaign and other cards throughout, so the
absolute tok/s values are "this box, this hour" — the *attribution* is what these runs are
about. VRAM free at each start is in `before_driver.log` / the run lines below.

Bundles visible to the two configurations:

| config | CPU bundle | Vulkan bundle |
|---|---|---|
| `mixed` | `/work/t603-runtime/b11026-linux-x64-cpu` (the pinned `linux-x64-cpu` asset of `runtime.lock`, sha256 `219cf1c7…` verified after download) via `GGUFONE_RUNTIME_DIR` | `/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan` (installed), found via `GGUFONE_BENCH_RUNTIME_DIR` |
| `opoffload` | none — `cpu` therefore resolves to the *Vulkan* bundle's own directory (`backend_runtimes` registers every bundle under `cpu` too) | the same installed bundle |

Model: `/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf` (2,583,221,408 B) — the smallest
local GGUF, chosen so a CPU-class row finishes while the GPU is busy with E3.

Common flags: `--suite throughput --runs 1 --threads 4 --sizes 64 --json`
(`--sizes 64` keeps the CPU rows short; note `commands.reproduce` in the reports does not echo
`--sizes`, a pre-existing gap tracked elsewhere).

## BEFORE — tree `1205c0b` (`/work/t603-before`, parent of this card's commits)

```
# mixed  (VRAM free 4860 MiB)                                                       exit 134
GGUFONE_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu \
GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime \
  uv run --frozen ggufone bench --suite throughput --model …/Qwen3.5-4B-Q4_0.gguf \
    --backend all --runs 1 --threads 4 --sizes 64 --json      -> logs/before_mixed.raw

# op-offload  (VRAM free 4612 MiB)                                                  exit 0
GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime \
  uv run --frozen ggufone bench --suite throughput --model …/Qwen3.5-4B-Q4_0.gguf \
    --backend cpu --runs 1 --threads 4 --sizes 64 --json      -> logs/before_opoffload.raw
```

`exit 134` = `SIGABRT` **after** the report was written: the mixed process prints the whole
`ggufone.bench/v1` JSON and then dies with `double free or corruption (!prev)` during
teardown. The report is complete and was about to be published as `ok: true`.

## AFTER — this card's tree (`/work/ggufone-t603`, HEAD of the two fix commits)

```
# mixed  (VRAM free ~1000 MiB, E3 campaign running)                                 exit 134
GGUFONE_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu \
GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime \
  uv run --frozen ggufone bench --suite throughput --model …/Qwen3.5-4B-Q4_0.gguf \
    --backend all --runs 1 --threads 4 --sizes 64 --json      -> logs/after_mixed.raw

# op-offload  (VRAM free ~1000 MiB)                                                 exit 1
GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime \
  uv run --frozen ggufone bench --suite throughput --model …/Qwen3.5-4B-Q4_0.gguf \
    --backend cpu --runs 1 --threads 4 --sizes 64 --json      -> logs/after_opoffload.raw
```

`exit 1` is the CLI's "a gate failed" code (SPEC 2.5 / `cli._bench`): the report's `ok` is
`false` because a row's own engine log contradicts its label. The mixed run still carries the
pre-existing teardown abort (`exit 134`, same `double free`) — that is a separate defect of
loading two bundles in one process and is filed as its own card; the fix under test only
changes what the report *says*.

## Rendering

`logs/rows_before_after.txt` (row-level before/after, straight from the JSON),
`logs/rendered_after_mixed.md` and `logs/rendered_after_opoffload.md` (the markdown tables the
fixed tree renders for the same two runs); `render_report` was driven from this tree's
`src/` so the "after" tables are the real output of the shipped renderer.
