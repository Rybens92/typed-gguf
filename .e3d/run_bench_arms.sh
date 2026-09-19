#!/bin/bash
# E3d stage 3 (card t_d90404ac): the E2-style quality table, side by side, for the two shippable
# cue shapes — the same box, the same model, the same 60 committed items, the same context; only
# `--cue` moves. The rows come out of the engine's own path (`tools/e2_reproduce.py` ->
# `bench.suites.run_suite`), which is what a published table is read from.
#
# Placement: the shipped configuration (the Vulkan bundle, the device visible; the rows carry
# `effective_backend` and the `W_BACKEND_MISMATCH` warning the E3 machinery raises when the
# *claim* says `cpu` while Vulkan0 computed). The comparison itself is placement-invariant: both
# arms ran under the same placement, which is the only thing a side-by-side table needs.
#
# Reproduce (from the repo root, with the pinned runtime + the 4B GGUF):
#     bash .e3d/run_bench_arms.sh
set -u
cd "$(dirname "$0")/.." || exit 126
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export GGUFONE_HOME="${GGUFONE_HOME:-$PWD/.e3d/ggufone-home}"
export GGUFONE_RUNTIME_DIR="${GGUFONE_RUNTIME_DIR:-$HOME/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
mkdir -p "$GGUFONE_HOME" .e3d/logs
MODEL="${GGUFONE_E3D_MODEL:-$HOME/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"

for cue in shipped two_step; do
  echo "=== arm $cue ==="
  uv run --frozen python -u tools/e2_reproduce.py --suite quality \
    --model "$MODEL" --backend auto --threads 4 --items 60 --cue "$cue" \
    --out ".e3d/bench_${cue}.json" || echo "ARM FAILED: $cue"
done
echo "=== arms done ==="
