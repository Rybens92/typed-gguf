#!/bin/bash
# E3d (card t_d90404ac): the three cue shapes on the committed 60-item dev set, 4B, CPU-only.
#
# Same items, same model, same context, one batched decode per item for all three shapes, so the
# shapes can differ in where the readout sits and in nothing else. The engine's own ranked readout
# (`labels.score_paths` + restricted softmax) is taken for every shape, so *agreement* is measured
# per shape rather than inferred from coverage. `--hide-devices` keeps the box's accelerator for
# the sibling campaigns and fixes the placement for every cell of the table — the E3d comparison
# is a comparison (`docs/evidence/e3d_cue_decision_4b.md`).
#
# Reproduce (from the repo root, with the pinned runtime + the 4B GGUF):
#     bash .e3d/run_full.sh
set -u
cd "$(dirname "$0")/.." || exit 126
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export GGUFONE_RUNTIME_DIR="${GGUFONE_RUNTIME_DIR:-$HOME/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
mkdir -p .e3d/logs .e3d/states
MODEL="${GGUFONE_E3D_MODEL:-$HOME/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"

exec uv run --frozen python -u tools/e3c_cue_shapes.py run \
  --model "$MODEL" \
  --shapes shipped two_step_shipped json_field \
  --rank shipped=bare --rank two_step_shipped=bare --rank json_field=bare \
  --backend cpu --gpu-layers 0 --hide-devices --threads 4 \
  --states-home .e3d/states \
  --out .e3d/full.json --report .e3d/full.md
