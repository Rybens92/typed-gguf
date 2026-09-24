#!/bin/bash
# E3d stage 3 (card t_d90404ac), re-run after the framing fix (card t_6de5fc53): the E2-style
# quality table, side by side, for the cue shapes — the same box, the same model, the same 60
# committed items, the same context; only `--cue` moves.
#
# Card t_6de5fc53: **this script is the post-fix recipe.** Before the fix `LiveModel.decide`
# planned the executed context from the live session, and a `ModelSession` resolves no chat
# template, so the arms it wrote then measured the *plain* E1b framing. Those runs are kept as
# `.e3d/bench_plain_<cue>.json` (all three cues, this box, this command — they reproduce the
# published `.e3d/bench_shipped.json` / `.e3d/bench_two_step.json` numbers of card t_d90404ac);
# the arms written here are `.e3d/bench_templated_<cue>.json`, measured with the model's own chat
# template: the prompt the serving path (`ggufone ask`/`run`) sends. To re-measure the plain arms,
# run this script from the pre-fix commit (`git worktree add /tmp/prefix <base-commit>`) and point
# `--out` at the `bench_plain_` names.
# `docs/evidence/e3d_cue_decision_4b.md` §6 renders both instruments under their framing markers,
# and `docs/evidence/e2_fix_t_6de5fc53_framing.md` carries the card's before/after tables.
#
# Placement: the shipped configuration (the Vulkan bundle, the device visible; the rows carry
# `effective_backend` and the `W_BACKEND_MISMATCH` warning the E3 machinery raises when the
# *claim* says `cpu` while Vulkan0 computed). The comparison itself is placement-invariant: every
# arm runs under the same placement, which is the only thing a side-by-side table needs.
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

for cue in shipped two_step json_field; do
  echo "=== arm $cue ==="
  uv run --frozen python -u tools/e2_reproduce.py --suite quality \
    --model "$MODEL" --backend auto --threads 4 --items 60 --cue "$cue" \
    --out ".e3d/bench_templated_${cue}.json" || echo "ARM FAILED: $cue"
done
echo "=== arms done ==="
