#!/bin/bash
# t_9bcbecff — the OPTIONAL Occamy second pass the card names ("Tiel (and Occamy)").
#
# The card's row is the Tiel challenger. This run asks the same E3e question of the other
# qwen35moe [host] model, on the same 60 committed dev items, the same placement ask and the same
# `--backend vulkan` instrument. Occamy has *no published row under the corrected instrument*, so
# both cells of the E3e pair are measured here (the shipped/answer-sheet baseline and the
# `role_split` + `json_instructed` challenger) and are compared with each other, paired by item —
# neither cell is comparable with the container-era E3 Occamy row of card t_a431be85.
#
#   bash .t9bcb/run_occamy.sh
#
# Run from an unlimited scope (MemoryMax=infinity), never the 4 GiB worker scope.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t9bcb || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=${MODEL:-/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf}
NGL=${NGL:-9}
OUT_DIR=${OUT_DIR:-docs/evidence/t9bcbecff_occamy_chunks}
echo "occamy pass start $(date -Is) pid $$"
echo "model=$MODEL ngl=$NGL"

TAG=occ_base_ CUE=shipped CHAT=answer_sheet OUT_DIR="$OUT_DIR" \
  MODEL="$MODEL" NGL="$NGL" bash .t9bcb/run_arm.sh || exit 1
TAG=occ_e3e_ CUE=json_instructed CHAT=role_split OUT_DIR="$OUT_DIR" \
  MODEL="$MODEL" NGL="$NGL" bash .t9bcb/run_arm.sh || exit 1
echo "occamy pass done $(date -Is)"
