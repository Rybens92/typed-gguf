#!/bin/bash
# t_9bcbecff — an AUXILIARY arm of the same [host] campaign (never the card's row).
#
# The card's row is the challenger (`role_split` + `json_instructed`), measured by
# `.t9bcb/run_chunks.sh`. This driver measures one *extra* cell of the same E3e design on the same
# 60 committed items, same model file, same placement ask, same `--backend vulkan` instrument — so
# a reading that would otherwise stay an extrapolation from the 4B gets a Tiel number with the
# cell it belongs to. It is labelled `aux_<tag>_` on disk and published as auxiliary.
#
#   TAG=aux_role_split_ CUE=shipped     CHAT=role_split  ...  # the placement alone
#   TAG=aux_two_step_   CUE=two_step    CHAT=role_split  ...  # the 4B's collapse cell
#
# Run from an unlimited scope (MemoryMax=infinity), never the 4 GiB worker scope.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t9bcb || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=${MODEL:-/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf}
NGL=${NGL:-9}
TAG=${TAG:?set TAG (e.g. aux_role_split_)}
CUE=${CUE:?set CUE (shipped | two_step | json_instructed)}
CHAT=${CHAT:-role_split}
CONTRACT=${CONTRACT:-question}
LOG_DIR=.t9bcb/logs
OUT_DIR=docs/evidence/t9bcbecff_tiel_chunks
mkdir -p "$LOG_DIR" "$OUT_DIR"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "tiel aux arm ${TAG} start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "model=$MODEL ngl=$NGL cue=$CUE chat_format=$CHAT json_contract=$CONTRACT"
sha256sum "$MODEL"
for n in 001 002 003 004 005 006; do
  echo "=== chunk $n start $(date -Is)"
  echo "    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"
  start=$(date +%s)
  python3 .t9bcb/tiel_e3e_arm.py --suite quality \
    --cue "$CUE" --chat-format "$CHAT" --json-contract "$CONTRACT" \
    --model "$MODEL" \
    --backend vulkan --gpu-layers "$NGL" --threads 4 \
    --devset "docs/evidence/tiel_corrected_chunks/devset_${n}.jsonl" \
    --out "$OUT_DIR/${TAG}report_${n}.json" \
    --placement-out "$OUT_DIR/${TAG}placement_${n}.json" > "$LOG_DIR/${TAG}run_${n}.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== chunk $n exit=$rc wall=$((end - start))s $(date -Is)"
  grep -m1 '^- framing:' "$LOG_DIR/${TAG}run_${n}.log" || true
  grep -m1 'compute buffer size' "$LOG_DIR/${TAG}run_${n}.log" || true
  if [ "$rc" -ne 0 ]; then
    echo "!! chunk $n failed; stopping so the gap is never published"
    exit "$rc"
  fi
done
echo "tiel aux arm ${TAG} done $(date -Is)"
sha256sum "$MODEL"
