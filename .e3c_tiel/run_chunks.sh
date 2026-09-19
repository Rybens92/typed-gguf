#!/bin/bash
# E3c-Tiel host run (card t_a58f8b67): the six dev chunks of the committed 60-item set,
# measured with the E3 lens (Occamy 1.0) on Tiel-Coder-35B-A3B.
#
#   * runtime: the pinned b11026 Vulkan bundle (same build both earlier campaigns used)
#   * ICD: the manifest naming libEGL_nvidia.so.0 (E3 §6.2 recipe), VK_INSTANCE_LAYERS dropped
#   * placement: the Tiel fit plan's n_gpu_layers=9 (free-VRAM aware) — every chunk asks for the
#     same placement so the six rows are comparable
#   * --backend vulkan --threads 4, one process per chunk (E3's shape), reports raw
#   * resource snapshot per chunk boundary: free VRAM / MemAvailable / cgroup memory.max
#
# Run this script from a scope that is NOT the kanban worker's own: that scope is capped at
# 4 GiB (`hermes-worker-kanban-<task>.scope`, memory.max=4294967296), and a 21 GB model inside
# it re-reads its weights from disk forever — measured 608 s for one 10-item chunk vs 91 s for
# two items in an unlimited scope. The scope's own memory.max is printed below so the run's
# provenance carries the proof rather than the claim.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
NGL=${NGL:-9}
LOG_DIR=/var/home/rybens/.e3c_tiel
OUT_DIR=docs/evidence/tiel_chunks
mkdir -p "$LOG_DIR" "$OUT_DIR"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "tiel host run start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "GGUFONE_RUNTIME_DIR=$GGUFONE_RUNTIME_DIR"
echo "VK_DRIVER_FILES=$VK_DRIVER_FILES"
echo "model=$MODEL ngl=$NGL"
for n in 001 002 003 004 005 006; do
  echo "=== chunk $n start $(date -Is)"
  echo "    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"
  start=$(date +%s)
  python3 tools/e3c_tiel_reproduce.py --suite quality \
    --model "$MODEL" \
    --backend vulkan --gpu-layers "$NGL" --threads 4 \
    --devset "$OUT_DIR/devset_${n}.jsonl" \
    --out "$OUT_DIR/report_${n}.json" \
    --placement-out "$OUT_DIR/placement_${n}.json" > "$LOG_DIR/run_${n}.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== chunk $n exit=$rc wall=$((end - start))s $(date -Is)"
  if [ "$rc" -ne 0 ]; then
    echo "!! chunk $n failed; stopping so the gap is never published"
    exit "$rc"
  fi
done
echo "tiel host run done $(date -Is)"
