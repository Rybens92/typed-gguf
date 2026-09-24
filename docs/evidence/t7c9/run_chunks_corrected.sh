#!/bin/bash
# t_7c926398 — Tiel-Coder-35B-A3B [host] 60-item quality re-run under the CORRECTED instrument.
#
# The pre-fix row (docs/BENCHMARKS.md §7.4, 0.517 = 31/60) was measured with the plain E1b
# framing; card t_6de5fc53 proved the bench planned its executed context from the live session
# (no chat template) while `ggufone ask`/`run` send the model's chat template. This campaign
# re-measures the same 60 committed dev items, same model file, same placement, against the
# committed fixed tree (worktree at 00265ea, branch t7c926398-tiel-corrected) — nothing else
# moves between the two rows.
#
#   * runtime: the pinned b11026 Vulkan bundle (same build both earlier campaigns used)
#   * ICD: the manifest naming libEGL_nvidia.so.0 (E3 §6.2 recipe), VK_INSTANCE_LAYERS dropped
#   * placement: the Tiel fit plan's n_gpu_layers=9 — the SAME ask the pre-fix row used, so the
#     pair isolates the framing (a placement-matched pair is what makes the comparison readable)
#   * --backend vulkan --threads 4, one process per chunk, reports raw
#   * resource snapshot per chunk boundary: free VRAM / MemAvailable / cgroup memory.max
#
# Run from an unlimited scope, NOT the kanban worker's own scope (memory.max=4 GiB): a 21 GB
# model inside that scope re-reads its weights from disk forever (608 s for one chunk vs ~150 s).
# systemd-run --user --property=MemoryMax=infinity is that scope, and the run prints its own
# memory.max so the provenance carries the proof rather than the claim.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t7c9 || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
NGL=${NGL:-9}
LOG_DIR=.t7c9/logs
OUT_DIR=docs/evidence/tiel_corrected_chunks
mkdir -p "$LOG_DIR" "$OUT_DIR"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "tiel corrected-instrument run start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "GGUFONE_RUNTIME_DIR=$GGUFONE_RUNTIME_DIR"
echo "VK_DRIVER_FILES=$VK_DRIVER_FILES"
echo "model=$MODEL ngl=$NGL"
sha256sum "$MODEL"
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
  grep -m1 '^- framing:' "$LOG_DIR/run_${n}.log" || true
  if [ "$rc" -ne 0 ]; then
    echo "!! chunk $n failed; stopping so the gap is never published"
    exit "$rc"
  fi
done
echo "tiel corrected-instrument run done $(date -Is)"
sha256sum "$MODEL"
