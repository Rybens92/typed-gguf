#!/bin/bash
# t_9bcbecff — the 60-item [host] challenger arm: the corrected instrument + the two E3e flags.
#
# Baseline (not re-measured here, by the card's rule): the corrected-instrument Tiel row of card
# `t_7c926398` — 22/60 = 0.367, `low_mass` 57/60, 59/60 refusals at the cue, `measured` 3/60 —
# whose per-chunk reports are committed at `docs/evidence/tiel_corrected_chunks/report_00{1..6}.json`.
# This arm is the SAME instrument (same model file, same 60 committed dev items, same placement
# ask, `--backend vulkan --threads 4`, temperature 0, one process per chunk) with exactly the two
# E3e policy flags added: `--chat-format role_split --cue json_instructed --json-contract question`.
#
#   * chunks + instrument: `.t7c9/run_chunks_corrected.sh` (ten-item chunks, `--backend vulkan
#     --gpu-layers 9 --threads 4`, model SHA checked before/after)
#   * flag override: `.t9bcb/tiel_e3e_arm.py` (the committed driver's own observer + the flags)
#   * resource snapshot per chunk boundary: free VRAM / MemAvailable / cgroup memory.max
#   * the compute path is grep-able in each chunk log (`... compute buffer size`), one backend named
#
# Run from an unlimited scope, NOT the kanban worker's own scope (memory.max=4 GiB); this script
# prints its own memory.max so the provenance carries the proof rather than the claim.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t9bcb || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=${MODEL:-/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf}
NGL=${NGL:-9}
CUE=${CUE:-json_instructed}
CHAT=${CHAT:-role_split}
CONTRACT=${CONTRACT:-question}
LOG_DIR=.t9bcb/logs
OUT_DIR=docs/evidence/t9bcbecff_tiel_chunks
mkdir -p "$LOG_DIR" "$OUT_DIR"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "tiel e3e-arm campaign start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "GGUFONE_RUNTIME_DIR=$GGUFONE_RUNTIME_DIR"
echo "VK_DRIVER_FILES=$VK_DRIVER_FILES"
echo "model=$MODEL ngl=$NGL cue=$CUE chat_format=$CHAT json_contract=$CONTRACT"
sha256sum "$MODEL"
sha256sum docs/evidence/tiel_corrected_chunks/devset_00*.jsonl > "$OUT_DIR/devset_sha.txt"
for n in 001 002 003 004 005 006; do
  echo "=== chunk $n start $(date -Is)"
  echo "    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"
  start=$(date +%s)
  python3 .t9bcb/tiel_e3e_arm.py --suite quality \
    --cue "$CUE" --chat-format "$CHAT" --json-contract "$CONTRACT" \
    --model "$MODEL" \
    --backend vulkan --gpu-layers "$NGL" --threads 4 \
    --devset "docs/evidence/tiel_corrected_chunks/devset_${n}.jsonl" \
    --out "$OUT_DIR/report_${n}.json" \
    --placement-out "$OUT_DIR/placement_${n}.json" > "$LOG_DIR/run_${n}.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== chunk $n exit=$rc wall=$((end - start))s $(date -Is)"
  grep -m1 '^- framing:' "$LOG_DIR/run_${n}.log" || true
  grep -m1 'compute buffer size' "$LOG_DIR/run_${n}.log" || true
  if [ "$rc" -ne 0 ]; then
    echo "!! chunk $n failed; stopping so the gap is never published"
    exit "$rc"
  fi
done
echo "tiel e3e-arm campaign done $(date -Is)"
sha256sum "$MODEL"
