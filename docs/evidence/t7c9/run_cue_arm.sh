#!/bin/bash
# t_7c926398 (auxiliary arm) — the same Tiel campaign at `--cue two_step`, six chunks.
#
# Not the card's row: the card asks for the shipped cue (the product's default), measured in
# `.t7c9/run_chunks_corrected.sh`. This arm answers the question the shipped-cue row raises —
# *is the collapse the cue ROW?* — with a measurement, over the same 60 committed items, the same
# placement ask and the same corrected instrument. Everything else is the card's driver.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t7c9 || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
NGL=${NGL:-9}
CUE=${CUE:-two_step}
LOG_DIR=.t7c9/logs
OUT_DIR=docs/evidence/tiel_corrected_chunks
mkdir -p "$LOG_DIR" "$OUT_DIR"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "tiel cue arm ($CUE) start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "model=$MODEL ngl=$NGL cue=$CUE"
for n in 001 002 003 004 005 006; do
  echo "=== chunk $n start $(date -Is)"
  start=$(date +%s)
  python3 .t7c9/tiel_cue_arm.py --suite quality --cue "$CUE" \
    --model "$MODEL" \
    --backend vulkan --gpu-layers "$NGL" --threads 4 \
    --devset "$OUT_DIR/devset_${n}.jsonl" \
    --out "$OUT_DIR/${CUE}_report_${n}.json" \
    --placement-out "$OUT_DIR/${CUE}_placement_${n}.json" > "$LOG_DIR/${CUE}_run_${n}.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== chunk $n exit=$rc wall=$((end - start))s $(date -Is)"
  grep -m1 '^- framing:' "$LOG_DIR/${CUE}_run_${n}.log" || true
  if [ "$rc" -ne 0 ]; then
    echo "!! chunk $n failed; stopping so the gap is never published"
    exit "$rc"
  fi
done
echo "tiel cue arm ($CUE) done $(date -Is)"
