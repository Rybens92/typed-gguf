#!/bin/bash
# t_7c926398 — Tiel [host] quality re-run under the CORRECTED instrument (card t_6de5fc53's fix).
# Smoke: the first two dev items, so the wiring (framing label, prefix tokens, placement capture)
# is verified before the 30-minute campaign. Same flags as the published pre-fix row.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t7c9 || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
OUT=.t7c9
mkdir -p "$OUT"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "smoke start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"
python3 tools/e3c_tiel_reproduce.py --suite quality \
  --model "$MODEL" \
  --backend vulkan --gpu-layers 9 --threads 4 \
  --devset docs/evidence/tiel_chunks/devset_001.jsonl --items 2 \
  --out "$OUT/smoke_report.json" --placement-out "$OUT/smoke_placement.json" \
  > "$OUT/smoke_run.log" 2>&1
rc=$?
echo "smoke exit=$rc $(date -Is)"
tail -25 "$OUT/smoke_run.log"
echo "smoke done $(date -Is)"
exit "$rc"
