#!/bin/bash
# t_9bcbecff — the two 6-item smokes, before the 60-item campaign.
#
# Card t_4c48f40a's own recommended falsification order (its [host] hand-off), on the committed
# first chunk of the corrected Tiel dev set:
#
#   1. collapse cell  — `--chat-format role_split --cue two_step`: on the 4B this reads 28/60 with
#      `low_mass` 60/60 because under the role split the LABEL sits at the cue row and `two_step`'s
#      one-token advance reads past it. If Tiel reproduces that mechanism, the role split is doing
#      what the 4B's reading says it does on this family too.
#   2. challenger     — `--chat-format role_split --cue json_instructed --json-contract question`.
#
# Run from an unlimited scope, NOT the kanban worker's own scope (memory.max=4 GiB): a 21 GB model
# inside that scope re-reads its weights from disk forever. `systemd-run --user
# --property=MemoryMax=infinity` is that scope and this script prints its own memory.max.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t9bcb || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
NGL=${NGL:-9}
OUT=.t9bcb
mkdir -p "$OUT/logs"
CGROUP=$(cat /proc/self/cgroup | cut -d: -f3)
echo "tiel e3e-arm smoke start $(date -Is) pid $$"
echo "cgroup=$CGROUP memory.max=$(cat /sys/fs/cgroup$CGROUP/memory.max 2>/dev/null)"
echo "vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"
sha256sum "$MODEL"
rc_all=0
for cell in "collapse:two_step" "challenger:json_instructed"; do
  name=${cell%%:*}
  cue=${cell#*:}
  echo "=== smoke $name: --chat-format role_split --cue $cue $(date -Is)"
  start=$(date +%s)
  python3 .t9bcb/tiel_e3e_arm.py --suite quality \
    --cue "$cue" --chat-format role_split --json-contract question \
    --model "$MODEL" \
    --backend vulkan --gpu-layers "$NGL" --threads 4 \
    --devset docs/evidence/tiel_corrected_chunks/devset_001.jsonl --items 6 \
    --out "$OUT/smoke_${name}.json" --placement-out "$OUT/smoke_${name}_placement.json" \
    > "$OUT/logs/smoke_${name}.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== smoke $name exit=$rc wall=$((end - start))s $(date -Is)"
  grep -m1 '^- framing:' "$OUT/logs/smoke_${name}.log" || true
  tail -20 "$OUT/logs/smoke_${name}.log"
  if [ "$rc" -ne 0 ]; then
    echo "!! smoke $name failed; stopping here so no claim is made from a failed cell"
    rc_all="$rc"
  fi
done
sha256sum "$MODEL"
echo "tiel e3e-arm smoke done $(date -Is) rc=$rc_all"
exit "$rc_all"
