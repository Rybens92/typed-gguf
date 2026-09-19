#!/bin/bash
# E3c-Tiel smoke: two dev items, the Tiel fit placement (ngl=9), on the host.
# Purpose: validate the placement-capturing driver end to end and get a per-item wall
# before the 60-item campaign is committed to.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
NGL=${NGL:-9}
start=$(date +%s)
python3 tools/e3c_tiel_reproduce.py --suite quality \
  --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
  --backend vulkan --gpu-layers "$NGL" --threads 4 \
  --devset .e3c_tiel/smoke_devset.jsonl \
  --out .e3c_tiel/smoke_report.json \
  --placement-out .e3c_tiel/smoke_placement.json > .e3c_tiel/smoke_run.log 2>&1
rc=$?
end=$(date +%s)
echo "smoke ngl=$NGL exit=$rc wall=$((end - start))s"
exit "$rc"
