#!/bin/bash
# E3c-Tiel smoke under an UNLIMITED systemd scope. The worker's own scope is capped at 4 GiB
# (`hermes-worker-kanban-t_a58f8b67-run-169.scope`), which makes the 21 GB model thrash on every
# forward; E3 (t_6d2e084d) escaped that by running outside the worker shell's session.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
start=$(date +%s)
python3 tools/e3c_tiel_reproduce.py --suite quality \
  --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
  --backend vulkan --gpu-layers 9 --threads 4 \
  --devset .e3c_tiel/smoke_devset.jsonl \
  --out .e3c_tiel/smoke2_report.json \
  --placement-out .e3c_tiel/smoke2_placement.json > .e3c_tiel/smoke2_run.log 2>&1
rc=$?
end=$(date +%s)
echo "smoke2 exit=$rc wall=$((end - start))s cgroup=$(cat /proc/self/cgroup)"
exit "$rc"
