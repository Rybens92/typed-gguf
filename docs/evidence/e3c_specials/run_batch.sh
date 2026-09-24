#!/bin/bash
# Card t_635124bf: one serving-shaped batch (`ggufone run`, 20 questions, one state).
#
#   $1 = tree root (the source under test)      $2 = scratch dir (state + response)
#   $3 = model path                             $4 = backend label (cpu | vulkan)
#   $5 = n_seq_max
#
# Same shape as the Tiel campaign's `.e3c_tiel/run_batch.sh`: the committed E3 driver's
# `--suite batch`, `--items 20`, one state, `--backend <one backend>` (never `all` on this box).
set -u
TREE="$1"; SCRATCH="$2"; MODEL="$3"; BACKEND="${4:-cpu}"; NSEQ="${5:-8}"
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export PYTHONUNBUFFERED=1
# the batch child is `sys.executable -m ggufone`: the source tree must be on its path explicitly
# (the repo has zero runtime deps, so the system python serves) — exactly what the host runner did
export PYTHONPATH="$TREE/src"
if [ "$BACKEND" = "vulkan" ]; then
  export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/work/e3scratch/nvidia_egl_icd.json}
  export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
else
  export VK_DRIVER_FILES=/nonexistent/no-vulkan-icd.json
  export VK_ICD_FILENAMES=/nonexistent/no-vulkan-icd.json
fi
mkdir -p "$SCRATCH"
cd "$TREE" || exit 126
echo "tree=$TREE scratch=$SCRATCH model=$(basename "$MODEL") backend=$BACKEND n_seq_max=$NSEQ"
echo "start $(date -Is) cgroup=$(cat /sys/fs/cgroup/memory.max) pids=$(cat /sys/fs/cgroup/pids.current)"
python3 tools/e3_reproduce.py --suite batch \
  --model "$MODEL" --backend "$BACKEND" --threads 4 --items 20 --n-seq-max "$NSEQ" \
  --work-dir "$SCRATCH" --out "$SCRATCH/batch_response.json"
rc=$?
echo "batch exit=$rc $(date -Is)"
exit "$rc"
