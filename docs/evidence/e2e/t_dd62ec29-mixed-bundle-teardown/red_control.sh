#!/bin/sh
# Card t_dd62ec29, the RED side: the *parent* tree (1b7192f) running the operator's exact mixed
# command — two bundles in one process, which is what the fix stops doing.
#
#   sh red_control.sh <model.gguf> <label>
#
# PYTHONPATH pins the pre-card source tree (`worktree add /work/t603-dd62-red 1b7192f`); the venv
# is this clone's, so only `src/` differs between the two runs.
set -u
cd /work/t603-dd62-1451 || exit 1
LOG=.e2e/t_dd62ec29-mixed-bundle-teardown/logs
export GGUFONE_RUNTIME_DIR=/work/t603-runtime/b11026-linux-x64-cpu
export GGUFONE_BENCH_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
export PYTHONPATH=/work/t603-dd62-red/src
.venv/bin/python -m ggufone bench --suite throughput --model "$1" --backend all \
    --runs 1 --threads 4 --sizes 64 --json > "$LOG/red_$2.raw" 2>&1
echo "EXIT=$?" > "$LOG/red_$2.exit"
echo "red_$2: $(cat "$LOG/red_$2.exit")"
tail -c 400 "$LOG/red_$2.raw"
