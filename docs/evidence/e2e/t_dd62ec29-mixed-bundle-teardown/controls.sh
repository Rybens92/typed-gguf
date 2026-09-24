#!/bin/sh
# Card t_dd62ec29: the operator-box controls behind docs/evidence/e2_fix_t_dd62ec29_*.md.
#
# Every run is the *documented* command on this box, with the two bundles the operator installs:
#   cpu  = the pinned linux-x64-cpu asset of runtime.lock, unpacked at /work/t603-runtime
#   vulk = the installed Vulkan b11026 bundle under ~/.local/share/ggufone/runtime
# The container needs VK_DRIVER_FILES to see the operator's GPU (the NVIDIA EGL vendor library as
# ICD); without it the Vulkan rows still run, on the host CPU, and are flagged.
set -u
cd /work/t603-dd62-1451 || exit 1
CPU=/work/t603-runtime/b11026-linux-x64-cpu
HOST_RT=/var/home/rybens/.local/share/ggufone/runtime
VULK=$HOST_RT/b11026-linux-x64-vulkan
BIG=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf
SMALL=/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf
LOG=.e2e/t_dd62ec29-mixed-bundle-teardown/logs
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOG"

(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader || echo "nvidia-smi n/a") \
    > "$LOG/vram_before_controls.txt" 2>&1

# (A) the Vulkan bundle alone — the very row the mixed run isolates, in its own process, one bundle
GGUFONE_RUNTIME_DIR=$VULK .venv/bin/python -m ggufone bench --suite throughput --model "$BIG" \
    --backend vulkan --runs 1 --threads 4 --sizes 64 --json > "$LOG/vulkan_alone.raw" 2>&1
echo "EXIT=$?" > "$LOG/vulkan_alone.exit"

# (B) the operator's mixed command again (two bundles, the 4B model)
GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT \
    .venv/bin/python -m ggufone bench --suite throughput --model "$BIG" --backend all \
    --runs 1 --threads 4 --sizes 64 --json > "$LOG/after_mixed_2.raw" 2>&1
echo "EXIT=$?" > "$LOG/after_mixed_2.exit"

# (C) the same mixed command on the small model: both rows measured, on the same box
GGUFONE_RUNTIME_DIR=$CPU GGUFONE_BENCH_RUNTIME_DIR=$HOST_RT \
    .venv/bin/python -m ggufone bench --suite throughput --model "$SMALL" --backend all \
    --runs 1 --threads 4 --sizes 64 --json > "$LOG/after_mixed_small.raw" 2>&1
echo "EXIT=$?" > "$LOG/after_mixed_small.exit"

(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader || echo "nvidia-smi n/a") \
    >> "$LOG/vram_before_controls.txt" 2>&1
for name in vulkan_alone after_mixed_2 after_mixed_small; do
    printf '%s %s\n' "$name" "$(cat "$LOG/$name.exit")"
done
