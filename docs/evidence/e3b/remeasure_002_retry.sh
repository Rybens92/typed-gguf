#!/bin/bash
# E3b re-measure chunk 002 driver (t_6952f0dd), v2: host VRAM pressure edition.
#
# This worker container shares the host's RTX 3060 Ti (8 GiB) with the desktop and sibling
# containers, and the host's pids cgroup (256) is near its cap from other tenants — neither is
# under this card's control. Chunk 002 already died once with `E_BACKEND_OOM ... needed ~218 MiB`
# while a sibling held the board, and a launch that waits for a *clean* box never starts.
#
# So the rule is: take a short, bounded look at the board; if there is room for a GPU placement,
# use it (the loader's own ladder degrades anyway); otherwise measure CPU-only. Placement never
# changes what the card measures — coverage is read from the cue row of the same weights and the
# same prompt — it changes only the wall clock, and the run record states `placement` either way.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
LOG=.e3b/logs/remeasure_002.log
: >> "$LOG"

for attempt in 1 2; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "$attempt" = 1 ] && [ "$free" -ge 3072 ]; then
        layers=7
    else
        layers=0
    fi
    echo "=== attempt $attempt $(date -u '+%H:%M:%S') gpu_layers=$layers vram_free=${free}MiB pids=$(cat /sys/fs/cgroup/pids.current 2>/dev/null)/256 ===" >> "$LOG"
    uv run --frozen python -u tools/e3b_label_policy.py run \
        --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
        --devset docs/evidence/e3_chunks/devset_002.jsonl \
        --cues shipped --rank shipped=bare --rank shipped=newline \
        --threads 4 --gpu-layers "$layers" \
        --out .e3b/after_002.json --report .e3b/after_002.md --report-dir .e3b/reports002 >> "$LOG" 2>&1
    rc=$?
    echo "attempt $attempt exit=$rc" >> "$LOG"
    [ "$rc" = 0 ] && break
    sleep 30
done
echo "EXIT=$rc" >> "$LOG"
exit "$rc"
