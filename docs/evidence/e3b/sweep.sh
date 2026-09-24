#!/bin/bash
# E3b sweep: 6 dev items (2 per type) from the E3 chunk 001, every cue x label variant,
# the two cheap ranked policies, the kept-prefix control on the first 2 items, one cross-check.
set -x
cd /workspace/ggufone
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
exec uv run --frozen python tools/e3b_label_policy.py run \
  --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
  --devset docs/evidence/e3_chunks/devset_001.jsonl --per-type 2 \
  --rank shipped=bare --rank shipped=newline \
  --cross-check 1 --extra-prefix kept --extra-prefix-items 2 \
  --threads 4 --gpu-layers 7 \
  --out .e3b/sweep.json --report .e3b/sweep.md --report-dir .e3b/reports
