#!/bin/bash
# E3b re-measure (t_6952f0dd), chunk 002: the second 10 of the 20 published E3 items, written to
# a separate report dir so the per-policy files of the two chunks can be merged per policy.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p .e3b/logs .e3b/reports002
exec uv run --frozen python -u tools/e3b_label_policy.py run \
  --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
  --devset docs/evidence/e3_chunks/devset_002.jsonl \
  --cues shipped --rank shipped=bare --rank shipped=newline \
  --threads 4 --gpu-layers 7 \
  --out .e3b/after_002.json --report .e3b/after_002.md --report-dir .e3b/reports002
