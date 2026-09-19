#!/bin/bash
# E3b sweep (t_6952f0dd, run 161): 6 dev items (2 per type) from the E3 chunk 001, every cue x
# label variant, the two cheap ranked policies, the kept-prefix control on the first 2 items,
# one cross-check. Same shape as sweep.sh; adds TMPDIR + unbuffered output for the background log.
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p .e3b/logs .e3b/reports
exec uv run --frozen python -u tools/e3b_label_policy.py run \
  --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
  --devset docs/evidence/e3_chunks/devset_001.jsonl --per-type 2 \
  --rank shipped=bare --rank shipped=newline \
  --cross-check 1 --extra-prefix kept --extra-prefix-items 2 \
  --threads 4 --gpu-layers 7 \
  --out .e3b/sweep.json --report .e3b/sweep.md --report-dir .e3b/reports
