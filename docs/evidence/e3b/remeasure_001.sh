#!/bin/bash
# E3b re-measure (t_6952f0dd), chunk 001: the 20 published E3 items split 10+10 (the chunked
# campaign shape — a killed run loses one chunk, never the campaign).
# Both shipped-policy rankings (bare = the published policy, newline = the two-step readout),
# all five label renderings' coverage from the same cue rows, no extra prefix, no cross-check
# (the 6-item sweep cross-checked the probe against DecisionEngine: |d coverage| = 8.37e-08).
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p .e3b/logs .e3b/reports
exec uv run --frozen python -u tools/e3b_label_policy.py run \
  --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
  --devset docs/evidence/e3_chunks/devset_001.jsonl \
  --cues shipped --rank shipped=bare --rank shipped=newline \
  --threads 4 --gpu-layers 7 \
  --out .e3b/after_001.json --report .e3b/after_001.md --report-dir .e3b/reports
