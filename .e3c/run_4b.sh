#!/bin/bash
# E3c cue-shape sweep (card t_6c119626), the 4B control run: the same 6 items E3b (t_6952f0dd)
# measured, the shipped cue (control) plus the mid-answer cue shapes, all five label renderings
# from the same rows, and the ranked readout for three policies (the shipped policy E3b published
# and the two shapes this card proposes).
#
# CPU-only and deliberately so: the E2 4B baseline (`coverage ~1%`, 60 items) is a CPU number, and
# `--hide-devices` pins the run to the CPU (the sibling campaign owns the GPU on this box).
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
mkdir -p .e3c/logs
exec uv run --frozen python -u tools/e3c_cue_shapes.py run \
  --model /var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf \
  --devset docs/evidence/e3_chunks/devset_001.jsonl \
  --ids c01 c02 s01 s02 n01 n02 \
  --rank shipped=bare --rank two_step_shipped=bare --rank answer_is=bare \
  --backend cpu --gpu-layers 0 --hide-devices --threads 4 \
  --states-home /work/e3c/states-4b \
  --out .e3c/spark4b.json --report .e3c/spark4b.md
