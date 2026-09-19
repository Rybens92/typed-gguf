#!/bin/bash
# E3c cue-shape sweep (card t_6c119626), the Occamy run: the same 6 items E3b (t_6952f0dd)
# measured, on the same box, with the same placement (Vulkan, 7 GPU layers) — the model E3b
# measured `<|im_end|>` at p ≈ 1.0 on. Chunked 3+3 (a killed run loses one chunk, never the
# campaign), merged by `.e3c_scratch/merge_runs.py` before the report is rendered.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/tmp
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p .e3c/logs
chunk="${1:?usage: run_occamy.sh A|B}"
case "$chunk" in
  A) ids="c01 c02 s01" ;;
  B) ids="s02 n01 n02" ;;
  *) echo "unknown chunk: $chunk" >&2; exit 2 ;;
esac
exec uv run --frozen python -u tools/e3c_cue_shapes.py run \
  --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
  --devset docs/evidence/e3_chunks/devset_001.jsonl \
  --ids $ids \
  --rank shipped=bare --rank two_step_shipped=bare --rank answer_is=bare \
  --backend vulkan --gpu-layers 7 --threads 4 \
  --states-home /work/e3c/states-occamy \
  --out ".e3c/occamy_${chunk}.json" --report ".e3c/occamy_${chunk}.md"
