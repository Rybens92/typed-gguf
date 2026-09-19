#!/bin/bash
# E3c cue-shape sweep (card t_6c119626), the Occamy run: the same 6 items E3b (t_6952f0dd)
# measured, on the same box — the model E3b measured `<|im_end|>` at p ≈ 1.0 on.
#
# One dev item per invocation: a 23 GB model on the CPU costs minutes per item, and the box's
# cgroup pids cap (256) plus the sibling campaigns make a long-lived process a liability — a
# killed chunk loses one item, never the campaign. Pass `all` or an explicit id list.
#
# Placement: CPU (`--hide-devices`, `--gpu-layers 0`). E3b's run had 7 layers on the GPU; the GPU
# is held by a sibling campaign here (a Vulkan context still reserves ~1 GB of *device* memory
# even with 0 layers offloaded, which is what OOMs), so this run pins the CPU and the report
# states the claim plus the engine's own compute-buffer line. The label mass at a fixed row is a
# weight-level quantity; the placement difference is recorded, not hidden.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export TMPDIR=/tmp
export OMP_NUM_THREADS=4
export GGUFONE_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
mkdir -p .e3c/logs
ids="${1:?usage: run_occamy_items.sh all|c01 [c02 ...]}"
[ "$ids" = "all" ] && ids="c01 c02 s01 s02 n01 n02"
failures=0
for id in $ids; do
  log=".e3c/logs/occamy_${id}.log"
  out=".e3c/occamy_${id}.json"
  for attempt in 1 2 3; do
    # the pids cap is shared: wait for a shoehorn before spawning llama's thread pool
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
      [ "$(cat /sys/fs/cgroup/pids.current)" -lt 200 ] && break
      sleep 20
    done
    uv run --frozen python -u tools/e3c_cue_shapes.py run \
      --model /var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
      --devset docs/evidence/e3_chunks/devset_001.jsonl \
      --ids "$id" \
      --rank shipped=bare --rank two_step_shipped=bare --rank answer_is=bare \
      --backend cpu --gpu-layers 0 --hide-devices --threads 4 \
      --states-home /work/e3c/states-occamy \
      --out "$out" --report ".e3c/occamy_${id}.md" >> "$log" 2>&1 && break
    if ! grep -q "Resource temporarily unavailable" "$log"; then
      echo "$id: failed for a reason that is not thread pressure (attempt $attempt)" >&2
      break
    fi
    echo "$id: thread pressure, retrying (attempt $attempt)" >&2
    sleep 30
  done
  if [ -f "$out" ]; then echo "$id: ok -> $out"; else echo "$id: FAILED" >&2; failures=$((failures + 1)); fi
done
echo "failures: $failures"
exit $((failures > 0))
