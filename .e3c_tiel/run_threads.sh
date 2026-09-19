#!/bin/bash
# E3c-Tiel deliverable 5: the threads probe — `llama-bench` at the fitted placement (ngl 9) for
# -t 4 / 8 / 12, the same probe E3 ran on Occamy (E3 §6.5: 4 won).
#
# E3's bundle carries `libllama-bench-impl.so` (the dd-linked shape) rather than a `llama-bench`
# executable; the probe uses whichever the bundle provides. `-ngl` is the placement the quality
# campaign used, `-p 64`/`-n 8` are E3's own sizes so the two probes line up.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
unset VK_INSTANCE_LAYERS
BIN=${LLAMA_BENCH:-$GGUFONE_RUNTIME_DIR/llama-bench}
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
NGL=${NGL:-9}
echo "threads probe start $(date -Is); bin=$BIN"
if [ ! -x "$BIN" ]; then
  echo "!! no llama-bench executable in the bundle: $(ls "$GGUFONE_RUNTIME_DIR" | grep -i bench || echo none)"
  exit 127
fi
for t in 4 8 12; do
  echo "=== ngl=$NGL threads=$t ==="
  "$BIN" -m "$MODEL" -ngl "$NGL" -t "$t" -p 64 -n 8 -r 2 2>&1 | grep -E "qwen35moe|model|t/s" | head -5
done
echo "threads probe done $(date -Is)"
