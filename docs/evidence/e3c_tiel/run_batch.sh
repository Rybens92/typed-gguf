#!/bin/bash
# E3c-Tiel deliverable 4: the 20-question batch through the production `ggufone run` path
# (A-E3-2 shape: one state, N questions, waves when n_seq_max < 1+candidates).
#
#   * n_seq_max 8 = the fit plan's own concurrency bound (`ggufone fit` output)
#   * --backend vulkan --threads 4, one process, one state, --out the raw response
#   * run from an unlimited scope (see run_chunks.sh): the worker's own scope is 4 GiB
#   * PYTHONPATH=src: the batch child runs `python3 -m ggufone`, and the repo venv's python is a
#     system-python symlink whose site-packages do not carry ggufone — the source tree must be on
#     the child's path explicitly
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
export PYTHONPATH=/var/home/rybens/workspace/ggufone/src
export PYTHONUNBUFFERED=1
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
NSEQ=${NSEQ:-8}
echo "tiel batch start $(date -Is) pid $$ n_seq_max=$NSEQ"
python3 tools/e3c_tiel_reproduce.py --suite batch \
  --model "$MODEL" --backend vulkan --threads 4 --items 20 --n-seq-max "$NSEQ" \
  --work-dir .e3c_tiel --out .e3c_tiel/batch_response.json 2>&1
rc=$?
echo "tiel batch exit=$rc $(date -Is)"
exit "$rc"
