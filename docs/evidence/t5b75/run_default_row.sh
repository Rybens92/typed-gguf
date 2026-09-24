#!/bin/bash
# Policy v2 (card t_5b754458) — the re-measured 4B quality row under the **defaults**.
#
# The recipe is `.e3e/run_arms.sh`'s `json_instructed_role_split` arm with the flags removed: same
# box, same 4B, same committed 60 items, same `--backend vulkan --threads 4 --runs 5` instrument,
# plus one deliberate difference — **no `--cue` / `--chat-format` / `--json-contract` at all**, so
# what is measured is what a request that names nothing renders. The item-level comparison against
# the published arm is `.t5b75/compare_default_row.py`.
#
#     bash .t5b75/run_default_row.sh              # the 60-item row (writes the evidence JSON)
#     GGUFONE_T5B75_ITEMS=6 bash .t5b75/run_default_row.sh   # a smoke of the same recipe
set -u
cd "$(dirname "$0")/.." || exit 126
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export GGUFONE_HOME="${GGUFONE_HOME:-$PWD/.t5b75/ggufone-home}"
export GGUFONE_RUNTIME_DIR="${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
mkdir -p "$GGUFONE_HOME" .t5b75/logs
MODEL="${GGUFONE_T5B75_MODEL:-/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"
ITEMS="${GGUFONE_T5B75_ITEMS:-60}"
OUT="${GGUFONE_T5B75_OUT:-docs/evidence/e2_quality_v2_t_5b754458.json}"

if [ ! -f "$MODEL" ]; then
    echo "model is not a file: $MODEL" >&2
    exit 126
fi

{
    echo "=== default-row run $(date -Is) ==="
    echo "model:  $MODEL"
    echo "sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "runtime: $GGUFONE_RUNTIME_DIR"
    echo "items: $ITEMS  backend: vulkan  threads: 4  runs: 5 (harness default)  temperature: 1.0"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
} >> .t5b75/logs/run_default_row.log

uv run --frozen python -u tools/e2_reproduce.py --suite quality \
    --model "$MODEL" --backend vulkan --threads 4 --items "$ITEMS" \
    --out "$OUT" >".t5b75/logs/default_row.stdout" 2>&1
rc=$?
echo "e2_reproduce exit=$rc (stdout: .t5b75/logs/default_row.stdout, report: $OUT)"
exit "$rc"
