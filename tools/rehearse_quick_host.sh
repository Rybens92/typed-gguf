#!/usr/bin/env bash
# Host-side rehearsal of the `--quick` evidence (card t_f46cec41) for the models the *worker
# container cannot see*.
#
# The container's /var/home/rybens/.hermes/models/ carries Spark-X2.5-4B only; the operator's
# bigger GGUFs (Occamy 1.0, 23 GiB, `qwen35moe`; Tiel-Coder-35B-A3B, 21 GiB) are host-only, and a
# 23 GiB file cannot be measured inside the 8 GiB memory cgroup anyway. This script is the named
# vehicle for those numbers: run it on the host and the per-suite wall times land in one log dir.
#
#   tools/rehearse_quick_host.sh [log-dir] [threads]
#
# Every model that exists is measured with `tools/rehearse_quick.sh` (the same five-suite quick
# campaign the container ran); a model that is absent prints `skip (absent)` instead of failing.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
LOG="${1:-$(mktemp -d /tmp/quick-host-XXXX)}"
THREADS="${2:-4}"
mkdir -p "$LOG"
echo "log dir: $LOG"
echo "threads: $THREADS (this box's best-measured setting; docs/BENCHMARKS.md 3.5)"

MODELS=(
    "/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf"
    "/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"
    "/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf"
    "/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
)

for model in "${MODELS[@]}"; do
    name="$(basename "$model" .gguf)"
    echo
    echo "################ $name"
    if [ ! -f "$model" ]; then
        echo "skip (absent): $model"
        continue
    fi
    bash tools/rehearse_quick.sh "$LOG/$name" "$model" "$THREADS"
done

echo
echo "################ summary (suite <tab> exit <tab> report wall time in s)"
for walls in "$LOG"/*/quick_walls.txt; do
    [ -f "$walls" ] || continue
    echo "# $(dirname "$walls")"
    cat "$walls"
done
