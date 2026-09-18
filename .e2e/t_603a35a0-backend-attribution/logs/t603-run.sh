#!/usr/bin/env bash
# Card t_603a35a0 evidence driver: the two ways a bench row's backend label can lie.
#
#   usage: bash /work/t603-run.sh <tree> <tag>
#
# Both runs are the operator box's own setup: the installed Vulkan b11026 bundle
# (/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan, found by the default
# scan) plus the pinned CPU bundle from runtime.lock (linux-x64-cpu, sha256 verified) unpacked at
# /work/t603-runtime/b11026-linux-x64-cpu.
#
#   mixed      -> GGUFONE_RUNTIME_DIR=<cpu bundle> (explicit) + the default scan -> the auditor's
#                 two-bundle case: `--backend all` must not emit a `vulkan` row that ran on CPU
#   opoffload  -> only the Vulkan bundle is visible, so `cpu` resolves to *its* directory and
#                 llama.cpp's op offload runs the graph on the device under a `cpu` label
set -u
tree="${1:?usage: bash /work/t603-run.sh <tree> <tag>}"
tag="${2:?usage: bash /work/t603-run.sh <tree> <tag>}"

out=/work/t603-evidence
mkdir -p "$out"
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
model=/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf
cpu_bundle=/work/t603-runtime/b11026-linux-x64-cpu
vulkan_root=/var/home/rybens/.local/share/ggufone/runtime

cd "$tree" || exit 1

one() {  # one <name> <env-assignments> <args...>
    local name="$1"; shift
    local envs="$1"; shift
    local raw="$out/${tag}_${name}.raw"
    printf '=== %s %s | %s | ggufone bench %s\n' "$name" "$(date -u +%H:%M:%SZ)" "$envs" "$*" \
        | tee -a "$out/${tag}_driver.log"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader \
        >> "$out/${tag}_driver.log" 2>&1
    # shellcheck disable=SC2086
    env $envs timeout 1200 uv run --frozen ggufone bench --suite throughput --model "$model" \
        --runs 1 --threads 4 --sizes 64 --json "$@" > "$raw" 2>&1
    local rc=$?
    printf '=== %s exit=%s %s (raw: %s, %s bytes)\n' "$name" "$rc" "$(date -u +%H:%M:%SZ)" \
        "$raw" "$(wc -c < "$raw")" | tee -a "$out/${tag}_driver.log"
    return 0
}

one mixed "GGUFONE_RUNTIME_DIR=$cpu_bundle GGUFONE_BENCH_RUNTIME_DIR=$vulkan_root" --backend all
one opoffload "GGUFONE_BENCH_RUNTIME_DIR=$vulkan_root" --backend cpu
