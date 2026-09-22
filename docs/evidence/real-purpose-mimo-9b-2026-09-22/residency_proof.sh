#!/usr/bin/env bash
# Residency proof capture (card t_d199e09c AC2): cold-ish load of the MiMo Q4_K_M model
# through the same `typed-gguf run` invocation the driver uses, with the VRAM numbers and the
# keep host's own placement block captured while the model is resident.
#
# Usage: BASE=/work/t_d199e09c/mimo bash residency_proof.sh
set -u
BASE="${BASE:-/work/t_d199e09c/mimo}"
MODEL="${MODEL:-/work/t_d199e09c/models/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export REAL_PURPOSE_BASE="$BASE"
export TYPED_GGUF_HOME="${TYPED_GGUF_HOME:-$BASE/home}"
export TYPED_GGUF_RUNTIME_DIR="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
cd /workspace/ggufone || exit 1
mkdir -p "$BASE/proof"
out="$BASE/proof/residency_proof.txt"
FIT_KEYS="n_gpu_layers,n_ctx,kv_type,n_seq_max,est_weights_bytes,est_kv_bytes,est_total_bytes,budget_bytes,source,warnings,notes"
KEEP_KEYS="model,model_path,state,pid,requests,model_load_ms,placement,devices,key,loaded_s"
ENG_KEYS="engine.runtime,engine.backend,engine.effective_backend,engine.devices,engine.readout,engine.cue,engine.chat_format,engine.template,engine.kv_unified,engine.n_ctx,engine.n_seq_max,engine.kv_type,engine.n_gpu_layers,engine.placement,engine.fit,engine.keep,engine.device_buffers"
{
  echo "== residency proof (card t_d199e09c) =="
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model: $MODEL"
  echo "size: $(stat -c %s "$MODEL") bytes"
  echo "sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo
  echo "-- nvidia-smi before the load (desktop/voice-app floor already counted) --"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
  echo
  echo "-- fit plan for the placement this load will use --"
  echo "cmd: uv run typed-gguf fit $MODEL --print --json --no-cache --n-ctx 4096 --fit-ctx 4096 --fit-target 512"
  uv run typed-gguf fit "$MODEL" --print --json --no-cache --n-ctx 4096 --fit-ctx 4096 --fit-target 512 > "$BASE/proof/fit_4096_target512.json"
  python3 "$HERE/json_pluck.py" "$FIT_KEYS" "$BASE/proof/fit_4096_target512.json"
  echo
  echo "-- one real request through the driver's exact invocation --"
  cmd=(uv run typed-gguf run --questions "$BASE/questions.json" --state @"$BASE/items/t07.txt"
       --model "$MODEL" --threads 4 --fit-target 512 --out "$BASE/proof/t07.json" --keep-alive 1m)
  echo "cmd: ${cmd[*]}"
  start=$(date +%s%3N)
  "${cmd[@]}"
  code=$?
  end=$(date +%s%3N)
  echo "exit=$code wall_ms=$((end - start))"
  echo
  echo "-- nvidia-smi while the model is resident --"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
  echo
  echo "-- keep host's own placement block (from keep status --json) --"
  uv run typed-gguf keep status --json > "$BASE/proof/keep_status.json"
  python3 "$HERE/json_pluck.py" "$KEEP_KEYS" "$BASE/proof/keep_status.json"
  echo
  echo "-- the response's own engine block (from the payload it wrote) --"
  python3 "$HERE/json_pluck.py" "$ENG_KEYS" "$BASE/proof/t07.json"
  echo
  echo "-- keep host stopped --"
  uv run typed-gguf keep stop --json
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
} > "$out" 2>&1
echo "wrote $out"
