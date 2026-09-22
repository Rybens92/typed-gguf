#!/usr/bin/env bash
# exp1 driver — one `typed-gguf run` per item against the warm keep host.
# Every engine call: exit code asserted, wall clock + exact command line recorded.
set -u
export TYPED_GGUF_HOME=/work/t977-typed-gguf/home
export TYPED_GGUF_RUNTIME_DIR=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
MODEL=/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf
BASE=/work/t977-typed-gguf/exp1
cd /workspace/ggufone || exit 1
mkdir -p "$BASE/out"
: > "$BASE/run.log"
fail=0
for f in "$BASE"/items/*.txt; do
  id=$(basename "$f" .txt)
  cmd=(uv run typed-gguf run --questions "$BASE/questions.json" --state "@$f" --model "$MODEL"
       --out "$BASE/out/$id.json" --keep-alive 10m)
  start=$(date +%s%3N)
  "${cmd[@]}" > "$BASE/out/$id.stdout" 2> "$BASE/out/$id.stderr"
  code=$?
  end=$(date +%s%3N)
  wall=$((end - start))
  printf '%s exit=%s wall_ms=%s :: %s\n' "$id" "$code" "$wall" "${cmd[*]}" >> "$BASE/run.log"
  if [ "$code" -ne 0 ]; then
    fail=$((fail + 1))
    echo "FAILED $id exit=$code" >&2
    tail -3 "$BASE/out/$id.stderr" >&2
  fi
done
echo "items=$(ls "$BASE"/items/*.txt | wc -l) failed=$fail"
date -u +"END %Y-%m-%dT%H:%M:%SZ"
