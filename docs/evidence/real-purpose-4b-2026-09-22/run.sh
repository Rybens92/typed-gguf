#!/usr/bin/env bash
# Real-purpose run driver (card t_977ad206): one `typed-gguf run` per item, warm keep host.
# Every engine call: exit code asserted, wall clock + exact command line recorded in run.log.
#
# BASE resolution: $REAL_PURPOSE_BASE, else /work/t977-typed-gguf/exp1 when that run directory
# exists (the run this evidence came from), else the directory of this script.
# The item states are exported from items.jsonl (see export_states.py) — nothing else is needed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export TYPED_GGUF_HOME="${TYPED_GGUF_HOME:-/work/t977-typed-gguf/home}"
export TYPED_GGUF_RUNTIME_DIR="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
MODEL="${MODEL:-/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"
DEFAULT_BASE=/work/t977-typed-gguf/exp1
if [ -n "${REAL_PURPOSE_BASE:-}" ]; then BASE="$REAL_PURPOSE_BASE"
elif [ -f "$DEFAULT_BASE/items.jsonl" ]; then BASE="$DEFAULT_BASE"
else BASE="$HERE"; fi
mkdir -p "$BASE"
for f in items.jsonl questions.json; do
  [ -f "$BASE/$f" ] || cp "$HERE/$f" "$BASE/$f" || { echo "cannot stage $f into $BASE" >&2; exit 1; }
done
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO" || exit 1
mkdir -p "$BASE/out"
python3 "$HERE/export_states.py" || exit 1
: > "$BASE/run.log"
fail=0
shopt -s nullglob
states=("$BASE"/items/*.txt)
if [ "${#states[@]}" -eq 0 ]; then echo "no item state files under $BASE/items" >&2; exit 1; fi
for f in "${states[@]}"; do
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
