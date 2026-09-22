#!/usr/bin/env bash
# Real-purpose run driver (card t_977ad206): one `typed-gguf run` per item, warm keep host.
# Every engine call: exit code asserted, wall clock + exact command line recorded in run.log.
#
# BASE resolution: $REAL_PURPOSE_BASE, else the directory of this script. That is the only knob
# the driver needs — its data home (keep socket, states, fit cache) defaults to "$BASE/home", i.e.
# inside the run's own scratch dir, so no machine-local path is ever named. Reruns on any host
# where this checkout + the 4B model exist. Point TYPED_GGUF_HOME elsewhere to reuse an existing
# fit cache / resident host instead.
# The 4B model and the extracted llama.cpp runtime default to this box's paths; MODEL and
# TYPED_GGUF_RUNTIME_DIR override both.
#
# Exit status: 0 only when every item's engine call exits 0 *and* leaves its --out payload;
# otherwise the failing ids are listed on stderr and the driver exits 1. A reproduction driver
# that cannot fail is not a gate.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${REAL_PURPOSE_BASE:-$HERE}"
export REAL_PURPOSE_BASE="$BASE"          # export_states.py / analyze.py resolve the same way
mkdir -p "$BASE" || { echo "cannot create BASE=$BASE" >&2; exit 1; }
export TYPED_GGUF_HOME="${TYPED_GGUF_HOME:-$BASE/home}"
export TYPED_GGUF_RUNTIME_DIR="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
MODEL="${MODEL:-/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"
for f in items.jsonl questions.json; do
  [ -f "$BASE/$f" ] || cp "$HERE/$f" "$BASE/$f" || { echo "cannot stage $f into $BASE" >&2; exit 1; }
done
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO" || exit 1
mkdir -p "$BASE/out"
python3 "$HERE/export_states.py" || exit 1
: > "$BASE/run.log"
echo "BASE=$BASE home=$TYPED_GGUF_HOME model=$MODEL"
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
  elif [ ! -s "$BASE/out/$id.json" ]; then
    fail=$((fail + 1))
    echo "FAILED $id exit=0 but wrote no payload to $BASE/out/$id.json" >&2
  fi
done
items=$(ls "$BASE"/items/*.txt | wc -l)
echo "items=$items failed=$fail"
if [ "$fail" -ne 0 ]; then
  echo "REPRODUCE FAILED: $fail/$items items — see $BASE/run.log" >&2
  exit 1
fi
date -u +"END %Y-%m-%dT%H:%M:%SZ"
