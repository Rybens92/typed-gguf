#!/usr/bin/env bash
# Real-purpose run driver for the Tiel-Coder-35B arm (card t_0d5db2ef).
#
# Same protocol as the committed 4B arm (docs/evidence/real-purpose-4b-2026-09-22/run.sh):
# one `typed-gguf run` per item against a warm keep host, every engine call's exit code
# asserted and wall clock + exact command line recorded in $BASE/run.log.
#
# This script is self-contained (does not source the 4B receipt): the item states are
# exported from the items.jsonl sitting next to it, and items.jsonl/questions.json are
# byte-identical copies of the frozen 4B inputs (sha256 in the evidence doc).
#
# BASE resolution: $REAL_PURPOSE_BASE, else /work/t0d5-tiel when that run directory exists
# (the run this evidence came from), else the directory of this script.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BASE=/work/t0d5-tiel
if [ -n "${REAL_PURPOSE_BASE:-}" ]; then BASE="$REAL_PURPOSE_BASE"
elif [ -f "$DEFAULT_BASE/items.jsonl" ]; then BASE="$DEFAULT_BASE"
else BASE="$HERE"; fi
export TYPED_GGUF_HOME="${TYPED_GGUF_HOME:-$BASE/home}"
export TYPED_GGUF_RUNTIME_DIR="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
MODEL="${MODEL:-/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf}"
ITEM_TIMEOUT="${ITEM_TIMEOUT:-600}"
# THREADS: empty = engine default (os.cpu_count(), i.e. 24 in the worker sandbox whose CPU quota
# is 2). The Tiel arm runs at 4 — the same `--threads 4` the repository's published Tiel rows use
# (BENCHMARKS.md §7.4/§7.4.2) — because the engine's default spawns 24 ggml threads against a
# 2-CPU cgroup quota and stalls; the deviation is a performance knob only and is recorded in the
# evidence doc. Set THREADS= to reproduce the 4B arm's default exactly.
THREADS="${THREADS-4}"
EXTRA=()
[ -n "$THREADS" ] && EXTRA=(--threads "$THREADS")
mkdir -p "$BASE" || exit 1
for f in items.jsonl questions.json; do
  [ -f "$BASE/$f" ] || cp "$HERE/$f" "$BASE/$f" || { echo "cannot stage $f into $BASE" >&2; exit 1; }
done
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO" || exit 1
mkdir -p "$BASE/out"
python3 "$HERE/export_states.py" || exit 1
echo "HOST $(uname -n) $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL home=$TYPED_GGUF_HOME runtime=$TYPED_GGUF_RUNTIME_DIR" > "$BASE/run.log"
fail=0
shopt -s nullglob
states=("$BASE"/items/*.txt)
if [ "${#states[@]}" -eq 0 ]; then echo "no item state files under $BASE/items" >&2; exit 1; fi
for f in "${states[@]}"; do
  id=$(basename "$f" .txt)
  cmd=(uv run typed-gguf run --questions "$BASE/questions.json" --state "@$f" --model "$MODEL"
       "${EXTRA[@]}" --out "$BASE/out/$id.json" --keep-alive 10m)
  start=$(date +%s%3N)
  timeout "$ITEM_TIMEOUT" "${cmd[@]}" > "$BASE/out/$id.stdout" 2> "$BASE/out/$id.stderr"
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
[ "$fail" -eq 0 ]
