#!/usr/bin/env bash
# Real-purpose run driver for the MiMo-V2.6-Distill-Qwen-9B arm (card t_d199e09c).
#
# Same protocol as the committed 4B arm (docs/evidence/real-purpose-4b-2026-09-22/run.sh) and
# its Tiel sibling (docs/evidence/real-purpose-tiel-35b-2026-09-22/run_tiel.sh): one
# `typed-gguf run` per item against a warm keep host, every engine call's exit code asserted,
# wall clock + exact command line recorded in $BASE/run.log, and a run that cannot fail is not
# a gate — exit 0 only when all 30 items exit 0 *and* leave their --out payload.
#
# This script is self-contained: the item states are exported from the items.jsonl sitting next
# to it, and items.jsonl/questions.json are byte-identical copies of the frozen 4B inputs
# (sha256 68a8984e…/b3b72582…, quoted in the evidence doc).
#
# BASE resolution: $REAL_PURPOSE_BASE, else the directory of this script.
#
# TWO documented deviations from the 4B arm's protocol, both placement/performance knobs, neither
# a decision knob (same prompt, same labels, same scoring — see the evidence doc §1):
#
#   --threads 4      the engine default is os.cpu_count() = 24 host CPUs, but the worker sandbox
#                    has a 2-CPU cgroup quota; 4 is the setting the repository's published Tiel
#                    rows use (docs/BENCHMARKS.md §7.4/§7.4.2). Same deviation the Tiel arm made.
#
#   --fit-target 512 the placement knob that makes the OWNER'S request possible: at the engine's
#                    default margin (1024 MiB) the fit plan offloads 29 of 32 layers; at 512 MiB
#                    it offloads 32/32 (fully GPU-resident). Everything else about the plan stays
#                    at its default (n_ctx 4096, n_seq_max 8, kv_type auto -> q4_0 by the ladder).
#                    Set FIT_TARGET= to reproduce the default-margin plan (29/32 layers).
#
# MODEL: the production path first, then the workspace copy this arm downloaded (the sandbox's
# ~/.hermes/models is read-only, so host pickup is a separate step).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${REAL_PURPOSE_BASE:-$HERE}"
export REAL_PURPOSE_BASE="$BASE"          # export_states.py / report_mimo.py resolve the same way
mkdir -p "$BASE" || { echo "cannot create BASE=$BASE" >&2; exit 1; }
export TYPED_GGUF_HOME="${TYPED_GGUF_HOME:-$BASE/home}"
export TYPED_GGUF_RUNTIME_DIR="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
MODEL="${MODEL:-/var/home/rybens/.hermes/models/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf}"
[ -f "$MODEL" ] || MODEL="$BASE/../models/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
ITEM_TIMEOUT="${ITEM_TIMEOUT:-600}"
THREADS="${THREADS-4}"
FIT_TARGET="${FIT_TARGET-512}"
EXTRA=()
[ -n "$THREADS" ] && EXTRA+=(--threads "$THREADS")
[ -n "$FIT_TARGET" ] && EXTRA+=(--fit-target "$FIT_TARGET")
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
for f in items.jsonl questions.json; do
  [ -f "$BASE/$f" ] || cp "$HERE/$f" "$BASE/$f" || { echo "cannot stage $f into $BASE" >&2; exit 1; }
done
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO" || exit 1
mkdir -p "$BASE/out"
python3 "$HERE/export_states.py" || exit 1
{
  echo "HOST $(uname -n) $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL home=$TYPED_GGUF_HOME runtime=$TYPED_GGUF_RUNTIME_DIR"
  echo "PLACEMENT threads=${THREADS:-default} fit_target_mb=${FIT_TARGET:-default} "
} > "$BASE/run.log"
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