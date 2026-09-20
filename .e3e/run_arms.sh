#!/bin/bash
# E3e (card t_4c48f40a) stage 4: the policy table — one arm per cell of
# {shipped, two_step, json_instructed} x {answer_sheet, role_split}, plus the amendment's two
# `json_contract=system` variants. The recipe is the E3d post-fix one (`.e3d/run_bench_arms.sh`) —
# the same box, the same 4B, the same committed 60 items, temperature 0, `--runs 1`, seed fixed —
# with one deliberate difference the card's rules ask for: **`--backend vulkan`, not `auto`**.
#
# Why the difference. On this box `--backend auto` has selected the `cpu` *bundle* while the
# engine log shows Vulkan0 doing the compute (`W_BACKEND_MISMATCH`), and one earlier arm in this
# very campaign (`.e3e/logs/bench_two_step_role_split.log`, 21:14) fell through to `CPU compute
# buffer size` rows — a genuine CPU placement. Two falls of the same dice are not a table: the
# card says single-backend runs, so every cell here is measured with the device named. The
# baseline cell (`shipped`/`answer_sheet`) is re-measured under the same flag as well, and the
# *freeze* the card asks for is then the comparison of that re-run against the committed
# `.e3d/bench_templated_shipped.json` (which the E3d card measured with `--backend auto`, i.e.
# the Vulkan-compute row behind a `cpu` claim): if the two agree item for item, the backend
# *label* moved no number and the published rows stand.
#
# Reproduce (from the repo root, with the pinned runtime + the 4B GGUF — the host must be free:
# a second >16 GiB mapped model on a 31 GiB box makes both readings unreadable):
#     bash .e3e/run_arms.sh                 # the freeze probe + every arm (resumes finished arms)
#     bash .e3e/run_arms.sh probe           # just the six-item freeze probe
#     bash .e3e/run_arms.sh arm <label>     # one arm
set -u
cd "$(dirname "$0")/.." || exit 126
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export GGUFONE_HOME="${GGUFONE_HOME:-$PWD/.e3e/ggufone-home}"
export GGUFONE_RUNTIME_DIR="${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan}"
mkdir -p "$GGUFONE_HOME" .e3e/logs
MODEL="${GGUFONE_E3E_MODEL:-/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"
BACKEND="${GGUFONE_E3E_BACKEND:-vulkan}"

if [ ! -f "$MODEL" ]; then
    echo "model is not a file: $MODEL" >&2
    exit 126
fi

# label|extra flags
ARMS=(
  "shipped_answer_sheet|--cue shipped --chat-format answer_sheet"
  "shipped_role_split|--cue shipped --chat-format role_split"
  "two_step_answer_sheet|--cue two_step --chat-format answer_sheet"
  "two_step_role_split|--cue two_step --chat-format role_split"
  "json_instructed_answer_sheet|--cue json_instructed --chat-format answer_sheet"
  "json_instructed_role_split|--cue json_instructed --chat-format role_split"
  "json_instructed_answer_sheet_system|--cue json_instructed --chat-format answer_sheet --json-contract system"
  "json_instructed_role_split_system|--cue json_instructed --chat-format role_split --json-contract system"
)

campaign=.e3e/logs/campaign.log
{
    echo "=== campaign start $(date -Is) ==="
    echo "model:  $MODEL"
    echo "sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "runtime: $GGUFONE_RUNTIME_DIR"
    echo "backend: $BACKEND  threads: 4  runs: 1  temperature: 0"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
} >>"$campaign" 2>&1

probe() {
  # The freeze probe is the *baseline's own recipe*: `.e3d/bench_templated_shipped.json` was
  # measured with `--backend auto`, so only that command can be byte-identical to it. The table's
  # own re-score of the same cell (`.e3e/bench_shipped_answer_sheet.json`, `--backend vulkan`) is
  # the arm below, and `tools/e3e_roles_decision.py --placement-probe` reads the two together.
  echo "=== freeze probe: the default cell, six items, --backend auto (the baseline's recipe) ===" \
    | tee -a "$campaign"
  # shellcheck disable=SC2086
  uv run --frozen python -u tools/e2_reproduce.py --suite quality \
    --model "$MODEL" --backend auto --threads 4 --items 6 \
    --out .e3e/probe_default.json >.e3e/logs/probe_default.log 2>&1 \
    && echo "probe ok" || echo "PROBE FAILED"
}

arm() {
  local label="$1" flags="$2"
  local out=".e3e/bench_${label}.json"
  local state
  state=$(python3 .e3e/arm_ready.py "$out" 2>&1) && {
      echo "=== arm $label: already measured ($state) ===" | tee -a "$campaign"
      return 0
  }
  echo "=== arm $label ($state) $(date -Is) ===" | tee -a "$campaign"
  # shellcheck disable=SC2086
  uv run --frozen python -u tools/e2_reproduce.py --suite quality \
    --model "$MODEL" --backend "$BACKEND" --threads 4 --items 60 $flags \
    --out "$out" >".e3e/logs/bench_${label}.log" 2>&1 \
    && echo "arm ok: $label ($(date -Is))" | tee -a "$campaign" \
    || echo "ARM FAILED: $label" | tee -a "$campaign"
}

case "${1:-all}" in
  probe) probe ;;
  arm)
    shift
    for entry in "${ARMS[@]}"; do
      [ "${entry%%|*}" = "${1:-}" ] && arm "${entry%%|*}" "${entry#*|}"
    done ;;
  *)
    probe
    for entry in "${ARMS[@]}"; do
      arm "${entry%%|*}" "${entry#*|}"
    done
    echo "=== arms done $(date -Is) ===" | tee -a "$campaign" ;;
esac
