#!/usr/bin/env bash
# E1c FIX host gate (card t_8cb0a05e requirement 5): the fit path on a BUSY desktop, plus a
# fake-allocation-failure world that needs no GPU.
#
#   tools/host_gate_e1c_fit.sh [LOG_DIR]
#
# Default LOG_DIR: $HOME/.ggufone-fit-gate-<utc timestamp>. Raw stdout/stderr, exit codes, timings
# and the free-VRAM reading of every step land there; tools/host_gate_e1c_fit_summary.py folds the
# directory into host_gate_e1c_fit.json.
#
# Why this gate: the operator's box reported 8192 MiB *total* / 1112 MiB *free* (the desktop held
# ~6.8 GB) and the default `--fit` run died with a 1.06 GB allocation it could never satisfy — and
# the error said `E_MODEL_ARCH_UNSUPPORTED`. This gate requires, for every fit-touching step:
#   (a) the free-VRAM number as the driver reports it, before and after,
#   (b) the plan bounded by that number (`budget_bytes`, `n_gpu_layers`),
#   (c) `--fit-target` actually changing the plan,
#   (d) a run that degrades instead of dying, with `engine.placement` + the warnings,
#   (e) `--no-fit` saying "CPU only" out loud,
#   (f) the fake-OOM worlds: degrade-to-CPU succeeds, nothing-fits answers `E_BACKEND_OOM`.
#
# Steps (all raw output kept):
#   0  host facts + the free-VRAM reading (nvidia-smi --query-gpu=memory.total,memory.free)
#   1  pytest -q                      offline suite
#   2  fit --json --no-cache          the plan + `host.vram_free_bytes`
#   3  fit --json --no-cache --fit-target 5200   the target must bound the plan
#   4  run --no-fit-cache             a real (busy) desktop run: exit 0, placement named
#   5  run --no-fit                   "fit disabled, CPU only" in engine.placement
#   6  fake OOM bundle                probe: degrade to CPU succeeds
#   7  fake OOM bundle (all rungs)    probe: E_BACKEND_OOM with free/needed bytes + hints
#
# Sandbox rehearsal: set GGUFONE_GATE_FAKE_DRIVER=<dir> to prepend a fake `nvidia-smi` (the busy
# desktop), and GGUFONE_GATE_MODEL=<file> to point at the pinned model. The summary records which
# vehicle produced the run (`is_host_run`), so sandbox numbers can never be read as host numbers.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${1:-$HOME/.ggufone-fit-gate-$STAMP}"
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 2

MODEL="${GGUFONE_GATE_MODEL:-$HOME/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"
FAKE_DRIVER="${GGUFONE_GATE_FAKE_DRIVER:-}"
if [ -n "$FAKE_DRIVER" ]; then PATH="$FAKE_DRIVER:$PATH"; fi

echo "ggufone E1c fit host gate"
echo "  repo    : $REPO"
echo "  logs    : $LOG_DIR"
echo "  model   : $MODEL"
echo "  started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------- plumbing
step() {  # step <name> <budget-seconds> <cmd...>
    local name="$1" budget="$2"; shift 2
    local start end code
    echo
    echo "== [$name] $* (budget ${budget}s)"
    start=$(date +%s)
    timeout "$budget" "$@" >"$LOG_DIR/$name.out" 2>"$LOG_DIR/$name.err"
    code=$?
    end=$(date +%s)
    echo "$code" >"$LOG_DIR/$name.exit"
    echo "$((end - start))" >"$LOG_DIR/$name.elapsed"
    echo "   -> exit=$code elapsed=$((end - start))s  ($LOG_DIR/$name.out)"
    return 0
}

free_vram() {  # prints one line: "total_mib free_mib source"
    local total free value
    if command -v nvidia-smi >/dev/null 2>&1; then
        value="$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
        if [ -n "$value" ]; then echo "$value nvidia-smi"; return 0; fi
    fi
    if [ -r /sys/class/drm/card0/device/mem_info_vram_total ]; then
        total="$(cat /sys/class/drm/card0/device/mem_info_vram_total)"
        free="$(( (total - $(cat /sys/class/drm/card0/device/mem_info_vram_used)) / 1048576 ))"
        echo "$(( total / 1048576 )), $free amdgpu-sysfs"
        return 0
    fi
    echo "?, ? none"
}

# ---------------------------------------------------------------- 0. host facts + free VRAM
{
    echo "# host facts"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -a)"
    echo "git_head: $(git -C "$REPO" rev-parse HEAD 2>&1) $(git -C "$REPO" status --porcelain | head -3)"
    echo "python: $(python3 -V 2>&1)  uv: $(uv --version 2>&1)"
    echo "nvidia_smi_which: $(command -v nvidia-smi || echo '<absent>')"
    echo "fake_driver: ${FAKE_DRIVER:-<none>}"
    echo "free_vram_before: $(free_vram)"
    echo "nvidia_smi_L:"; nvidia-smi -L 2>&1 | sed 's/^/  /'
    echo "nvidia_smi_query_raw:"; nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv 2>&1 | sed 's/^/  /'
    echo "dri_nodes:"; ls -l /dev/dri 2>&1 | sed 's/^/  /'
    echo "ggufone_home: $(python3 - <<'PY'
import os, pathlib
print(os.environ.get("GGUFONE_HOME") or (os.environ.get("XDG_DATA_HOME")
      or pathlib.Path.home() / ".local" / "share") / "ggufone")
PY
)"
    echo "model: $MODEL ($(stat -c '%s bytes' "$MODEL" 2>&1))"
    echo "runtime_dir: ${GGUFONE_RUNTIME_DIR:-<unset>}"
} >"$LOG_DIR/host_facts.txt" 2>&1
echo
sed -n '1,25p' "$LOG_DIR/host_facts.txt"

# ---------------------------------------------------------------- 1. offline suite
step pytest_before 900 uv run pytest -q

# ---------------------------------------------------------------- 2 + 3. the plan and the target
STATE_FILE="$LOG_DIR/state.txt"
printf '%s\n' \
  "Payment flow degradation. At 09:12 the checkout page began returning HTTP 500 for every" \
  "customer; the error rate is 100% on /checkout, 0% elsewhere. No deploy happened in the last" \
  "24 hours and the on-call engineer is already awake." >"$STATE_FILE"

if [ -f "$MODEL" ]; then
    step fit_plan 300 uv run ggufone fit "$MODEL" --no-cache --json
    step fit_target_bounded 300 uv run ggufone fit "$MODEL" --no-cache --json --fit-target 5200
    step run_busy_desktop 900 uv run ggufone ask --state "@$STATE_FILE" \
        --choice "area=Which team owns this incident?:billing|technical" --model "$MODEL" \
        --no-fit-cache --out "$LOG_DIR/run_busy_desktop.json"
    step run_no_fit 900 uv run ggufone ask --state "@$STATE_FILE" \
        --choice "area=Which team owns this incident?:billing|technical" --model "$MODEL" \
        --no-fit --format native --out "$LOG_DIR/run_no_fit.json"
else
    echo "warning: $MODEL is not on this box — steps 2-5 are recorded as SKIPPED" \
        | tee "$LOG_DIR/model_missing.txt"
    for name in fit_plan fit_target_bounded run_busy_desktop run_no_fit; do
        echo "SKIPPED: no model at $MODEL" >"$LOG_DIR/$name.err"
        echo 97 >"$LOG_DIR/$name.exit"
    done
fi

# ---------------------------------------------------------------- 4. the fake allocation failure
FIXTURE="$LOG_DIR/fake-bundle"
mkdir -p "$FIXTURE"
if command -v cc >/dev/null 2>&1; then
    cc -shared -fPIC -O1 -o "$FIXTURE/libllama.so" tools/fixtures/fit_oom_bundle.c 2>"$LOG_DIR/fake_bundle_build.err" \
        && cc -shared -fPIC -O1 -o "$FIXTURE/libggml.so" tools/fixtures/fit_oom_bundle.c 2>>"$LOG_DIR/fake_bundle_build.err"
    echo "$?" >"$LOG_DIR/fake_bundle_build.exit"
else
    echo "cc is not installed: the fake-OOM steps cannot build their bundle" \
        >"$LOG_DIR/fake_bundle_build.err"
    echo 97 >"$LOG_DIR/fake_bundle_build.exit"
fi

PROBE_MODEL="$MODEL"
if [ ! -f "$PROBE_MODEL" ]; then
    # Without the pinned model the probe still needs a real GGUF header: synthesise one (header +
    # tensor index only; the fake bundle never reads a tensor byte).
    uv run python tools/fit_oom_probe.py --make-gguf "$LOG_DIR/probe.gguf" >"$LOG_DIR/probe_gguf.out" 2>&1
    echo "$?" >"$LOG_DIR/probe_gguf.exit"
    PROBE_MODEL="$LOG_DIR/probe.gguf"
fi

if [ "$(cat "$LOG_DIR/fake_bundle_build.exit")" = "0" ] && [ -f "$PROBE_MODEL" ]; then
    echo
    echo "== [fake_oom_degrade] the loader against a bundle that cannot allocate"
    timeout 300 env -u GGUFONE_FAKE_OOM_ALL uv run python tools/fit_oom_probe.py \
        --model "$PROBE_MODEL" --runtime "$FIXTURE" --free-mib 1112 \
        --json "$LOG_DIR/fake_oom_degrade.json" \
        >"$LOG_DIR/fake_oom_degrade.out" 2>"$LOG_DIR/fake_oom_degrade.err"
    echo "$?" >"$LOG_DIR/fake_oom_degrade.exit"
    echo "   -> exit=$(cat "$LOG_DIR/fake_oom_degrade.exit")"

    echo
    echo "== [fake_oom_all_rungs] nothing fits -> E_BACKEND_OOM"
    timeout 300 env GGUFONE_FAKE_OOM_ALL=1 uv run python tools/fit_oom_probe.py \
        --model "$PROBE_MODEL" --runtime "$FIXTURE" --free-mib 1112 \
        --json "$LOG_DIR/fake_oom_all_rungs.json" \
        >"$LOG_DIR/fake_oom_all_rungs.out" 2>"$LOG_DIR/fake_oom_all_rungs.err"
    echo "$?" >"$LOG_DIR/fake_oom_all_rungs.exit"
    echo "   -> exit=$(cat "$LOG_DIR/fake_oom_all_rungs.exit")"
else
    for name in fake_oom_degrade fake_oom_all_rungs; do
        echo 97 >"$LOG_DIR/$name.exit"
    done
fi

# ---------------------------------------------------------------- 5. free VRAM after
{
    echo "free_vram_after: $(free_vram)"
    echo "# the plan files the run produced"
    for file in "$LOG_DIR"/run_busy_desktop.json "$LOG_DIR"/run_no_fit.json; do
        [ -f "$file" ] || continue
        echo "== $file"
        python3 - "$file" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
engine = payload.get("engine", {})
print("  n_gpu_layers:", engine.get("n_gpu_layers"))
print("  placement   :", json.dumps(engine.get("placement"), sort_keys=True))
print("  kv_type     :", engine.get("kv_type"))
print("  warnings    :", json.dumps(payload.get("warnings")))
print("  fit.budget  :", (engine.get("fit") or {}).get("budget_bytes"))
PY
    done
} >"$LOG_DIR/after.txt" 2>&1
echo
cat "$LOG_DIR/after.txt"

# ---------------------------------------------------------------- summary
python3 tools/host_gate_e1c_fit_summary.py "$LOG_DIR" || echo "warning: summary generation failed"
echo
echo "host gate finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "raw logs + host_gate_e1c_fit.json: $LOG_DIR"
