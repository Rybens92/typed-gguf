#!/usr/bin/env bash
# Sandbox rehearsal of the two `.github/workflows/runtime-matrix.yml` steps added by
# card t_31b3943a ("bench + placement retry"), run with THIS box's pinned bundle and models.
#
# CI substitutes (stated, never hidden):
#   * /tmp/smoke.gguf  -> the local pinned 0.8B GGUF (the CI job downloads qwen2.5-0.5b)
#   * /tmp/ggufone-rt  -> /var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu (the pinned bundle)
# The commands, the assertions and the fake-OOM bundle are the ones the workflow runs.
#
#   tools/rehearse_bench_placement_ci.sh <log-dir>
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
LOG="${1:-$(mktemp -d /tmp/bench-placement-ci-XXXX)}"
mkdir -p "$LOG"
MODEL="${GGUFONE_GATE_MODEL:-/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf}"
RT="${GGUFONE_RUNTIME_DIR:-/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu}"
echo "rehearsal log dir: $LOG"
echo "model: $MODEL"
echo "runtime: $RT"

run_step() {  # run_step <name> <budget> <cmd...>
    local name="$1" budget="$2"; shift 2
    echo
    echo "== [$name] $* (budget ${budget}s)"
    local start end code
    start=$(date +%s)
    timeout "$budget" "$@" >"$LOG/$name.out" 2>"$LOG/$name.err"
    code=$?
    end=$(date +%s)
    echo "$code" >"$LOG/$name.exit"
    echo "   -> exit=$code elapsed=$((end - start))s"
    return 0
}

# ------------------------------------------------------------------ step A: the real bundle
run_step bench_placement 1800 env GGUFONE_RUNTIME_DIR="$RT" uv run ggufone bench --suite latency \
    --model "$MODEL" --gpu-layers 4 --runs 1 --sizes 256 --threads 2 \
    --json --out "$LOG/placement.json"
python3 -c "import json;r=json.load(open('$LOG/placement.json'));p=r['placement'];assert r['model_load']['n']==1,r['model_load'];assert p['requested']=='n_gpu_layers=4',p;assert p['used'] is not None,'the bench row must carry the placement the loader used';assert r['per_question'],'no decision rows'"
echo "   assertions: OK"

# ------------------------------------------------------------------ step B: the fake-OOM bundle
mkdir -p "$LOG/fake-bundle"
cc -shared -fPIC -O1 -o "$LOG/fake-bundle/libllama.so" tools/fixtures/fit_oom_bundle.c \
    2>"$LOG/fake_bundle_build.err" \
    && cc -shared -fPIC -O1 -o "$LOG/fake-bundle/libggml.so" tools/fixtures/fit_oom_bundle.c \
       2>>"$LOG/fake_bundle_build.err"
echo "$?" >"$LOG/fake_bundle_build.exit"
uv run python tools/fit_oom_probe.py --make-gguf "$LOG/synthetic.gguf" >"$LOG/make_gguf.out" 2>&1
echo "$?" >"$LOG/make_gguf.exit"
run_step bench_placement_oom 900 env -u GGUFONE_RUNTIME_DIR GGUFONE_FAKE_OOM_ALL=1 \
    GGUFONE_BENCH_RUNTIME_DIR="$LOG/fake-bundle" \
    uv run ggufone bench --suite throughput --model "$LOG/synthetic.gguf" --gpu-layers 4 \
    --runs 1 --json --out "$LOG/placement-oom.json"
test "$(cat "$LOG/bench_placement_oom.exit")" = 1
python3 -c "import json;r=json.load(open('$LOG/placement-oom.json'));row=r['backends'][0];reason=row['reason'];assert row['measured'] is False,row;assert 'E_BACKEND_OOM' in reason and '3 placement(s)' in reason,reason;assert 'AttributeError' not in reason and 'E_INTERNAL' not in reason,reason"
echo "   assertions: OK"

echo
echo "rehearsal finished: $LOG"
