#!/usr/bin/env bash
# Pre-fix control for the two CI steps added by card t_31b3943a: run the SAME commands the workflow
# runs, in a worktree of the parent commit 4e1d549, and show that the automation would have caught
# the regression (exit 4 / an AttributeError reason) instead of passing.
#
#   tools/rehearse_bench_placement_ci_red.sh <log-dir> [<rev>]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REV="${2:-4e1d549}"
LOG="${1:-$(mktemp -d /tmp/bench-placement-red-XXXX)}"
mkdir -p "$LOG"
LOG="$(cd "$LOG" && pwd)"          # absolute: the steps below cd into the worktree
TREE="$(mktemp -d /tmp/red-tree-XXXX)/repo"
echo "parent-rev control: rev=$REV log=$LOG tree=$TREE"
cd "$REPO" || exit 2
git worktree add --detach "$TREE" "$REV" >"$LOG/worktree.out" 2>"$LOG/worktree.err" || exit 3
git -C "$TREE" log --oneline -1

MODEL="${TYPED_GGUF_GATE_MODEL:-/var/home/rybens/.cache/llama.cpp/Qwen3.5-0.8B-UD-Q4_K_XL.gguf}"
RT="${TYPED_GGUF_RUNTIME_DIR:-/var/home/rybens/.hermes/runtime/b11026-linux-x64-cpu}"

# ---- CI step 1: bench --gpu-layers 4 on the pinned bundle (a crash happens before any load)
cd "$TREE" || exit 2
timeout 900 env TYPED_GGUF_RUNTIME_DIR="$RT" uv run typed-gguf bench --suite latency \
    --model "$MODEL" --gpu-layers 4 --runs 1 --sizes 256 --threads 2 \
    --json --out "$LOG/placement.json" >"$LOG/ci_step1.out" 2>"$LOG/ci_step1.err"
echo "$?" >"$LOG/ci_step1.exit"

# ---- CI step 2: the fake-alloc bundle, where the retry must be a typed row
mkdir -p "$LOG/fake-bundle"
cc -shared -fPIC -O1 -o "$LOG/fake-bundle/libllama.so" tools/fixtures/fit_oom_bundle.c \
    2>"$LOG/fake_bundle_build.err" \
    && cc -shared -fPIC -O1 -o "$LOG/fake-bundle/libggml.so" tools/fixtures/fit_oom_bundle.c \
       2>>"$LOG/fake_bundle_build.err"
echo "$?" >"$LOG/fake_bundle_build.exit"
uv run python tools/fit_oom_probe.py --make-gguf "$LOG/synthetic.gguf" >"$LOG/make_gguf.out" 2>&1
echo "$?" >"$LOG/make_gguf.exit"
timeout 600 env -u TYPED_GGUF_RUNTIME_DIR TYPED_GGUF_FAKE_OOM_ALL=1 \
    TYPED_GGUF_BENCH_RUNTIME_DIR="$LOG/fake-bundle" \
    uv run typed-gguf bench --suite throughput --model "$LOG/synthetic.gguf" --gpu-layers 4 \
    --runs 1 --json --out "$LOG/placement-oom.json" >"$LOG/ci_step2.out" 2>"$LOG/ci_step2.err"
echo "$?" >"$LOG/ci_step2.exit"

echo
echo "== step 1 (pinned bundle, --gpu-layers 4): exit=$(cat "$LOG/ci_step1.exit")"
tail -2 "$LOG/ci_step1.err" | sed 's/^/   | /'
echo "== step 2 (fake-alloc bundle): exit=$(cat "$LOG/ci_step2.exit")"
head -c 400 "$LOG/placement-oom.json" | sed 's/^/   | /'
echo
echo "control finished: $LOG   (the CI assertions MUST fail on this rev: that is the point)"
