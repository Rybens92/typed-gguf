#!/bin/sh
# Card t_57cc0179 — the *teardown-time* starve: hold VRAM while the child exits.
#
# `starve_and_run.sh` starves the device *before* the child starts, which is the right recipe for
# the other half of the shape (`E_BACKEND_OOM`: the loader's own fit ladder refuses the placement —
# `logs/pre_starved.raw`). It cannot reproduce the *crash*, because the 4B placement needs the
# device to hold ~2.5 GiB of weights + ~0.45 GiB of compute buffers *plus* the 1 GiB
# `--fit-target` margin at load time (~3.9 GiB free), and the defect happens at **teardown** with
# the device gone tight. So:
#
#   1. wait for a load window (>= LOAD_FREE MiB free);
#   2. start the one Vulkan child, and wait until its own stderr says the context exists
#      (`… Vulkan0 compute buffer size …`) — the model is loaded, the device is committed;
#   3. *then* start the dummy allocator, holding everything above LEAVE_FREE MiB, and keep holding
#      it while the child measures and tears down;
#   4. record the child's exit code, streams and the nvidia-smi snapshots.
#
#   sh starve_at_teardown.sh <tree> <tag> [load_free_mib] [leave_free_mib] [gdb]
set -u
TREE=${1:?tree}
TAG=${2:?tag}
LOAD_FREE=${3:-4200}
LEAVE_FREE=${4:-300}
GDB_MODE=${5:-off}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs
VULK=/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
MODEL=${REPRO_MODEL:-/var/home/rybens/.hermes/models/Qwen3.5-4B-Q4_0.gguf}
PY=/work/t57cc-ggufone/.venv/bin/python
export VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json
mkdir -p "$LOGS"

smi() { nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader \
        || echo "nvidia-smi n/a"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR==1 {print $1}' | tr -dc '0-9'; }

smi > "$LOGS/${TAG}_vram_before.txt"
WAITED=0
while [ "$WAITED" -lt 1800 ]; do
    NOW=$(free_mib)
    echo "$(date -u '+%H:%M:%S') free=${NOW}MiB want=${LOAD_FREE}MiB" >> "$LOGS/${TAG}_window.txt"
    [ "${NOW:-0}" -ge "$LOAD_FREE" ] && break
    sleep 15
    WAITED=$((WAITED + 15))
done
smi > "$LOGS/${TAG}_vram_window.txt"

cd "$TREE" || exit 1
export PYTHONPATH=$TREE/src
export PYTHONFAULTHANDLER=1
case "$GDB_MODE" in
    gdb) RUNNER="timeout 900 gdb -batch -nx -ex 'set pagination off' -ex run -ex 'bt 40' --args env"
         set -- env "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
             "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
             "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
             --runs 1 --threads 4 --sizes 64 --json ;;
    *)   RUNNER="env"
         set -- env "GGUFONE_RUNTIME_DIR=$VULK" "PYTHONPATH=$TREE/src" \
             "VK_DRIVER_FILES=$VK_DRIVER_FILES" "PYTHONFAULTHANDLER=1" \
             "$PY" -m ggufone bench --suite throughput --model "$MODEL" --backend vulkan \
             --runs 1 --threads 4 --sizes 64 --json ;;
esac

$RUNNER "$@" > "$LOGS/${TAG}.raw" 2> "$LOGS/${TAG}.stderr" &
CHILD=$!

# the child is loading: wait for its own log to prove the context exists (or for it to be gone)
LOADED=0
for _ in $(seq 1 120); do
    if ! kill -0 "$CHILD" 2>/dev/null; then break; fi
    if grep -q "compute buffer size" "$LOGS/${TAG}.stderr" 2>/dev/null; then LOADED=1; break; fi
    sleep 2
done
# …then wait for the *graph* buffers too: the context line prints before they are allocated, and a
# hold taken there starves the child's own run (measured: `E_BACKEND_OOM` mid-run, exit 1). The
# engine's Vulkan log goes quiet once the allocation phase is over, so the marker is "no new line
# for 4 s" — the measurement itself prints nothing.
QUIET=0
LAST=$(wc -c < "$LOGS/${TAG}.stderr" 2>/dev/null || echo 0)
for _ in $(seq 1 90); do
    kill -0 "$CHILD" 2>/dev/null || break
    NOW_BYTES=$(wc -c < "$LOGS/${TAG}.stderr" 2>/dev/null || echo 0)
    if [ "$NOW_BYTES" = "$LAST" ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
    LAST=$NOW_BYTES
    [ "$QUIET" -ge 4 ] && break
    sleep 1
done
STARVE_HOLD=$(free_mib)
HOLD=$((STARVE_HOLD - LEAVE_FREE))
[ "$HOLD" -lt 0 ] && HOLD=0
"$HERE/vram_hog" "$HOLD" > "$LOGS/${TAG}_hog.log" 2>&1 &
HOG=$!
sleep 2
smi > "$LOGS/${TAG}_vram_starved.txt"

wait "$CHILD"
echo "$?" > "$LOGS/${TAG}.exit"
if [ "$GDB_MODE" = "gdb" ]; then
    # gdb's own exit status is not the child's: the raw carries `Program received signal …`
    grep -c "Program received signal" "$LOGS/${TAG}.raw" > "$LOGS/${TAG}_signal_count.txt" 2>/dev/null
fi

kill -TERM "$HOG" 2>/dev/null
sleep 1
kill -KILL "$HOG" 2>/dev/null
smi > "$LOGS/${TAG}_vram_after.txt"

echo "tag=$TAG tree=$TREE model=$MODEL loaded=$LOADED hold=${HOLD}MiB leave_free=${LEAVE_FREE}MiB"
echo "exit=$(cat "$LOGS/${TAG}.exit")  vram at window: $(cat "$LOGS/${TAG}_vram_window.txt")"
echo "vram starved: $(cat "$LOGS/${TAG}_vram_starved.txt")"
echo "vram after:   $(cat "$LOGS/${TAG}_vram_after.txt")"
echo "hog: $(head -c 160 "$LOGS/${TAG}_hog.log" | tr '\n' ' ')"
echo "--- stderr tail ---"; tail -c 400 "$LOGS/${TAG}.stderr"
