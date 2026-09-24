#!/usr/bin/env bash
# `serve` host live gate (card t_f5d8b6c7, acceptance 6): the operator's box, the official SDK,
# the real 4B model. In-container execution is not the acceptance — this script IS.
#
#   tools/host_gate_serve.sh [LOG_DIR]
#
# Default LOG_DIR: $HOME/.typed-gguf-host-gate-serve-<utc timestamp>. Raw stdout/stderr, exit
# codes, timings, the served body and the client's verdict all land there.
#
# What it runs, in order:
#   0  host facts        python/uv/git head, the pinned model, the data home, nvidia-smi (if any)
#   1  build the venv    fresh venv in LOG_DIR, `pip install typesafe-sdk==0.7.1`, and the
#                        installed version is read back — a different SDK aborts the run
#   2  offline suite     uv run pytest -q tests/test_serve.py (CI shape, network blocked)
#   3  model + registry  the 4B as this host's registry knows it (aliases, not paths)
#   4  serve             `uv run typed-gguf serve --host 127.0.0.1 --port N` in the background,
#                        /health polled to readiness (no fixed sleeps)
#   5  the SDK call      LOG_DIR/venv/bin/python tools/host_gate_serve_client.py --base-url …
#                        cold + warm mixed choice/score/noul on the real model, typed answers
#                        printed, exit 2 on the first mismatch, exit 3 on an SDK mismatch
#   6  warm host proof   the same call a second time stays inside the keep window, and the
#                        server's own log says the host answered (`served_by=host`)
#   7  teardown          the server is stopped, `keep stop` is called, and no record/process is
#                        left behind (a leftover host is a failure, not a note)
#
# Exit 0 only when every step above is 0. Nothing outside LOG_DIR and the typed-gguf data home
# is written; the repository is never modified (no git commands that write).
#
# What ONLY this host can prove (recorded in the run's own verdict): that a real
# `typesafe-sdk==0.7.1` client, told nothing but TYPESAFE_BASE_URL, gets typed answers from the
# 4B through the warm keep host — the drop-in claim. In-container we prove the wire, the mapping,
# the cold/warm path, the teardown and the SDK field sets against `tests/fake_keep_host.py`
# (offline, no GPU, no model) — see `docs/evidence/t_f5d8b6c7_serve_gates.md`.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${1:-$HOME/.typed-gguf-host-gate-serve-$STAMP}"
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 2

SDK_PIN="typesafe-sdk==0.7.1"
PORT="${TYPED_GGUF_GATE_PORT:-8088}"
MODEL="${TYPED_GGUF_GATE_MODEL:-$HOME/.hermes/models/Spark-X2.5-4B-Q8_0.gguf}"
FAILED=0

echo "typed-gguf serve host gate"
echo "  repo    : $REPO"
echo "  logs    : $LOG_DIR"
echo "  sdk pin : $SDK_PIN"
echo "  port    : $PORT"
echo "  model   : $MODEL"
echo "  started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
    [ "$code" = 0 ] || FAILED=1
    return 0
}

# ---------------------------------------------------------------- 0. host facts
{
    echo "# host facts"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -a)"
    echo "git_head: $(git -C "$REPO" rev-parse HEAD 2>&1)"
    echo "git_dirty: $(git -C "$REPO" status --porcelain | head -5)"
    echo "python: $(python3 -V 2>&1)  uv: $(uv --version 2>&1)"
    echo "nvidia_smi_L:"; nvidia-smi -L 2>&1 | sed 's/^/  /'
    echo "gate_model_realpath: $(readlink -f "$MODEL" 2>&1)"
    echo "gate_model_bytes: $(stat -c %s "$MODEL" 2>&1)"
    echo "TYPED_GGUF_HOME: ${TYPED_GGUF_HOME:-<unset: the default data home is used>}"
} >"$LOG_DIR/host_facts.txt" 2>&1
sed -n '1,20p' "$LOG_DIR/host_facts.txt"

# ---------------------------------------------------------------- 1. the pinned SDK venv
echo
echo "== [venv] python3 -m venv $LOG_DIR/venv + pip install '$SDK_PIN'"
if python3 -m venv "$LOG_DIR/venv" >"$LOG_DIR/venv.out" 2>"$LOG_DIR/venv.err" \
   && "$LOG_DIR/venv/bin/pip" install -q "$SDK_PIN" >>"$LOG_DIR/venv.out" \
                                                                  2>>"$LOG_DIR/venv.err"; then
    INSTALLED="$("$LOG_DIR/venv/bin/python" -c 'import typesafe_sdk; print(typesafe_sdk.__version__)' \
                 2>>"$LOG_DIR/venv.err")"
    echo "   -> installed typesafe-sdk $INSTALLED"
    echo "$INSTALLED" >"$LOG_DIR/sdk_version.txt"
    if [ "$INSTALLED" != "0.7.1" ]; then
        echo "FATAL: this gate measures the typesafe-sdk 0.7.1 wire; the venv resolved" >&2
        echo "'$INSTALLED'. A different SDK answers a different wire — refusing to report." >&2
        FAILED=1
    fi
else
    echo "FATAL: could not build the SDK venv (see $LOG_DIR/venv.err)" >&2
    FAILED=1
fi

# ---------------------------------------------------------------- 2. the offline serve gates
step pytest_serve_gates 900 env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 \
    uv run --extra dev pytest -q tests/test_serve.py

# ---------------------------------------------------------------- 3. the model through the registry
uv run typed-gguf models ls --json >"$LOG_DIR/models.out" 2>"$LOG_DIR/models.err"
echo
echo "== [models] the registry as this host sees it"
sed -n '1,20p' "$LOG_DIR/models.out"
if [ ! -f "$MODEL" ]; then
    echo "FATAL: no model at $MODEL — set TYPED_GGUF_GATE_MODEL or pull it first." >&2
    FAILED=1
fi

# ---------------------------------------------------------------- 4. serve, in the background
SERVER_LOG="$LOG_DIR/serve.log"
echo
echo "== [serve] uv run typed-gguf serve --host 127.0.0.1 --port $PORT"
env -u PYTHONPATH nohup uv run typed-gguf serve --host 127.0.0.1 --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$LOG_DIR/serve.pid"

# a python poll rather than a curl dependency: same interpreter, no extra tool to install
health_poll() {  # health_poll <path> <seconds> <out-file>
    env -u PYTHONPATH uv run python - "$PORT" "$1" "$2" "$3" <<'PY'
import pathlib, sys, time, urllib.request

port, route, budget, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), pathlib.Path(sys.argv[4])
deadline = time.monotonic() + budget
last = "never tried"
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=3) as response:
            out.write_bytes(response.read())
            sys.exit(0)
    except Exception as exc:                      # noqa: BLE001 — the poll wants any failure
        last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
print(f"no answer from {route} within {budget:g}s (last: {last})", file=sys.stderr)
sys.exit(1)
PY
}

READY=0
if health_poll /health 90 "$LOG_DIR/health.json"; then
    READY=1
    echo "   -> /health answered: $(cat "$LOG_DIR/health.json")"
else
    echo "FATAL: /health never answered on 127.0.0.1:$PORT (see $SERVER_LOG)" >&2
    FAILED=1
fi

# ---------------------------------------------------------------- 5 + 6. the official SDK client
if [ "$READY" = 1 ]; then
    step sdk_client 1800 env TYPESAFE_API_KEY="typed-gguf-host-gate" \
        TYPESAFE_BASE_URL="http://127.0.0.1:$PORT" \
        "$LOG_DIR/venv/bin/python" tools/host_gate_serve_client.py \
        --base-url "http://127.0.0.1:$PORT" --out "$LOG_DIR/served_body.json"
    echo
    echo "-- the SDK's own answers (cold then warm) --"
    grep -E "^== \[|^   ok:|^PASS|^MISMATCH" "$LOG_DIR/sdk_client.out" || true
    echo "-- who answered, from the server's own log --"
    grep -E "served_by=" "$SERVER_LOG" | head -4 || true
    if ! grep -q "served_by=host" "$SERVER_LOG"; then
        echo "FATAL: the server's log never names the keep host as the answerer — the warm path" >&2
        echo "did not go through the resident host (SPEC 2.12)." >&2
        FAILED=1
    fi
    if health_poll /v1/models 30 "$LOG_DIR/v1_models.json"; then
        echo
        echo "== [route] GET /v1/models (no SDK in the loop)"
        head -c 600 "$LOG_DIR/v1_models.json"; echo
    fi
fi

# ---------------------------------------------------------------- 7. teardown
echo
echo "== [teardown] stop the server, then the keep host"
kill "$SERVER_PID" 2>/dev/null || true
for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
if kill -0 "$SERVER_PID" 2>/dev/null; then kill -9 "$SERVER_PID" 2>/dev/null; fi
wait "$SERVER_PID" 2>/dev/null || true

step keep_stop 180 env -u PYTHONPATH uv run typed-gguf keep stop --json
env -u PYTHONPATH uv run python - >"$LOG_DIR/leftover.txt" 2>"$LOG_DIR/leftover.err" <<'PY'
import sys

from typed_gguf.keep import state

record = state.read_record()
if record is None:
    print("none")
else:
    print(f"pid={record.pid} alive={state.pid_alive(record.pid)}")
PY
echo "   -> record after teardown: $(cat "$LOG_DIR/leftover.txt" 2>/dev/null)"
if [ "$(cat "$LOG_DIR/leftover.txt" 2>/dev/null)" != "none" ]; then
    echo "FATAL: a keep record survived teardown — a leaked host (see $LOG_DIR/leftover.txt)" >&2
    FAILED=1
else
    echo "   -> nothing resident: no record, no process"
fi

{
    echo "# serve host gate verdict"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "sdk: $(cat "$LOG_DIR/sdk_version.txt" 2>/dev/null || echo '<venv step failed>')"
    echo "port: $PORT"
    echo "model: $MODEL"
    for name in pytest_serve_gates sdk_client keep_stop; do
        printf '%s_exit: %s\n' "$name" "$(cat "$LOG_DIR/$name.exit" 2>/dev/null || echo '<not run>')"
    done
    printf 'leaked_record: %s\n' "$(cat "$LOG_DIR/leftover.txt" 2>/dev/null)"
    echo "verdict: $([ "$FAILED" = 0 ] && echo PASS || echo FAIL)"
    echo
    echo "# only this host can prove: a real typesafe-sdk 0.7.1 client, given nothing but"
    echo "# TYPESAFE_BASE_URL, got typed answers from the 4B through the warm keep host."
    echo "# the offline gates prove the wire, the mapping, the typed refusals, the cold/warm"
    echo "# reuse and the teardown: tests/test_serve.py over tests/fake_keep_host.py."
} >"$LOG_DIR/verdict.txt"
echo
cat "$LOG_DIR/verdict.txt"
echo
echo "host gate finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)  logs: $LOG_DIR"
exit "$FAILED"
