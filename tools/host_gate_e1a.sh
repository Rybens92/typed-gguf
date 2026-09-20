#!/usr/bin/env bash
# E1a host live gate (card t_eae35404): the operator's GPU host, real downloads, real probes.
#
#   tools/host_gate_e1a.sh [LOG_DIR]
#
# Default LOG_DIR: $HOME/.typed-gguf-host-gate-<utc timestamp>. Everything the run produced —
# raw stdout/stderr, exit codes, timings, the runtime record, the probe facts — lands there,
# and tools/host_gate_summary.py turns the directory into host_gate_e1a.json.
#
# What it runs, in order (card requirements 2 + 3 + 4):
#   0  host facts          nvidia-smi, /dev/dri, Vulkan ICD, python/uv/git head
#   1  pytest -q           offline suite BEFORE the install (A-E1a-11, GPU world)
#   2  init --dry-run      the variant detection picks on this box
#   3  guard               every fake compiler on PATH is rejected (A-E1a-2 setup check)
#   4  init                real download + probe + cuda -> vulkan -> cpu fallback (timed)
#   5  doctor --json       working backend, backends list, fallback reason (exit code kept)
#   6  version --json      what the record says is installed
#   7  pytest -q           offline suite with the runtime installed (oracle section B is live)
#   8  oracle              python3 docs/verify_runtime_contract.py — exit 0, section B no SKIP
#   9  pytest --run-network live tests against the installed runtime and the pinned model
#  10  poisoned-PATH init  A-E1a-2: 0 compiler shims, <= 180 s (fresh home, offline cache)
#  11  fallback evidence   runtime.json + the archive hashes in the downloads dir
#
# It is safe to re-run: the only writes are under LOG_DIR and, for steps 4/10, under the
# typed-gguf data home (delete `<data home>/runtime` to force a fresh install).
# Nothing in the repository is modified (no git commands that write).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${1:-$HOME/.typed-gguf-host-gate-$STAMP}"
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 2

echo "typed-gguf E1a host gate"
echo "  repo    : $REPO"
echo "  logs    : $LOG_DIR"
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

# ---------------------------------------------------------------- 0. host facts
{
    echo "# host facts"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -a)"
    echo "git_head: $(git -C "$REPO" rev-parse HEAD 2>&1) $(git -C "$REPO" status --porcelain | head -3)"
    echo "python: $(python3 -V 2>&1)  uv: $(uv --version 2>&1)"
    echo "freedisk_home_bytes: $(df -B1 --output=avail "$HOME" 2>/dev/null | tail -1 | tr -d ' ')"
    echo "nvidia_smi_which: $(command -v nvidia-smi || echo '<absent>')"
    echo "nvidia_smi_L:"; nvidia-smi -L 2>&1 | sed 's/^/  /'
    echo "dri_nodes:"; ls -l /dev/dri 2>&1 | sed 's/^/  /'
    echo "vulkan_icd:"; ls /usr/share/vulkan/icd.d 2>&1 | sed 's/^/  /'
    echo "cudart_in_ldconfig:"; (ldconfig -p 2>/dev/null | grep -E 'libcudart|libcublas|libcuda\.so' || echo "  <none>") | sed 's/^/  /'
    echo "typed_gguf_home: $(python3 - <<'PY'
import os, pathlib
print(os.environ.get("TYPED_GGUF_HOME") or (os.environ.get("XDG_DATA_HOME")
      or pathlib.Path.home() / ".local" / "share") / "typed-gguf")
PY
)"
} >"$LOG_DIR/host_facts.txt" 2>&1
echo
sed -n '1,40p' "$LOG_DIR/host_facts.txt"

# ---------------------------------------------------------------- 1 + 2 pre-install
step pytest_before 900 uv run pytest -q
step init_dry_run 120 uv run typed-gguf init --dry-run --json

# ---------------------------------------------------------------- 3. poison the toolchain
POISON="$LOG_DIR/poison"
mkdir -p "$POISON"
for tool in cc gcc g++ clang clang++ nvcc cmake ninja make ld ar rustc cargo meson bazel; do
    cat >"$POISON/$tool" <<'SHIM'
#!/bin/sh
echo "$(basename "$0")" >> "${SHIM_LOG:-/dev/null}"
echo "poisoned: $(basename "$0") must not be called by typed-gguf init" >&2
exit 127
SHIM
    chmod +x "$POISON/$tool"
done
echo
echo "== [poison] $(ls "$POISON" | wc -l) compiler shims installed in $POISON"
PATH="$POISON:$PATH" bash -c 'set -u; for t in gcc cmake nvcc; do PATH= command -v "$t" >/dev/null && echo "$t still reachable"; done; echo "poison check: $(gcc --version 2>&1 | head -1)"'

# ---------------------------------------------------------------- 4. the real install
step init 1800 uv run typed-gguf init --json

# ---------------------------------------------------------------- 5 + 6 reporting
step doctor 600 uv run typed-gguf doctor --json
step version 120 uv run typed-gguf version --json

# ---------------------------------------------------------------- 7. suite with the runtime
step pytest_after 900 uv run pytest -q

# ---------------------------------------------------------------- 8. the oracle
step oracle 900 python3 docs/verify_runtime_contract.py

# ---------------------------------------------------------------- 9. live tests
step pytest_network 1800 uv run pytest -q --run-network

# ---------------------------------------------------------------- 10. A-E1a-2 poisoned PATH
DATA_HOME="$(python3 - <<'PY'
import os, pathlib
print(os.environ.get("TYPED_GGUF_HOME") or (os.environ.get("XDG_DATA_HOME")
      or pathlib.Path.home() / ".local" / "share") / "typed-gguf")
PY
)"
POISON_HOME="$LOG_DIR/poison-home"
if [ -d "$DATA_HOME/downloads" ]; then
    export TYPED_GGUF_OFFLINE_CACHE="$DATA_HOME/downloads"   # both pinned archives after step 4
fi
export TYPED_GGUF_HOME="$POISON_HOME"
export SHIM_LOG="$LOG_DIR/poison_calls.log"
: >"$LOG_DIR/poison_calls.log"
step init_poisoned_path 180 env PATH="$POISON:$PATH" uv run typed-gguf init --json
echo "   poison shims invoked: $(wc -l <"$LOG_DIR/poison_calls.log")"
unset TYPED_GGUF_HOME TYPED_GGUF_OFFLINE_CACHE SHIM_LOG

# ---------------------------------------------------------------- 11. fallback evidence
{
    echo "# installed runtime record (<data home>/runtime.json)"
    cat "$DATA_HOME/runtime.json" 2>&1
    echo
    echo "# installed runtime dirs"
    ls -l "$DATA_HOME/runtime" 2>&1
    echo
    echo "# downloaded archives + sha256 (the pins the installer verified)"
    for archive in "$DATA_HOME"/downloads/*.tar.gz "$DATA_HOME"/downloads/*.zip; do
        [ -e "$archive" ] || continue
        echo "$(sha256sum "$archive")"
    done
    echo
    echo "# section B of the oracle: skips?"
    sed -n '/\[B\]/,/\[C\]/p' "$LOG_DIR/oracle.out" | grep -c "SKIP" | sed 's/^/  skips_in_section_B: /'
} >"$LOG_DIR/fallback_evidence.txt" 2>&1
echo
cat "$LOG_DIR/fallback_evidence.txt"

# ---------------------------------------------------------------- summary
python3 tools/host_gate_summary.py "$LOG_DIR" || echo "warning: summary generation failed"
echo
echo "host gate finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "raw logs + host_gate_e1a.json: $LOG_DIR"
