#!/usr/bin/env bash
# runtime update|rollback CLI-level probes (read-only against the repo, /tmp only)
set -u
cd /workspace/ggufone
BIN=".venv/bin/typed-gguf"
[ -x "$BIN" ] || { echo "no $BIN"; exit 9; }

echo "=== 1. empty home: runtime update --check ==="
HOME1=$(mktemp -d /tmp/rt-empty-XXXX)
TYPED_GGUF_HOME="$HOME1" "$BIN" runtime update --check; echo "EXIT=$?"
echo "home contents: $(find "$HOME1" | sort | tr '\n' ' ')"

echo
echo "=== 2. TYPED_GGUF_RUNTIME_DIR set: rung-1 refusal ==="
TYPED_GGUF_RUNTIME_DIR=/some/managed/dir TYPED_GGUF_HOME="$HOME1" "$BIN" runtime update --check; echo "EXIT=$?"
TYPED_GGUF_RUNTIME_DIR=/some/managed/dir TYPED_GGUF_HOME="$HOME1" "$BIN" runtime update; echo "EXIT=$?"

echo
echo "=== 3. rollback with nothing recorded ==="
TYPED_GGUF_HOME="$HOME1" "$BIN" runtime rollback; echo "EXIT=$?"
echo "home contents after: $(find "$HOME1" | sort | tr '\n' ' ')"

echo
echo "=== 4. a fake installed runtime, offline --check (net blocked) ==="
HOME2=$(mktemp -d /tmp/rt-fake-XXXX)
mkdir -p "$HOME2/runtime/b11026-linux-x64-cpu" "$HOME2/downloads"
: > "$HOME2/runtime/b11026-linux-x64-cpu/libllama.so"
cat > "$HOME2/runtime.json" <<JSON
{"schema": "typed_gguf.runtime/v1", "dir": "$HOME2/runtime/b11026-linux-x64-cpu",
 "tag": "b11026", "build": 11026, "variant": "linux-x64-cpu", "installed_at": "2026-09-17T00:00:00Z"}
JSON
BEFORE=$(sha256sum "$HOME2/runtime.json" | cut -d' ' -f1)
env -u PYTHONPATH TYPED_GGUF_TEST_BLOCK_NET=1 TYPED_GGUF_HOME="$HOME2" "$BIN" runtime update --check; echo "EXIT=$?"
AFTER=$(sha256sum "$HOME2/runtime.json" | cut -d' ' -f1)
echo "runtime.json byte-identical: $([ "$BEFORE" = "$AFTER" ] && echo yes || echo NO)"
echo "downloads dir: $(ls -A "$HOME2/downloads" | wc -l) entries; staging dirs: $(ls -A "$HOME2/runtime" | tr '\n' ' ')"

echo
echo "=== 5. the same fake home, live --check (no download; reads the release list) ==="
BEFORE2=$(sha256sum "$HOME2/runtime.json" | cut -d' ' -f1)
timeout 60 env -u PYTHONPATH TYPED_GGUF_HOME="$HOME2" "$BIN" runtime update --check --json > /tmp/rt-check.json 2>/tmp/rt-check.err; echo "EXIT=$?"
head -3 /tmp/rt-check.err
tail -c 900 /tmp/rt-check.json
AFTER2=$(sha256sum "$HOME2/runtime.json" | cut -d' ' -f1)
echo "runtime.json byte-identical: $([ "$BEFORE2" = "$AFTER2" ] && echo yes || echo NO)"
echo "downloads dir: $(ls -A "$HOME2/downloads" | wc -l) entries"
