#!/usr/bin/env bash
# Run gateA (the updated task-0 file) on every fresh copy; log to $LOGS.
set -u
ROOT=/workspace/tmp/e1a-fix-t83ee1eed
LOGS=/workspace/tmp/e1a-fix-t83ee1eed-logs
mkdir -p "$LOGS"
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache

for name in base m08 m09 m10 m11; do
  echo "########## $name ##########"
  echo "--- whoami ---"
  ( cd "$ROOT/$name" && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider tests/_whoami_test.py ) \
    2>&1 | tail -3 | tee "$LOGS/$name.whoami.log"
  echo "--- gateA: tests/test_host_purity.py (updated) ---"
  ( cd "$ROOT/$name" && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider tests/test_host_purity.py -rf ) \
    2>&1 | tee "$LOGS/$name.gateA.log" | tail -12
  echo
done
echo "logs in $LOGS"
