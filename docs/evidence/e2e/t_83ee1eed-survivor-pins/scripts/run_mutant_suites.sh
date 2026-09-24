#!/usr/bin/env bash
# Full offline suite on every fresh copy (canonical-style gate), plus the exact new node ids on base.
set -u
ROOT=/workspace/tmp/e1a-fix-t83ee1eed
LOGS=/workspace/tmp/e1a-fix-t83ee1eed-logs
export HOME=/work/agent-home
export UV_CACHE_DIR=/work/.uv-cache

for name in base m08 m09 m10 m11; do
  echo "########## $name : full suite ##########"
  ( cd "$ROOT/$name" && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider \
      --ignore=tests/_whoami_test.py -rf ) \
    2>&1 | tee "$LOGS/$name.suite.log" | tail -8
  echo
done

echo "########## base : the five new node ids ##########"
( cd "$ROOT/base" && uv run --project /workspace/ggufone pytest -q -p no:cacheprovider -v \
    "tests/test_host_purity.py::test_backends_answers_the_system_the_caller_named" \
    "tests/test_host_purity.py::test_backends_never_asks_this_host_for_a_system_the_caller_supplied" \
    "tests/test_host_purity.py::test_an_unnamed_machine_is_a_caller_error_not_a_platform_machine_read" \
    "tests/test_host_purity.py::test_omitted_dri_nodes_never_list_the_real_dev_dri" \
    "tests/test_host_purity.py::test_an_empty_injected_vram_probe_never_reaches_the_real_driver" ) \
  2>&1 | tee "$LOGS/base.new_nodes.log" | tail -12
echo "logs in $LOGS"
