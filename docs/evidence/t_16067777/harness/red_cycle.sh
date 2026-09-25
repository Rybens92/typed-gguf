#!/bin/sh
# Card t_16067777 — independent RED reproduction on the real tree.
# Swap the five changed src files back to HEAD, run the pin set, restore (always).
set -u
cd /workspace/ggufone || exit 1
OUT=/workspace/t_16067777
BAK=$OUT/redbak
mkdir -p "$BAK"

FILES="src/typed_gguf/keep/client.py src/typed_gguf/keep/state.py src/typed_gguf/registry/hf.py src/typed_gguf/runtime/install.py src/typed_gguf/runtime/update.py tools/host_gate_serve_client.py"
# The pre-fix tree. While the card is still uncommitted that is `HEAD`; after the card is committed
# (44582bd) it is the commit before it — `$1` overrides both.
BASE="${1:-HEAD~1}"
echo "pre-fix baseline: $BASE ($(git log -1 --format=%h "$BASE" 2>/dev/null))"
for f in $FILES; do
  mkdir -p "$BAK/$(dirname "$f")"
  cp "$f" "$BAK/$f" || exit 1
done
restore() {
  for f in $FILES; do cp "$BAK/$f" "$f"; done
}
trap restore EXIT INT TERM

for f in $FILES; do git show "$BASE:$f" > "$f" || exit 1; done
echo "== RED: the six source files at the pre-fix baseline, pins from the working tree =="
env -u PYTHONPATH -u TYPED_GGUF_HOME TYPED_GGUF_TEST_BLOCK_NET=1 \
  TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-023 \
  .venv/bin/python -m pytest -q -p no:randomly -rf \
  tests/test_runtime_update.py::test_rollback_keeps_the_probe_facts_the_update_recorded \
  tests/test_runtime_update.py::test_the_github_asset_leg_names_github_and_sends_the_real_user_agent \
  tests/test_hf.py::test_the_user_agent_carries_the_packaged_version \
  tests/test_hf.py::test_a_huggingface_failure_still_names_huggingface \
  tests/test_hf.py::test_the_unreachable_download_leg_names_the_host_the_product_talks_to \
  tests/test_hf.py::test_an_unknown_transport_failure_is_still_the_typed_download_error \
  tests/test_keep_client.py::test_stop_takes_the_ledgers_log_with_it \
  tests/test_serve.py::test_keep_stop_still_works_after_serving_and_leaves_nothing_behind \
  tests/test_serve.py::test_the_host_gate_driver_raises_the_sdk_timeout_above_its_ten_second_default
echo "RED exit=$?"
restore
echo "== GREEN: src restored to the fixed tree (same pins) =="
env -u PYTHONPATH -u TYPED_GGUF_HOME TYPED_GGUF_TEST_BLOCK_NET=1 \
  TYPED_GGUF_BENCH_RUNTIME_DIR=/tmp/offline-bundle-023 \
  .venv/bin/python -m pytest -q -p no:randomly \
  tests/test_runtime_update.py::test_rollback_keeps_the_probe_facts_the_update_recorded \
  tests/test_runtime_update.py::test_the_github_asset_leg_names_github_and_sends_the_real_user_agent \
  tests/test_hf.py::test_the_user_agent_carries_the_packaged_version \
  tests/test_hf.py::test_a_huggingface_failure_still_names_huggingface \
  tests/test_hf.py::test_the_unreachable_download_leg_names_the_host_the_product_talks_to \
  tests/test_hf.py::test_an_unknown_transport_failure_is_still_the_typed_download_error \
  tests/test_keep_client.py::test_stop_takes_the_ledgers_log_with_it \
  tests/test_serve.py::test_keep_stop_still_works_after_serving_and_leaves_nothing_behind \
  tests/test_serve.py::test_the_host_gate_driver_raises_the_sdk_timeout_above_its_ten_second_default
echo "GREEN exit=$?"
echo "== restored? (the six changed source files must show as modified) =="
git status --porcelain -- src tools | cat
