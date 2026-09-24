#!/usr/bin/env bash
# Tier-M sweep, card t_a0fa2dc0 (UX: a usable default after `models pull`).
#
#   pair: source_paths = ["src/typed_gguf/registry/store.py"]
#         pytest_add_cli_args_test_selection = ["tests/test_default_model.py",
#                                               "tests/test_registry_store.py",
#                                               "tests/test_cli_e1a.py"]
#
# Why this pair: the change's decision rule is `store.find_default` (`current`, else the registry's
# sole alias) plus the `resolve(..., use_current=True)` every "the request named no model" path
# goes through. The module is small (285 lines) and its three gate files are in-process and fast,
# which is what a Tier-M sweep needs on this box. `cli.py` is deliberately not in source_paths (the
# standing convention): its two hunks — the no-model error's text and the `requested -> resolved`
# note — are wiring the CLI gates drive end to end.
#
# The driver applies the container's xattr workaround (mutmut's own `shutil.copy2` of the tree dies
# on `security.selinux` and on the committed evidence venvs' dangling links); `--max-children 2` is
# the pid-cap-safe width, and a failed `os.fork()` takes the whole run down, so the loop retries.
set -u
cd /workspace/ggufone
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date -u +%H:%M:%S)"
  if uv run --extra dev --with mutmut python tools/mutmut_driver.py run --max-children 2; then
    echo "=== sweep finished (attempt $attempt) $(date -u +%H:%M:%S)"
    break
  fi
  echo "=== mutmut exited non-zero (attempt $attempt) $(date -u +%H:%M:%S) — retrying"
  sleep 5
done
echo "=== results $(date -u +%H:%M:%S)"
uv run --extra dev --with mutmut python tools/mutmut_driver.py results 2>&1 | tail -20
