#!/usr/bin/env bash
# Tier-M sweep, card t_176614c6 (a) — the swap's drain lives in `keep/client.py`.
#
#   pair: source_paths = ["src/typed_gguf/keep/client.py"]
#         pytest_add_cli_args_test_selection = ["tests/test_keep_client.py"]
#
# Why this pair and not the whole keep package: the (a) fix is the only change in this module, its
# own gate file drives every branch of it (spawn / reuse / swap / drain / SIGKILL escalation), and
# the keep package's full surface (client + host + state + identity) with all four keep gate files
# cannot finish inside this box's pid cap and this card's wall clock. The pair is time-boxed by the
# caller (`timeout`), and `mutmut results` is read afterwards: a partial sweep is reported with its
# not-run count, never as a full-sweep score (the local convention, card t_e29734e6).
#
# The driver applies the container's xattr workaround (mutmut's own `shutil.copy2` of the tree dies
# on `security.selinux`); `--max-children 2` is the pid-cap-safe width, and a failed `os.fork()`
# takes the whole run down, so the loop retries.
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
