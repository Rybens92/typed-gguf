#!/bin/sh
# Card t_97f1bc93 — coverage of the changed modules, subprocess-aware.
#
# The gates drive the entry point in *child* processes on purpose (real exit codes), and a child
# that ends via `os._exit` writes no coverage data at all — that IS the remedy under test. So the
# measurement starts coverage in every process that can start it (`COVERAGE_PROCESS_START` +
# a `sitecustomize.py` on PYTHONPATH), folds the parts that exist with `coverage combine`, and the
# evidence doc says which line the missing part explains.
#
#   sh .e2e/t_97f1bc93-vulkan-teardown/coverage.sh
set -u
cd /work/t97-ggufone || exit 1
PY=.venv/bin/python
export HOME=/var/home/rybens TMPDIR=/tmp
export COVERAGE_PROCESS_START=/work/t97-scratch/coveragerc-d
export PYTHONPATH=/work/t97-scratch/covsite:/work/t97-ggufone/src
echo "\$ .venv/bin/python -m pytest -q tests/test_cli_teardown.py -p no:randomly"
$PY -m pytest -q tests/test_cli_teardown.py -p no:randomly
echo "pytest EXIT=$?"
echo "\$ .venv/bin/python -m ggufone --help      # the __main__.py path, no bundle -> normal shutdown"
$PY -m ggufone --help > /dev/null
echo "ggufone --help EXIT=$?"
$PY -m coverage combine --data-file=/tmp/cov-t97-d/.coverage
echo "combine EXIT=$?"
$PY -m coverage report --data-file=/tmp/cov-t97-d/.coverage -m \
    --include='*/ggufone/runtime/teardown.py,*/ggufone/cli.py,*/ggufone/__main__.py'
echo "report EXIT=$?"
