#!/bin/sh
# Setup for card t_a696ce02: pytest-randomly in the private clone's venv.
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
cd /work/t_a696ce02/repo || exit 1
uv pip install -q -p .venv pytest-randomly || exit 2
ls .venv/lib/python3.11/site-packages | grep -i -E "randomly" || exit 3
.venv/bin/python -m pytest --version || exit 4
echo "SETUP OK"
