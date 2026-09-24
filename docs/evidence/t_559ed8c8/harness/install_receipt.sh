#!/usr/bin/env bash
# Story 1, leg 0: the clean-install receipt — wheel built at the card's head, installed into a
# fresh venv with no repo on the path, then `version` and `serve --help` out of that venv.
set -u
WHEEL=/workspace/ggufone/dist/typed_gguf-0.2.3-py3-none-any.whl
cd /workspace/e2e-t_559ed8c8

echo "### repo head"
git -C /workspace/ggufone rev-parse HEAD
git -C /workspace/ggufone status --porcelain | wc -l
echo
echo "### the wheel (built with: env -u PYTHONPATH uv build)"
ls -la /workspace/ggufone/dist/
sha256sum "$WHEEL"
echo
echo "### command: python3 -m venv venv && venv/bin/pip install --force-reinstall --no-deps <wheel>"
python3 --version
env -u PYTHONPATH venv/bin/pip install --force-reinstall --no-deps "$WHEEL"
echo "EXIT=$?"
echo
echo "### command: venv/bin/pip show typed-gguf"
env -u PYTHONPATH venv/bin/pip show typed-gguf
echo
echo "### command: (no PYTHONPATH, cwd outside the repo) venv/bin/python src/import_probe.py"
env -u PYTHONPATH venv/bin/python src/import_probe.py
echo "EXIT=$?"
echo
echo "### command: venv/bin/typed-gguf version"
env -u PYTHONPATH venv/bin/typed-gguf version
echo "EXIT=$?"
echo
echo "### command: venv/bin/typed-gguf serve --help"
env -u PYTHONPATH venv/bin/typed-gguf serve --help
echo "EXIT=$?"
echo
echo "### command: venv/bin/typed-gguf runtime update --help"
env -u PYTHONPATH venv/bin/typed-gguf runtime update --help
echo "EXIT=$?"
