"""Read the offline-gate matrix out of ci.yml the way the workflow itself would (card t_b5872762).

Also answers AC2's second half: does `runtime-matrix.yml` carry a python-version matrix the same
reasoning would have to widen?
"""
import pathlib
import sys

import yaml

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/ggufone")
WORKFLOWS = ROOT / ".github" / "workflows"

ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
gate = ci["jobs"]["gate"]
print("ci.yml  gate.matrix.python-version =", gate["strategy"]["matrix"]["python-version"])
print("ci.yml  gate.env.UV_PYTHON        =", gate["env"]["UV_PYTHON"])
print("ci.yml  gate.name                 =", gate["name"])

matrix = yaml.safe_load((WORKFLOWS / "runtime-matrix.yml").read_text(encoding="utf-8"))
for name, job in matrix["jobs"].items():
    keys = sorted((job.get("strategy") or {}).get("matrix", {}))
    print(f"runtime-matrix.yml  job {name!r}: matrix keys = {keys or '(none)'}")
