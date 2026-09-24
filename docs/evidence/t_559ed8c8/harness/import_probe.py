"""Receipt: where does `typed_gguf` come from in this interpreter (must be site-packages)."""
from __future__ import annotations

import importlib
import pathlib
import sys

import typed_gguf

module = pathlib.Path(typed_gguf.__file__).resolve()
print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("typed_gguf.__file__:", module)
print("typed_gguf.__version__:", typed_gguf.__version__)
print("sys.path[:6]:", sys.path[:6])
repo = pathlib.Path("/workspace/ggufone").resolve()
inside = repo in module.parents
print("imports from the repo checkout:", inside)
print("PYTHONPATH:", repr(__import__("os").environ.get("PYTHONPATH")))
lock = importlib.import_module("typed_gguf.runtime.pins")
print("packaged lock:", lock.packaged_lock_path())
