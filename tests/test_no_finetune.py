"""A2 gate: typed-gguf must never be able to fine-tune anything (SPEC 2.11).

Fails if training code or training dependencies ever enter the critical path.
"""
from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_DEPS = [
    "torch", "peft", "trl", "unsloth", "bitsandbytes", "deepspeed", "accelerate",
    "lightning", "sentence-transformers", "axolotl",
]
FORBIDDEN_SOURCE = re.compile(r"\b(llama_sampler_(init|chain)|def\s+train\w*\(|optimizer)", re.I)

ROOT = Path(__file__).resolve().parents[1]


def test_no_training_dependency_declared() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    deps_block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert deps_block.strip() == "", "core runtime dependencies must stay empty (stdlib only)"
    for dep in FORBIDDEN_DEPS:
        assert not re.search(rf'["\']{re.escape(dep)}[\[=><~!]', text), dep


def test_no_training_or_sampling_code_in_src() -> None:
    src = ROOT / "src"
    if not src.exists():
        return
    for path in src.rglob("*.py"):
        body = path.read_text()
        assert not FORBIDDEN_SOURCE.search(body), f"{path} touches sampling/training"
