"""E3e: the role-split tool's report shape and its per-family verdicts (canned records).

The engine gates are in `tests/test_e3e_roles.py`. The tool (`tools/e3e_role_render.py`) is the
*evidence* instrument — it renders every GGUF on the box offline and records whether the role split
is expressible for it — so what is gated here is the part a mutation would silently change: the
report must name the fallback for a family that cannot render the shape, and it must not hide a
family's status behind a same-named row.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tool():
    """Import `tools/e3e_role_render.py` (the repo keeps tools outside the package)."""
    spec = importlib.util.spec_from_file_location("e3e_role_render",
                                                  ROOT / "tools" / "e3e_role_render.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rendered_row(**overrides) -> dict:
    row = {"file": "Spark-X2.5-4B-Q8_0.gguf", "arch": "spark2_5", "family": "spark2_5",
           "status": "rendered", "prefix_chars": 552, "tail_chars": [363, 359], "dropped": "\n",
           "checks": {"state_words_in_prefix": True, "question_words_in_its_own_tail": True,
                      "question_absent_from_prefix": True,
                      "prefix_plus_tail_is_the_render_plus_opener": True, "no_open_think": True},
           "tail_head_excerpt": "<|User|>QUESTION:",
           "tail_tail_excerpt": "</think>\n{\"choice\": \"",
           "prefix_tail_excerpt": "<|end_of_sentence|>"}
    row.update(overrides)
    return row


def test_the_report_names_every_family_and_its_status() -> None:
    tool = _tool()
    record = {"schema": tool.SCHEMA, "generated_at": "2026-09-19T00:00:00Z",
              "chat_format": "role_split", "cue": "json_instructed", "models_dir": "/m",
              "dev_items": 2, "counts": {"rendered": 2, "refused": 0, "not-renderable": 1,
                                         "checks-failed": 0},
              "families": [
                  rendered_row(),
                  rendered_row(file="Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf", family="qwen35moe",
                               arch="qwen35moe", status="not-renderable", prefix_chars=0,
                               tail_chars=[], dropped="", checks=None, prefix_tail_excerpt="",
                               tail_head_excerpt="", tail_tail_excerpt="",
                               reason="E_TEMPLATE_UNRESOLVED: unsupported construct",
                               fallback="--template plain")]}
    text = tool.render_report(record)
    assert "### spark2_5" in text and "### qwen35moe" in text
    assert "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf" in text          # the file names the row
    assert "not-renderable" in text and "E_TEMPLATE_UNRESOLVED" in text
    assert "--template plain" in text                            # the fallback is stated
    assert "5/5" in text                                         # the acceptance count


def test_a_live_block_is_rendered_when_present() -> None:
    tool = _tool()
    record = {"schema": tool.SCHEMA, "generated_at": "2026-09-19T00:00:00Z",
              "chat_format": "role_split", "cue": "json_instructed", "models_dir": "/m",
              "dev_items": 2, "counts": {"rendered": 1, "refused": 0, "not-renderable": 0,
                                         "checks-failed": 0},
              "families": [rendered_row()],
              "live": {"item": "c01", "type": "choice", "model": {"name": "m.gguf"},
                       "backend": "vulkan", "gpu_layers": -1, "load_ms": 900.0, "wall_s": 30.0,
                       "prefix_tokens": 120, "tail_tokens": [80], "warnings": ["W_LOW_MASS"],
                       "prefix_tokens_reported": 120, "expected": "billing", "agrees": True,
                       "answer": {"type": "choice", "choice": "billing",
                                  "cue": {"verdict": "answered", "refused": False}}}}
    text = tool.render_report(record)
    assert "### live check" in text and "m.gguf" in text and "answered" in text


def test_the_offline_record_refuses_a_directory_without_models(tmp_path) -> None:
    tool = _tool()

    class Args:
        models = str(tmp_path)
        cue = "json_instructed"

    with pytest.raises(SystemExit) as caught:
        tool.offline_record(Args())
    assert "no .gguf files" in str(caught.value)
