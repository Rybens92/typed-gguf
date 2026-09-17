"""E0 scaffold gate: the package imports, the layout is right, the CLI answers.

This is the only test that may exist before E1a; it protects the contract surfaces
the SPEC freezes (module layout, command names, error codes) from silent drift.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ggufone import __version__, cli
from ggufone.errors import ERROR_CODES, WARNING_CODES

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "ggufone", "ggufone.errors", "ggufone.schema", "ggufone.cli",
    "ggufone.engine", "ggufone.engine.prompt", "ggufone.engine.readout",
    "ggufone.engine.session", "ggufone.engine.decide",
    "ggufone.runtime", "ggufone.runtime.finder", "ggufone.runtime.ctypes_binding",
    "ggufone.runtime.capability", "ggufone.runtime.install", "ggufone.runtime.fit",
    "ggufone.registry", "ggufone.registry.gguf", "ggufone.registry.hf",
    "ggufone.registry.store", "ggufone.registry.recommend",
    "ggufone.calibration", "ggufone.api", "ggufone.bench",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_version_is_str() -> None:
    assert isinstance(__version__, str) and __version__


def test_spec_and_oracle_present() -> None:
    assert (ROOT / "SPEC.md").exists()
    assert (ROOT / "docs" / "verify_runtime_contract.py").exists()
    assert (ROOT / "docs" / "evidence" / "poc-ctypes-20260917.py").exists()


def test_cli_version_and_unknown_command(capsys) -> None:
    assert cli.main(["version"]) == 0
    assert __version__ in capsys.readouterr().out
    assert cli.main(["nope"]) == 2
    # E1b implemented `run`/`ask`: a bare `run` is now a user error (missing --questions),
    # and the still-unimplemented commands keep the frozen "stub" exit code 3.
    assert cli.main(["run"]) == 2
    assert cli.main(["serve"]) == 3


def test_cli_command_set_frozen() -> None:
    assert set(cli.COMMANDS) == {
        "init", "doctor", "models", "run", "ask", "serve", "mcp", "bench",
        "fit", "calibrate", "version",
    }
    assert set(cli.MODELS_SUBCOMMANDS) == {
        "search", "pull", "use", "ls", "rm", "verify", "recommend-quant",
    }


def test_error_catalog_frozen() -> None:
    assert "E_MODEL_ARCH_UNSUPPORTED" in ERROR_CODES
    assert "W_LOW_MASS" in WARNING_CODES
