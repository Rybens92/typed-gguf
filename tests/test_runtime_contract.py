"""The oracle (docs/verify_runtime_contract.py) as pytest cases (SPEC A6 / A-E1a-1).

Two levels:
* offline: the evidence pins + arithmetic mirror + package surface must be green without a
  runtime (skips in section B are fine here);
* live: with a runtime installed (GGUFONE_RUNTIME_DIR or `ggufone init`), section B must be
  green with **no SKIP** — that is the A-E1a-1 gate, and this test is how CI enforces it.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "docs" / "verify_runtime_contract.py"


def run_oracle(env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ORACLE)], capture_output=True, text=True,  # noqa: S603
                          cwd=str(ROOT), check=False, env=env, timeout=600)


def test_oracle_runs_offline_and_exits_zero() -> None:
    result = run_oracle()
    assert result.returncode == 0, result.stdout[-4000:]
    assert "failures: 0" in result.stdout
    assert "[A] distribution + model evidence pins" in result.stdout
    assert "[C] arithmetic mirror" in result.stdout


def test_oracle_section_d_sees_the_package() -> None:
    result = run_oracle()
    tail = result.stdout.split("[D] package + contract surface")[-1]
    assert "module ggufone.runtime imports" in tail
    assert "module ggufone.registry imports" in tail


def _installed_runtime() -> pathlib.Path | None:
    import os
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    candidates = [pathlib.Path.home() / ".local" / "share" / "ggufone" / "runtime"]
    for root in candidates:
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (child / "libllama.so").exists():
                    return child
    return None


def test_oracle_live_section_is_green_without_skips() -> None:
    runtime = _installed_runtime()
    if runtime is None:
        pytest.skip("no runtime installed (run `ggufone init` or set GGUFONE_RUNTIME_DIR)")
    import os
    env = {**os.environ, "GGUFONE_RUNTIME_DIR": str(runtime)}
    result = run_oracle(env=env)
    assert result.returncode == 0, result.stdout[-4000:]
    live = result.stdout.split("[B] live runtime probes")[-1].split("[C] arithmetic mirror")[0]
    skips = [line for line in live.splitlines() if re.match(r"\s*SKIP ", line)]
    assert skips == [], f"section B skipped: {skips}"
    assert "resolves all 34 required symbols" in live
    assert "llama-fit-params available for auto-fit" in live
