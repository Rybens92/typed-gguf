"""The oracle (docs/verify_runtime_contract.py) as pytest cases (SPEC A6 / A-E1a-1).

Two levels:
* offline: the evidence pins + arithmetic mirror + package surface must be green without a
  runtime (skips in section B are fine here);
* live: with a runtime installed (TYPED_GGUF_RUNTIME_DIR or `typed-gguf init`), section B must be
  green with **no SKIP** — that is the A-E1a-1 gate, and this test is how CI enforces it.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import time

import pytest

from typed_gguf.runtime import pressure

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "docs" / "verify_runtime_contract.py"

#: What the oracle prints/traces when the *kernel* refused one of its own forks (card t_a696ce02:
#: this container's shared pid cgroup answers EAGAIN at its cap). That is the box, not the
#: contract, so the run is retried; a contract failure is returned as-is and asserted.
FORK_REFUSED_TOKENS = ("Resource temporarily unavailable", "BlockingIOError",
                       "fork system call failed")


def fork_refused(result: subprocess.CompletedProcess[str]) -> bool:
    """Did this oracle run die because the box would not let it fork?"""
    if result.returncode == 0:
        return False
    blob = (result.stdout or "") + (result.stderr or "")
    return any(token in blob for token in FORK_REFUSED_TOKENS)


def run_oracle(env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the oracle; retry only a fork-starved run, never a failing contract.

    The oracle is a frozen artifact (it *is* the contract) and forks internally, so the only
    pressure-proof place left is here: the outer spawn goes through `pressure.spawn`, and a run
    whose own fork was refused is retried for the same bounded budget.
    """
    for attempt in range(1, pressure.SPAWN_ATTEMPTS + 1):
        result = pressure.spawn([sys.executable, str(ORACLE)], capture_output=True, text=True,
                                cwd=str(ROOT), check=False, env=env, timeout=600)
        if not fork_refused(result) or attempt == pressure.SPAWN_ATTEMPTS:
            return result
        time.sleep(pressure.SPAWN_BACKOFF * 2 ** (attempt - 1))
    raise AssertionError("unreachable: the loop returns on its last attempt")


@pytest.mark.needs_fork
def test_oracle_runs_offline_and_exits_zero() -> None:
    result = run_oracle()
    assert result.returncode == 0, result.stdout[-4000:]
    assert "failures: 0" in result.stdout
    assert "[A] distribution + model evidence pins" in result.stdout
    assert "[C] arithmetic mirror" in result.stdout


@pytest.mark.needs_fork
def test_oracle_section_d_sees_the_package() -> None:
    result = run_oracle()
    tail = result.stdout.split("[D] package + contract surface")[-1]
    assert "module typed_gguf.runtime imports" in tail
    assert "module typed_gguf.registry imports" in tail


def _installed_runtime() -> pathlib.Path | None:
    import os
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if env and (pathlib.Path(env) / "libllama.so").exists():
        return pathlib.Path(env)
    candidates = [pathlib.Path.home() / ".local" / "share" / "typed-gguf" / "runtime"]
    for root in candidates:
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (child / "libllama.so").exists():
                    return child
    return None


@pytest.mark.needs_fork
def test_oracle_live_section_is_green_without_skips() -> None:
    runtime = _installed_runtime()
    if runtime is None:
        pytest.skip("no runtime installed (run `typed-gguf init` or set TYPED_GGUF_RUNTIME_DIR)")
    import os
    env = {**os.environ, "TYPED_GGUF_RUNTIME_DIR": str(runtime)}
    result = run_oracle(env=env)
    assert result.returncode == 0, result.stdout[-4000:]
    live = result.stdout.split("[B] live runtime probes")[-1].split("[C] arithmetic mirror")[0]
    skips = [line for line in live.splitlines() if re.match(r"\s*SKIP ", line)]
    assert skips == [], f"section B skipped: {skips}"
    assert "resolves all 32 required symbols" in live       # 32 libllama.so symbols …
    assert "resolves the backend loader (2/2)" in live       # … + 2 libggml.so symbols = 34
    assert "llama-fit-params available for auto-fit" in live
