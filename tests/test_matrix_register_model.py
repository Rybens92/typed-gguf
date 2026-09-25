"""`tools/matrix_register_model.py` — how a served decision gets a model to resolve (AC2/AC3).

`serve`'s TypeSafe routes resolve **registry aliases only** (SPEC 2.9: a remote client must never be
able to point the server at a file), and the pinned SDK sends its own default model —
`jev-latest`, which the server maps to the registry's `current`
(`typed_gguf.api.http.COMPAT_MODEL`).
Both serve jobs download the 0.5B smoke GGUF **by URL** (no `models pull`, no HuggingFace client in
the job), so nothing has told the registry that the file exists; without this step every served
decision would answer `422 E_MODEL_NOT_FOUND` and the macOS/Windows serve stories would be red for
a reason that has nothing to do with the platform.

The tool builds exactly the `store.Entry` `models pull` builds (`arch`/`quant`/`file_type` read from
the file's own GGUF header, `size` from its stat) and makes the alias the registry's `current`. The
gates below drive it against a synthetic GGUF and then *use* the result through the served route —
"the registry can resolve what the SDK sends" is the claim, so it is asserted through the SDK's own
body, not by reading the registry back.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from tests.test_cli_e1a import make_gguf
from tests.test_serve import ALIAS as SERVED_ALIAS
from tests.test_serve import SDK_MIXED_BODY, Engine, Log, body_of, post

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "matrix_register_model.py"
SMOKE = "smoke-0.5b"


def load_tool():
    if not TOOL.exists():
        raise AssertionError(
            f"{TOOL.relative_to(ROOT)} is missing: the macOS and Windows serve steps download "
            "their smoke GGUF by URL, so the served routes have no alias to resolve without it "
            "(card t_f96fed7f)")
    spec = importlib.util.spec_from_file_location("matrix_register_model", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke(tmp_path: pathlib.Path) -> pathlib.Path:
    """A real (synthetic) GGUF file, the way the job's `Invoke-WebRequest` lands one."""
    path = tmp_path / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    path.write_bytes(make_gguf())
    return path


def _run(smoke: pathlib.Path, *args: str, home: pathlib.Path | None = None):
    env = None
    if home is not None:
        import os
        env = {**os.environ, "TYPED_GGUF_HOME": str(home)}
    return subprocess.run([sys.executable, str(TOOL), "--model", str(smoke), *args],
                          capture_output=True, text=True, timeout=120, env=env)


def test_the_smoke_model_becomes_a_current_alias(smoke: pathlib.Path,
                                                 tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    result = _run(smoke, "--alias", SMOKE, "--json", home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["alias"] == SMOKE and payload["current"] == SMOKE, payload
    assert payload["path"] == str(smoke)
    assert payload["arch"] == "spark2_5" and payload["quant"] == "Q8_0", payload
    assert payload["size"] == smoke.stat().st_size, payload


def test_the_registered_alias_is_what_the_sdk_s_own_body_resolves(
        smoke: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim the CI step exists for: `model: "jev-latest"` (the SDK's default) answers 200.

    Driven through the *served* route with the SDK's own mixed body, so the gate covers the mapping
    too — registration alone would not prove the SDK can reach it.
    """
    from typed_gguf.api import http as serve

    home = tmp_path / "home"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(home))
    result = _run(smoke, "--alias", SMOKE, "--json", home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    app = serve.App(decide=Engine(), home=home, log=Log())
    response = post(app, SDK_MIXED_BODY)
    assert response.status == 200, body_of(response)
    assert body_of(response)["model"] == SMOKE, "the served body echoes the resolved alias"


def test_registering_the_same_file_twice_is_idempotent(smoke: pathlib.Path,
                                                       tmp_path: pathlib.Path) -> None:
    """A re-run of the job must not leave `smoke-0.5b-2` behind and point `current` at it."""
    home = tmp_path / "home"
    first = _run(smoke, "--alias", SMOKE, "--json", home=home)
    second = _run(smoke, "--alias", SMOKE, "--json", home=home)
    assert first.returncode == 0 and second.returncode == 0, second.stdout + second.stderr
    assert json.loads(second.stdout)["alias"] == SMOKE, second.stdout
    assert json.loads(second.stdout)["current"] == SMOKE, second.stdout


def test_a_different_file_under_a_taken_alias_gets_its_own_name(
        smoke: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Two models must never share an alias: `store.add_entry` dedupes, and the tool reports it."""
    other = tmp_path / "other.gguf"
    other.write_bytes(make_gguf())
    home = tmp_path / "home"
    assert _run(smoke, "--alias", SMOKE, "--json", home=home).returncode == 0
    result = _run(other, "--alias", SMOKE, "--json", home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["alias"] != SMOKE and payload["alias"].startswith(SMOKE), payload


def test_a_missing_file_is_refused_loudly(tmp_path: pathlib.Path) -> None:
    result = _run(tmp_path / "absent.gguf", "--alias", SMOKE, "--json", home=tmp_path / "home")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "absent.gguf" in result.stderr, result.stderr


def test_a_file_that_is_not_a_gguf_is_refused_with_the_typed_code(tmp_path: pathlib.Path) -> None:
    """The jobs download a 0.35 GB file over the network: a truncated one must say so."""
    broken = tmp_path / "truncated.gguf"
    broken.write_bytes(b"GGUF" + b"\x00" * 4)
    result = _run(broken, "--alias", SMOKE, "--json", home=tmp_path / "home")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "E_GGUF_CORRUPT" in result.stderr, result.stderr


def test_the_registry_the_tool_writes_is_the_one_serve_reads(smoke: pathlib.Path,
                                                            tmp_path: pathlib.Path) -> None:
    """One data home, one registry: `TYPED_GGUF_HOME` is what both sides resolve."""
    from typed_gguf.registry import store

    home = tmp_path / "home"
    assert _run(smoke, "--alias", SMOKE, "--json", home=home).returncode == 0
    registry, _warnings = store.load_registry(store.registry_path(home))
    assert registry.current == SMOKE, registry.current
    assert registry.aliases[SMOKE].path == str(smoke)


def test_the_default_alias_is_derived_from_the_file_name(smoke: pathlib.Path,
                                                         tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    result = _run(smoke, "--json", home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["alias"] == "qwen2.5-0.5b-instruct-q4_k_m", result.stdout
    assert json.loads(result.stdout)["alias"] != SERVED_ALIAS, (
        "a sanity check on the fixture import: the served fixture's alias is a different one")
