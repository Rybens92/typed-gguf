"""The public name is `typed-gguf` (card t_5f9c15fe): no trace of the old one in the living surface.

A receipt is a record, not a name. `docs/evidence/**` and the dev-run dirs (`.e2e/`, `.e3*/`,
`.t*/`, `.gauntlet/`, `state/`) keep the exact commands, paths and schema strings they were
produced with — `test_the_receipts_keep_their_history` pins that — while everything a *reader or
an operator* touches (the distribution, the import package, the console script, the env vars, the
default data home, the schema strings, the workflows, the living docs) takes the new name. The
gate that keeps it that way is `test_the_living_surface_carries_no_old_name`: a later card that
re-introduces the old name fails here instead of in someone's shell.

`OLD_NAME` is assembled at run time on purpose: this file *is* living surface, and a literal
would make the gate trip on the very test that runs it.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW_NAME = "typed-gguf"
NEW_MODULE = "typed_gguf"
OLD_NAME = "gguf" + "one"
#: env spellings: the new override, and the old one this file proves is *not* silently aliased
#: (a fallback would hide the migration and leave the old name in the living source).
NEW_HOME_ENV = "TYPED_GGUF_HOME"
OLD_HOME_ENV = OLD_NAME.upper() + "_HOME"

#: frozen history — never swept. `docs/evidence/**` and the E-run dirs carry what they were
#: produced with, `state/` is the coordination ledger, `mutants*/` is mutation time (it holds a
#: copy of the pre-rename tree).
RECEIPT_DIRS = {"docs/evidence", ".e2e", ".gauntlet", "state"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
#: the one allowance inside the living surface: the "formerly …" line README/SPEC carry so that
#: a reader who knows the old name can still find the project.
FORMERLY = "formerly"


def _is_frozen(rel: tuple[str, ...]) -> bool:
    """Is this path (relative parts) receipt/history, i.e. out of the sweep's reach?"""
    if not rel:
        return False
    top = rel[0]
    if top in SKIP_DIRS or top in RECEIPT_DIRS or top.startswith("mutants"):
        return True
    return rel[:2] == ("docs", "evidence") or top.startswith((".e3", ".t"))


def _living_files() -> list[pathlib.Path]:
    """Every file of the living surface — the walk prunes the frozen dirs *before* descending
    (the `mutants*/` tree copies are large and are not repo content)."""
    files: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = pathlib.Path(dirpath).relative_to(ROOT).parts
        if _is_frozen(rel):
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if not _is_frozen(rel + (name,))]
        files.extend(pathlib.Path(dirpath) / name for name in filenames)
    return sorted(files)


# ------------------------------------------------------------------- the surface


def test_the_import_package_is_typed_gguf() -> None:
    module = importlib.import_module(NEW_MODULE)
    assert pathlib.Path(module.__file__) == ROOT / "src" / NEW_MODULE / "__init__.py"
    assert isinstance(module.__version__, str) and module.__version__


def test_the_old_module_is_not_importable() -> None:
    """A leftover alias would make every internal rename cosmetic: the tree must not answer to
    the old import name (neither from `src/` nor from a stale editable install)."""
    assert importlib.util.find_spec(OLD_NAME) is None


def test_the_distribution_and_console_script_are_named_typed_gguf() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert project["name"] == NEW_NAME
    assert list(project["scripts"]) == [NEW_NAME]
    assert project["scripts"][NEW_NAME] == f"{NEW_MODULE}.cli:run"
    assert payload["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        f"src/{NEW_MODULE}"]
    assert project["urls"]["Homepage"] == f"https://github.com/Rybens92/{NEW_NAME}"


def test_the_default_data_home_carries_the_new_name(monkeypatch, tmp_path) -> None:
    store = importlib.import_module(f"{NEW_MODULE}.registry.store")
    monkeypatch.delenv(NEW_HOME_ENV, raising=False)
    monkeypatch.delenv(OLD_HOME_ENV, raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert store.data_home() == tmp_path / ".local" / "share" / NEW_NAME
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert store.data_home() == tmp_path / "xdg" / NEW_NAME
    monkeypatch.setenv(NEW_HOME_ENV, "~/typed-home")
    assert store.data_home() == tmp_path / "typed-home"
    # the derived dirs land under the same home: states/downloads/runtime are one root
    monkeypatch.setenv(NEW_HOME_ENV, str(tmp_path / "home"))
    assert store.states_dir() == tmp_path / "home" / "states"
    assert store.downloads_dir() == tmp_path / "home" / "downloads"
    # …and the old override is *not* honoured: the rename is a rename, not an alias
    monkeypatch.delenv(NEW_HOME_ENV)
    monkeypatch.setenv(OLD_HOME_ENV, str(tmp_path / "legacy"))
    assert store.data_home() == tmp_path / "xdg" / NEW_NAME


def test_the_cli_names_the_tool(capsys) -> None:
    cli = importlib.import_module(f"{NEW_MODULE}.cli")
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{NEW_NAME} ")
    assert f"usage: {NEW_NAME} <command> [options]" in out
    assert cli.main(["version"]) == 0
    version_out = capsys.readouterr().out
    assert OLD_NAME not in (out + version_out).lower()


def test_python_dash_m_is_the_typed_gguf_entry_point() -> None:
    proc = subprocess.run([sys.executable, "-m", NEW_MODULE, "--help"], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith(f"{NEW_NAME} ")


def test_the_runtime_lock_schema_is_renamed() -> None:
    lock = json.loads((ROOT / "runtime.lock").read_text(encoding="utf-8"))
    assert lock["schema"] == f"{NEW_MODULE}.runtime.lock/v1"


# ------------------------------------------------------------------- the gates


def test_the_living_surface_carries_no_old_name() -> None:
    """The sweep's grep proof, executable: every living file, every line, one allowance."""
    offenders = []
    for path in _living_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                      # binary (compiled artifacts, images): not a name surface
        for number, line in enumerate(text.splitlines(), start=1):
            if OLD_NAME in line.lower() and FORMERLY not in line.lower():
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:120]}")
    assert offenders == [], (
        f"the old name survives in {len(offenders)} living line(s) of the "
        f"{NEW_NAME} tree:\n" + "\n".join(offenders[:40]))


def test_the_receipts_keep_their_history() -> None:
    """The other half of the contract: `docs/evidence/**` is a record. If this ever goes to zero,
    someone rewrote history to make a grep pretty."""
    evidence = ROOT / "docs" / "evidence"
    kept = [path for path in evidence.rglob("*")
            if path.is_file() and OLD_NAME.encode() in path.read_bytes()]
    assert len(kept) >= 50, f"only {len(kept)} evidence files still carry the old name"
