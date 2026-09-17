"""Locate libllama/libggml (.so/.dylib/.dll) and the active runtime dir

Milestone: E1a.

Two sources of truth (SPEC 2.7/4):
* `$GGUFONE_RUNTIME_DIR` — read-only consumption of an existing runtime (also how the oracle
  is pointed at a bundle);
* `<data-home>/runtime/<tag>-<variant>/` — what `ggufone init` installs.

`runtime.json` (same data home) records what was installed and what the probe saw; it is
written by `init` and read by `doctor`/`version`.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
from dataclasses import dataclass

from ggufone.errors import RuntimeMissingError
from ggufone.registry import store

_LIB_NAMES: dict[str, dict[str, str]] = {
    "linux": {"llama": "libllama.so", "ggml": "libggml.so", "ggml_base": "libggml-base.so"},
    "darwin": {"llama": "libllama.dylib", "ggml": "libggml.dylib",
               "ggml_base": "libggml-base.dylib"},
    "windows": {"llama": "llama.dll", "ggml": "ggml.dll", "ggml_base": "ggml-base.dll"},
}
RUNTIME_RECORD_SCHEMA = "ggufone.runtime/v1"
TOOL_NAMES = ("llama-cli", "llama-fit-params", "llama-tokenize")


def library_names(system: str | None = None) -> dict[str, str]:
    system = (system or platform.system()).lower()
    if system not in _LIB_NAMES:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: no library naming rule for platform {system!r} "
            f"(known: {', '.join(sorted(_LIB_NAMES))})")
    return dict(_LIB_NAMES[system])


def library_glob(system: str | None = None) -> str:
    """Glob for the architecture/backend libs (libggml-cpu.so, libggml-vulkan.so, ...)."""
    ext = {"linux": ".so", "darwin": ".dylib", "windows": ".dll"}.get(
        (system or platform.system()).lower(), ".so")
    return f"libggml-*{ext}" if ext != ".dll" else "*ggml-*.dll"


@dataclass(frozen=True)
class RuntimeLayout:
    directory: pathlib.Path
    libllama: pathlib.Path
    libggml: pathlib.Path
    ggml_base: pathlib.Path | None
    tools: dict[str, pathlib.Path]

    @property
    def complete(self) -> bool:
        return self.libllama.exists() and self.libggml.exists()


def layout(directory: str | os.PathLike[str], *, system: str | None = None) -> RuntimeLayout:
    directory = pathlib.Path(directory)
    names = library_names(system)
    tools = {name: directory / name for name in TOOL_NAMES if (directory / name).exists()}
    base = directory / names["ggml_base"]
    return RuntimeLayout(directory=directory, libllama=directory / names["llama"],
                         libggml=directory / names["ggml"],
                         ggml_base=base if base.exists() else None, tools=tools)


def runtime_dirs(home: pathlib.Path | None = None) -> list[pathlib.Path]:
    home = home or store.data_home()
    root = home / "runtime"
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)


def find_runtime(*, home: pathlib.Path | None = None,
                 system: str | None = None) -> pathlib.Path | None:
    """The active runtime directory, or None when nothing is installed.

    `$GGUFONE_RUNTIME_DIR` is honoured strictly: if the user points it somewhere without
    `libllama`, that is an error, not a silent fallback to another install.

    Otherwise the variant `runtime.json` records wins (that is what `init` proved works on
    this host), and only then a scan of `<home>/runtime/*` — which also covers hand-made
    installs the record does not know about.
    """
    env = os.environ.get("GGUFONE_RUNTIME_DIR")
    names = library_names(system)
    if env:
        candidate = pathlib.Path(os.path.expanduser(env))
        if (candidate / names["llama"]).exists():
            return candidate
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: GGUFONE_RUNTIME_DIR={candidate} does not contain "
            f"{names['llama']}; point it at an extracted llama.cpp bundle or unset it")
    recorded = (runtime_record(home) or {}).get("dir")
    if recorded and (pathlib.Path(recorded) / names["llama"]).exists():
        return pathlib.Path(recorded)
    for candidate in runtime_dirs(home):
        if (candidate / names["llama"]).exists():
            return candidate
    return None


def resolve_runtime(*, home: pathlib.Path | None = None, system: str | None = None,
                    required: bool = True) -> pathlib.Path | None:
    found = find_runtime(home=home, system=system)
    if found is None and required:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: no llama.cpp runtime installed under "
            f"{(home or store.data_home()) / 'runtime'}; run `ggufone init` (no compiler "
            f"needed) or set GGUFONE_RUNTIME_DIR")
    return found


def runtime_record(home: pathlib.Path | None = None) -> dict | None:
    path = (home or store.data_home()) / "runtime.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_runtime_record(record: dict, home: pathlib.Path | None = None) -> pathlib.Path:
    home = home or store.data_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / "runtime.json"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
