"""Locate libllama/libggml (.so/.dylib/.dll) and the active runtime dir

Milestone: E1a.

Two sources of truth (SPEC 2.7/4):
* `$TYPED_GGUF_RUNTIME_DIR` — read-only consumption of an existing runtime (also how the oracle
  is pointed at a bundle);
* `<data-home>/runtime/<tag>-<variant>/` — what `typed-gguf init` installs.

`runtime.json` (same data home) records what was installed and what the probe saw; it is
written by `init` and read by `doctor`/`version`.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
from dataclasses import dataclass

from typed_gguf.errors import RuntimeMissingError
from typed_gguf.registry import store
from typed_gguf.runtime.pins import RuntimeLock

_LIB_NAMES: dict[str, dict[str, str]] = {
    "linux": {"llama": "libllama.so", "ggml": "libggml.so", "ggml_base": "libggml-base.so"},
    "darwin": {"llama": "libllama.dylib", "ggml": "libggml.dylib",
               "ggml_base": "libggml-base.dylib"},
    "windows": {"llama": "llama.dll", "ggml": "ggml.dll", "ggml_base": "ggml-base.dll"},
}
#: the ggml accelerator backend library a bundle must carry to run a graph on that backend. A
#: bundle with none of them is a complete llama.cpp bundle whose only compute path is the CPU.
#: Ordered: a bundle carrying several is classified by the first hit (`backend_of_bundle`).
_ACCELERATOR_NAMES: dict[str, tuple[tuple[str, str], ...]] = {
    "linux": (("vulkan", "libggml-vulkan.so"), ("cuda", "libggml-cuda.so")),
    "darwin": (("metal", "libggml-metal.dylib"), ("vulkan", "libggml-vulkan.dylib"),
               ("cuda", "libggml-cuda.dylib")),
    "windows": (("vulkan", "ggml-vulkan.dll"), ("cuda", "ggml-cuda.dll"),
                ("metal", "ggml-metal.dll")),
}
RUNTIME_RECORD_SCHEMA = "typed_gguf.runtime/v1"
#: The *roles* a bundle's command-line tools fill. The file name is the platform's (`llama-cli.exe`
#: on Windows — `tool_name` / `tool_names`), so a caller asks for the role and never for a name:
#: the pinned Windows zip ships `llama-cli.exe`, which is why `build_number` could not read a build
#: there while the Linux name is bare (card t_8dab8b3a).
TOOL_NAMES = ("llama-cli", "llama-fit-params", "llama-tokenize")


def tool_name(tool: str, system: str | None = None) -> str:
    """`llama-cli` -> `llama-cli.exe` on Windows; the same name everywhere else."""
    if (system or platform.system()).lower() == "windows":
        return f"{tool}.exe"
    return tool


def tool_names(system: str | None = None) -> tuple[str, ...]:
    """`TOOL_NAMES` as the platform spells the *files* (SPEC 2.7's bundle layout)."""
    return tuple(tool_name(name, system) for name in TOOL_NAMES)


def accelerator_names(system: str | None = None) -> tuple[tuple[str, str], ...]:
    """`((backend, library), …)` for the platform — which ggml backend a bundle advertises."""
    resolved = (system or platform.system()).lower()
    if resolved not in _ACCELERATOR_NAMES:
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: no accelerator naming rule for platform {resolved!r} "
            f"(known: {', '.join(sorted(_ACCELERATOR_NAMES))})")
    return _ACCELERATOR_NAMES[resolved]


def backend_of_bundle(directory: str | os.PathLike[str] | None, *,
                      system: str | None = None) -> str | None:
    """`vulkan | cuda | metal | cpu` — what backends the bundle at `directory` can run.

    The claim side of the device attribution (card t_80f1a4c6): a *complete* bundle (it carries
    `libllama`) with no accelerator library answers `cpu`, and a directory that is not a bundle
    at all answers `None` — a run that loaded nothing cannot be labelled from what it loaded.
    What the engine *really* used is `typed_gguf.runtime.devices`, read from its own log; this is
    only the label that log is checked against.
    """
    if directory is None:
        return None
    directory = pathlib.Path(directory)
    if not (directory / library_names(system)["llama"]).exists():
        return None
    for backend, library in accelerator_names(system):
        if (directory / library).exists():
            return backend
    return "cpu"


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
    # keys are the ROLES (`llama-cli`), values the platform's file: a caller asks for the role and
    # gets whichever file this platform ships (card t_8dab8b3a).
    tools = {role: directory / tool_name(role, system) for role in TOOL_NAMES
             if (directory / tool_name(role, system)).exists()}
    base = directory / names["ggml_base"]
    return RuntimeLayout(directory=directory, libllama=directory / names["llama"],
                         libggml=directory / names["ggml"],
                         ggml_base=base if base.exists() else None, tools=tools)


def required_files(lock: RuntimeLock, *, system: str | None = None) -> tuple[str, ...]:
    """`runtime.lock`'s `required_files`, spelled for `system`.

    The lock pins the three libraries by their canonical names — the Linux SONAMEs the oracle and
    `tests/test_pins.py` verify. The same three libraries ship under every platform's own names
    (`llama.dll` / `ggml.dll` / `ggml-base.dll` on Windows, `*.dylib` on macOS), so a distribution
    check that reads the lock literally declares a *complete* Windows bundle incomplete and reports
    `E_RUNTIME_MISSING` for a bundle whose `llama.dll` is right there — what the first live matrix
    run proved (card t_8dab8b3a, job 108153215424). The naming table above is the single source of
    those names; this maps the lock's roles onto it, and a name the table does not know is passed
    through unchanged (a future lock entry is never silently dropped).
    """
    resolved = (system or platform.system()).lower()
    if resolved == "linux":
        return tuple(lock.required_files)
    names = library_names(resolved)               # raises for an unknown platform, like the rest
    role_of = {filename: role for role, filename in _LIB_NAMES["linux"].items()}
    return tuple(names[role_of[name]] if name in role_of else name for name in lock.required_files)


def runtime_dirs(home: pathlib.Path | None = None) -> list[pathlib.Path]:
    home = home or store.data_home()
    root = home / "runtime"
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)


def find_runtime(*, home: pathlib.Path | None = None,
                 system: str | None = None) -> pathlib.Path | None:
    """The active runtime directory, or None when nothing is installed.

    `$TYPED_GGUF_RUNTIME_DIR` is honoured strictly: if the user points it somewhere without
    `libllama`, that is an error, not a silent fallback to another install.

    Otherwise the variant `runtime.json` records wins (that is what `init` proved works on
    this host), and only then a scan of `<home>/runtime/*` — which also covers hand-made
    installs the record does not know about.
    """
    env = os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    names = library_names(system)
    if env:
        candidate = pathlib.Path(os.path.expanduser(env))
        if (candidate / names["llama"]).exists():
            return candidate
        raise RuntimeMissingError(
            f"E_RUNTIME_MISSING: TYPED_GGUF_RUNTIME_DIR={candidate} does not contain "
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
            f"{(home or store.data_home()) / 'runtime'}; run `typed-gguf init` (no compiler "
            f"needed) or set TYPED_GGUF_RUNTIME_DIR")
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
