"""Registry store: aliases, XDG paths, fit-plan cache, atomic writes

Milestone: E1a.

Layout (SPEC 2.7, XDG): `models/` (files), `registry.json` (aliases), `runtime/<tag>-<variant>/`,
`runtime.json` (active runtime + probe results), `states/`, `downloads/` (resume parts).
`GGUFONE_HOME` overrides the whole base directory.

Durability rules: every write is `tmp + fsync + os.replace` (a killed process never leaves a
half-written registry), and a registry file we cannot parse is *quarantined*, never deleted:
`ModelNotFoundError`-style data loss is not acceptable for something the user paid a 4 GB
download for.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ggufone.errors import ModelNotFoundError, RegistryCorruptError

SCHEMA = "ggufone.registry/v1"
REGISTRY_NAME = "registry.json"
QUARANTINE_PREFIX = "registry.json.corrupt-"


# --------------------------------------------------------------------- paths
def data_home() -> pathlib.Path:
    """Base data dir: `$GGUFONE_HOME` > `$XDG_DATA_HOME/ggufone` > `~/.local/share/ggufone`."""
    raw = os.environ.get("GGUFONE_HOME")
    if raw:
        return pathlib.Path(os.path.expanduser(raw))
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return pathlib.Path(os.path.expanduser(xdg)) / "ggufone"
    return pathlib.Path.home() / ".local" / "share" / "ggufone"


def models_dir() -> pathlib.Path:
    return data_home() / "models"


def downloads_dir() -> pathlib.Path:
    return data_home() / "downloads"


def runtime_root() -> pathlib.Path:
    return data_home() / "runtime"


def states_dir() -> pathlib.Path:
    return data_home() / "states"


def registry_path() -> pathlib.Path:
    return data_home() / REGISTRY_NAME


def runtime_record_path() -> pathlib.Path:
    return data_home() / "runtime.json"


def calibration_path() -> pathlib.Path:
    return data_home() / "calibration.json"


# --------------------------------------------------------------------- model
ENTRY_REQUIRED = ("alias", "path")
ENTRY_FIELDS = ("alias", "path", "sha256", "arch", "quant", "size", "license", "source",
                "added_at", "fit_plan", "file_type", "repo")


@dataclass
class Entry:
    """One registry alias -> the GGUF file it points at (SPEC 2.1)."""

    alias: str
    path: str
    sha256: str | None = None
    arch: str | None = None
    quant: str | None = None
    size: int | None = None
    license: str | None = None
    source: str | None = None
    added_at: str | None = None
    fit_plan: dict[str, Any] | None = None
    file_type: int | None = None
    repo: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Entry:
        missing = [k for k in ENTRY_REQUIRED if not payload.get(k)]
        if missing:
            raise ValueError(f"missing field(s): {', '.join(missing)}")
        known = {k: payload.get(k) for k in ENTRY_FIELDS}
        known["size"] = int(known["size"]) if known.get("size") is not None else None
        known["file_type"] = (int(known["file_type"])
                              if known.get("file_type") is not None else None)
        return cls(**known)  # type: ignore[arg-type]


@dataclass
class Registry:
    """The alias table plus the `current` default model (SPEC 2.7 / 2.8)."""

    aliases: dict[str, Entry] = field(default_factory=dict)
    current: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "current": self.current,
                "aliases": {name: entry.to_dict() for name, entry in self.aliases.items()}}


# --------------------------------------------------------------------- io
def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _quarantine(path: pathlib.Path) -> pathlib.Path | None:
    """Move a broken registry aside, keeping every byte (recoverable, no data loss)."""
    if not path.exists():
        return None
    stamp = int(time.time())
    target = path.with_name(f"{QUARANTINE_PREFIX}{stamp}")
    n = 1
    while target.exists():
        target = path.with_name(f"{QUARANTINE_PREFIX}{stamp}-{n}")
        n += 1
    try:
        os.replace(path, target)
    except OSError:
        try:
            target.write_bytes(path.read_bytes())
            path.unlink()
        except OSError:
            return None
    return target


def load_registry(path: pathlib.Path | None = None, *,
                  recover: bool = True) -> tuple[Registry, list[str]]:
    """Read `registry.json`.

    Returns `(registry, warnings)`. With `recover=True` (the default, used by every CLI
    path) a corrupt file is quarantined and an empty registry is returned so the tool stays
    usable; with `recover=False` the file is left untouched and `E_REGISTRY_CORRUPT` raised.
    """
    path = path or registry_path()
    if not path.exists():
        return Registry(), []
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        reason = f"invalid JSON ({exc.__class__.__name__}: {exc})"
        if not recover:
            raise RegistryCorruptError(
                f"E_REGISTRY_CORRUPT: {path}: {reason}; fix or move the file and retry") from exc
        target = _quarantine(path)
        return Registry(), [
            f"E_REGISTRY_CORRUPT: {path}: {reason}; "
            + (f"quarantined to {target}, starting with an empty registry"
               if target else "could not quarantine the file (left in place)")]
    warnings: list[str] = []
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA \
            or not isinstance(payload.get("aliases"), dict):
        reason = "unexpected schema (not a ggufone registry)"
        if not recover:
            raise RegistryCorruptError(
                f"E_REGISTRY_CORRUPT: {path}: {reason}; fix or move the file and retry")
        target = _quarantine(path)
        return Registry(), [
            f"E_REGISTRY_CORRUPT: {path}: {reason}; "
            + (f"quarantined to {target}, starting with an empty registry"
               if target else "could not quarantine the file (left in place)")]
    registry = Registry()
    for name, entry_payload in payload["aliases"].items():
        try:
            registry.aliases[name] = Entry.from_dict(dict(entry_payload))
        except (TypeError, ValueError) as exc:
            warnings.append(f"skipping registry entry {name!r}: {exc}")
    current = payload.get("current")
    registry.current = current if current in registry.aliases else None
    return registry, warnings


def save_registry(registry: Registry, path: pathlib.Path | None = None) -> pathlib.Path:
    path = path or registry_path()
    _atomic_write(path, json.dumps(registry.to_dict(), indent=1, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------- aliases
def slugify(name: str) -> str:
    """`Spark-X2.5-4B-Q8_0.gguf` -> `spark-x2.5-4b-q8_0`."""
    stem = pathlib.PurePosixPath(name).name
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "model"


def add_entry(registry: Registry, entry: Entry, *, alias: str | None = None) -> Entry:
    """Insert `entry`, de-duplicating the alias and defaulting `current` to the first entry."""
    want = slugify(alias or entry.alias or entry.path)
    taken = set(registry.aliases)
    name = want
    n = 2
    while name in taken:
        name = f"{want}-{n}"
        n += 1
    entry.alias = name
    if not entry.added_at:
        entry.added_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    registry.aliases[name] = entry
    if registry.current is None:
        registry.current = name
    return entry


def remove_entry(registry: Registry, alias: str) -> Entry:
    if alias not in registry.aliases:
        raise ModelNotFoundError(
            f"unknown alias {alias!r}; known: "
            f"{', '.join(sorted(registry.aliases)) or '<none>'}")
    entry = registry.aliases.pop(alias)
    if registry.current == alias:
        registry.current = next(iter(registry.aliases), None)
    return entry


def resolve(registry: Registry, ref: str | None, *, use_current: bool = False) -> Entry | None:
    """Resolve an alias, an absolute/relative path, or the `current` alias."""
    if ref is None:
        if use_current and registry.current:
            return registry.aliases.get(registry.current)
        return None
    if ref in registry.aliases:
        return registry.aliases[ref]
    if os.sep in ref or ref.endswith(".gguf"):
        wanted = os.path.abspath(os.path.expanduser(ref))
        for entry in registry.aliases.values():
            if os.path.abspath(os.path.expanduser(entry.path)) == wanted:
                return entry
    return None
