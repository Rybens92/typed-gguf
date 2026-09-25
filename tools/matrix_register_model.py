#!/usr/bin/env python3
"""Register a local GGUF as a registry alias a served decision can name (card t_f96fed7f).

`serve`'s TypeSafe routes resolve **registry aliases only** (SPEC 2.9: a remote client must never be
able to point the server at a file), and the pinned SDK sends its own default model —
`jev-latest`, which the server maps to the registry's `current`
(`typed_gguf.api.http.COMPAT_MODEL`). The matrix jobs download the 0.5B smoke GGUF *by URL* (no
`models pull`, no HuggingFace client, no 4.4 GB Spark download), so nothing has told the registry
that the file exists: without this step every served decision answers `422 E_MODEL_NOT_FOUND`, and
the macOS/Windows serve stories would be red for a reason that has nothing to do with the platform.

This is that wiring, and nothing else. It builds the same `store.Entry` `models pull` builds —
`arch` / `file_type` / `quant` read from the file's own GGUF header, `size` from its stat — and
makes the alias the registry's `current`, so the SDK's `jev-latest` resolves to it. Registering the
same file again under the same alias is idempotent (a re-run of the job must not leave
`smoke-0.5b-2` behind and point `current` at it); a *different* file under a taken alias gets the
deduplicated name `store.add_entry` gives it, and the tool reports which name won.

    tools/matrix_register_model.py --model /tmp/smoke.gguf --alias smoke-0.5b --json

Exit 0 = the alias resolves. 2 = the file cannot be registered (missing, truncated, not a GGUF),
with the product's own typed code in the message.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from typed_gguf.errors import TypedGgufError
from typed_gguf.registry import gguf, store

SCHEMA = "typed_gguf.matrix.registered_model/v1"


class InputError(Exception):
    """The file cannot be registered, and the message is the whole story (exit 2)."""


def register(model: str, *, alias: str | None = None,
             home: pathlib.Path | None = None) -> dict:
    """Put `model` in the registry under `alias` and make it `current`. Returns the record."""
    path = pathlib.Path(str(model)).expanduser()
    if not path.is_file():
        raise InputError(f"{path} is not a file: this step registers a GGUF that is already on "
                         f"disk (the job downloads it first)")
    try:
        kv = gguf.parse_gguf_metadata(path)["kv"]
    except TypedGgufError as exc:
        raise InputError(f"{path} is not a readable GGUF ({exc})") from exc
    wanted = alias or store.slugify(path.name)
    registry, warnings = store.load_registry(store.registry_path(home))
    existing = registry.aliases.get(wanted)
    if existing is not None and pathlib.Path(existing.path).resolve() == path.resolve():
        # the same file under the same alias: a re-run, not a second model
        registry.current = existing.alias
        store.save_registry(registry, store.registry_path(home))
        return _record(existing, registry.current, reused=True, warnings=warnings)
    file_type = gguf.file_type_of(kv)
    entry = store.Entry(alias=wanted, path=str(path), sha256=None, arch=gguf.arch_of(kv),
                        quant=gguf.quant_label(file_type) if file_type is not None else None,
                        size=path.stat().st_size, license=None, source="local",
                        added_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        fit_plan=None, file_type=file_type, repo=None)
    entry = store.add_entry(registry, entry, alias=wanted)
    registry.current = entry.alias
    store.save_registry(registry, store.registry_path(home))
    return _record(entry, registry.current, reused=False, warnings=warnings)


def _record(entry: store.Entry, current: str, *, reused: bool, warnings: list[str]) -> dict:
    return {
        "schema": SCHEMA,
        "alias": entry.alias,
        "current": current,
        "path": entry.path,
        "arch": entry.arch,
        "quant": entry.quant,
        "file_type": entry.file_type,
        "size": entry.size,
        "home": str(store.data_home()),
        "reused": reused,
        "warnings": list(warnings),
        "note": ("the served TypeSafe route resolves aliases only and the pinned SDK sends "
                 "`jev-latest`, which the server maps to the registry's `current` (SPEC 2.9)"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a local GGUF so `typed-gguf serve` can resolve it (card "
                    "t_f96fed7f): the serve jobs download their smoke model by URL.")
    parser.add_argument("--model", required=True, help="the GGUF file already on disk")
    parser.add_argument("--alias", default=None,
                        help="the registry alias (default: slugified from the file name)")
    parser.add_argument("--home", default=None,
                        help="the data home (default: $TYPED_GGUF_HOME / the product's own)")
    parser.add_argument("--json", action="store_true", help="print the record as JSON")
    opts = parser.parse_args(argv)

    try:
        record = register(opts.model, alias=opts.alias,
                          home=pathlib.Path(opts.home) if opts.home else None)
    except InputError as exc:
        print(f"cannot register the smoke model: {exc}", file=sys.stderr)
        return 2
    for warning in record["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if opts.json:
        print(json.dumps(record, indent=2))
    else:
        print(f"registered {record['alias']} -> {record['path']}")
        print(f"  current  {record['current']}   arch {record['arch']} quant {record['quant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
