#!/usr/bin/env python3
"""Post-sweep repair: the mechanical rule turned every bare old name into the *command*
spelling (`typed-gguf`), but some of those literals are *module/path component* names and must
be `typed_gguf` (the import package / source-dir name). One exact edit each, no regex.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path("/workspace/ggufone")

#: (path, old, new, count)
FIXES = [
    # src: `-m <module>` and source-dir components
    ("src/typed_gguf/bench/isolation.py", '"-m", "typed-gguf", "bench"', '"-m", "typed_gguf", "bench"', 1),
    ("src/typed_gguf/runtime/isolated.py", '/ "typed-gguf" / "runtime"', '/ "typed_gguf" / "runtime"', 1),
    ("src/typed_gguf/runtime/probe_child.py", '/ "typed-gguf" / "runtime"', '/ "typed_gguf" / "runtime"', 1),
    # the doctor payload's version key follows the schema namespace (`typed_gguf.doctor/v1`)
    ("src/typed_gguf/cli.py", '"typed-gguf": __version__', '"typed_gguf": __version__', 1),
    # tools
    ("tools/e3_reproduce.py", '"-m", "typed-gguf", "run"', '"-m", "typed_gguf", "run"', 1),
    ("tools/t80_span_check.py", 'parts.index("typed-gguf")', 'parts.index("typed_gguf")', 1),
    # tests
    ("tests/test_bench.py", '/ "src" / "typed-gguf" / "bench"', '/ "src" / "typed_gguf" / "bench"', 1),
    ("tests/test_bench_isolation.py", '"-m", "typed-gguf", "bench"', '"-m", "typed_gguf", "bench"', 1),
    ("tests/test_bench_isolation_live.py", '"-m", "typed-gguf", "bench"', '"-m", "typed_gguf", "bench"', 1),
    ("tests/test_bench_vulkan_teardown_live.py", '"-m", "typed-gguf", "bench"', '"-m", "typed_gguf", "bench"', 1),
    ("tests/test_cli.py", '"-m", "typed-gguf"', '"-m", "typed_gguf"', 5),
    ("tests/test_cli_teardown.py", 'runpy.run_module("typed-gguf"', 'runpy.run_module("typed_gguf"', 1),
    ("tests/test_engine_fork.py", '/ "src" / "typed-gguf"', '/ "src" / "typed_gguf"', 1),
    ("tests/test_runtime_install.py", '(ROOT / "src" / "typed-gguf")', '(ROOT / "src" / "typed_gguf")', 1),
    ("tests/test_scaffold.py", '"typed-gguf", "typed_gguf.errors"', '"typed_gguf", "typed_gguf.errors"', 1),
]


def main() -> int:
    problems = []
    for name, old, new, count in FIXES:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        seen = text.count(old)
        if seen != count:
            problems.append(f"{name}: expected {count} of {old!r}, found {seen}")
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
    print("\n".join(problems) if problems else f"applied {len(FIXES)} edits")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
