#!/usr/bin/env python3
"""Group the survivors of a mutmut 3.8 tree by the function they sit in.

A key in `*.py.meta` looks like

    typed_gguf.keep.host.xǁServerǁstatus__mutmut_25

i.e. `<module>.<function>` for a module-level function, `<module>.x<Class>ǁ<method>` for a method
(`ǁ` is U+01C1, mutmut's separator), and `__mutmut_<n>` names the mutation inside it. Grouping by
that prefix is what a triage needs: "which function is under-tested" is a question about functions,
not about individual mutants.

Usage: python3 .e2e/t_7e24cea4-warm-host/mutmut_triage.py [mutants-dir] [--top N] [--list PREFIX]
"""
import collections
import json
import pathlib
import sys

SEP = "ǁ"          # mutmut's class/method separator
MARK = "__mutmut_"


def split_key(key: str) -> str:
    """`typed_gguf.keep.host.xǁServerǁhandle_line__mutmut_1` -> `host.Server.handle_line`.

    The shape is `<dotted module>.<function>__mutmut_<n>` for a module-level function and
    `<dotted module>.xǁ<Class>ǁ<method>__mutmut_<n>` for a method — the `xǁ` prefix is mutmut's
    class-scope marker (`ǁ` is U+01C1). Only the last module segment is useful as context, so the
    package path is dropped: a triage wants "host.Server.handle_line", not the import path. Note the
    marker is `x` *and* the separator: a module-level `x_host_status` is a plain function name.
    """
    name = key.split(MARK, 1)[0]
    parts = name.split(".")
    tail = parts[-1]
    stem = parts[-2] if len(parts) > 1 else "?"
    if tail.startswith("x" + SEP):
        pieces = [piece for piece in tail[2:].split(SEP) if piece]
        return ".".join([stem, *pieces])
    return f"{stem}.{tail}"


def main(argv: list[str]) -> int:
    rest = argv[1:]
    top = 40
    if "--top" in rest:                    # pop the flag *and* its value before any positional
        index = rest.index("--top")
        top = int(rest[index + 1])
        del rest[index:index + 2]
    root = pathlib.Path(rest[0]) if rest else pathlib.Path("mutants")
    killed: collections.Counter[str] = collections.Counter()
    survived: collections.Counter[str] = collections.Counter()
    unrun: collections.Counter[str] = collections.Counter()
    for meta in sorted(root.rglob("*.py.meta")):
        if "keep" not in str(meta):
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        for key, code in (data.get("exit_code_by_key") or {}).items():
            bucket = unrun if code is None else (survived if code == 0 else killed)
            bucket[split_key(key)] += 1
    rows = []
    for name in sorted(set(killed) | set(survived) | set(unrun)):
        k, s, u = killed[name], survived[name], unrun[name]
        scored = k + s
        if scored == 0:
            continue
        rows.append((s, scored, k, u, name))
    rows.sort(reverse=True)
    print(f"{'survivors':>9} {'scored':>6} {'killed':>6} {'k/s':>6}  function")
    for s, scored, k, u, name in rows[:top]:
        print(f"{s:9d} {scored:6d} {k:6d} {100.0 * k / scored:5.1f}%  {name}")
    total_k = sum(killed.values())
    total_s = sum(survived.values())
    print(f"keep package: killed={total_k} survived={total_s} unrun={sum(unrun.values())} "
          f"-> {100.0 * total_k / (total_k + total_s):.1f}% of the scored mutants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
