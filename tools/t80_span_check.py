"""Do any mutmut survivors cover the lines THIS card changed? (fixed key mapping, t_80f1a4c6)

    python3 tools/t80_span_check.py <mutants-dir> <module.py> <needle> [<needle> ...]

`tools/mutation_span_check.py` maps the `.spans` file's keys straight into the `.meta` dict, but
the two use different key shapes: `.spans` holds `x_backend_claim__mutmut_1` while
`.meta`'s `exit_code_by_key` holds `ggufone.engine.session.x_backend_claim__mutmut_1`. Every
lookup therefore misses, and every mutant reads as "unrun" — the tool cannot distinguish a
survivor from an unseen mutant, which is the one question it exists to answer. This version
qualifies the span keys with `<package>.<module>.` first (derived from the module's own path) and
reports, per needle line: mutants covering it, how many survived, how many are unrun, and the
survivors by name (with the surviving span so a diff can be pulled from the run file).
"""
from __future__ import annotations

import json
import pathlib
import sys


def qualify(module_path: str) -> str:
    """`src/ggufone/engine/session.py` -> `ggufone.engine.session.`"""
    parts = pathlib.Path(module_path).with_suffix("").parts
    start = parts.index("ggufone")
    return ".".join(parts[start:]) + "."


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    root = pathlib.Path(argv[1])
    module = root / argv[2]
    needles = argv[3:]
    prefix = qualify(argv[2])
    sources = module.read_text(encoding="utf-8").splitlines()
    spans = json.loads((module.parent / (module.name + ".spans")).read_text())["spans"]
    codes = json.loads((module.parent / (module.name + ".meta")).read_text())["exit_code_by_key"]
    print(f"# {argv[2]} — {len(spans)} spans, prefix {prefix!r}")
    total_survivors = 0
    total_unrun = 0
    for needle in needles:
        hits = [index + 1 for index, line in enumerate(sources) if needle in line]
        print(f"== {needle!r} at mutants-copy line(s) {hits or '<not found>'}")
        for line_no in hits:
            covering = [key for key, (start, end) in spans.items() if start <= line_no <= end]
            mutants = [key for key in covering if "__mutmut_" in key and "orig" not in key]
            survivors = [key for key in mutants if codes.get(prefix + key) == 0]
            unrun = [key for key in mutants if codes.get(prefix + key) is None]
            total_survivors += len(survivors)
            total_unrun += len(unrun)
            print(f"   line {line_no}: {len(mutants)} mutants, {len(survivors)} survived, "
                  f"{len(unrun)} unrun")
            for key in survivors:
                print(f"     SURVIVED {key} span={spans[key]}")
    print(f"# TOTAL survivors covering the needles: {total_survivors} (unrun: {total_unrun})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
