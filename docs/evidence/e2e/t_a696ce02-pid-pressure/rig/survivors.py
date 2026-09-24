"""List surviving mutants with their source range (card t_a696ce02).

mutmut 3.8 keeps `<module>.py.meta` (`exit_code_by_key`, where 0 = survived and `None` = not run)
and `<module>.py.spans` (`{key: [start, end]}` source ranges). `tools/mutation_span_check.py`
mismatches qualified/unqualified keys, so this reads the two artifacts directly.
"""
from __future__ import annotations

import json
import pathlib
import sys

STATUS = {0: "survived", 1: "killed", 3: "killed", 5: "no-tests", 2: "interrupted", None: "not-run",
          33: "no-tests", 34: "skipped", 36: "timeout"}


def main() -> int:
    mutants = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "mutants")
    wanted = sys.argv[2] if len(sys.argv) > 2 else "survived"
    for module in sorted(mutants.rglob("*.py.meta")):
        meta = json.loads(module.read_text())
        spans_path = module.with_suffix("").with_suffix(".py.spans")
        spans = json.loads(spans_path.read_text()) if spans_path.exists() else {}
        source = module.parent / module.name[:-len(".meta")]
        lines = source.read_text().splitlines()
        codes = meta.get("exit_code_by_key", meta)
        for key, code in codes.items():
            name = STATUS.get(code, f"code={code}")
            if name != wanted:
                continue
            span = spans.get(key)
            if isinstance(span, list) and len(span) == 2:
                start, end = span
                text = " ".join(lines[start - 1:end]).strip()
            else:
                text = "<no span>"
            print(f"{name:9} {module.name[:-5]} {key}\n    {text[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
