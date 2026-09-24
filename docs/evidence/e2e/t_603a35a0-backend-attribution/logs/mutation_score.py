"""mutmut meta: distribution + survivors (values are the pytest exit code of the mutant run)."""
import json
import pathlib
import sys
from collections import Counter

meta = json.loads(pathlib.Path(sys.argv[1]).read_text())
codes: dict[str, int] = meta.get("exit_code_by_key", meta)
hist = Counter(codes.values())
print("exit-code histogram:", dict(sorted(hist.items())))
# mutmut 3.x writes the *pytest* exit code: 0 = the suite still passed -> SURVIVOR,
# 1 (or any non-zero) = the suite failed -> KILLED, 8 = timeout / no tests collected.
survivors = sorted(name for name, code in codes.items() if code == 0)
other = sorted(name for name, code in codes.items() if code not in (0, 1))
print(f"{len(codes)} mutants · {len(survivors)} survivors · "
      f"{100.0 * (len(codes) - len(survivors)) / len(codes):.1f}% killed/ran")
for name in survivors:
    print("  SURVIVED:", name)
for name in other:
    print(f"  code={codes[name]}:", name)
