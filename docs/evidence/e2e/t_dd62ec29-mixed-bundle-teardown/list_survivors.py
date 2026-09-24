"""List the survivors of a mutmut run's meta (for the round-1/round-2 evidence artifacts)."""
import json
import pathlib
import sys

meta = pathlib.Path(sys.argv[1])
codes = json.loads(meta.read_text(encoding="utf-8"))["exit_code_by_key"]
survivors = sorted(key for key, code in codes.items() if code == 0)
print(f"{meta}: {len(survivors)} survivors of {len(codes)} mutants "
      f"({100.0 * (len(codes) - len(survivors)) / len(codes):.1f} % killed)")
for key in survivors:
    print(f"  {key}")
