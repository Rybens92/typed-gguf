"""Print the mutmut verdicts of specific keys from the current run's meta."""
import json
import pathlib
import sys

meta = json.loads(pathlib.Path(
    "mutants/src/ggufone/bench/isolation.py.meta").read_text(encoding="utf-8"))
codes = meta["exit_code_by_key"]
for key in sys.argv[1:]:
    print(f"{codes.get(key)!r:>5}  {key}")
print(f"total {len(codes)} · survived {sum(1 for v in codes.values() if v == 0)}")
