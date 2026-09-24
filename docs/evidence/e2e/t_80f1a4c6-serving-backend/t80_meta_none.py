"""Count verdicts in the sweep's metas, and how many are None (never executed)."""
import json
import pathlib

for name in ("session", "decide"):
    meta = pathlib.Path(f"/work/t80serve/mutants/src/ggufone/engine/{name}.py.meta")
    data = json.loads(meta.read_text())
    codes = data["exit_code_by_key"]
    none = sum(1 for value in codes.values() if value is None)
    print(f"{name}: entries={len(codes)} none={none} "
          f"killed={sum(1 for v in codes.values() if v == 1)} "
          f"survived={sum(1 for v in codes.values() if v == 0)} "
          f"no-tests={sum(1 for v in codes.values() if v == 33)}")
