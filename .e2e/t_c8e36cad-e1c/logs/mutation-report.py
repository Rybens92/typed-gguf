"""Per-file mutation report from a mutmut 3.8 tree (E1c copy of the E1b script).

Usage: uv run python .e2e/t_c8e36cad-e1c/logs/mutation-report.py src/ggufone/engine/template.py

Reads `mutants/<relative>.meta` (written by mutmut per source file) and prints killed/survived
plus the survivor list, so the numbers in the evidence doc can be re-derived without re-running
the sweep. `repo` defaults to the current working directory.
"""
import collections
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path.cwd())
relative = sys.argv[1] if len(sys.argv) > 1 else "src/ggufone/engine/template.py"
meta = repo / "mutants" / relative
meta = meta.with_name(meta.name + ".meta")
if not meta.exists():
    print(f"no meta for {relative} (not reached by this sweep?)")
    raise SystemExit(1)
data = json.loads(meta.read_text())
scores = collections.Counter(data["exit_code_by_key"].values())
killed = scores.get(1, 0)
survived = scores.get(0, 0)
other = {code: n for code, n in scores.items() if code not in (0, 1)}
total = killed + survived + sum(other.values())
print(f"{relative} mutants: {total}  killed={killed}  survived={survived}  other={other}")
if total:
    print(f"mutation score: {100.0 * killed / total:.1f}%")
print("\nsurvivors (first 40):")
for i, (key, code) in enumerate(data["exit_code_by_key"].items()):
    if code != 1 and i < 40:
        print(f"  [{code}] {key}")
print("\nsource sha256:", hashlib.sha256((repo / relative).read_bytes()).hexdigest())
