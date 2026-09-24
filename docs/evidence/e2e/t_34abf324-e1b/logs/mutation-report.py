import collections
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path("/var/home/rybens/workspace/ggufone")
relative = sys.argv[1] if len(sys.argv) > 1 else "src/ggufone/engine/readout.py"
meta = repo / "mutants" / relative
meta = meta.with_name(meta.name + ".meta")
data = json.loads(meta.read_text())
scores = collections.Counter(data["exit_code_by_key"].values())
killed = scores.get(1, 0)
survived = scores.get(0, 0)
other = {code: n for code, n in scores.items() if code not in (0, 1)}
total = killed + survived + sum(other.values())
print(f"{relative} mutants: {total}  killed={killed}  survived={survived}  other={other}")
if total:
    print(f"mutation score: {100.0 * killed / total:.1f}%")
print("\nsurvivors:")
for key, code in data["exit_code_by_key"].items():
    if code != 1:
        print(f"  [{code}] {key}")
print("\nsource sha256:", hashlib.sha256((repo / relative).read_bytes()).hexdigest())
