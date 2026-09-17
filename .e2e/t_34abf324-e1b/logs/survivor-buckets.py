"""Fast line-scanner bucketing of the schema.py survivors (no AST, no full-file parsing)."""
import collections
import json
import pathlib
import re

REPO = pathlib.Path("/var/home/rybens/workspace/ggufone")
MUT = REPO / "mutants" / "src" / "ggufone" / "schema.py"
ORIG = REPO / "src" / "ggufone" / "schema.py"


def bodies(path: pathlib.Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        match = re.match(r"^def (\w+)\(", line)
        if match:
            current = match.group(1)
            out[current] = []
        elif current is not None:
            if line and not line[0].isspace() and not line.startswith(")"):
                current = None
                continue
            out[current].append(line)
    return out


mut_bodies = bodies(MUT)
orig_bodies = bodies(ORIG)
data = json.loads(MUT.with_suffix(".py.meta").read_text())

buckets = collections.Counter()
per_function = collections.Counter()
samples: dict[str, list] = {}
for key, code in data["exit_code_by_key"].items():
    if code == 1:
        continue
    short = key.rsplit(".", 1)[-1]
    fn = short.split("__mutmut_")[0].removeprefix("x_")
    mutated = mut_bodies.get(short, [])
    original = orig_bodies.get(fn, [])
    changed = [line.strip() for line in mutated if line.strip() not in
               {l.strip() for l in original}]
    text = " ".join(changed)
    if not changed:
        kind = "body-not-found"
    elif re.fullmatch(r"raise [\w.]*\(.*\)", " ".join(changed)) or "_fail(" in text:
        if "XX" in text or "None," in text or text.upper() == text:
            kind = "message-text"
        else:
            kind = "message-other"
    elif "return " in text and len(changed) == 1:
        kind = "return-value"
    elif changed and all(re.match(r"(if |elif |while )", line) for line in changed):
        kind = "branch-condition"
    else:
        kind = "other"
    buckets[kind] += 1
    per_function[f"{kind}:{fn}"] += 1
    samples.setdefault(kind, []).append((key, changed[:3]))

print("total survivors:", sum(buckets.values()))
print("buckets:", dict(buckets))
print("\ntop:", per_function.most_common(10))
for kind in ("other", "return-value", "branch-condition", "message-other"):
    if samples.get(kind):
        print(f"\n--- {kind} samples ---")
        for key, changed in samples[kind][:8]:
            print("  ", key)
            for line in changed:
                print("       ", line)
