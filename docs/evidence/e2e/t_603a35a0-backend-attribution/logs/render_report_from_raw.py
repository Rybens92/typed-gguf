"""Render the report inside a bench raw log (log + JSON) with the tree's own renderer."""
import json
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from ggufone.bench import harness  # noqa: E402


def payload_of(path: pathlib.Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    json_at = next(i for i, line in enumerate(lines) if line.strip() == "{" and i > 5)
    joined = "\n".join(lines[json_at:])
    return json.loads(joined[: joined.rfind("}") + 1])


for name in sys.argv[2:]:
    path = pathlib.Path(name)
    print("=" * 20, path.name)
    print(harness.render_report(payload_of(path)))
