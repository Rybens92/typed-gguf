"""Write one state file per item: <BASE>/items/<id>.txt, from <BASE>/items.jsonl.

BASE resolution (same rule as run.sh): $REAL_PURPOSE_BASE, else the directory this script lives in.
"""
import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path(os.environ.get("REAL_PURPOSE_BASE") or HERE)

items = [json.loads(line) for line in (BASE / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
out = BASE / "items"
out.mkdir(parents=True, exist_ok=True)
for item in items:
    (out / f"{item['id']}.txt").write_text(item["state"], encoding="utf-8")
ids = [item["id"] for item in items]
print(f"BASE={BASE}")
print(f"wrote {len(ids)} state files: {' '.join(ids)}")
print("queues:", {q: sum(1 for i in items if i["gold_queue"] == q) for q in ("billing", "technical", "account", "policy")})
print("escalate true:", sum(1 for i in items if i["gold_esc"]), "of", len(items))
print("severity:", {s: sum(1 for i in items if i["gold_sev"] == s) for s in (0, 1, 2)})
