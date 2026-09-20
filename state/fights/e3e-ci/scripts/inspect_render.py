import json
import pathlib
import sys

path = sys.argv[1]
rec = json.loads(pathlib.Path(path).read_text())
fam = rec["families"][2]
print("record keys:", list(rec.keys()))
print("family keys:", list(fam.keys()))
print(json.dumps({k: (v if not isinstance(v, str) or len(v) < 120 else v[:120] + "…")
                  for k, v in fam.items()}, indent=1, ensure_ascii=False)[:3000])
