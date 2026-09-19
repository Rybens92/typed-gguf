"""Print the verdict of named mutants out of a mutmut meta (card t_80f1a4c6 helper)."""
import json
import pathlib
import sys

meta_path = pathlib.Path(sys.argv[1])
if not meta_path.name.endswith(".meta"):
    meta_path = meta_path.with_suffix(meta_path.suffix + ".meta")
meta = json.loads(meta_path.read_text())["exit_code_by_key"]
STATUS = {0: "SURVIVED", 1: "killed", 33: "no-tests", None: "not-run"}
for needle in sys.argv[2:]:
    for key, value in meta.items():
        if needle in key:
            print(f"{STATUS.get(value, value):8} {key}")
