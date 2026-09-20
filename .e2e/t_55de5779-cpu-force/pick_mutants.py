"""Pick mutant keys by function + verdict from a mutmut meta file.

    python3 /work/t55/pick_mutants.py <meta> <fn-substring> <verdict> [limit]
"""
from __future__ import annotations

import json
import sys

VERDICT = {1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests",
           34: "skipped", 35: "suspicious", 36: "timeout", 24: "timeout", -24: "timeout",
           152: "timeout", 255: "timeout", 37: "type-check", -11: "segfault", -9: "segfault",
           None: "not run"}

meta = json.load(open(sys.argv[1]))
needle, want = sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
keys = [key for key, code in meta["exit_code_by_key"].items()
        if needle in key and VERDICT.get(code, str(code)) == want]
print(" ".join(keys[:limit] if limit else keys))
