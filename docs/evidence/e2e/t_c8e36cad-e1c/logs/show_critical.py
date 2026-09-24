"""Show sampled surviving mutants from the *criterion-critical* functions (E1c classification).

Usage: uv run --extra dev --with mutmut python .e2e/t_c8e36cad-e1c/logs/show_critical.py <n>
The functions below are the ones an acceptance criterion (A-E1c-1/2/4/5/6) depends on, so a
survivor here is only acceptable if it is provably equivalent.
"""
from __future__ import annotations

import collections
import json
import pathlib
import subprocess
import sys

repo = pathlib.Path("/workspace/ggufone")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

WANT = {
    "src/ggufone/engine/template.py": ["x_suppress_thinking", "x_resolve", "x__user_resolution",
                                       "x_detect_family", "x_no_open_think"],
    "src/ggufone/runtime/fit.py": ["x__kv_from_budget", "x_estimate_plan", "x_plan_from_binary",
                                   "x_measured_rss_bytes", "x_kv_bytes_per_token"],
}
for rel, wanted in WANT.items():
    meta = (repo / "mutants" / rel).with_name(pathlib.Path(rel).name + ".meta")
    codes = json.loads(meta.read_text())["exit_code_by_key"]
    survivors = [name for name, code in codes.items() if code == 0]
    per_fn: dict[str, list[str]] = collections.defaultdict(list)
    for name in survivors:
        body = name.split("__mutmut_")[0]
        if any(w in body for w in wanted):
            per_fn[body.split(".")[-1]].append(name)
    print(f"\n### {rel}: {len(survivors)} survivors; critical functions sampled: "
          f"{sorted(per_fn)}")
    for fn, names in sorted(per_fn.items(), key=lambda kv: -len(kv[1])):
        print(f"\n--- {fn}: {len(names)} survivors, showing {n} ---")
        for name in names[:n]:
            out = subprocess.run(["uv", "run", "--extra", "dev", "--with", "mutmut",
                                  "mutmut", "show", name],
                                 cwd=repo, capture_output=True, text=True, check=False,
                                 encoding="utf-8", errors="replace")
            body = (out.stdout or out.stderr).strip()
            print(f"  [{name}]\n" + "\n".join("    " + line for line in body.splitlines()[:18]))
