"""Show sampled surviving mutants (mutmut show) so the classes can be classified by hand.

Usage: uv run --extra dev --with mutmut python .e2e/t_c8e36cad-e1c/logs/show_survivors.py [n_per_fn]
"""
from __future__ import annotations

import collections
import json
import pathlib
import subprocess
import sys

repo = pathlib.Path("/workspace/ggufone")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2

WANT = {
    "src/ggufone/engine/template.py": ["x_suppress_thinking", "x_resolve", "x__user_resolution",
                                       "parse_comparison", "LoopState.attr", "x__scan"],
    "src/ggufone/runtime/fit.py": ["x_read_tensor_index", "x__kv_from_budget", "x_estimate_plan",
                                   "x_plan_for_model", "ModelFacts.read", "x_host_facts"],
}
for rel, wanted in WANT.items():
    meta = (repo / "mutants" / rel).with_name(pathlib.Path(rel).name + ".meta")
    codes = json.loads(meta.read_text())["exit_code_by_key"]
    survivors = [name for name, code in codes.items() if code == 0]
    per_fn: dict[str, list[str]] = collections.defaultdict(list)
    for name in survivors:
        body = name.split("__mutmut_")[0]
        parts = body.split("ǁ")
        fn = ".".join(parts[-2:]) if len(parts) > 1 else body
        if any(w in body for w in wanted):
            per_fn[body.split(".")[-1]].append(name)
    print(f"\n### {rel}: {len(survivors)} survivors; sampling from "
          f"{len(per_fn)} wanted functions")
    for fn, names in sorted(per_fn.items(), key=lambda kv: -len(kv[1])):
        print(f"\n--- {fn}: {len(names)} survivors, showing {n} ---")
        for name in names[:n]:
            out = subprocess.run(["uv", "run", "--extra", "dev", "--with", "mutmut",
                                  "mutmut", "show", name],
                                 cwd=repo, capture_output=True, text=True, check=False,
                                 encoding="utf-8", errors="replace")
            body = (out.stdout or out.stderr).strip()
            print(f"  [{name}]\n" + "\n".join("    " + line for line in body.splitlines()[:22]))
