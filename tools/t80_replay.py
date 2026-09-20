#!/usr/bin/env python3
"""Replay mutmut survivors of one module against the gates — with the mandatory control row.

    python3 tools/t80_replay.py <module-path> [<survivor-key-substring> ...]

Applies each surviving mutant to the module *alone*, runs the gate file, records
KILLED/SURVIVED plus the failing test line, and restores the file with `git checkout` (sha256
before/after printed, restore proven). Where the earlier version knew one module and one function,
this one takes the module path and derives each mutant's function from the run-dir block, so a
survivor inside a class method (`xǁModelSessionǁ__init____mutmut_7`) replays the same way as a
module-level one (`x_open_model__mutmut_161`).

Two rows are non-negotiable, both learned on this card:

* **the control row** — the gate file must PASS on the unmutated tree, or every "KILLED" is fake
  (a test that fails on HEAD fails for every mutant);
* **the restore line** — `git checkout -- <module>` runs in a `finally`, because a crash mid-table
  (the shared box forks `EAGAIN` in this container) used to leave a mutant in the working tree.

    python3 tools/t80_replay.py src/typed_gguf/engine/session.py 'ModelSessionǁ__init__'
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TEST = "tests/test_serving_attribution.py"
PYTEST = [".venv/bin/python", "-m", "pytest", "-q", "-p", "no:randomly", "--tb=line", TEST]


def blocks(text: str) -> dict[str, str]:
    """name -> source block, for every `x...__mutmut_<n>` / `..._orig` def (indent-aware)."""
    out: dict[str, str] = {}
    current: str | None = None
    indent = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^(\s*)def (x_?[\wǁ]*__mutmut_(?:orig|\d+))\(", line)
        if match:
            current = match.group(2)
            indent = len(match.group(1))
            out[current] = line
            continue
        if current is not None:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                current = None
            else:
                out[current] += line
    return out


def mutate(source: str, original: str, variant: str) -> str:
    """Apply the orig->variant edit to `source` (the blocks differ only at the mutation).

    The `def` line is dropped from both sides first: mutmut renames the function in each copy, so
    it always differs and is not part of the mutation.
    """
    stripped_orig = original.splitlines()[1:]
    stripped_variant = variant.splitlines()[1:]
    diff = list(difflib.unified_diff(stripped_orig, stripped_variant, lineterm="", n=0))
    removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    if not removed:
        raise SystemExit("no diff between orig and variant")
    for index, line in enumerate(removed):
        new = added[index] if index < len(added) else ""
        assert source.count(line) == 1, f"line not unique in the module: {line!r}"
        source = source.replace(line, new or line, 1) if new else source.replace(line + "\n", "", 1)
    return source


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    src = REPO / argv[1]
    filters = argv[2:]
    run_file = REPO / "mutants" / argv[1]
    meta_path = pathlib.Path(str(run_file) + ".meta")
    codes = json.loads(meta_path.read_text())["exit_code_by_key"]
    short_codes = {key.rsplit(".", 1)[-1]: value for key, value in codes.items()}
    survivors = sorted(key for key, value in short_codes.items()
                       if value == 0 and any(f in key for f in filters))
    print(f"# replaying {len(survivors)} survivors of {argv[1]} matching {filters or '<all>'}")
    original_text = src.read_text()
    sha_before = hashlib.sha256(original_text.encode()).hexdigest()
    print(f"# sha256 before: {sha_before}")
    env = {"PATH": "/usr/bin:/bin", "HOME": "/work/agent-home", "TMPDIR": "/tmp"}
    vk = os.environ.get("VK_DRIVER_FILES")
    if vk:
        env["VK_DRIVER_FILES"] = vk
    control = subprocess.run(PYTEST, cwd=REPO, capture_output=True, text=True, env=env)
    print(f"# CONTROL (unmutated tree): "
          f"{'PASS' if control.returncode == 0 else 'FAIL, aborting (kills would be fake)'}")
    if control.returncode != 0:
        print(control.stdout)
        return 2
    code = blocks(run_file.read_text(encoding="utf-8"))
    try:
        for short in survivors:
            variant = code.get(short)
            orig = code.get(re.sub(r"__mutmut_\d+$", "__mutmut_orig", short))
            if variant is None or orig is None:
                print(f"SKIP     {short}  (no block in the run dir)")
                continue
            src.write_text(mutate(original_text, orig, variant))
            proc = subprocess.run(PYTEST, cwd=REPO, capture_output=True, text=True, env=env)
            verdict = "KILLED" if proc.returncode != 0 else "SURVIVED"
            detail = ""
            if verdict == "KILLED":
                for line in proc.stdout.splitlines():
                    if "test_serving_attribution.py::" in line and "FAILED" in line:
                        detail = line.strip().removeprefix("FAILED ")
                        break
            print(f"{verdict:8} {short}  {detail}", flush=True)
    finally:
        subprocess.run(["git", "checkout", "--", argv[1]], cwd=REPO, check=True)
    sha_after = hashlib.sha256(src.read_text().encode()).hexdigest()
    print(f"# sha256 after:  {sha_after}   restored={sha_after == sha_before}")
    return 0 if sha_after == sha_before else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
