#!/usr/bin/env python3
"""Replay this card's `device_evidence` survivors against the new normalization gate.

Applies each surviving mutant to `src/ggufone/engine/decide.py` alone, runs the pinning test,
records KILLED/SURVIVED plus the failing test line, and restores the file with `git checkout`
(printing the sha256 before and after so the restore is provable).

    python3 tools/t80_replay.py            # run from the repo root; writes the table to stdout

The table this prints is saved (by the caller) next to the sweep's other raw material.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src/ggufone/engine/decide.py"
RUN = REPO / "mutants/src/ggufone/engine/decide.py"
META = REPO / "mutants/src/ggufone/engine/decide.py.meta"
TEST = "tests/test_serving_attribution.py"
PYTEST = [".venv/bin/python", "-m", "pytest", "-q", "-p", "no:randomly", "--tb=line", TEST]


def block(text: str, name: str) -> str:
    """The source block of one top-level function."""
    out: list[str] = []
    on = False
    for line in text.splitlines(True):
        m = re.match(r"^def (\w+)\(", line)
        if m:
            on = m.group(1) == name
            if on:
                out = [line]
            continue
        if on:
            if line.strip() and not line[:1].isspace():
                break
            out.append(line)
    assert out, f"block {name} not found"
    return "".join(out)


def main() -> int:
    meta = json.loads(META.read_text())
    codes = meta["exit_code_by_key"]
    survivors = sorted(key.rsplit(".", 1)[-1] for key, code in codes.items()
                       if code == 0 and "device_evidence" in key)
    print(f"# replaying {len(survivors)} device_evidence survivors: {survivors}")
    original = SRC.read_text()
    sha_before = hashlib.sha256(original.encode()).hexdigest()
    print(f"# src sha256 before: {sha_before}")
    env = {"PATH": "/usr/bin:/bin", "HOME": "/work/agent-home", "TMPDIR": "/tmp"}
    vk = os.environ.get("VK_DRIVER_FILES")
    if vk:
        env["VK_DRIVER_FILES"] = vk
    # CONTROL: the pin must PASS on the unmutated tree, or every "KILLED" below is a fake kill
    # (a test that fails on HEAD fails for every mutant). This card's first replay attempt made
    # exactly that mistake (a wrong expected count), so the control is part of the table.
    control = subprocess.run(PYTEST, cwd=REPO, capture_output=True, text=True, env=env)
    print(f"# CONTROL (unmutated tree): "
          f"{'PASS' if control.returncode == 0 else 'FAIL, aborting (kills would be fake)'}")
    if control.returncode != 0:
        print(control.stdout)
        return 2
    run_text = RUN.read_text()
    orig_block = block(original, "device_evidence")
    try:
        for short in survivors:
            mutant = short if short in run_text else f"x_{short}"
            mut_block = block(run_text, mutant)
            patched_block = re.sub(r"^def [\wǁ]+__mutmut_\d+\(", "def device_evidence(", mut_block)
            patched = original.replace(orig_block, patched_block)
            assert patched != original, short
            SRC.write_text(patched)
            proc = subprocess.run(PYTEST, cwd=REPO, capture_output=True, text=True, env=env)
            verdict = "KILLED" if proc.returncode != 0 else "SURVIVED"
            detail = ""
            if verdict == "KILLED":
                for line in proc.stdout.splitlines():
                    if "test_serving_attribution.py:" in line:
                        detail = line.strip()
                        break
            print(f"{verdict:8} {short}  {detail}", flush=True)
    finally:
        # A crash mid-table (this box forks EAGAIN under the sibling campaigns) must not leave a
        # mutant in the working tree: the restore runs even on the failure path.
        subprocess.run(["git", "checkout", "--", "src/ggufone/engine/decide.py"], cwd=REPO,
                       check=True)
    sha_after = hashlib.sha256(SRC.read_text().encode()).hexdigest()
    print(f"# src sha256 after:  {sha_after}   restored={sha_after == sha_before}")
    return 0 if sha_after == sha_before else 1


if __name__ == "__main__":
    sys.exit(main())
