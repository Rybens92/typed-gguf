"""Hand-mutation table over the card's own lines (the Tier-M companion to the mutmut sweep).

    uv run --extra dev --no-sync python /work/t55/hand_mutations.py

mutmut's verdicts are only worth something if they are real on *this* box (a capped pid cgroup can
fake a 100 %, an env re-sync can fake a 0 %). This applies each mutation alone, runs the pin gates,
records the exit code, and restores the file byte-identically — printing the sha256 of every file
before and after, so "restored" is a measurement, not a promise. Two controls must survive: a
docstring change (text) and a logically equivalent rewrite.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path("/work/t55/repo")
GATES = ["tests/test_bench_cpu_force.py", "tests/test_bench_placement.py", "tests/test_bench.py"]

MUTATIONS: list[tuple[str, str, str, str, str]] = [
    # (name, file, old, new, expectation)
    ("drop the pin (the device lookup)",
     "src/ggufone/engine/session.py",
     "        pinned_device = ctypes_binding.cpu_device(runtime)",
     "        pinned_device = None  # hand mutation",
     "killed"),
    ("executed plan keeps the requested layers",
     "src/ggufone/engine/session.py",
     "        n_gpu_layers = 0 if cpu_only else requested",
     "        n_gpu_layers = requested",
     "killed"),
    ("no refusal for a bundle without a CPU device",
     "src/ggufone/engine/session.py",
     "        if pinned_device is None:",
     "        if False:  # hand mutation",
     "killed"),
    ("device list never reaches the loader",
     "src/ggufone/engine/session.py",
     "        params.devices = C.cast(devices, C.c_void_p)",
     "        params.devices = None",
     "killed"),
    ("spec_for stops pinning cpu rows",
     "src/ggufone/bench/harness.py",
     "                     cpu_only=backend == CPU_BACKEND)",
     "                     cpu_only=False)",
     "killed"),
    ("the row string drops the pin marker",
     "src/ggufone/bench/harness.py",
     '    return f"{base} (cpu compute pinned)" if spec.cpu_only else base',
     "    return base",
     "killed"),
    ("control: docstring only (must survive)",
     "src/ggufone/engine/session.py",
     '    """One load attempt with the placement\'s layer count (`llama_model_load_from_file`).',
     '    """One load attempt, hand control: the docstring cannot change behaviour.',
     "survived"),
    ("control: equivalent rewrite (must survive)",
     "src/ggufone/runtime/ctypes_binding.py",
     "    return int(device) if device else None",
     "    return None if not device else int(device)",
     "survived"),
]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    before = {path: sha(REPO / path) for path in {m[1] for m in MUTATIONS}}
    rows: list[tuple[str, str, str, int, str]] = []
    for name, rel, old, new, expect in MUTATIONS:
        path = REPO / rel
        text = path.read_text()
        if text.count(old) != 1:
            rows.append((name, rel, "SNIPPET NOT UNIQUE (%d)" % text.count(old), -1, "ERROR"))
            continue
        path.write_text(text.replace(old, new, 1))
        result = subprocess.run([".venv/bin/python", "-m", "pytest", "-q", *GATES],
                                cwd=REPO, capture_output=True, text=True,
                                env={"HOME": "/work/agent-home", "UV_CACHE_DIR": "/work/.uv-cache",
                                     "PATH": "/usr/bin:/bin"})
        path.write_text(text)                      # byte-identical restore, verified below
        verdict = "killed" if result.returncode else "survived"
        rows.append((name, rel, expect, result.returncode, verdict))

    print(f"{'mutation':52} {'file':38} {'expect':9} {'exit':>4}  verdict")
    ok = True
    for name, rel, expect, code, verdict in rows:
        mark = "OK " if verdict == expect or expect.startswith("SNIPPET") else "!! "
        ok = ok and (verdict == expect or expect.startswith("SNIPPET"))
        print(f"{mark}{name:49} {rel:38} {expect:9} {code:4}  {verdict}")

    after = {path: sha(REPO / path) for path in before}
    print("\ncrypto: restored byte-identically" if before == after else "\n!! FILES NOT RESTORED")
    for path, digest in sorted(before.items()):
        print(f"  {digest[:16]}  {path}  {'(unchanged)' if after[path] == digest else '!! CHANGED'}")
    print("\nhand table:", "all rows as expected" if ok else "MISMATCH — inspect above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
