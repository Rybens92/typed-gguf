"""Card t_97f1bc93 — hand-mutation table over the remedy's own lines (Tier-M hardening).

mutmut 3.8's operator set emits only expression-to-`None` variants: for `runtime/teardown.py`
that is 4 mutants (`suppress(Exception) -> suppress(None)`, `os._exit(int(code)) ->
os._exit(None)`, `os._exit(int(None))`, and the `bool(...)` call). Those 4 are 4/4 killed, but the
*behavioural* mutations of this remedy — drop the flush, drop `os._exit`, lie about the loader,
hand the shell a 0, point the entry points back at `main` — are not generated at all. Each row
below applies one such mutation alone, runs the card's gate file, records KILLED/SURVIVED with the
failing test names, then restores the file byte-for-byte (sha256 printed per row).

    .venv/bin/python .e2e/t_97f1bc93-vulkan-teardown/hand_mutants.py

Two rows are controls: `e1` (equivalent — `code` is already an `int`) and `c1` (a docstring wording
change) must SURVIVE, or the table is lying about what a kill means.

Adapted from the sibling card's `t_57cc0179-vulkan-teardown/hand_mutants.py` vehicle.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEARDOWN = "src/ggufone/runtime/teardown.py"
MAIN = "src/ggufone/__main__.py"
PYPROJECT = "pyproject.toml"

FLUSH_LOOP = """    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):   # a closed or None stream must not change the code
            stream.flush()
"""
EXIT_CALL = "    os._exit(int(code))\n"
LOADED = "    return bool(ctypes_binding.loaded_runtimes())\n"
DOC_SENTENCE = "truncated document because the answer arrived in a block-buffered pipe).\n"

ROWS: list[tuple[str, str, str, str, str]] = [
    # (name, file, old, new, expected)
    ("h1_drop_the_flush", TEARDOWN, FLUSH_LOOP,
     "    pass  # hand mutant h1: the streams are never flushed before the process ends\n", "KILLED"),
    ("h2_drop_os_exit", TEARDOWN, EXIT_CALL,
     "    return None  # hand mutant h2: the pre-fix path, back to SystemExit\n", "KILLED"),
    ("h3_loader_always_false", TEARDOWN, LOADED,
     "    return False  # hand mutant h3: never end the process deliberately\n", "KILLED"),
    ("h4_loader_always_true", TEARDOWN, LOADED,
     "    return True  # hand mutant h4: end every process deliberately\n", "KILLED"),
    ("h5_silent_zero", TEARDOWN, EXIT_CALL,
     "    os._exit(0)  # hand mutant h5: the shell never sees the command's code\n", "KILLED"),
    ("h6_main_entry_point", MAIN, "from ggufone.cli import run\n",
     "from ggufone.cli import main as run  # hand mutant h6: the pre-fix target\n", "KILLED"),
    ("h7_console_script_target", PYPROJECT, 'ggufone = "ggufone.cli:run"\n',
     'ggufone = "ggufone.cli:main"  # hand mutant h7\n', "KILLED"),
    ("e1_int_is_redundant", TEARDOWN, EXIT_CALL,
     "    os._exit(code)  # hand mutant e1: `code` is already an int (equivalent)\n", "SURVIVED"),
    ("c1_docstring_wording", TEARDOWN, DOC_SENTENCE,
     "truncated document, because the answer may arrive in a block-buffered pipe).\n", "SURVIVED"),
]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gates() -> tuple[int, list[str]]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_cli_teardown.py", "-p", "no:randomly"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    failed = [line.split(" - ")[0].strip()
              for line in completed.stdout.splitlines() if line.startswith("FAILED")]
    tail = completed.stdout.strip().splitlines()[-1:] if completed.stdout.strip() else []
    return completed.returncode, failed, tail[0] if tail else ""


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    print(f"tree: {head}")
    print(f"gate file: tests/test_cli_teardown.py   python: {sys.version.split()[0]}\n")
    verdicts: dict[str, str] = {}
    for name, rel, old, new, expected in ROWS:
        path = ROOT / rel
        before = sha256(path)
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            print(f"{name}: ANCHOR ERROR — {old!r} appears {text.count(old)}x in {rel}")
            return 1
        path.write_text(text.replace(old, new), encoding="utf-8")
        try:
            code, failed, summary = run_gates()
            verdict = "KILLED" if code != 0 else "SURVIVED"
            names = ", ".join(failing.split("::")[-1] for failing in failed) or "(no named failure)"
            print(f"{name}: {verdict:<8} pytest exit={code}  {summary}")
            print(f"    failed: {names}")
        finally:
            path.write_text(text, encoding="utf-8")
        after = sha256(path)
        print(f"    restore: {after[:16]}… {'byte-identical' if after == before else 'CHANGED!'}")
        if after != before:
            return 1
        verdicts[name] = verdict
        if verdict != expected:
            print(f"    !! expected {expected} — the table and the gates disagree")
    print("\n== table ==")
    ok = True
    for name, *_ , expected in ROWS:
        mark = "ok" if verdicts[name] == expected else "MISMATCH"
        ok = ok and verdicts[name] == expected
        print(f"  {name:<26} {verdicts[name]:<8} (expected {expected})  {mark}")
    print("\nRESULT:", "every row as expected" if ok else "TABLE MISMATCH — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
