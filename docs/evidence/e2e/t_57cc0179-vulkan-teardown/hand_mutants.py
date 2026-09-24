"""Card t_57cc0179 — hand-mutation table over the diff (the sweep cannot converge on this box).

`mutmut run` on this container dies at `os.fork()` with `EAGAIN` whenever the shared pid cgroup
(256, with the E3 campaign, the voice-companion trees and the other kanban workers inside it) fills
up — 306 of 1162 mutants never ran. The sweep's *scored* half is reported as it stands
(`logs/mutation_score.txt`); this table answers the question the score alone cannot: does a
*behavioural* mutant of the new logic survive the gates?

Method (the repo's own replay discipline): for each mutant, splice it into the source, run the two
isolation gate files with the working tree at HEAD, record KILLED (a gate failed) or SURVIVED (all
green) with the failing test's name, restore the file, and print the file's sha256 before/after so
the restore is provable. A control mutant that must SURVIVE is included.

    python hand_mutants.py > .e2e/t_57cc0179-vulkan-teardown/logs/hand_mutants.txt
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/work/t57cc-ggufone")
SOURCE = ROOT / "src/ggufone/bench/isolation.py"
SUITES = ROOT / "src/ggufone/bench/suites.py"
PYTHON = ROOT / ".venv/bin/python"
GATES = ["tests/test_bench_isolation.py", "tests/test_bench_teardown_crash.py"]

MUTANTS = [
    ("M1 retry_placement: half the layers -> the same layers", SOURCE,
     "        half = start // 2", "        half = start"),
    ("M2 retry_placement: '< 0 means every layer' -> never promoted", SOURCE,
     "    if start < 0 and counted > 0:\n        start = counted", "    if False:\n        start = counted"),
    ("M3 retry_placement: the KV rung -> the same KV", SOURCE,
     "        lower = _next_kv(fit, asked_kv)", "        lower = asked_kv"),
    ("M4 teardown_crash: only a missing report is checked", SOURCE,
     "    if exit_code is None or int(exit_code) not in FATAL_SIGNALS:",
     "    if exit_code is None:"),
    ("M5 teardown_crash: a crash without a report is the shape too", SOURCE,
     "    if report is None:\n        return None", "    if False:\n        return None"),
    ("M6 _asked_layers: the model's own placement is ignored", SOURCE,
     "    value = used.get(\"n_gpu_layers\") if isinstance(used, Mapping) else None",
     "    value = None"),
    ("M7 starved: the margin is ignored", SOURCE,
     "    return memory.free_bytes - FIT_MARGIN_BYTES < int(weights_bytes)",
     "    return memory.free_bytes < int(weights_bytes)"),
    ("M8 the withheld row loses its warning", SOURCE,
     "    if child.warning is not None:\n        row[\"warnings\"] = [child.warning]",
     "    if child.warning is not None:\n        row[\"warnings\"] = []"),
    ("M9 the retry loop runs once: no retry ever", SOURCE,
     "    for index in range(1, attempts + 1):", "    for index in range(1, 2):"),
    ("M10 the recovered row hides the crash from process_block", SOURCE,
     "    if child.ok and len(child.attempts) > 1:", "    if False and len(child.attempts) > 1:"),
    ("M11 suites: the crash warning re-enters the mismatch aggregation", SUITES,
     "    mismatched = [row for row in isolation.verified_rows(rows) if row.get(\"warnings\")]",
     "    mismatched = [row for row in rows if row.get(\"warnings\")]"),
    ("M12 (control, must SURVIVE) crash_warning: the closing sentence reworded", SOURCE,
     '    parts.append("; the row is withheld (`measured: false`) and the report is not ok")',
     '    parts.append("; the row is withheld and the report is not ok")'),
]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_gates() -> tuple[int, str]:
    done = subprocess.run([str(PYTHON), "-m", "pytest", "-q", "-p", "no:randomly", "-x", *GATES],
                          cwd=ROOT, capture_output=True, text=True)
    failing = ""
    for line in done.stdout.splitlines():
        if line.startswith("FAILED "):
            failing = line.split("FAILED ", 1)[1].split(" ")[0]
            break
    return done.returncode, failing


def main() -> int:
    verdicts: dict[str, str] = {}
    for name, path, old, new in MUTANTS:
        before = sha256(path)
        text = path.read_text(encoding="utf-8")
        if old not in text:
            print(f"{name}: ANCHOR NOT FOUND ({old[:60]!r})", flush=True)
            verdicts[name] = "ANCHOR-MISS"
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        code, failing = run_gates()
        path.write_text(text, encoding="utf-8")
        after = sha256(path)
        verdict = "KILLED" if code != 0 else "SURVIVED"
        verdicts[name] = verdict
        print(f"{verdict:<8} {name}  [sha {before}->{after}"
              f"{'' if before == after else ' !! NOT RESTORED'}]"
              f"{('  failing: ' + failing) if failing else ''}", flush=True)
    print()
    print("KILLED:", sum(1 for v in verdicts.values() if v == "KILLED"),
          "| SURVIVED:", sum(1 for v in verdicts.values() if v == "SURVIVED"),
          "| other:", sum(1 for v in verdicts.values() if v not in ("KILLED", "SURVIVED")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
