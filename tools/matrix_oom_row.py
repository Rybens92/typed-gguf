"""`tools/matrix_oom_row.py` — the two fake-OOM worlds of the linux-cpu job, judged as data.

Card t_31b3943a added the step ("the placement retry answers a typed row, never E_INTERNAL") and
its assertion was `'3 placement(s)' in reason`. That number came from `fit_oom_probe`'s ladder; the
*bench* row it was asserted against cannot produce it — a `cpu` row is CPU-pinned (card t_55de5779)
and therefore has no rungs to walk. The assertion had never run, because the fixture was refused
one gate earlier (`E_RUNTIME_SYMBOLS: … cannot name its CPU device`), and the first live matrix run
(36159785190, job 108153215436) is where that met reality (card t_8dab8b3a).

So this tool pins BOTH worlds, each with the count it really has:

* `--bench-row` (the product's own row): `E_BACKEND_OOM`, `measured: false`, the CPU-pinned cpu row,
  **1** placement down to CPU-only, and never `E_INTERNAL`/`AttributeError`;
* `--probe-receipt` (the same bundle, a full-offload plan): the ladder itself — `E_BACKEND_OOM`
  over **3** placements (measured: `n_gpu_layers 36 -> 18 -> 0`).

    tools/matrix_oom_row.py --bench-row /tmp/placement-oom.json --bench-exit 1 \
        --probe-receipt /tmp/oom-probe.json [--json verdict.json]

Exit 0 = both worlds answered the typed row. 1 = they did not (reasons on stderr).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

SCHEMA = "typed_gguf.matrix.oom_row/v1"
#: The typed code both worlds must answer with (never a bare crash).
TYPED_CODE = "E_BACKEND_OOM"
#: Measured on the fixed head (the card's own runs): the bench row resolves to the `cpu` backend,
#: which `spec_for` pins to the bundle's CPU device — one attempt, no ladder; the probe's
#: full-offload plan walks 36 -> 18 -> 0.
BENCH_PLACEMENTS = 1
LADDER_PLACEMENTS = 3
#: Strings that mean the step's own reason for existing (the E2 crash) came back.
FORBIDDEN = ("E_INTERNAL", "AttributeError", "Traceback (most recent call last)")
_COUNT_RE = re.compile(r"tried (\d+) placement\(s\)")


def placements(reason: str) -> int | None:
    """The `tried N placement(s)` count a message carries, or `None` when it says nothing."""
    match = _COUNT_RE.search(reason)
    return int(match.group(1)) if match else None


def _common_reason_problems(reason: str) -> list[str]:
    problems: list[str] = []
    for needle in FORBIDDEN:
        if needle in reason:
            problems.append(f"the row carries {needle!r}: the crash this step exists for came "
                            f"back — reason={reason!r}")
    if TYPED_CODE not in reason:
        problems.append(f"the row's reason does not name {TYPED_CODE}: reason={reason!r}")
    if "down to CPU-only" not in reason:
        problems.append(f"the row's reason does not say it walked down to CPU-only: {reason!r}")
    return problems


def judge_bench(row_report: dict[str, Any], *, exit_code: int | None) -> tuple[list[str], dict]:
    """Every problem with the bench world's report. `([], facts)` = the pinned truth holds."""
    problems: list[str] = []
    rows = row_report.get("backends") or []
    if not rows:
        return ["the bench report carries no `backends` rows: the step's own JSON is not what this "
                "pin reads (`bench --json --out`)"], {}
    if exit_code != 1:
        problems.append(f"the bench exited {exit_code}, not 1: nothing was measured, so the run "
                        f"must report the row and fail (`exit_code` is `BackendOomError`'s)")
    row = rows[0]
    reason = str(row.get("reason") or "")
    if row.get("measured") is not False:
        problems.append(f"the row claims `measured: {row.get('measured')!r}: an OOM row measured "
                        f"nothing (reason={reason!r})")
    if row.get("backend") != "cpu":
        problems.append(f"the row's backend is {row.get('backend')!r}, not 'cpu': on a GPU-less "
                        f"runner the fake bundle resolves to cpu, and the pinned placement count "
                        f"below is that row's")
    if "(cpu compute pinned)" not in str(row.get("placement") or ""):
        problems.append(
            f"the row is not CPU-pinned (`placement={row.get('placement')!r}`): a cpu row resolves "
            f"the bundle's CPU device and therefore has NO ladder — if this changed, the count "
            f"this pin asserts must be re-read deliberately (card t_55de5779)")
    problems.extend(_common_reason_problems(reason))
    count = placements(reason)
    if count is None:
        problems.append(f"the reason carries no placement count: {reason!r}")
    elif count != BENCH_PLACEMENTS:
        problems.append(
            f"the row reports {count} placement(s), not {BENCH_PLACEMENTS}: a CPU-pinned row has "
            f"no ladder to walk (card t_55de5779) — the ladder claim belongs to the probe world, "
            f"which this step asserts too ({LADDER_PLACEMENTS} rungs)")
    return problems, {"backend": row.get("backend"), "measured": row.get("measured"),
                      "placement": row.get("placement"), "placements": count, "reason": reason}


def judge_ladder(receipt: dict[str, Any]) -> tuple[list[str], dict]:
    """Every problem with the probe world's receipt. `([], facts)` = the pinned truth holds."""
    problems: list[str] = []
    error = receipt.get("error") or {}
    code = error.get("code")
    message = str(error.get("message") or "")
    if receipt.get("world") != "oom-all":
        problems.append(
            f"the receipt's world is {receipt.get('world')!r}, not 'oom-all': the probe was not "
            f"run with TYPED_GGUF_FAKE_OOM_ALL=1")
    if code != TYPED_CODE:
        problems.append(f"the receipt's error code is {code!r}, not {TYPED_CODE!r}: the ladder "
                        f"answered something else (message={message!r})")
        return problems, {"code": code, "placements": placements(message), "message": message}
    if message:
        problems.extend(_common_reason_problems(message))
    count = placements(message)
    if count is None:
        problems.append(f"the ladder answer carries no placement count: {message!r}")
    elif count != LADDER_PLACEMENTS:
        problems.append(
            f"the ladder walked {count} placement(s), not {LADDER_PLACEMENTS}: this is the world "
            f"the old step's number came from (`n_gpu_layers 36 -> 18 -> 0`) — if the ladder "
            f"changed, re-read it deliberately")
    return problems, {"code": code, "placements": count, "message": message}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Judge the linux-cpu job's two fake-OOM worlds: the product's own typed row "
                    "(a CPU-pinned cpu row, one rung) and the ladder (a full-offload plan, three).")
    parser.add_argument("--bench-row", required=True,
                        help="the `bench --json --out` report of the fake-OOM run")
    parser.add_argument("--bench-exit", type=int, required=True, help="that run's exit code")
    parser.add_argument("--probe-receipt", required=True,
                        help="the `tools/fit_oom_probe.py --json` receipt of the same bundle")
    parser.add_argument("--json", default=None, help="write the verdict here")
    opts = parser.parse_args(argv)

    problems: list[str] = []
    facts: dict[str, Any] = {}
    try:
        row_report = json.loads(pathlib.Path(opts.bench_row).read_text(encoding="utf-8"))
        receipt = json.loads(pathlib.Path(opts.probe_receipt).read_text(encoding="utf-8"))
        if not isinstance(row_report, dict) or not isinstance(receipt, dict):
            raise ValueError("both inputs must be JSON objects")
        bench_problems, bench_facts = judge_bench(row_report, exit_code=opts.bench_exit)
        ladder_problems, ladder_facts = judge_ladder(receipt)
        problems = bench_problems + ladder_problems
        facts = {"bench": bench_facts, "ladder": ladder_facts}
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        problems = [f"the fake-OOM evidence could not be read ({exc.__class__.__name__}: {exc})"]

    print("fake-OOM worlds:")
    if facts:
        print(f"  bench (cpu-pinned row): {facts['bench'].get('placements')} placement(s), "
              f"reason={facts['bench'].get('reason')!r}")
        print(f"  ladder (full-offload)  : {facts['ladder'].get('placements')} placement(s), "
              f"code={facts['ladder'].get('code')!r}")
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    verdict = {"schema": SCHEMA, "facts": facts, "problems": problems, "ok": not problems}
    if opts.json:
        pathlib.Path(opts.json).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(f"fake-OOM row RED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("fake-OOM row OK: both worlds answered the typed row, never E_INTERNAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
