#!/usr/bin/env python3
"""Pin what `typed-gguf doctor` reports on Windows — and say the mismatch loudly (t_f96fed7f, AC2).

AC2 asks the Windows job to flatten the pinned zip and run `doctor` against it via
`TYPED_GGUF_RUNTIME_DIR`. What that report *is* is not a free choice, and it is not green:

* `runtime.lock` pins `required_files` as the Linux SONAMEs (`libllama.so`, `libggml.so`,
  `libggml-base.so` — `tests/test_pins.py` pins the same three), while the pinned
  `llama-b11026-bin-win-cpu-x64.zip` carries `llama.dll` / `ggml.dll` / `ggml-base.dll` **at the
  archive root** (verified against the real asset; there is no top-level directory to flatten). So
  `runtime.files` and `runtime.loadable` fail on a bundle that is perfectly complete for Windows;
* `TOOL_NAMES` is `llama-cli` / `llama-fit-params` / `llama-tokenize` — no `.exe` — while the zip
  ships `llama-cli.exe` / `llama-fit-params.exe`. So `build_number()` never runs the CLI, and the
  pinned `llama.dll` carries no `build 11026` string either (checked byte-wise on the real DLL):
  `runtime.build` reports "cannot determine the build number";
* what *does* work on Windows is the part the platform rules exist for: the bundle is found through
  `TYPED_GGUF_RUNTIME_DIR`, and `runtime.backends` classifies it (`*ggml-*.dll` ->
  `backends: cpu`) — `finder.library_names("windows")` / `library_glob("windows")`.

Those three sentences are the truth this card reports, and the failed distribution expectations are
a real finding for the SPEC's R12 ("CI smoke job per platform"): the *loader* is platform-aware, the
*distribution check* is not. This tool runs `doctor --json` on the runner, judges exactly the facts
above, prints a GitHub `::warning::` annotation naming the gap (so it is loud on every run) and
exits 0 while the pinned truth holds. A future card that makes doctor green on Windows (a
platform-aware `required_files`) makes this tool refuse: the pin is then updated on purpose,
never silently.

    tools/matrix_windows_doctor.py --cli "uv run typed-gguf" --json doctor_verdict.json

Exit 0 = the pinned Windows truth. 1 = doctor did not report it (and the reasons are on stderr).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

SCHEMA = "typed_gguf.matrix.windows_doctor/v1"
#: The `doctor` contract this pin rests on: A-E1a-3's statuses, where a `fail` check is "broken".
EXPECTED_EXIT = 1
#: The Linux SONAMEs `runtime.lock.required_files` demands of every bundle (`tests/test_pins.py`).
LINUX_SONAMES = ("libllama.so", "libggml.so", "libggml-base.so")
#: The bundle the pinned zip *does* carry, at its root.
WINDOWS_LIBS = ("llama.dll", "ggml.dll", "ggml-base.dll")
ANNOTATION = (
    "typed-gguf doctor cannot pass on Windows: runtime.lock's `required_files` are the Linux "
    "SONAMEs ({sonames}), so `runtime.files`/`runtime.loadable` fail on a bundle whose llama.dll "
    "is right there, and `runtime.build` cannot read a build from llama-cli.exe / llama.dll. The "
    "bundle IS found and classified ({backends}). The loader is platform-aware; the distribution "
    "check is not (SPEC R12). Card t_f96fed7f reports this instead of hiding it."
)


def checks_of(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`id -> check` (the report's own `checks` list, keyed)."""
    out: dict[str, dict[str, Any]] = {}
    for check in report.get("checks") or []:
        if isinstance(check, dict) and isinstance(check.get("id"), str):
            out[check["id"]] = check
    return out


def judge(report: dict[str, Any], *, exit_code: int) -> list[str]:
    """Every problem with a claimed Windows doctor report. `[]` = the pinned truth holds."""
    problems: list[str] = []
    checks = checks_of(report)
    if not checks:
        return [f"the report carries no checks (status={report.get('status')!r}): `doctor --json` "
                f"did not answer the shape this pin reads"]
    if exit_code != EXPECTED_EXIT:
        problems.append(
            f"`doctor --json` exited {exit_code}, not {EXPECTED_EXIT}: A-E1a-3 fixes 1 for "
            f"'broken' (a report with a `fail` check). If the distribution check became "
            f"platform-aware, this pin is stale and must be updated on purpose")
    present = checks.get("runtime.present")
    if present is None or present.get("status") != "ok":
        detail = (present or {}).get("detail")
        problems.append(
            f"`runtime.present` is {detail!r}, not ok: TYPED_GGUF_RUNTIME_DIR did not resolve to a "
            f"bundle the finder accepts (the job points it at the flattened zip)")
    files = checks.get("runtime.files")
    if files is None or files.get("status") != "fail":
        problems.append(
            f"`runtime.files` is {(files or {}).get('status')!r}, not fail: the pinned gap this "
            f"card reports is gone (a platform-aware `required_files`?) — update this pin "
            f"deliberately instead of letting the job pass for a reason nobody wrote down")
    else:
        detail = str(files.get("detail") or "")
        missing = [soname for soname in LINUX_SONAMES if soname not in detail]
        if missing:
            problems.append(
                f"`runtime.files` does not name {missing}: the reason it fails must be the Linux "
                f"SONAMEs, got {detail!r}")
    loadable = checks.get("runtime.loadable")
    loadable_detail = str((loadable or {}).get("detail") or "")
    if "E_RUNTIME_MISSING" not in loadable_detail:
        problems.append(
            f"`runtime.loadable` does not carry E_RUNTIME_MISSING ({loadable_detail!r}): the "
            f"loader must say the bundle is incomplete for the same reason")
    backends = checks.get("runtime.backends")
    if backends is None or backends.get("status") != "ok":
        problems.append(
            f"`runtime.backends` is {(backends or {}).get('detail')!r}, not ok: the DLL names the "
            f"finder looks for (llama.dll / *ggml-*.dll) were not found in the flattened bundle")
    elif "cpu" not in str(backends.get("detail") or ""):
        problems.append(
            f"`runtime.backends` does not classify the bundle as cpu: "
            f"{backends.get('detail')!r} — the pinned win-cpu zip has no accelerator DLL")
    return problems


def annotation(report: dict[str, Any]) -> str:
    """The loud line every run prints: what doctor cannot do here, and what it can."""
    backends = str((checks_of(report).get("runtime.backends") or {}).get("detail") or "unknown")
    return "::warning::" + ANNOTATION.format(sonames=", ".join(LINUX_SONAMES), backends=backends)


def run_doctor(prefix: Sequence[str], cwd: pathlib.Path) -> tuple[str, int]:
    """`<prefix> doctor --json` on this box: (stdout, exit code)."""
    result = subprocess.run([*prefix, "doctor", "--json"], capture_output=True, text=True,  # noqa: S603
                            cwd=str(cwd))
    return result.stdout, int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run `doctor --json` and pin the Windows truth (card t_f96fed7f): the bundle "
                    "resolves, the Linux SONAMEs are what fail, and that gap is reported loudly.")
    parser.add_argument("--cli", default="uv run typed-gguf",
                        help="the command prefix that runs the product "
                             "(default: `uv run typed-gguf`)")
    parser.add_argument("--workdir", default=None, help="where `doctor` is run (default: cwd)")
    parser.add_argument("--json", default=None, help="write the verdict here")
    opts = parser.parse_args(argv)

    prefix = shlex.split(opts.cli)
    workdir = pathlib.Path(opts.workdir or ".").resolve()
    problems: list[str] = []
    report: dict[str, Any] = {}
    exit_code: int | None = None
    try:
        raw, exit_code = run_doctor(prefix, workdir)
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError(f"the report is a {type(report).__name__}, not an object")
        problems = judge(report, exit_code=exit_code)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        problems = [f"`doctor --json` could not be read ({exc.__class__.__name__}: {exc})"]

    checks = checks_of(report)
    print(f"doctor exit {exit_code} status {report.get('status')!r}")
    for check in report.get("checks") or []:
        print(f"  {check.get('status'):5s} {check.get('id'):24s} {check.get('detail')}")
    if "runtime.files" in checks:
        print(annotation(report))
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    verdict = {
        "schema": SCHEMA,
        "cli": opts.cli,
        "exit_code": exit_code,
        "status": report.get("status"),
        "checks": checks,
        "annotation": annotation(report) if "runtime.files" in checks else None,
        "problems": problems,
        "ok": not problems,
    }
    if opts.json:
        pathlib.Path(opts.json).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(f"windows doctor RED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("windows doctor OK: the pinned Windows truth holds (the gap is reported, not hidden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
