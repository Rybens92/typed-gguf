#!/usr/bin/env python3
"""Pin what `typed-gguf doctor` reports on Windows — the platform-aware distribution check.

Two cards own this file, and it says which side of the truth it is pinning.

`t_f96fed7f` introduced it to pin the **gap**: `runtime.lock`'s `required_files` are the Linux
SONAMEs (`libllama.so`, `libggml.so`, `libggml-base.so`), `TOOL_NAMES` had no `.exe` leg and the
pinned `llama.dll` carries no build string, so on a *complete* Windows bundle `doctor` failed
`runtime.files`/`runtime.loadable` and reported "cannot determine the build number". That pin did
its job: the first live run (36159785190, job 108153215424) failed on exactly that, and its
`::warning::` named the gap on every run.

`t_8dab8b3a` closes it, and this file pins the **closed** truth instead — same style, inverted:

* `runtime.present`  — `TYPED_GGUF_RUNTIME_DIR` resolves (unchanged);
* `runtime.files`    — **ok**, naming `llama.dll` / `ggml.dll` / `ggml-base.dll`:
  `finder.required_files` maps the lock's canonical SONAMEs onto the platform's own names;
* `runtime.loadable` — **no `E_RUNTIME_MISSING`**: a complete bundle is not "incomplete" any more;
* `runtime.build`    — **ok**, reading the real `b11026`. `build_number` runs the platform's CLI
  (`llama-cli.exe --version`, which prints `version: 0.4.1-dev (build 11026, commit b49650adb)`);
  the DLL cannot answer this — `11026` occurs in **no** file of the zip, because
  `version: %s (build %d, commit %s)` formats a compiled integer;
* `runtime.backends` — **ok** and `cpu` (unchanged);
* the arch scan answers from the binary's own ABI decoration (`.?AUgraph@llama_model_qwen2@@` in a
  PE, `_ZTS17llama_model_qwen2\0` in an ELF) — measured byte-wise on both assets.

What the step must *not* do is pass quietly, so `judge()` refuses by name: a report with a `fail`
check anywhere, an exit code that is not 0 (clean) or 2 (warnings only), a distribution check that
went back to the Linux names, a build number that was not read, a bundle that was never found.
Everything this *runner* still cannot make green — no `runtime.json` to compare a SHA-256 against,
no model in the registry, a deep symbol probe that cannot dlopen a PE — stays a REPORTED
`::warning::` naming each check, printed on every run and never pinned to one runner image.

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
#: The `doctor` exit codes a *healthy* Windows bundle produces (A-E1a-3): 0 clean, 2 warnings-only.
#: 1 means a `fail` check was reported, which is a broken bundle and a red step.
OK_EXIT_CODES = (0, 2)
OK_STATUSES = ("ok", "warnings")
#: The bundle the pinned zip *does* carry, at its root — the names the distribution check must
#: accept and name (the lock's canonical SONAMEs are the Linux spellings of the same three roles).
WINDOWS_LIBS = ("llama.dll", "ggml.dll", "ggml-base.dll")
#: `runtime.lock`'s tag: the build `build_number` has to really read out of the bundle.
PINNED_TAG = "b11026"


def checks_of(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`id -> check` (the report's own `checks` list, keyed)."""
    out: dict[str, dict[str, Any]] = {}
    for check in report.get("checks") or []:
        if isinstance(check, dict) and isinstance(check.get("id"), str):
            out[check["id"]] = check
    return out


def warnings_of(report: dict[str, Any]) -> list[str]:
    """`id: detail` for every `warn` check — what this runner honestly cannot make green."""
    return [f"{cid}: {check.get('detail')}" for cid, check in checks_of(report).items()
            if check.get("status") == "warn"]


def judge(report: dict[str, Any], *, exit_code: int | None) -> list[str]:
    """Every problem with a claimed Windows doctor report. `[]` = the pinned truth holds."""
    problems: list[str] = []
    checks = checks_of(report)
    if not checks:
        return [f"the report carries no checks (status={report.get('status')!r}): `doctor --json` "
                f"did not answer the shape this pin reads"]
    if exit_code not in OK_EXIT_CODES:
        problems.append(
            f"`doctor --json` exited {exit_code}, not one of {OK_EXIT_CODES} (0 clean / 2 "
            f"warnings): A-E1a-3 fixes 1 for a report with a `fail` check, so this is a broken "
            f"bundle — or the report shape moved")
    status = report.get("status")
    if status not in OK_STATUSES:
        problems.append(f"the report's status is {status!r}, not one of {OK_STATUSES}")
    broken = sorted(cid for cid, check in checks.items() if check.get("status") == "fail")
    if broken:
        problems.append(
            f"`doctor` reports {len(broken)} FAIL check(s) for the pinned Windows bundle "
            f"({', '.join(broken)}): a `fail` is a broken bundle, never a note (the warnings are "
            f"the only thing this step reports without failing)")
    present = checks.get("runtime.present")
    if present is None or present.get("status") != "ok":
        detail = (present or {}).get("detail")
        problems.append(
            f"`runtime.present` is {detail!r}, not ok: TYPED_GGUF_RUNTIME_DIR did not resolve to a "
            f"bundle the finder accepts (the job points it at the flattened zip)")
    files = checks.get("runtime.files")
    if files is None or files.get("status") != "ok":
        problems.append(
            f"`runtime.files` is {(files or {}).get('status')!r} "
            f"({(files or {}).get('detail')!r}), not ok: the distribution check stopped using the "
            f"platform's names — the gap card t_8dab8b3a closed is back (a Linux-shaped "
            f"`required_files` read?)")
    else:
        detail = str(files.get("detail") or "")
        missing = [dll for dll in WINDOWS_LIBS if dll not in detail]
        if missing:
            problems.append(
                f"`runtime.files` does not name {missing}: the check must say which files it "
                f"found, by the platform's own names, got {detail!r}")
    loadable = checks.get("runtime.loadable")
    loadable_detail = str((loadable or {}).get("detail") or "")
    if loadable is not None and (loadable.get("status") == "fail"
                                 or "E_RUNTIME_MISSING" in loadable_detail):
        problems.append(
            f"`runtime.loadable` is {loadable.get('status')!r} ({loadable_detail!r}): a complete "
            f"Windows bundle must not be called incomplete (the t_f96fed7f gap)")
    build = checks.get("runtime.build")
    if build is None or build.get("status") != "ok":
        problems.append(
            f"`runtime.build` is {(build or {}).get('status')!r} "
            f"({(build or {}).get('detail')!r}), not ok: the build number has to be READ from the "
            f"platform's CLI (`llama-cli.exe --version`), never assumed")
    elif PINNED_TAG not in str(build.get("detail") or ""):
        problems.append(
            f"`runtime.build` does not report {PINNED_TAG}: got {build.get('detail')!r} — the "
            f"bundle under TYPED_GGUF_RUNTIME_DIR is not the pinned one")
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
    """The loud line every run prints: the facts verified here, and what is still only warned."""
    checks = checks_of(report)
    build_detail = str((checks.get("runtime.build") or {}).get("detail") or "unknown")
    warns = warnings_of(report)
    if not warns:
        return (f"::notice::typed-gguf doctor on Windows is fully green: "
                f"{', '.join(WINDOWS_LIBS)} validated by the platform-aware distribution check, "
                f"runtime.build says {build_detail!r}, and this runner reports no warnings.")
    return ("::warning::typed-gguf doctor on Windows: the platform-aware checks are green — "
            f"{', '.join(WINDOWS_LIBS)} validated by runtime.files and runtime.build says "
            f"{build_detail!r} — and this runner still warns {len(warns)} time(s), reported on "
            "every run instead of hidden: " + " | ".join(warns))


def run_doctor(prefix: Sequence[str], cwd: pathlib.Path) -> tuple[str, int]:
    """`<prefix> doctor --json` on this box: (stdout, exit code)."""
    result = subprocess.run([*prefix, "doctor", "--json"], capture_output=True, text=True,  # noqa: S603
                            cwd=str(cwd))
    return result.stdout, int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run `doctor --json` and pin the Windows truth: the bundle resolves, the "
                    "distribution check names the DLLs, the build number is read from the "
                    "platform's CLI, and whatever this runner still warns about is reported.")
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
    if checks:
        print(annotation(report))
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    verdict = {
        "schema": SCHEMA,
        "cli": opts.cli,
        "exit_code": exit_code,
        "status": report.get("status"),
        "checks": checks,
        "annotation": annotation(report) if checks else None,
        "warnings": warnings_of(report),
        "problems": problems,
        "ok": not problems,
    }
    if opts.json:
        pathlib.Path(opts.json).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(f"windows doctor RED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("windows doctor OK: the pinned Windows truth holds (warnings are reported, not hidden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
