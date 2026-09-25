#!/usr/bin/env python3
"""The runtime-matrix oracle step's known-skip filter (card t_f96fed7f, AC1).

`.github/workflows/runtime-matrix.yml` is the live acceptance harness: it unpacks the *pinned*
llama.cpp bundle, points `TYPED_GGUF_RUNTIME_DIR` at it and runs `docs/verify_runtime_contract.py`.
A skip in that run is treated as a failure on purpose — a distribution path that silently
regressed must not read green. The first run the harness ever had (`36151935399`, 2026-09-25)
failed on that rule in **both** Linux jobs, with exactly two skips:

    SKIP /home/runner/.hermes/models/Spark-X2.5-4B-Q8_0.gguf absent — sha256 download-verify
    evidence not re-run
    SKIP local Spark GGUF absent — header-parse pins not re-run

Both are *inherent* on a runner: they are about the 4.4 GB Spark GGUF, which this card's constraints
forbid downloading (and which no runner has). The assertion (`! grep -q "SKIP" <<< "$out"`) was
written before the harness ever ran, so the policy had never been exercised.

"Skips are failures" is still the right policy — the harness exists so a skip cannot hide a
distribution regression — but the exceptions have to be *named*, and the check has to be about
skip lines rather than about a substring over the whole output. This filter:

* counts a line as a skip only when it *starts* with `SKIP` (the oracle's own prefix), so a `FAIL`
  message that mentions the word is not one — the grep could not tell the difference;
* allows a skip only when it carries one of `KNOWN_SKIPS`' substrings. Substrings, not whole
  lines: the first one names an absolute path (`$HOME/.hermes/models/…`), which belongs to the box
  and not to the oracle, so the same skip has to be recognised under `/home/runner`, `/root` or
  `/work`;
* refuses everything else — an unexplained skip exits non-zero and is printed, with the two allowed
  ones listed beside it so the job log shows what was forgiven;
* refuses a run whose own trailing `failures: … skips: …` summary disagrees with the skip lines it
  printed, and a run with no summary at all: a capture that lost a line is a capture that cannot be
  trusted to have lost only that one.

Usage (as the workflow calls it — the oracle's stdout on stdin):

    python3 docs/verify_runtime_contract.py | python3 tools/matrix_oracle_skips.py

Exit 0 = every skip explained. 1 = an unexplained skip, or a run this filter cannot vouch for.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

#: The oracle's own skip prefix (`docs/verify_runtime_contract.py`, `skip()`).
SKIP_PREFIX = "SKIP "
#: The oracle's own trailing summary (`main()`): `failures: 0  skips: 2`.
SUMMARY_RE = re.compile(r"^failures:\s*(?P<failures>\d+)\s+skips:\s*(?P<skips>\d+)\s*$")
#: `(substring, why it is allowed)`. Exactly the two skips run 36151935399 printed; both are
#: model-absent skips, and the model is absent *by design* (SPEC 6 / this workflow's own header:
#: the harness downloads the pinned bundle and a small GGUF, never the 4.4 GB Spark file).
KNOWN_SKIPS: tuple[tuple[str, str], ...] = (
    ("sha256 download-verify evidence not re-run",
     "the 4.4 GB Spark-X2.5-4B-Q8_0.gguf is not on a runner (and must never be downloaded here), "
     "so the sha256 download-verify pins are re-run on the box that holds the file"),
    ("header-parse pins not re-run",
     "the same file, the same reason: the GGUF header-parse pins need it"),
)


@dataclasses.dataclass(frozen=True)
class Report:
    """What one captured oracle run says about its own skips."""

    #: `(skip line, the allowance's reason)` for every skip this card named.
    known: list[tuple[str, str]]
    #: Skip lines no allowance covers — each one fails the step.
    unknown: list[str]
    #: The oracle's own counts, or `None` when the summary line is missing.
    failures: int | None
    skips: int | None

    @property
    def summary_missing(self) -> bool:
        return self.skips is None

    @property
    def summary_disagrees(self) -> bool:
        return self.skips is not None and self.skips != len(self.known) + len(self.unknown)

    def problems(self) -> list[str]:
        """Everything that must fail the step, in the order the filter found it."""
        out: list[str] = []
        if self.summary_missing:
            out.append("the run carries no `failures: … skips: …` summary line: this capture is "
                       "not a complete `docs/verify_runtime_contract.py` run, so nothing about its "
                       "skips can be trusted")
        for line in self.unknown:
            out.append(f"unexplained skip (a skip in this harness is a failure): {line}")
        if self.summary_disagrees:
            out.append(f"the oracle's own summary says skips: {self.skips} but it printed "
                       f"{len(self.known) + len(self.unknown)} skip line(s) — the capture and the "
                       f"run disagree")
        return out


def skip_lines(text: str) -> list[str]:
    """Every line of `text` that *is* a skip: the oracle's prefix, nothing else."""
    return [stripped for stripped in (line.strip() for line in text.splitlines())
            if stripped.startswith(SKIP_PREFIX)]


def classify(line: str) -> str | None:
    """The reason this skip is allowed, or `None` when no allowance covers it."""
    for substring, reason in KNOWN_SKIPS:
        if substring in line:
            return reason
    return None


def examine(text: str) -> Report:
    """Sort one captured oracle run into allowed and unexplained skips."""
    known: list[tuple[str, str]] = []
    unknown: list[str] = []
    for line in skip_lines(text):
        reason = classify(line)
        if reason is None:
            unknown.append(line)
        else:
            known.append((line, reason))
    failures: int | None = None
    skips: int | None = None
    for line in reversed(text.splitlines()):
        match = SUMMARY_RE.match(line.strip())
        if match:
            failures, skips = int(match.group("failures")), int(match.group("skips"))
            break
    return Report(known=known, unknown=unknown, failures=failures, skips=skips)


def verdict(report: Report) -> list[str]:
    """The lines this filter prints about a run it accepts (the *why* of every allowance)."""
    lines = [f"known model-absent skips: {len(report.known)} (allowed — the 4.4 GB Spark GGUF "
             f"cannot be on a runner)"]
    for line, reason in report.known:
        lines.append(f"  allowed: {line}")
        lines.append(f"           ^ {reason}")
    lines.append(f"failures: {report.failures}  skips: {report.skips}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail the runtime-matrix oracle step on any skip except the two model-absent "
                    "ones (card t_f96fed7f).")
    parser.add_argument("--input", default=None,
                        help="the oracle's captured stdout (default: stdin)")
    parser.add_argument("--quiet", action="store_true",
                        help="print the verdict only when it is not green")
    opts = parser.parse_args(argv)

    if opts.input is None:
        text = sys.stdin.read()
    else:
        path = pathlib.Path(opts.input)
        if not path.exists():
            print(f"no such file: {path} (the oracle's stdout is the filter's input)",
                  file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")

    report = examine(text)
    problems = report.problems()
    if not opts.quiet or problems:
        for line in verdict(report):
            print(line)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"the oracle step is red: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
