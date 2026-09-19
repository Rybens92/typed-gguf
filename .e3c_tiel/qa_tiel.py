#!/usr/bin/env python3
"""E3c-Tiel QA: verify the published numbers against the raw artifacts. Read-only.

Checks, in order:
  1. the six chunk reports: ids, row count, every row `ok`, union = 60 unique ids;
  2. the six devset chunk copies: same ids as their report, and byte-identical to E3's
     `docs/evidence/e3_chunks/devset_00N.jsonl` (the pairing claim);
  3. the committed merge tool reproduces the committed `tiel_quality.json`
     (`--suite merge` into /tmp, then field-by-field compare);
  4. the three-way table's numbers recompute from the reports through the committed
     `ggufone.bench.compare` arithmetic;
  5. the quality rows for c01/c02 (for the batch-path comparison in §6);
  6. the batch response: 20/20 answers, usage, warnings, wall (`total_ms`).

Prints PASS/FAIL per check and the numbers it read; exits non-zero on any FAIL.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ggufone.bench import compare  # noqa: E402

EVID = ROOT / "docs" / "evidence"
CHUNKS = EVID / "tiel_chunks"
E3_CHUNKS = EVID / "e3_chunks"
failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures.append(name)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_of_jsonl(path: pathlib.Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(str(json.loads(line).get("id")))
    return out


def main() -> int:
    # 1 · chunk reports
    reports = []
    union: list[str] = []
    for n in range(1, 7):
        report = json.loads((CHUNKS / f"report_{n:03d}.json").read_text(encoding="utf-8"))
        rows = compare.rows_of(report)
        reports.append(report)
        union += [str(row.get("id")) for row in rows]
        bad = [str(row.get("id")) for row in rows if str(row.get("status", "ok")) != "ok"]
        check(f"chunk {n:03d} rows", len(rows) == 10 and not bad,
              f"{len(rows)} rows, not-ok {bad or 'none'}")
    check("union of chunk ids", len(union) == 60 and len(set(union)) == 60,
          f"{len(union)} rows, {len(set(union))} unique")

    # 2 · devset copies unchanged vs E3's
    same = []
    for n in range(1, 7):
        mine = ids_of_jsonl(CHUNKS / f"devset_{n:03d}.jsonl")
        theirs = ids_of_jsonl(E3_CHUNKS / f"devset_{n:03d}.jsonl")
        rows = [str(row.get("id")) for row in compare.rows_of(reports[n - 1])]
        same.append(sha256(CHUNKS / f"devset_{n:03d}.jsonl")
                    == sha256(E3_CHUNKS / f"devset_{n:03d}.jsonl"))
        check(f"devset {n:03d} pairing", mine == theirs == rows,
              f"ids equal {mine == theirs == rows} ({len(mine)} items)")
    check("devset bytes identical to E3 chunks", all(same), f"{sum(same)}/6 identical")

    # 3 · committed merge tool reproduces the committed merged report
    out = pathlib.Path("/tmp/qa_tiel_merge.json")
    committed = json.loads((EVID / "tiel_quality.json").read_text(encoding="utf-8"))
    label = committed.get("label")
    cmd = [sys.executable, str(ROOT / "tools" / "e3c_tiel_reproduce.py"), "--suite", "merge",
           "--reports", *[str(CHUNKS / f"report_{n:03d}.json") for n in range(1, 7)],
           "--out", str(out), "--quiet"]
    if label:
        cmd += ["--label", label]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    check("merge command", proc.returncode == 0 and out.exists(),
          f"rc={proc.returncode} label={label!r} out={out.exists()}")
    if out.exists():
        rebuilt = json.loads(out.read_text(encoding="utf-8"))
        keys = ["overall", "per_type", "low_mass", "measured", "coverage",
                "confidence_coverage_correlation", "chunks", "ok", "label"]
        diffs = [k for k in keys if rebuilt.get(k) != committed.get(k)]
        check("merged report == committed", not diffs,
              f"differing keys: {diffs or 'none'}")
        rows_new = {str(r.get("id")): r for r in compare.rows_of(rebuilt)}
        rows_old = {str(r.get("id")): r for r in compare.rows_of(committed)}
        rowdiff = [i for i in rows_new
                   if {k: rows_new[i].get(k) for k in ("correct", "coverage", "reliability")}
                   != {k: rows_old[i].get(k) for k in ("correct", "coverage", "reliability")}]
        check("merged rows field-equal", not rowdiff, f"{len(rows_new)} rows, diffs {rowdiff or 'none'}")

    # 4 · three-way arithmetic recomputed from the three reports
    triad = [EVID / "e2_quality.json", EVID / "e3_occamy_quality.json", EVID / "tiel_quality.json"]
    labels = ["4B default (E2)", "Occamy 1.0 (E3)", "Tiel-Coder (E3c)"]
    models = []
    for path, lab in zip(triad, labels, strict=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        models.append(compare.model_row(report, label=lab, coverage_floor=0.10))
    for model in models:
        print(f"     {model['label']}: overall {model['overall']['agreement']:.3f} "
              f"({model['overall']['correct']}/{model['overall']['n']}) "
              f"ci [{model['overall']['ci'][0]:.3f}-{model['overall']['ci'][1]:.3f}] · "
              f"low_mass {model[compare.LOW_MASS]['n']} · measured {model[compare.MEASURED]['n']}")
    check("three-way ties Tiel to Occamy overall",
          models[2]["overall"]["correct"] == models[1]["overall"]["correct"] == 31,
          f"Tiel {models[2]['overall']['correct']} vs Occamy {models[1]['overall']['correct']}")

    # 5 · the quality rows for the batch comparison
    merged_rows = {str(r.get("id")): r for r in compare.rows_of(committed)}
    for key in ("c01", "c02"):
        row = merged_rows.get(key, {})
        print(f"     quality row {key}: coverage {row.get('coverage')} "
              f"reliability {row.get('reliability')} cue {json.dumps(row.get('cue'))[:160]}")

    # 6 · batch
    batch = json.loads((ROOT / ".e3c_tiel" / "batch_response.json").read_text(encoding="utf-8"))
    answers = batch["answers"]
    nonempty = sum(1 for a in answers.values() if isinstance(a, dict) and a.get("choice") or a.get("answer"))
    check("batch 20/20 answers", len(answers) == 20 and nonempty == 20,
          f"{len(answers)} ids, {nonempty} with a choice/answer")
    print(f"     batch usage: {json.dumps(batch['usage'])}")
    print(f"     batch timings: {json.dumps(batch['timings'])}")
    print(f"     batch warnings: {json.dumps(batch['warnings'])}")
    rel = {}
    for key, value in answers.items():
        rel[str(value.get("reliability"))] = rel.get(str(value.get("reliability")), 0) + 1
    print(f"     batch reliability tally: {rel}")
    covs = sorted(float(value.get("coverage") or 0.0) for value in answers.values())
    print(f"     batch coverage min {covs[0]:.3e} median {covs[len(covs) // 2]:.3e} max {covs[-1]:.3e}")
    refused = [k for k, v in answers.items() if (v.get("cue") or {}).get("refused")]
    print(f"     batch cue.refused rows: {len(refused)} {refused or ''}")

    print()
    print(f"{'ALL CHECKS PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
