"""Where the quick-preset wall time went (card t_f46cec41 evidence).

Reads the JSON reports written by tools/rehearse_quick.sh and prints:
* the per-suite wall time as the reports measured it;
* for latency, every row's p50 (and the implied measurement count);
* a diff against a full-campaign report of the same model, when one is given.

    python3 tools/quick_breakdown.py <quick-dir> [full-latency.json]
"""
from __future__ import annotations

import json
import pathlib
import sys

SUITES = ("latency", "throughput", "quality", "calibration", "determinism")


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    quick_dir = pathlib.Path(argv[1])
    total = 0.0
    print(f"quick reports in {quick_dir}")
    for suite in SUITES:
        report = load(quick_dir / f"quick_{suite}.json")
        total += float(report["wall_ms"])
        print(f"  {suite:<12s} {report['wall_ms'] / 1000:8.1f} s   quick={report['quick']} "
              f"truncated={report['truncated']}")
    print(f"  {'total':<12s} {total / 1000:8.1f} s")
    print(f"  reproduce: {load(quick_dir / 'quick_latency.json')['commands']['reproduce']}")
    print()
    latency = load(quick_dir / "quick_latency.json")
    print("quick latency rows (p50 ms, n):")
    load_p50 = latency["model_load"]
    print(f"  model load          {load_p50['p50']:10.1f}  n={load_p50['n']}")
    for row in latency["prefill"]:
        print(f"  prefill {row['tokens']:>5} tok   {row['ms']['p50']:10.1f}  n={row['ms']['n']}")
    for row in latency["per_question"]:
        print(f"  candidates {row['candidates']:>2}      {row['ms']['p50']:10.1f}  "
              f"n={row['ms']['n']}  (warm-up call not counted)")
    for row in latency["wave_scaling"]:
        print(f"  N={row['questions']:<2} questions    {row['ms']['p50']:10.1f}  n={row['ms']['n']}"
              f"  (warm-up call not counted)")
    print(f"  warm cache prefill  {latency['warm_cache']['prefill_ms']['p50']:10.1f}"
          f"  questions {latency['warm_cache']['questions_ms']['p50']:.1f}")
    print(f"  amortisation: serve {latency['load_amortisation']['serve_ms_per_request']:.1f} ms, "
          f"one-shot {latency['load_amortisation']['one_shot_ms_per_request']:.1f} ms")
    print()
    if len(argv) > 2:
        full = load(pathlib.Path(argv[2]))
        print(f"full campaign latency report: {argv[2]} (runs={full['config']['runs']}, "
              f"threads={full['config']['threads']})")
        full_load = full["model_load"]
        print(f"  model load p50      {full_load['p50']:10.1f}  n={full_load['n']}")
        for row in full["prefill"]:
            print(f"  prefill {row['tokens']:>5} tok   {row['ms']['p50']:10.1f}  "
                  f"n={row['ms']['n']}")
        for row in full["per_question"]:
            print(f"  candidates {row['candidates']:>2}      {row['ms']['p50']:10.1f}")
        print(f"  wave rows N=1..{full['wave_scaling'][-1]['questions']}: "
              f"{sum(row['ms']['p50'] for row in full['wave_scaling']) / 1000:.1f} s of p50 sum")
        if full.get("truncated"):
            print(f"  the full run is *truncated*: {len(full['skipped'])} rows never started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
