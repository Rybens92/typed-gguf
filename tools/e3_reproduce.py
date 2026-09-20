#!/usr/bin/env python3
"""E3 campaign driver: chunked Occamy runs, the 20-question batch, and the comparison table.

E3 measures a 23 GB MoE (Occamy 1.0, `qwen35moe`) on a box whose *container* sees 8 GiB of RAM:
the model cannot be cached, so every item costs a full weight sweep from disk. A single
60-item pass is therefore hours long and must survive interruption, which is why this tool

* **slices the committed dev set in stratified chunks** (`--write-chunks`) so each chunk is a
  complete little campaign with its own report (a killed run loses one chunk, never the campaign);
* **merges** the chunk reports into the one `quality` report the published table reads
  (`--suite merge`), refusing to double-count an item;
* **renders the comparison** 4B default vs Occamy (`--suite compare`) with Wilson intervals and the
  `low_mass` split (`typed_gguf.bench.compare`);
* **runs the A-E3-2 batch**: 20 dev questions on one state, with `n_seq_max` constrained, through
  the production `typed-gguf run` path (`--suite batch`).

    # 1. the chunks (each one is a `bench --suite quality` run of the same model/subset)
    python3 tools/e3_reproduce.py --write-chunks .e3/chunks --chunk 10
    TYPED_GGUF_RUNTIME_DIR=<bundle> python3 tools/e3_reproduce.py --suite quality \
        --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \
        --backend vulkan --gpu-layers 7 --threads 4 \
        --devset .e3/chunks/devset_001.jsonl --out .e3/chunks/report_001.json

    # 2. one report over everything that landed, then the published table
    python3 tools/e3_reproduce.py --suite merge --reports .e3/chunks/report_*.json \
        --label "Occamy 1.0" --out docs/evidence/e3_occamy_quality.json
    python3 tools/e3_reproduce.py --suite compare --a docs/evidence/e2_quality.json \
        --b docs/evidence/e3_occamy_quality.json --labels "4B default" "Occamy 1.0" \
        --out docs/evidence/e3_comparison.md

    # 3. the batch gate
    TYPED_GGUF_RUNTIME_DIR=<bundle> python3 tools/e3_reproduce.py --suite batch \
        --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf --items 20 --n-seq-max 4
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf.bench import compare, devset, harness, suites  # noqa: E402


# --------------------------------------------------------------------- chunks
def write_chunks(directory: pathlib.Path, *, size: int) -> list[pathlib.Path]:
    """Write the stratified dev-set chunks as JSONL files (the same record shape as the set)."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for number, chunk in enumerate(devset.stratified_chunks(devset.load(), size), start=1):
        path = directory / f"devset_{number:03d}.jsonl"
        path.write_text("".join(json.dumps(item.to_json()) + "\n" for item in chunk),
                        encoding="utf-8")
        written.append(path)
    return written


def build_config(args: argparse.Namespace, suite: str) -> harness.BenchConfig:
    return harness.BenchConfig(
        suite=suite,
        model_path=harness.resolve_model_path(args.model),
        backend=args.backend,
        runs=args.runs,
        threads=args.threads,
        devset=args.devset,
        items=args.items,
        n_seq_max=args.n_seq_max,
        kv_type=args.kv_type,
        gpu_layers=args.gpu_layers)


# --------------------------------------------------------------------- batch (A-E3-2)
def run_batch(args: argparse.Namespace) -> int:
    """`typed-gguf run` with a 20-question batch on one state; the command is what the gate asks."""
    payload = devset.batch_payload(devset.load(), model=args.model, limit=args.items,
                                   n_seq_max=args.n_seq_max)
    state = payload.pop("state")
    work = pathlib.Path(args.work_dir or ".e3")
    work.mkdir(parents=True, exist_ok=True)
    questions_path = work / "batch_questions.json"
    questions_path.write_text(json.dumps({"questions": payload["questions"]}, indent=2),
                              encoding="utf-8")
    state_path = work / "batch_state.txt"
    state_path.write_text(state, encoding="utf-8")
    out_path = pathlib.Path(args.out) if args.out else work / "batch_response.json"
    command = [sys.executable, "-m", "typed_gguf", "run",
               "--questions", str(questions_path), "--state", f"@{state_path}",
               "--model", str(payload["model"]), "--threads", str(args.threads or 4)]
    if args.n_seq_max:
        command += ["--n-seq-max", str(args.n_seq_max)]
    if args.backend:
        command += ["--backend", args.backend]
    if args.no_fit:
        command += ["--no-fit"]
    if args.fit_target is not None:
        command += ["--fit-target", str(args.fit_target)]
    command += ["--out", str(out_path)]
    if args.audit:
        command += ["--audit", str(args.audit)]
    print("$ " + " ".join(command), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - started
    sys.stderr.write(completed.stderr[-4000:])
    if completed.returncode != 0:
        print(f"batch failed: exit {completed.returncode} after {wall_s:.1f}s", file=sys.stderr)
        return completed.returncode
    response = json.loads(out_path.read_text(encoding="utf-8"))
    usage = response.get("usage") or {}
    engine = response.get("engine") or {}
    answers = response.get("answers") or {}
    record = {"wall_s": round(wall_s, 1), "questions": len(payload["questions"]),
              "n_seq_max": args.n_seq_max, "usage": usage,
              "n_gpu_layers": engine.get("n_gpu_layers"), "placement": engine.get("placement"),
              "answers": len(answers), "ok": True}
    print(json.dumps(record, indent=2))
    return 0


# --------------------------------------------------------------------- merge / compare
def merge(args: argparse.Namespace) -> int:
    reports = [json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
               for path in expand(args.reports)]
    merged = compare.merge_reports(reports, label=args.label)
    if args.out:
        harness.write_report(merged, args.out)
        print(f"report: {args.out}")
    print(json.dumps({"items": merged["overall"]["n"], "correct": merged["overall"]["correct"],
                      "agreement": merged["overall"]["agreement"],
                      "per_type": merged["per_type"], "chunks": merged["chunks"]}, indent=2))
    return 0


def expand(patterns: list[str]) -> list[str]:
    """Every `--reports` argument, with shell-style globs expanded (sorted, deterministic)."""
    found: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        found.extend(matches or [pattern])
    return found


def render_comparison(args: argparse.Namespace) -> int:
    baseline = json.loads(pathlib.Path(args.a).read_text(encoding="utf-8"))
    challenger = json.loads(pathlib.Path(args.b).read_text(encoding="utf-8"))
    labels = args.labels or ("baseline", "challenger")
    if len(labels) != 2:
        raise SystemExit("--labels takes exactly two labels")
    paired = None
    if args.align:
        paired = compare.align(baseline, challenger)
        baseline, challenger = paired["baseline"], paired["challenger"]
    table = compare.comparison(baseline, challenger, labels=tuple(labels))
    rendered = compare.render_comparison(table)
    if paired is not None:
        rendered = (rendered + f"\n\nPaired comparison: {paired['items']} dev items measured by "
                    f"both models (dropped {paired['dropped']['baseline']} unpaired baseline "
                    f"row(s) and {paired['dropped']['challenger']} unpaired challenger row(s) so "
                    f"the two sides ask the same questions).\n")
    if args.out:
        pathlib.Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"table: {args.out}")
    print(rendered)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="quality",
                        help="quality | merge | compare | batch")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--devset", default=None)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--n-seq-max", dest="n_seq_max", type=int, default=None)
    parser.add_argument("--kv-type", dest="kv_type", default="auto")
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=None)
    parser.add_argument("--write-chunks", default=None,
                        help="write the stratified dev-set chunks into this directory")
    parser.add_argument("--chunk", type=int, default=10, help="items per chunk (default 10)")
    parser.add_argument("--reports", nargs="+", default=[], help="chunk reports to merge")
    parser.add_argument("--labels", nargs="*", default=None, help="two labels, baseline first")
    parser.add_argument("--a", default=None, help="baseline report for --suite compare")
    parser.add_argument("--b", default=None, help="challenger report for --suite compare")
    parser.add_argument("--align", action="store_true",
                        help="cut both reports to the items they both measured (paired)")
    parser.add_argument("--label", default=None, help="model label of the merged report")
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--out", default=None, help="write the JSON report (or response) here")
    parser.add_argument("--fit-target", dest="fit_target", type=int, default=None)
    parser.add_argument("--no-fit", dest="no_fit", action="store_true")
    parser.add_argument("--audit", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.write_chunks:
        written = write_chunks(pathlib.Path(args.write_chunks), size=args.chunk)
        for path in written:
            print(f"chunk: {path}")
        return 0
    if args.suite == "merge":
        return merge(args)
    if args.suite == "compare":
        if not (args.a and args.b):
            raise SystemExit("--suite compare needs --a and --b")
        return render_comparison(args)
    if args.suite == "batch":
        if not args.model:
            raise SystemExit("--suite batch needs --model <path.gguf>")
        return run_batch(args)
    config = build_config(args, args.suite)
    report = suites.run_suite(config, factory=suites.live_factory)
    if args.out:
        harness.write_report(report, args.out)
        print(f"report: {args.out}")
    if not args.quiet:
        print(harness.render_report(report))
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
