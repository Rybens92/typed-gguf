#!/usr/bin/env python3
"""Reproduce the E2.5 measurements — one command per published table (SPEC 5 / A-E2p5-*).

Three subcommands, each of which is the exact invocation a row of
`docs/BENCHMARKS.md` §2.10 names:

    # 1. fit + store a calibration on a real model (A-E2p5-1/2/3/6)
    GGUFONE_HOME=<home> GGUFONE_RUNTIME_DIR=<bundle> \\
        python3 tools/e2p5_reproduce.py calibrate --model <path.gguf> --threads 2 \\
        --out docs/evidence/e2p5_calibration_qwen08.json

    # 2. a real `--route auto` decision, end to end (A-E2p5-4/7/8)
    GGUFONE_HOME=<home> GGUFONE_RUNTIME_DIR=<bundle> \\
        python3 tools/e2p5_reproduce.py route --model <alias|path> \\
        --out docs/evidence/e2p5_route.json

    # 3. the escalation delta on the committed dev set (A-E2p5-5)
    GGUFONE_HOME=<home> GGUFONE_RUNTIME_DIR=<bundle> \\
        python3 tools/e2p5_reproduce.py escalate --primary <path.gguf> --target <path.gguf> \\
        --threads 2 --out docs/evidence/e2p5_escalation.json

`calibrate` and `route` drive the shipped CLI (`ggufone calibrate`, `ggufone run --route auto`) in
process, so the evidence exercises the real code path; `escalate` measures the policy the CLI
wires up (`routing.escalation_candidates` / `apply_escalation`) over the bench harness, because a
60-item measurement needs each model loaded once instead of once per question.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ggufone import cli  # noqa: E402
from ggufone.bench import devset as devset_module  # noqa: E402
from ggufone.calibration import calibrate, routing  # noqa: E402
from ggufone.registry import store  # noqa: E402
from ggufone.runtime import finder, fit  # noqa: E402

SCHEMA = "ggufone.evidence.e2p5/v1"
DEFAULT_THRESHOLD = routing.DEFAULT_ESCALATION_THRESHOLD


def _write(payload: dict[str, Any], path: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False)
    if path:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(text)


def _runtime() -> str:
    found = finder.find_runtime()
    if found is None:
        raise SystemExit("no runtime installed: set GGUFONE_RUNTIME_DIR to a llama.cpp bundle")
    return str(found)


def _print_tail(label: str, text: str, lines: int = 12) -> None:
    tail = "\n".join(text.splitlines()[-lines:])
    print(f"--- {label} (tail) ---\n{tail}\n")


# --------------------------------------------------------------------- calibrate
def cmd_calibrate(args: argparse.Namespace) -> int:
    out = args.out
    runs: list[dict[str, Any]] = []
    started = time.time()
    argv = ["calibrate", "--model", args.model, "--json", "--threads", str(args.threads)]
    if args.from_report:
        argv += ["--from-report", args.from_report]
    if args.items:
        argv += ["--items", str(args.items)]
    if args.devset:
        argv += ["--devset", args.devset]
    if args.rows_out:
        argv += ["--out", args.rows_out]
    if args.dry_run:
        argv.append("--dry-run")
    for repeat in range(max(1, args.repeat)):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(argv)
        text = buffer.getvalue()
        if code != 0 and not (code == 1 and args.allow_reject):
            print(text[-2000:], file=sys.stderr)
            raise SystemExit(f"`ggufone calibrate` exited {code}")
        payload = json.loads(text)
        runs.append({"code": code, "params_hash": payload["params_hash"],
                     "accepted": payload["accepted"],
                     "temperatures": payload["params"]["types"],
                     "verdicts": {name: {"accepted": entry["accepted"],
                                         "reason": entry["reason"],
                                         "temperature": entry["temperature"],
                                         "holdout_ece": entry["ece"]["holdout"]}
                                  for name, entry in payload["types"].items()}})
        print(f"repeat {repeat}: code={code} accepted={payload['accepted']} "
              f"params={payload['params_hash']}")
        for name, verdict in runs[-1]["verdicts"].items():
            print(f"  {name}: T={verdict['temperature']:.4f} "
                  f"holdout ECE {verdict['holdout_ece']['before']:.4f} -> "
                  f"{verdict['holdout_ece']['after']:.4f} :: {verdict['reason']}")
    stored = calibrate.load_table(store.calibration_path(),
                                  calibrate.model_key_for(args.model))
    report = {
        "schema": SCHEMA,
        "command": " ".join(argv),
        "seconds": round(time.time() - started, 1),
        "model": args.model,
        "runtime": _runtime(),
        "home": str(store.data_home()),
        "rows_out": args.rows_out,
        "repeats": runs,
        "reproducible": len({entry["params_hash"] for entry in runs}) == 1,
        "stored": bool(stored),
        "table": json.loads(json.dumps(_table_payload(args, stored))),
    }
    _write(report, out)
    return 0


def _table_payload(args: argparse.Namespace, stored: Any) -> dict[str, Any]:
    """The table this run stored (or the one `--from-report --dry-run` would have stored)."""
    if stored is not None:
        return stored.to_json()
    rows = calibrate.load_rows(args.from_report) if args.from_report else []
    table = calibrate.fit_table(rows, model_key=calibrate.model_key_for(args.model),
                                model_path=args.model)
    return table.to_json()


# ------------------------------------------------------------------------- route
def ensure_registry(entries: list[str], current: str | None = None) -> None:
    """Register local GGUF files as aliases (`--register alias=path`) so routing has a choice.

    A registry is the router's input (SPEC 2.10: "picks ... from the registry"), and `models
    pull` is the only other way to fill it — which would download models this box already has.
    """
    registry, _warnings = store.load_registry(store.registry_path())
    for spec in entries:
        alias, _, path = spec.partition("=")
        if not alias or not path:
            raise SystemExit(f"--register needs alias=path (got {spec!r})")
        if alias in registry.aliases:
            continue
        size = pathlib.Path(path).stat().st_size
        try:
            facts = fit.ModelFacts.read(path, want_sha256=False)
            arch, quant = facts.arch, None
        except Exception:                              # noqa: BLE001 - a header we cannot read
            arch, quant = None, None
        store.add_entry(registry, store.Entry(alias=alias, path=path, arch=arch, quant=quant,
                                              size=size))
        print(f"registered {alias} -> {path} ({size / (1024 ** 3):.2f} GiB, arch={arch})")
    if current:
        registry.current = current
    store.save_registry(registry, store.registry_path())


def cmd_route(args: argparse.Namespace) -> int:
    ensure_registry(args.register or [], current=args.current)
    questions = {
        "owner": {"type": "choice", "instructions": "Which team owns this incident?",
                  "criteria": {"billing": "payments, invoices and refunds",
                               "technical": "api, infrastructure and deploys",
                               "data": "pipelines, warehouses and reporting"}},
        "severity": {"type": "score", "instructions": "How severe is this?",
                     "criteria": ["cosmetic", "degraded", "customer-visible", "data-loss"]},
        "page": {"type": "noul", "instructions": "Should we page the on-call engineer?",
                 "criteria": {"true": "someone must act now", "false": "it can wait for the next "
                                                                     "business day"}},
    }
    state = ("At 09:12 the checkout page started returning HTTP 500 for every customer. "
             "The error rate on /checkout is 100%, everywhere else 0%. No deploy happened in "
             "the last 24 hours and the database is reachable.")
    body = {"state": state, "questions": questions}
    payload_path = pathlib.Path(args.workdir) / "e2p5_route_request.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    argv = ["run", "--questions", str(payload_path), "--route", "auto",
            "--threads", str(args.threads), "--json"]
    if args.model:
        argv += ["--model", args.model]
    if args.audit:
        argv += ["--audit", args.audit]
    if args.escalate_model:
        argv += ["--escalate", "--max-escalations", str(args.max_escalations),
                 "--escalation-model", args.escalate_model]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    text = buffer.getvalue()
    if code != 0:
        print(text[-2000:], file=sys.stderr)
        raise SystemExit(f"`ggufone run --route auto` exited {code}")
    response = json.loads(text)
    engine = response.get("engine", {})
    audit_records = []
    if args.audit:
        audit_path = pathlib.Path(args.audit) / "audit.jsonl"
        if audit_path.exists():
            audit_records = [json.loads(line) for line in
                             audit_path.read_text(encoding="utf-8").splitlines()]
    report = {
        "schema": SCHEMA,
        "command": " ".join(argv),
        "model_request": args.model,
        "registry": _registry_snapshot(),
        "runtime": _runtime(),
        "route": engine.get("route"),
        "calibration": response.get("calibration"),
        "calibrated": response.get("calibrated"),
        "escalations": engine.get("escalations"),
        "fit": engine.get("fit"),
        "answers": response.get("answers"),
        "warnings": response.get("warnings"),
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "audit": audit_records[-1] if audit_records else None,
    }
    _write(report, args.out)
    return 0


def _registry_snapshot() -> dict[str, Any]:
    registry, _warnings = store.load_registry(store.registry_path())
    return {"current": registry.current,
            "aliases": {name: {"path": entry.path, "quant": entry.quant, "arch": entry.arch,
                               "size": entry.size}
                        for name, entry in sorted(registry.aliases.items())}}


# --------------------------------------------------------------------- escalate
def _measure(path: str, *, threads: int, devset_path: str | None = None,
             items: int | None = None) -> list[Any]:
    """One dev-set pass **through the serving path** (`cli.calibration_rows`).

    The escalation measurement must see the same distributions `run`/`ask` report, so it reuses
    the CLI's collector instead of re-implementing a load: the bench harness has its own
    placement seam (and its own open bug, card t_31b3943a) that E2.5 deliberately does not depend
    on.
    """
    options = {"threads": threads, "devset": devset_path, "items": items, "kv_type": "auto"}
    rows, _source = cli.calibration_rows(options, path)
    return rows


def _answers_of(rows: list[Any]) -> dict[str, dict[str, Any]]:
    return {row.id: row.as_answer() for row in rows}


def _subset_rows(items: list[devset_module.DevItem], ids: set[str]) -> dict[str, Any]:
    """A dev-set view with only `ids` (the escalated questions of a measurement)."""
    return {item.id: item for item in items if item.id in ids}


def cmd_escalate(args: argparse.Namespace) -> int:
    started = time.time()
    items = devset_module.load(args.devset)
    if args.items:
        items = items[: args.items]
    rows = _measure(args.primary, threads=args.threads, devset=args.devset, items=args.items)
    by_id = {item.id: item for item in items}
    answers = _answers_of(rows)
    decisions = routing.escalation_candidates(answers, threshold=args.threshold,
                                              max_escalations=args.limit)
    print(f"primary {args.primary}: {len(rows)} items, {len(decisions)} low-confidence "
          f"({', '.join(decision.question for decision in decisions)})")

    subset = _subset_rows(items, {decision.question for decision in decisions})
    replacement: dict[str, Any] = {}
    target_seconds = 0.0
    target_rows: list[Any] = []
    if subset:
        target_started = time.time()
        target_rows = _measure(args.target, threads=args.threads, items=len(subset),
                               devset_path=_write_subset(subset, args.workdir))
        replacement = _answers_of(target_rows)
        target_seconds = time.time() - target_started
    merged, log = routing.apply_escalation(
        answers, replacement, decisions,
        target={"alias": pathlib.Path(args.target).stem, "path": args.target,
                "model": pathlib.Path(args.target).stem},
        limit=args.limit, threshold=args.threshold)

    def correctness(answer_map: dict[str, Any]) -> dict[str, Any]:
        hits = 0
        for item in items:
            answer = answer_map.get(item.id)
            if answer is None:
                continue
            hits += int(decision_of(answer) == devset_module.gold_key(item))
        low, high = calibrate.stats.wilson(hits, len(items))
        return {"n": len(items), "correct": hits, "agreement": hits / len(items),
                "ci": [low, high]}

    before = correctness(answers)
    after = correctness(merged)
    split = calibrate.split_rows(rows, holdout_fraction=calibrate.HOLDOUT_FRACTION)
    split_ids = {
        "fit": {row.id for row in split.fit},
        "holdout": {row.id for row in split.holdout},
    }

    def split_view(ids: set[str], answer_map: dict[str, Any]) -> dict[str, Any]:
        subset_items = [item for item in items if item.id in ids]
        hits = sum(1 for item in subset_items
                   if decision_of(answer_map.get(item.id) or {}) == devset_module.gold_key(item))
        low, high = calibrate.stats.wilson(hits, len(subset_items))
        return {"n": len(subset_items), "correct": hits,
                "agreement": (hits / len(subset_items)) if subset_items else 0.0, "ci": [low, high]}

    report = {
        "schema": SCHEMA,
        "seconds": round(time.time() - started, 1),
        "target_seconds": round(target_seconds, 1),
        "primary": {"model": args.primary, "runtime": _runtime()},
        "target": {"model": args.target},
        "limit": args.limit,
        "threshold": args.threshold,
        "escalations": log.to_dict(),
        "agreement": {"before": before, "after": after,
                      "delta": after["agreement"] - before["agreement"]},
        "split": {name: {"before": split_view(ids, answers), "after": split_view(ids, merged)}
                  for name, ids in split_ids.items()},
        "escalated": [
            {"id": decision.question, "reason": decision.reason,
             "correct_before": decision_of(answers.get(decision.question) or {})
                               == devset_module.gold_key(by_id[decision.question]),
             "correct_after": decision_of(merged.get(decision.question) or {})
                              == devset_module.gold_key(by_id[decision.question])}
            for decision in decisions
        ],
    }
    _write(report, args.out)
    return 0


def decision_of(answer: dict[str, Any]) -> str:
    """The shipped decision rule (`routing.decision_of`) — one rule for the tool and the CLI."""
    return routing.decision_of(answer)


def _write_subset(subset: dict[str, Any], workdir: str) -> str:
    path = pathlib.Path(workdir) / "e2p5_escalation_subset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in subset.values():
            handle.write(json.dumps(item.to_json()) + "\n")
    return str(path)


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    work = sub.add_parser("calibrate", help="fit and store a calibration on a real model")
    work.add_argument("--model", required=True)
    work.add_argument("--threads", type=int, default=2)
    work.add_argument("--out")
    work.add_argument("--from-report", help="fit from a stored `--suite calibration` report")
    work.add_argument("--devset", help="a dev-set JSONL to measure instead of the committed one")
    work.add_argument("--items", type=int, help="measure only the first N dev items")
    work.add_argument("--rows-out", help="write the measured rows (the raw, refittable artifact)")
    work.add_argument("--dry-run", action="store_true")
    work.add_argument("--repeat", type=int, default=1,
                      help="run the whole fit again to show the params hash is identical")
    work.add_argument("--allow-reject", action="store_true",
                      help="exit 0 even when the gate rejected the fit (nothing stored)")
    work.set_defaults(func=cmd_calibrate)

    route = sub.add_parser("route", help="one real `--route auto` decision")
    route.add_argument("--model", help="alias|path (omit to let the registry decide)")
    route.add_argument("--threads", type=int, default=2)
    route.add_argument("--workdir", default="/tmp/e2p5")
    route.add_argument("--audit", help="write the decision record into this directory too")
    route.add_argument("--escalate-model", help="also run the escalation path on this model")
    route.add_argument("--max-escalations", type=int, default=1)
    route.add_argument("--register", action="append",
                       help="alias=path to register in the store before routing (repeatable)")
    route.add_argument("--current", help="the registry's current alias")
    route.add_argument("--out")
    route.set_defaults(func=cmd_route)

    escalate = sub.add_parser("escalate", help="measure the escalation delta on the dev set")
    escalate.add_argument("--primary", required=True)
    escalate.add_argument("--target", required=True)
    escalate.add_argument("--threads", type=int, default=2)
    escalate.add_argument("--limit", type=int, default=1)
    escalate.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    escalate.add_argument("--items", type=int)
    escalate.add_argument("--devset")
    escalate.add_argument("--workdir", default="/tmp/e2p5")
    escalate.add_argument("--out")
    escalate.set_defaults(func=cmd_escalate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
