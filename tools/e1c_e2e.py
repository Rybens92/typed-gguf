#!/usr/bin/env python3
"""A-E1c-8: the four documented example question sets, end to end, on this box.

Each set is a realistic *decision* request (a state + typed questions whose labels are the
options a human would pick), run through the same path the CLI uses — fit plan -> template
resolution -> one prefill -> fork readout — and recorded with the full distributions, the
confidence, the coverage, the warnings, the engine surface and the timings.

Output: `docs/evidence/e1c_e2e.json` (machine-readable) + a printed table.

Usage::

    TYPED_GGUF_RUNTIME_DIR=<bundle> uv run python tools/e1c_e2e.py [--model PATH] [--out FILE]
                                                              [--threads N] [--no-fit]
                                                              [--thinking]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf import cli  # noqa: E402

SCHEMA = "typed_gguf.evidence.e1c-e2e/v1"
DEFAULT_MODEL = pathlib.Path.home() / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"

# --------------------------------------------------------------------------- the sets
# Every set is documented here (state + questions verbatim) so a reader can re-run it with the
# printed `typed-gguf run` command; `docs/evidence/e1c_t_*.md` carries the recorded answers.
QUESTION_SETS: tuple[dict, ...] = (
    {
        "id": "incident-triage",
        "doc": "Incident triage: which team owns it, how severe, should on-call be paged.",
        "request": {
            "state": "Payment flow degradation. At 09:12 the checkout page began returning HTTP "
                     "500 for every customer; the error rate is 100% on /checkout, 0% elsewhere. "
                     "No deploy happened in the last 24h. The on-call engineer is already awake.",
            "questions": {
                "area": {"type": "choice", "instructions": "Which team owns this incident?",
                         "criteria": {"billing": "payments, invoices, refunds",
                                      "technical": "api, infrastructure, deployment",
                                      "sales": "contracts and pricing",
                                      "support": "customer conversations and tickets"}},
                "severity": {"type": "score", "instructions": "How severe is it?",
                             "criteria": ["cosmetic", "annoying", "degrading",
                                          "critical", "catastrophic"]},
                "page": {"type": "noul",
                         "instructions": "Should we page the on-call engineer right now?",
                         "criteria": {"true": "yes, page immediately",
                                      "false": "no, handle it in business hours"}},
            },
        },
    },
    {
        "id": "support-routing",
        "doc": "Support routing: queue, urgency and a yes/no on the refund request.",
        "request": {
            "state": "A customer on the Business plan writes: \"I was charged twice for the same "
                     "seat this month. I already asked last week and got no answer. I need the "
                     "refund before the end of the quarter or I have to escalate to my CFO.\" "
                     "Their account shows two charges 3 days apart.",
            "questions": {
                "queue": {"type": "choice", "instructions": "Which queue should own this ticket?",
                          "criteria": {"billing": "duplicate charges and refunds",
                                       "technical": "product bugs",
                                       "account": "plan changes and seat management",
                                       "escalation": "formal complaints and account reviews"}},
                "urgency": {"type": "score", "instructions": "How urgent is it?",
                            "criteria": ["whenever", "this week", "this quarter",
                                         "this week with a deadline"]},
                "refund": {"type": "noul",
                           "instructions": "Should we refund the duplicate charge now?",
                           "criteria": {"true": "yes, refund immediately",
                                        "false": "no, ask for more information first"}},
            },
        },
    },
    {
        "id": "release-readiness",
        "doc": "Release readiness: ship or hold, how confident, and the biggest risk area.",
        "request": {
            "state": "Release 4.2 candidate is built. CI is green except a flaky integration "
                     "test that failed twice today and passed on retry. Two of the last five "
                     "releases carried a data-migration bug. The changelog lists one schema "
                     "migration and no public API change. Freeze window starts in 48 hours.",
            "questions": {
                "decision": {"type": "choice", "instructions": "What should we do with 4.2?",
                             "criteria": {"ship": "ship as planned",
                                          "hold": "hold until the flaky test is fixed",
                                          "ship-migration": "ship without the migration"}},
                "confidence": {"type": "score",
                               "instructions": "How much do we trust the release candidate?",
                               "criteria": ["no trust", "low", "medium", "high"]},
            },
        },
    },
    {
        "id": "risk-assessment",
        "doc": "Risk assessment: the dominant risk, the residual exposure and a go/no-go.",
        "request": {
            "state": "Proposal: move the nightly batch from the self-hosted cluster to a managed "
                     "queue service. The service SLA is 99.95%, priced per message, and its "
                     "regional availability excludes one region we serve. The team has run the "
                     "self-hosted cluster for four years and has one expert on it. Migration "
                     "would take six engineer-weeks.",
            "questions": {
                "risk": {"type": "choice", "instructions": "Which risk dominates this proposal?",
                         "criteria": {"cost": "unit economics at peak volume",
                                      "availability": "regional coverage and SLA",
                                      "skills": "losing in-house operational knowledge",
                                      "migration": "the migration effort itself"}},
                "exposure": {"type": "score",
                             "instructions": "How exposed are we if we do nothing?",
                             "criteria": ["negligible", "manageable", "material", "existential"]},
                "proceed": {"type": "noul", "instructions": "Should we start the migration now?",
                            "criteria": {"true": "yes, start now",
                                         "false": "no, keep the current cluster"}},
            },
        },
    },
)


def _model_path(explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit).expanduser()
    return DEFAULT_MODEL


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "e1c_e2e.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args(argv)

    model = _model_path(args.model)
    if not model.exists():
        print(f"error: {model} is not on this box", file=sys.stderr)
        return 2

    runs = []
    for entry in QUESTION_SETS:
        payload = json.loads(json.dumps(entry["request"]))
        payload["model"] = str(model)
        payload["options"] = {"threads": args.threads, "thinking": args.thinking}
        started = time.perf_counter()
        response = cli.decide_payload(payload, fit_enabled=not args.no_fit)
        wall_ms = (time.perf_counter() - started) * 1000.0
        answers = response["answers"]
        print(f"\n== {entry['id']}  ({wall_ms:.0f} ms wall)")
        for qid, answer in answers.items():
            if answer["type"] == "choice":
                best = max(answer["probabilities"], key=answer["probabilities"].get)
                print(f"   {qid:<10} choice={answer['choice']:<14} "
                      f"p={answer['probabilities'][best]:.3f} conf={answer['confidence']:.3f} "
                      f"coverage={answer['coverage']:.3f} reliability={answer['reliability']}")
            elif answer["type"] == "score":
                print(f"   {qid:<10} score={answer['score']:.3f} conf={answer['confidence']:.3f} "
                      f"coverage={answer['coverage']:.3f} p={answer['probabilities']}")
            else:
                print(f"   {qid:<10} noul={answer['noul']:.3f} coverage={answer['coverage']:.3f} "
                      f"p={answer['probabilities']}")
        runs.append({
            "id": entry["id"],
            "doc": entry["doc"],
            "request": payload,
            "answers": answers,
            "usage": response["usage"],
            "timings": response["timings"],
            "warnings": response["warnings"],
            "engine": {key: response["engine"][key] for key in
                       ("runtime", "backend", "n_ctx", "n_seq_max", "prefix_tokens",
                        "prefill_reused", "kv_type", "n_gpu_layers", "template")},
            "wall_ms": wall_ms,
        })

    document = {
        "schema": SCHEMA,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model),
        "model_sha256": cli.fit_plan_for(str(model), use_cache=True, kv_type="auto").model_sha256,
        "fit_enabled": not args.no_fit,
        "thinking": args.thinking,
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "runs": runs,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    print(json.dumps({"runs": len(runs),
                      "first_run_timings": runs[0]["timings"],
                      "template": runs[0]["engine"]["template"],
                      "kv_type": runs[0]["engine"]["kv_type"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
