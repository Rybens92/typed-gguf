"""The TypeSafe client half of `tools/host_gate_serve.sh` (card `t_f5d8b6c7`, acceptance 6).

This file runs **inside the throwaway venv the gate builds**, against the pinned official SDK
(`typesafe-sdk==0.7.1`) and a real `typed-gguf serve` on this host:

    <log-dir>/venv/bin/python tools/host_gate_serve_client.py --base-url http://127.0.0.1:8088

It is the drop-in claim, measured: the client is told nothing but `TYPESAFE_BASE_URL` (and a
non-empty `TYPESAFE_API_KEY`, which the SDK insists on and our server never reads), asks one
mixed `choice`/`score`/`noul` question set about a real state, and the typed answers come back.

Exit codes: `0` when every check in `verify` passes, `2` on the first failing check, `3` when the
installed SDK is not the pinned one — a *different* SDK answers a different wire, so a mismatch
there must be loud, never a silently-passing run.

`verify` is a pure function over the SDK's own `model_dump()`, so the gate's judgements are
drivable offline (`tests/test_serve.py` does exactly that) instead of only on the host.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

PINNED_SDK = "0.7.1"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
API_KEY_ENV = "TYPESAFE_API_KEY"

#: the SDK's own wire for a response: these three keys, nothing else (SPEC 2.9 [sdk-0.7.1])
SERVICE_KEYS = frozenset({"model", "answers", "usage"})
USAGE_KEYS = frozenset({"input_tokens", "output_tokens"})
#: per-answer key sets, by the value the `type` discriminator carries. `noul` is the SDK's own
#: `NoulAnswer` ({type, noul}): the wire also carries `probabilities` (SPEC 2.9) and the SDK drops
#: it — its models ignore unknown keys — so `model_dump()` never shows it. `verify` tolerates it.
ANSWER_KEYS = {
    "noul": frozenset({"type", "noul"}),
    "choice": frozenset({"type", "choice", "confidence", "probabilities"}),
    "score": frozenset({"type", "score", "confidence", "legend", "probabilities"}),
}
#: keys the wire may carry that the SDK's typed view does not model
UNMODELED_KEYS = {"noul": frozenset({"probabilities"})}
#: the wire rounds to 6 significant digits (SPEC 2.5), so a K-level distribution sums to
#: 1 ± K·5e-7 — generous on purpose: this is not a renormalization test
PROBABILITY_SUM_TOLERANCE = 1e-5

#: the three questions this gate asks, one of each type
GATE_QUESTIONS = {"renew_style": "choice", "risk": "score", "escalate": "noul"}

#: the state the gate sends. Real prose about a real decision — long enough that the readout is
#: doing the work a user's state would ask of it, short enough to stay under the 4B's context.
STATE = (
    "Customer: Brightline Utilities (B2B, 3 years, EUR 84k ARR, 41 seats).\n"
    "Renewal due 2026-11-30. Health: weekly active seats down 41% -> 26% over two quarters;\n"
    "two of the three admin accounts have not logged in since June. Support: 6 tickets in 90\n"
    "days, 4 of them about the CSV export the October release broke, all still open. Champion\n"
    "(K. Alvarez, Head of Ops) left in July; the new buyer is the CFO, who has never seen the\n"
    "product and asked for a security review before any renewal talk. Billing: last invoice\n"
    "disputed for 11 days over seat counts, then paid late. No expansion signals on the account.\n"
    "The vendor's CSM changed twice this year; the current one has 9 other accounts at the same\n"
    "stage and no QBR has happened since Q1."
)


def expected_answer_keys(kind: str) -> frozenset[str]:
    return ANSWER_KEYS[kind]


def verify(payload: dict[str, Any], *,
           questions: dict[str, str] | None = None) -> list[str]:
    """Every problem with a served body, in the order they are found. Empty list = it conforms.

    `questions` maps a question id to the answer type the gate expects (`choice`/`score`/`noul`);
    it defaults to the three this file asks.
    """
    expected = questions or dict(GATE_QUESTIONS)
    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"the body is not a JSON object: {type(payload).__name__}"]
    if set(payload) != SERVICE_KEYS:
        problems.append(f"top-level keys are {sorted(payload)}, not {sorted(SERVICE_KEYS)}")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        problems.append(f"`model` must echo the resolved name, got {model!r}")
    usage = payload.get("usage")
    if not isinstance(usage, dict) or set(usage) != USAGE_KEYS:
        problems.append(f"usage keys are {None if not isinstance(usage, dict) else sorted(usage)}, "
                        f"not {sorted(USAGE_KEYS)}")
    else:
        for key, value in usage.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(f"usage.{key} must be a non-negative integer, got {value!r}")
        if not usage.get("input_tokens"):
            problems.append("usage.input_tokens is 0: a served request read a real prompt")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return [*problems, f"`answers` must be an object, got {type(answers).__name__}"]
    if set(answers) != set(expected):
        problems.append(f"answers are {sorted(answers)}, not the questions asked "
                        f"({sorted(expected)})")
    for qid, kind in expected.items():
        answer = answers.get(qid)
        if not isinstance(answer, dict):
            problems.append(f"answers.{qid} is missing or not an object")
            continue
        problems.extend(_verify_answer(qid, kind, answer))
    return problems


def _verify_answer(qid: str, kind: str, answer: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    keys = set(answer)
    missing = expected_answer_keys(kind) - keys
    unexpected = keys - expected_answer_keys(kind) - UNMODELED_KEYS.get(kind, frozenset())
    if missing:
        problems.append(f"answers.{qid} ({kind}) is missing {sorted(missing)}")
    if unexpected:
        problems.append(f"answers.{qid} ({kind}) carries unexpected keys {sorted(unexpected)}")
    if answer.get("type") != kind:
        problems.append(f"answers.{qid} says type={answer.get('type')!r}, the question was {kind}")
    probabilities = answer.get("probabilities")
    if kind == "noul" and probabilities is None:
        # the SDK's NoulAnswer does not model `probabilities`; what it must model is the yes-rate
        value = answer.get("noul")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"answers.{qid}.noul is {value!r}, not a number")
        elif not 0.0 <= float(value) <= 1.0:
            problems.append(f"answers.{qid}.noul = {value} is outside [0, 1]")
        return problems
    if not isinstance(probabilities, dict) or not probabilities:
        problems.append(f"answers.{qid} carries no probability distribution")
        return problems
    for label, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"answers.{qid}.probabilities[{label!r}] is {value!r}, not a number")
        elif not 0.0 <= float(value) <= 1.0:
            problems.append(f"answers.{qid}.probabilities[{label!r}] = {value} is outside [0, 1]")
    total = sum(float(v) for v in probabilities.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool))
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        problems.append(f"answers.{qid}.probabilities sum to {total!r}, not 1.0")
    if kind == "noul":
        value = answer.get("noul")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"answers.{qid}.noul is {value!r}, not a number")
        elif not 0.0 <= float(value) <= 1.0:
            problems.append(f"answers.{qid}.noul = {value} is outside [0, 1]")
        if "confidence" in keys:
            problems.append(f"answers.{qid}: noul carries no confidence (keys={sorted(keys)})")
        if set(probabilities) != {"yes", "no"}:
            problems.append(f"answers.{qid}: noul probabilities are {sorted(probabilities)}, "
                            "not ['no', 'yes']")
    if kind == "choice":
        choice = answer.get("choice")
        if not isinstance(choice, str) or not choice:
            problems.append(f"answers.{qid}.choice is {choice!r}, not a criterion name")
        else:
            if choice not in probabilities:
                problems.append(f"answers.{qid}.choice = {choice!r} is not one of "
                                f"{sorted(probabilities)}")
            # SPEC 2.5: `choice == argmax(probabilities)` (ties keep the deterministic tie-break)
            best = max(float(v) for v in probabilities.values()
                       if isinstance(v, (int, float)) and not isinstance(v, bool))
            winners = {label for label, value in probabilities.items() if float(value) == best}
            if choice not in winners:
                problems.append(f"answers.{qid}.choice = {choice!r} is not the argmax "
                                f"{sorted(winners)}")
        problems.extend(_verify_confidence(qid, answer))
    if kind == "score":
        score = answer.get("score")
        legend = answer.get("legend")
        if not isinstance(legend, dict) or not legend:
            problems.append(f"answers.{qid}.legend is {legend!r}, not a level->description map")
        else:
            # the wire sends level numbers as *string* keys (SPEC 2.9); pydantic coerces them to
            # the ints its `dict[int, str]` model declares, so both spellings have to pass here
            if {str(key) for key in legend} != {str(i) for i in range(len(legend))}:
                problems.append(f"answers.{qid}.legend keys are {sorted(legend)}, not level "
                                "numbers as strings")
            # SPEC 2.5: `score` is the probability-weighted mean of the level numbers, so it
            # lands anywhere in [0, K-1] — 1.30 for [0, .7, .3], never a rounded level
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                problems.append(f"answers.{qid}.score is {score!r}, not a number")
            elif not 0.0 <= float(score) <= len(legend) - 1:
                problems.append(f"answers.{qid}.score = {score} is outside [0, {len(legend) - 1}]")
        problems.extend(_verify_confidence(qid, answer))
    return problems


def _verify_confidence(qid: str, answer: dict[str, Any]) -> list[str]:
    value = answer.get("confidence")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"answers.{qid}.confidence is {value!r}, not a number"]
    if not 0.0 <= float(value) <= 1.0:
        return [f"answers.{qid}.confidence = {value} is outside [0, 1]"]
    return []


def build_questions() -> dict[str, Any]:
    """The three questions, built with the SDK's own public types (never a raw dict)."""
    from typesafe_sdk import Choice, Noul, Score

    return {
        "renew_style": Choice(
            instructions="Pick the renewal play this account gets.",
            criteria={
                "save_with_discount": "offer a discount to keep the logo at any cost",
                "exec_sponsor_plan": "rebuild with the new CFO before any commercial talk",
                "walk_away": "decline to renew and hand the account to collections",
            },
        ),
        "risk": Score(
            instructions="How bad is this renewal, on the vendor's own scale?",
            criteria=["healthy", "watch", "at risk", "lost cause"],
        ),
        "escalate": Noul(
            instructions="Should the vendor escalate to an exec-level conversation this week?",
            criteria={"true": "yes, escalate now", "false": "no, the CSM keeps it for now"},
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=None,
                        help=f"the served root (default: ${BASE_URL_ENV})")
    parser.add_argument("--out", default=None, help="write the served body here as JSON")
    parser.add_argument("--expect-sdk", default=PINNED_SDK)
    opts = parser.parse_args(argv)

    try:
        import typesafe_sdk
    except ModuleNotFoundError:
        print(f"REFUSING TO RUN: no typesafe-sdk in {sys.executable}. This gate measures the "
              f"pinned {opts.expect_sdk} wire; install `typesafe-sdk=={opts.expect_sdk}` first.",
              file=sys.stderr)
        return 3

    installed = getattr(typesafe_sdk, "__version__", "<none>")
    if installed != opts.expect_sdk:
        print(f"REFUSING TO RUN: this gate measures typesafe-sdk {opts.expect_sdk}; the venv has "
              f"{installed}. A different SDK answers a different wire — reinstall the pin before "
              f"trusting any result from this run.", file=sys.stderr)
        return 3
    base_url = opts.base_url or os.environ.get(BASE_URL_ENV, "").strip()
    if not base_url:
        print(f"REFUSING TO RUN: pass --base-url or set {BASE_URL_ENV}.", file=sys.stderr)
        return 3
    if not os.environ.get(API_KEY_ENV, "").strip():
        print(f"REFUSING TO RUN: {API_KEY_ENV} must be a non-empty string — the SDK demands one "
              "and this server never reads it (SPEC 2.9, S-17).", file=sys.stderr)
        return 3

    from typesafe_sdk import TypeSafeClient

    questions = build_questions()
    print(f"typesafe-sdk {installed} -> {base_url}")
    print(f"state: {len(STATE)} chars; questions: "
          + ", ".join(f"{qid}({kind})" for qid, kind in GATE_QUESTIONS.items()))
    with TypeSafeClient(base_url=base_url) as client:
        for attempt in ("cold", "warm"):
            started = time.monotonic()
            result = client.system_one(STATE, questions)
            elapsed = time.monotonic() - started
            payload = result.model_dump()
            print(f"\n== [{attempt}] client.system_one returned in {elapsed:.2f}s")
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            problems = verify(payload)
            if problems:
                print(f"\nMISMATCH after the {attempt} call:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                return 2
            print(f"   ok: {len(payload['answers'])} typed answers, keys and ranges conform")
            if opts.out and attempt == "warm":
                pathlib.Path(opts.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nPASS: the served wire answered both calls through the official SDK, typed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
