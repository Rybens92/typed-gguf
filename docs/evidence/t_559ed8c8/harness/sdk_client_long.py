"""The gate's own SDK client, with an explicit HTTP timeout.

`tools/host_gate_serve_client.py` is the card's client and it is unchanged *except* that the
decision here is slow: the in-container engine runs the real 0.8B through the Vulkan build
backed by the software rasteriser (lavapipe), and one mixed 3-question body takes ~30 s
on the wire. The SDK's *default* HTTP timeout is 10 s, so the gate client gives up on a body the
server does answer (see logs/serve_story.log: three 200s, `served_by=host`, 32.1/28.6/28.5 s).

This driver imports the gate tool itself — the same `STATE`, the same `build_questions()` (built
with the SDK's own public types) and the same `verify()` — and only passes
`TypeSafeClient(timeout=..., retry=RetryPolicy(max_retries=0))`, so the checks are the gate's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path("/workspace/ggufone")
PINNED_SDK = "0.7.1"


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "gate_client", REPO / "tools" / "host_gate_serve_client.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gate_client"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.environ.get("TYPESAFE_BASE_URL", ""))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default=None)
    opts = parser.parse_args()

    import typesafe_sdk
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    installed = getattr(typesafe_sdk, "__version__", "<none>")
    if installed != PINNED_SDK:
        print(f"REFUSING TO RUN: the gate measures typesafe-sdk {PINNED_SDK}; this venv has "
              f"{installed}", file=sys.stderr)
        return 3
    if not opts.base_url or not os.environ.get("TYPESAFE_API_KEY", "").strip():
        print("REFUSING TO RUN: need --base-url/TYPESAFE_BASE_URL and TYPESAFE_API_KEY",
              file=sys.stderr)
        return 3

    gate = load_gate()
    questions = gate.build_questions()
    print(f"typesafe-sdk {installed} -> {opts.base_url} (timeout={opts.timeout}s, no retries)")
    print(f"state: {len(gate.STATE)} chars; questions: "
          + ", ".join(f"{qid}({kind})" for qid, kind in gate.GATE_QUESTIONS.items()))
    with TypeSafeClient(base_url=opts.base_url, timeout=opts.timeout,
                        retry=RetryPolicy(max_retries=0)) as client:
        for attempt in ("cold", "warm"):
            started = time.monotonic()
            result = client.system_one(gate.STATE, questions)
            elapsed = time.monotonic() - started
            payload = result.model_dump()
            print(f"\n== [{attempt}] client.system_one returned in {elapsed:.2f}s "
                  f"(server-side: see logs/serve_long.log)")
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            problems = gate.verify(payload)
            if problems:
                print(f"\nMISMATCH after the {attempt} call:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                return 2
            print(f"   ok: {len(payload['answers'])} typed answers, keys and ranges conform "
                  f"(the gate's own verify())")
            if opts.out:
                pathlib.Path(f"{opts.out}.{attempt}.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8")
    print("\nPASS: both calls were answered through the official SDK, typed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
