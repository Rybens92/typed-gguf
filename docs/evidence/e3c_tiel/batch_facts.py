#!/usr/bin/env python3
"""E3c-Tiel deliverable 4 facts: the raw facts of the 20-question batch response.

Prints, from `.e3c_tiel/batch_response.json` only (nothing re-measured):
  * answers keyed c01..c20 with a non-empty check and a first-token peek,
  * usage (questions / waves / forks / decode_steps / tokens),
  * timings (model_load_ms / prefill_ms / questions_ms / total_ms),
  * engine facts (runtime, backend, effective_backend, n_gpu_layers, n_ctx,
    n_seq_max, placement, escalations, kv_type, state_id, prefill_reused),
  * warnings and calibration flags.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
RESPONSE = ROOT / ".e3c_tiel" / "batch_response.json"


def main() -> int:
    payload = json.loads(RESPONSE.read_text(encoding="utf-8"))
    answers = payload["answers"]
    ids = sorted(answers)
    print(f"answers: {len(answers)} ids {ids[0]}..{ids[-1]}")
    empty = [key for key, value in answers.items()
             if not (value.get("answer") if isinstance(value, dict) else value)]
    print(f"empty answers: {empty or 'none'}")
    print("\n-- first two answers verbatim")
    for key in ids[:2]:
        print(f"{key}: {json.dumps(answers[key], ensure_ascii=False)[:600]}")
    print("\n-- answer lengths")
    lengths = []
    for key in ids:
        value = answers[key]
        text = value.get("answer") if isinstance(value, dict) else value
        lengths.append(len(str(text)))
    print(f"min {min(lengths)} · median {sorted(lengths)[len(lengths) // 2]} · max {max(lengths)}")

    print("\n-- usage: " + json.dumps(payload["usage"], ensure_ascii=False))
    print("-- timings: " + json.dumps(payload["timings"], ensure_ascii=False))
    engine = payload["engine"]
    keys = ["runtime", "backend", "backend_source", "effective_backend", "kv_type",
            "n_gpu_layers", "n_ctx", "n_seq_max", "prefix_tokens", "state_id",
            "prefill_reused", "template"]
    for key in keys:
        if key in engine:
            print(f"-- engine.{key}: {json.dumps(engine[key], ensure_ascii=False)[:300]}")
    for key in ("placement", "fit", "escalations", "devices", "device_buffers"):
        if key in engine:
            print(f"-- engine.{key}: {json.dumps(engine[key], ensure_ascii=False)[:700]}")
    print("-- warnings: " + json.dumps(payload["warnings"], ensure_ascii=False)[:600])
    print(f"-- calibrated: {payload['calibrated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
