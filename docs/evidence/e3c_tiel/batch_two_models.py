#!/usr/bin/env python3
"""The same 20-item batch path on the two 35B-A3B models: Occamy (E3) vs Tiel (E3c).

Both files are the raw `--suite batch` responses for the *same* 20 questions
(`.e3c_tiel/batch_questions.json` == `docs/evidence/e3_batch_questions.json`). Prints the
per-item coverage/reliability/cue verdict side by side, plus usage/warnings/timings, so the
mass question can be stated as a path fact rather than a model guess.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
PAIRS = [("Occamy 1.0 (E3 batch)", ROOT / "docs" / "evidence" / "e3_batch.json"),
         ("Tiel-Coder (E3c batch)", ROOT / ".e3c_tiel" / "batch_response.json")]


def summarise(label: str, payload: dict) -> dict[str, dict]:
    answers = payload["answers"]
    tally: dict[str, int] = {}
    for value in answers.values():
        key = str(value.get("reliability"))
        tally[key] = tally.get(key, 0) + 1
    covs = sorted(float(value.get("coverage") or 0.0) for value in answers.values())
    refused = [k for k, v in answers.items() if (v.get("cue") or {}).get("refused")]
    print(f"\n== {label}")
    print(f"   model {payload.get('model')} · items {len(answers)}")
    print(f"   usage   {json.dumps(payload.get('usage'))}")
    print(f"   timings {json.dumps(payload.get('timings'))}")
    print(f"   warnings {json.dumps(payload.get('warnings'))}")
    print(f"   reliability {tally}")
    print(f"   coverage min {covs[0]:.3e} median {covs[len(covs) // 2]:.3e} max {covs[-1]:.3e}")
    print(f"   cue.refused {len(refused)} {refused}")
    token = {}
    for value in answers.values():
        cue = value.get("cue") or {}
        token[str(cue.get("token"))] = token.get(str(cue.get("token")), 0) + 1
    print(f"   cue tokens {token}")
    return answers


def main() -> int:
    both = {}
    for label, path in PAIRS:
        if not path.exists():
            print(f"missing: {path}")
            return 2
        both[label] = summarise(label, json.loads(path.read_text(encoding="utf-8")))
    first, second = [keys for keys in both.values()]
    print("\n== per item")
    print("id  | occamy cov/rel            | tiel cov/rel              | tiel cue token/verse")
    for key in sorted(first):
        one, two = first[key], second.get(key, {})
        cue = two.get("cue") or {}
        print(f"{key} | {float(one.get('coverage') or 0):.3e} / {one.get('reliability'):<9} | "
              f"{float(two.get('coverage') or 0):.3e} / {str(two.get('reliability')):<9} | "
              f"{cue.get('token')} mass {cue.get('mass')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
