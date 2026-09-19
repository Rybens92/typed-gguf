#!/usr/bin/env python3
"""Card t_635124bf: verdicts for *recorded* cue rows — the fix's rule applied to a committed run.

A `ggufone run` response already records, per answer, the cue row's argmax (`cue.token`) and its
full-vocabulary mass (`cue.mass`) — that is the whole row the verdict reads. The fix changes the
*verdict*, not the row, so a committed pre-fix response can be re-verdict-ed exactly:

    refused  ==  cue.mass >= floor  and  cue.token in closer_map(session)

`closer_map` comes from the fixed `engine/cue.py` over the model's real vocabulary (the names and
the class are the engine's own, not this script's), and the floor is `cue.REFUSAL_FLOOR` — the
engine's coverage floor. Nothing here reconstructs a logit row or guesses a result.

    python3 derive_verdicts.py --response <run.json> --model <path.gguf> [--runtime <bundle>]
                               [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

HIDDEN = "/nonexistent/no-vulkan-icd.json"
os.environ.setdefault("VK_DRIVER_FILES", HIDDEN)
os.environ.setdefault("VK_ICD_FILENAMES", HIDDEN)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone.engine import cue as cue_module  # noqa: E402
from ggufone.runtime import ctypes_binding  # noqa: E402


class RecordedVocab:
    """Just enough of a session for `cue.closer_map`: the vocabulary's own two tables."""

    def __init__(self, model: str, runtime_dir: str) -> None:
        self.runtime = ctypes_binding.load_libraries(runtime_dir)
        llama = self.runtime.llama
        params = llama.llama_model_default_params()
        params.n_gpu_layers = 0
        params.vocab_only = True
        self.handle = llama.llama_model_load_from_file(str(model).encode(), params)
        self.vocab = llama.llama_model_get_vocab(self.handle)
        self.n_vocab = int(llama.llama_vocab_n_tokens(self.vocab))

    def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
        return ctypes_binding.tokenize(self.runtime, self.vocab, text, add_special=add_special)

    def special_tokens(self) -> list[tuple[int, str]]:
        return ctypes_binding.special_tokens(self.runtime, self.vocab, self.n_vocab)

    def close(self) -> None:
        self.runtime.llama.llama_model_free(self.handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--response", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime",
                        default="/var/home/rybens/.local/share/ggufone/runtime/"
                                "b11026-linux-x64-vulkan")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    payload = json.loads(pathlib.Path(args.response).read_text(encoding="utf-8"))
    recorded = payload["answers"]
    vocab = RecordedVocab(args.model, args.runtime)
    try:
        catalogue = cue_module.single_token_closers(vocab.tokenize)
        specials = dict(vocab.special_tokens())
        mapping = cue_module.closer_map(vocab)
        floor = cue_module.REFUSAL_FLOOR
        rows: list[dict[str, Any]] = []
        for key in sorted(recorded):
            answer = recorded[key]
            before = answer.get("cue") or {}
            token, mass = int(before["token"]), float(before["mass"])
            refused_after = mass >= floor and token in mapping
            rows.append({
                "id": key, "type": answer.get("type"),
                "coverage": answer.get("coverage"), "reliability": answer.get("reliability"),
                "answer": answer.get("choice") or answer.get("score") or answer.get("noul"),
                "token": token, "mass": mass,
                "catalogue_before": catalogue.get(token),
                "special_after": specials.get(token),
                "refused_before": bool(before.get("refused")),
                "closer_before": before.get("closer"),
                "dominating": bool(mass >= floor),
                "refused_after": bool(refused_after),
                "closer_after": mapping.get(token) if refused_after else None,
            })
    finally:
        vocab.close()
    newly = [r for r in rows if not r["refused_before"] and r["refused_after"]]
    print(f"== {args.label or args.response}: {len(rows)} recorded rows · vocabulary "
          f"{vocab.n_vocab} ids · catalogue {len(catalogue)} ids · specials {len(specials)} · "
          f"floor {floor}")
    print(f"refused before {sum(r['refused_before'] for r in rows)}/{len(rows)} → after "
          f"{sum(r['refused_after'] for r in rows)}/{len(rows)} · newly flagged {len(newly)} "
          f"{[r['id'] for r in newly]}")
    print(f"rows clearing the floor {sum(r['dominating'] for r in rows)}/{len(rows)} · "
          f"already-refused rows whose verdict changed "
          f"{sum(1 for r in rows if r['refused_before'] and not r['refused_after'])}")
    print("| id | token | catalogue before | vocabulary special | mass | dominating "
          "| refused before → after | closer after |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['token']} | {r['catalogue_before']} | {r['special_after']} "
              f"| {r['mass']:.4f} | {r['dominating']} | {r['refused_before']} → "
              f"{r['refused_after']} | {r['closer_after']} |")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "response": args.response, "model": args.model, "label": args.label,
            "n_vocab": vocab.n_vocab, "n_catalogue": len(catalogue),
            "catalogue": {str(k): v for k, v in catalogue.items()},
            "n_special": len(specials), "floor": floor,
            "refused_before": sum(r["refused_before"] for r in rows),
            "refused_after": sum(r["refused_after"] for r in rows),
            "newly_flagged": [r["id"] for r in newly],
            "unchanged_by_the_fix": [r["id"] for r in rows
                                     if r["refused_before"] != r["refused_after"]][len(newly):],
            "rows": rows,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
