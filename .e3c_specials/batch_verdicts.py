#!/usr/bin/env python3
"""Card t_635124bf: what the fix changed on the serving-shaped batch — before vs after.

Reads two `ggufone run` responses of the SAME shape (pre-fix tree, post-fix tree) and reports, per
item: the two cue verdicts, the coverage/reliability pair (which must be *identical* — the fix
touches the verdict, not the row), the warnings; then the aggregates the card asks for: refusal
verdicts before/after, the `measured` share before/after, and the rows that were silently low-mass
and now carry a reason.

With `--model` (and `--runtime`) it also names every cue token the two runs saw: the vocabulary's
own text, the GGUF's `token_type`, and whether the fixed classifier calls it special. The
vocabulary is read with `vocab_only=true` (metadata only — a 22 GB model costs 1.6 s, no tensors).

    python3 batch_verdicts.py --before <pre>/batch_response.json \\
        --after <post>/batch_response.json --model <path.gguf> --runtime <bundle> [--out <json>]
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

ROOT = pathlib.Path("/workspace/ggufone")
sys.path.insert(0, str(ROOT / "src"))

from ggufone.registry import gguf as gguf_module  # noqa: E402
from ggufone.runtime import ctypes_binding  # noqa: E402

TYPES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def row_of(response: dict, key: str) -> dict:
    value = response["answers"][key]
    return value if isinstance(value, dict) else {"answer": value}


def vocab_facts(model: str, runtime_dir: str, wanted: set[int]) -> dict[str, Any]:
    """`{token: {...}}` for the ids the two runs saw, plus the vocabulary's special count."""
    runtime = ctypes_binding.load_libraries(runtime_dir)
    llama = runtime.llama
    params = llama.llama_model_default_params()
    params.n_gpu_layers = 0
    params.vocab_only = True
    handle = llama.llama_model_load_from_file(str(model).encode(), params)
    vocab = llama.llama_model_get_vocab(handle)
    n_vocab = int(llama.llama_vocab_n_tokens(vocab))
    specials = dict(ctypes_binding.special_tokens(runtime, vocab, n_vocab))
    types = [int(value) for value in
             gguf_module.parse_gguf_metadata(model)["kv"]["tokenizer.ggml.token_type"]]
    facts = {}
    for token in sorted(wanted):
        attr = ctypes_binding.token_attr(runtime, vocab, token)
        facts[str(token)] = {
            "text": ctypes_binding.token_text(runtime, vocab, token),
            "piece": ctypes_binding.token_piece(runtime, vocab, token),
            "gguf_type": TYPES.get(int(types[token]), "?"),
            "special": bool(attr & ctypes_binding.SPECIAL_TOKEN_ATTRS),
            "special_text": specials.get(token),
        }
    llama.llama_model_free(handle)
    return {"n_vocab": n_vocab, "n_special": len(specials), "tokens": facts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--runtime",
                        default="/var/home/rybens/.local/share/ggufone/runtime/"
                                "b11026-linux-x64-vulkan")
    parser.add_argument("--label", default="serving-shaped batch")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    before, after = load(args.before), load(args.after)
    ids = sorted(before["answers"])
    assert ids == sorted(after["answers"]), "the two runs do not ask the same items"

    rows = []
    for key in ids:
        b, a = row_of(before, key), row_of(after, key)
        bc, ac = b.get("cue") or {}, a.get("cue") or {}
        rows.append({
            "id": key,
            "type": b.get("type"),
            "coverage_before": b.get("coverage"),
            "coverage_after": a.get("coverage"),
            "reliability_before": b.get("reliability"),
            "reliability_after": a.get("reliability"),
            "answer_before": b.get("choice") or b.get("score") or b.get("noul"),
            "answer_after": a.get("choice") or a.get("score") or a.get("noul"),
            "token": ac.get("token", bc.get("token")),
            "mass": ac.get("mass", bc.get("mass")),
            "closer_before": bc.get("closer"),
            "closer_after": ac.get("closer"),
            "refused_before": bool(bc.get("refused")),
            "refused_after": bool(ac.get("refused")),
            "hint_after": "hint" in ac,
        })
    same_coverage = all(r["coverage_before"] == r["coverage_after"] for r in rows)
    same_reliability = all(r["reliability_before"] == r["reliability_after"] for r in rows)
    same_answer = all(r["answer_before"] == r["answer_after"] for r in rows)

    def share(kind: str, side: str) -> str:
        values = [r[f"reliability_{side}"] for r in rows]
        return f"{sum(1 for value in values if value == kind)}/{len(values)}"

    flagged = [r for r in rows if not r["refused_before"] and r["refused_after"]]
    unflagged = [r for r in rows if r["refused_before"] and not r["refused_after"]]
    tokens: dict[str, int] = {}
    for r in rows:
        tokens[str(r["token"])] = tokens.get(str(r["token"]), 0) + 1

    print(f"== {args.label}: {len(rows)} items")
    print(f"coverage identical in the two runs: {same_coverage} · reliability identical: "
          f"{same_reliability} · answers identical: {same_answer}")
    print(f"refusals   before {sum(r['refused_before'] for r in rows)}/{len(rows)}"
          f" · after {sum(r['refused_after'] for r in rows)}/{len(rows)}")
    print(f"measured   before {share('measured', 'before')} · after {share('measured', 'after')}")
    print(f"low_mass   before {share('low_mass', 'before')} · after {share('low_mass', 'after')}")
    print(f"cue top token histogram: {json.dumps(tokens)}")
    print(f"silently low-mass rows now flagged: {len(flagged)} {[r['id'] for r in flagged]}")
    print(f"refusals the fix removed (must be none): {len(unflagged)} "
          f"{[r['id'] for r in unflagged]}")
    facts = None
    if args.model:
        facts = vocab_facts(args.model, args.runtime, {int(k) for k in tokens})
        print(f"vocabulary: {facts['n_vocab']} ids · specials {facts['n_special']}")
        print("cue tokens the runs saw:")
        print("| token | vocabulary text | piece | gguf type | special? | items |")
        print("|---|---|---|---|---|---|")
        for token, count in sorted(tokens.items(), key=lambda pair: -pair[1]):
            info = facts["tokens"][token]
            print(f"| {token} | `{info['text']}` | `{info['piece']}` | {info['gguf_type']} "
                  f"| {info['special']} | {count} |")
    print()
    print("| id | type | coverage (both) | reliability | cue token | closer before → after "
          "| mass | refused before → after |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['type']} | {r['coverage_before']:.3e} | {r['reliability_after']} "
              f"| {r['token']} | {r['closer_before']} → {r['closer_after']} "
              f"| {float(r['mass']):.4f} | {r['refused_before']} → {r['refused_after']} |")
    print()
    print(f"warnings before: {json.dumps(before.get('warnings'))}")
    print(f"warnings after:  {json.dumps(after.get('warnings'))}")
    print(f"usage before {json.dumps(before.get('usage'))}")
    print(f"usage after  {json.dumps(after.get('usage'))}")
    print(f"coverage before: min {min(r['coverage_before'] for r in rows):.3e} "
          f"max {max(r['coverage_before'] for r in rows):.3e}")
    print(f"coverage after:  min {min(r['coverage_after'] for r in rows):.3e} "
          f"max {max(r['coverage_after'] for r in rows):.3e}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "label": args.label,
            "items": len(rows),
            "coverage_identical": same_coverage, "reliability_identical": same_reliability,
            "answers_identical": same_answer,
            "refused_before": sum(r["refused_before"] for r in rows),
            "refused_after": sum(r["refused_after"] for r in rows),
            "measured_before": share("measured", "before"),
            "measured_after": share("measured", "after"),
            "now_flagged": [r["id"] for r in flagged],
            "refusals_removed": [r["id"] for r in unflagged],
            "cue_tokens": tokens, "vocabulary": facts,
            "warnings_before": before.get("warnings"), "warnings_after": after.get("warnings"),
            "wall_s_before": before.get("timings", {}).get("total_ms"),
            "wall_s_after": after.get("timings", {}).get("total_ms"),
            "rows": rows,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
