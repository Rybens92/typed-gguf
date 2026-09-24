"""Summarise a native response JSON (quickstart receipts)."""
from __future__ import annotations

import json
import sys


def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    engine = body["engine"]
    print("model           ", body["model"])
    print("engine.runtime  ", engine["runtime"], "backend", engine["backend"],
          "source", engine.get("backend_source"))
    print("engine.cue      ", engine["cue"])
    print("chat_format     ", engine["chat_format"])
    print("template        ", engine["template"])
    print("state_id        ", engine["state_id"], "prefill_reused", engine["prefill_reused"],
          "prefix_tokens", engine["prefix_tokens"])
    print("n_ctx/n_seq_max ", engine["n_ctx"], engine["n_seq_max"], "kv_type", engine["kv_type"],
          "n_gpu_layers", engine["n_gpu_layers"])
    print("usage           ", body["usage"])
    print("timings         ", body["timings"])
    print("warnings        ", body["warnings"])
    for key, answer in body["answers"].items():
        row = {name: answer[name] for name in
               ("type", "choice", "score", "noul", "confidence", "coverage", "reliability")
               if name in answer}
        print(f"answer {key:<10}", row, "cue:", answer.get("cue"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
