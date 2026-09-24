"""Emit the README's quickstart JSON block from a real `ask` response.

The block is trimmed (fields a reader needs, `…` where lines are dropped) but every number,
string and key in it comes from the response file — nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys


def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    engine = body["engine"]
    out: list[str] = []
    out.append("{")
    out.append(f'  "model": "{body["model"]}",')
    out.append('  "engine": {')
    out.append(f'    "runtime": "{engine["runtime"]}",')
    out.append(f'    "backend": "{engine["backend"]}", "backend_source": "{engine["backend_source"]}",')
    out.append(f'    "devices": {json.dumps(engine["devices"])},')
    out.append(f'    "effective_backend": "{engine["effective_backend"]}",')
    out.append(f'    "readout": "{engine["readout"]}",')
    out.append(f'    "cue": "{engine["cue"]}",')
    chat = dict(engine["chat_format"])
    out.append('    "chat_format": {"kind": "%s", "question_turn": "%s", "contract": "%s",'
               % (chat.pop("kind"), chat.pop("question_turn"), chat.pop("contract")))
    out.append(f'                    "prefix_chars": {chat.pop("prefix_chars")},'
               f' "dropped": {json.dumps(chat.pop("dropped"))}}},')
    template = engine["template"]
    out.append('    "template": {"kind": "%s", "renderer": "%s", "source": "%s", "family": "%s",'
               % (template["kind"], template["renderer"], template["source"], template["family"]))
    out.append(f'                 "thinking": "{template["thinking"]}"}},')
    out.append(f'    "n_ctx": {engine["n_ctx"]}, "n_seq_max": {engine["n_seq_max"]},'
               f' "kv_type": "{engine["kv_type"]}", "n_gpu_layers": {engine["n_gpu_layers"]},')
    state_id = engine["state_id"]
    out.append(f'    "prefix_tokens": {engine["prefix_tokens"]},'
               f' "state_id": "{state_id[:11]}…", "prefill_reused": {json.dumps(engine["prefill_reused"])}')
    out.append('  },')
    out.append('  "answers": {')
    keys = list(body["answers"])
    for index, key in enumerate(keys):
        answer = body["answers"][key]
        row = {name: answer[name] for name in
               ("choice", "score", "noul", "probabilities", "confidence", "coverage",
                "reliability", "legend") if name in answer}
        out.append(f'    "{key}": {{')
        out.append(f'      "type": "{answer["type"]}",')
        out.append("      " + json.dumps(row)[1:-1] + ",")
        out.append('      "cue": ' + json.dumps(answer["cue"]) + ",")
        out.append(f'      "decode_steps": {answer["decode_steps"]}')
        out.append("    }" + ("," if index < len(keys) - 1 else ""))
    out.append('  },')
    out.append(f'  "usage": {json.dumps(body["usage"])},')
    out.append(f'  "timings": {json.dumps(body["timings"])},')
    out.append(f'  "warnings": {json.dumps(body["warnings"])}')
    out.append("}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
