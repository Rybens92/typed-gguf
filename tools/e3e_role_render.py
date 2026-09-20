#!/usr/bin/env python3
"""E3e (card t_4c48f40a): the role split, per family — does the template render two user turns?

`--chat-format role_split` promises the model a conversation its own chat template defines: the
state turn, then the question as a **second user turn**. Whether a family's template can render
that is a fact about the template, and this tool measures it per family instead of assuming it.

Offline (default, no model load, no socket):

    python3 tools/e3e_role_render.py --models ~/.hermes/models --cue json_instructed \
        --json /work/.../e3e_role_render.json --report /work/.../e3e_role_render.md

For every `*.gguf` it reads `tokenizer.chat_template` out of the metadata, resolves it through the
E1c chain, and renders the two-user-turn conversation as the *prefix* (state) plus the *tail*
(question + generation prompt + the JSON opener). It records, for each family:

* `status` — `rendered`, `refused` (`E_ROLE_SPLIT_UNSUPPORTED`: the template merges, drops or
  reorders the turns, or leaves a thinking block open) or `not-renderable` (`E_TEMPLATE_UNRESOLVED`:
  the template is outside the E1c internal subset, e.g. Tiel's `{%- set x -%}` at line 172 — the
  live path resolves that one through the builtin bridge or `--template plain`, and this tool says
  so rather than substituting silently);
* the five acceptance checks the engine runs, plus the byte pins (`prefix + tail == the render +
  the opener`, the generation prompt's own bytes, the opener);
* the shared prefix and the tail, with head/tail excerpts so a human can read the actual bytes;
* the residual the state-only render cannot share (`dropped`) — Spark's template ends every render
  with a newline, and that newline belongs at the *end* of a prompt, not in the middle of one.

Live (`--live --model PATH`): the same shape through a loaded model's own template chain and
tokenizer (the bench's own `harness.LiveModel`, so the live check uses the instrument the arms
use), one real decision on one dev item — the token ids of the split, the response's
`engine.chat_format` block, and the answer with its warnings and value verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typed_gguf import schema  # noqa: E402
from typed_gguf.bench import devset as devset_module  # noqa: E402
from typed_gguf.bench import harness  # noqa: E402
from typed_gguf.engine import prompt  # noqa: E402
from typed_gguf.engine import template as template_module  # noqa: E402
from typed_gguf.errors import UserError  # noqa: E402

SCHEMA = "typed_gguf.e3e-role-render/v1"
#: the state/dev items the offline render uses: the first two dev rows, so the question text is the
#: one the bench asks (not a toy) and the two tails can be compared
OFFLINE_ITEMS = 2


def _excerpt(text: str, *, head: int = 60, tail: int = 60) -> str:
    """`head…tail` of a render for the report — never the whole prompt (the JSON carries it)."""
    if len(text) <= head + tail + 3:
        return text
    return f"{text[:head]}…{text[-tail:]}"


def offline_family(path: pathlib.Path, *, cue: str,
                   json_contract: str = schema.JSON_CONTRACT) -> dict:
    """One family's offline verdict: the acceptance checks, the pins and the bytes."""
    from typed_gguf.registry import gguf
    kv = gguf.parse_gguf_metadata(path)["kv"]
    arch = gguf.arch_of(kv)
    text = kv.get(template_module.TEMPLATE_KEY)
    row: dict = {
        "file": path.name,
        "arch": arch,
        "template_chars": len(text) if isinstance(text, str) else 0,
        "template_sha256": (hashlib.sha256(text.encode("utf-8")).hexdigest()
                            if isinstance(text, str) else None),
        "cue": cue,
        "json_contract": json_contract if cue == "json_instructed" else None,
    }
    if not isinstance(text, str) or not text:
        row.update({"status": "not-renderable", "family": None,
                    "reason": "the GGUF carries no tokenizer.chat_template"})
        return row
    items = list(devset_module.load())[:OFFLINE_ITEMS]
    messages = prompt.chat_messages(items[0].state,
                                    framing=prompt.framing_for(cue, json_contract))
    questions = [_item_question(item, cue=cue, contract=json_contract) for item in items]
    try:
        resolution = template_module.resolve(messages=messages, model_template=text, arch=arch)
    except template_module.TemplateUnresolvedError as exc:
        row.update({"status": "not-renderable", "family": template_module.detect_family(arch, text),
                    "reason": str(exc).splitlines()[0],
                    "fallback": "--template plain (role_split keeps the plain USER:/ASSISTANT: "
                                "framing), or a live run, where the builtin bridge may render it"})
        return row
    row["family"] = resolution.family
    row["resolution"] = resolution.to_dict()
    try:
        role = prompt.role_split_render(items[0].state, questions, resolution=resolution, cue=cue,
                                        contract=json_contract)
    except UserError as exc:
        row.update({"status": "refused", "code": exc.code, "reason": str(exc).splitlines()[0]})
        return row
    # the acceptance the engine runs, re-checked here on the whole two-question render
    block = prompt.question_block(questions[0], cue=cue, contract=json_contract)
    full = template_module.render_prompt(
        [*messages, {"role": "user", "content": block}], resolution,
        add_generation_prompt=True, enable_thinking=False)
    opener = prompt.json_opener(questions[0].type) if cue in prompt.JSON_CUES else ""
    tails = list(role.tails)
    checks = {
        "state_words_in_prefix":
            prompt.normalised(str(messages[-1]["content"])) in prompt.normalised(role.prefix),
        "question_words_in_its_own_tail":
            prompt.normalised(block) in prompt.normalised(tails[0]),
        "question_absent_from_prefix":
            prompt.normalised(block) not in prompt.normalised(role.prefix),
        "prefix_plus_tail_is_the_render_plus_opener":
            role.prefix + tails[0] == full + opener,
        "no_open_think": template_module.no_open_think(role.prefix + tails[0]),
    }
    row.update({
        "status": "rendered" if all(checks.values()) else "checks-failed",
        "checks": checks,
        "question_id": questions[0].id,
        "prefix_chars": len(role.prefix),
        "tail_chars": [len(tail) for tail in tails],
        "distinct_tails": len(set(tails)) == len(tails),
        "dropped": role.dropped,
        "opener": opener,
        "prefix_excerpt": _excerpt(role.prefix),
        "prefix_tail_excerpt": role.prefix[-80:],
        "tail_head_excerpt": tails[0][:80],
        "tail_tail_excerpt": tails[0][-80:],
        "prompt_chars": len(role.prefix) + len(tails[0]),
        "full_render_chars": len(full),
    })
    return row


def _item_question(item: devset_module.DevItem, *, cue: str,
                   contract: str = schema.JSON_CONTRACT) -> schema.Question:
    """The dev row's own question body as a `Question` — the offline render asks the real text."""
    payload = devset_module.request_for(item, model="offline", cue=cue,
                                        chat_format=schema.ROLE_SPLIT, json_contract=contract)
    return schema.parse_request(payload).questions[0]


def offline_record(args: argparse.Namespace) -> dict:
    models_dir = pathlib.Path(args.models).expanduser()
    paths = sorted(models_dir.glob("*.gguf"))
    if not paths:
        raise SystemExit(f"no .gguf files under {models_dir}")
    record: dict = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cue": args.cue,
        "chat_format": schema.ROLE_SPLIT,
        "models_dir": str(models_dir),
        "dev_items": OFFLINE_ITEMS,
        "families": [offline_family(path, cue=args.cue, json_contract=args.json_contract)
                     for path in paths],
    }
    record["counts"] = {status: sum(1 for row in record["families"] if row["status"] == status)
                        for status in ("rendered", "refused", "not-renderable", "checks-failed")}
    return record


def live_check(args: argparse.Namespace, record: dict) -> dict:
    """One real decision in the role-split shape, through the bench's own model wrapper."""
    import os
    if args.hide_devices:
        icd = os.environ.get("TYPED_GGUF_HIDDEN_ICD")
        if icd:
            os.environ["VK_DRIVER_FILES"] = icd
            os.environ["VK_ICD_FILENAMES"] = icd
    model_path = str(pathlib.Path(os.path.expanduser(args.model)))
    runtime = args.runtime or os.environ.get("TYPED_GGUF_RUNTIME_DIR")
    if not runtime:
        raise SystemExit("--live needs TYPED_GGUF_RUNTIME_DIR (or --runtime)")
    item = next(iter([row for row in devset_module.load()
                      if args.id is None or str(row.id) == args.id]), None)
    if item is None:
        raise SystemExit(f"no dev item {args.id!r}")
    spec = harness.ModelSpec(path=model_path, backend=args.backend, runtime_dir=str(runtime),
                             threads=args.threads, kv_type=args.kv_type,
                             n_gpu_layers=args.gpu_layers if args.gpu_layers is not None else -1)
    request = schema.parse_request(devset_module.request_for(
        item, model=model_path, cue=args.cue, chat_format=schema.ROLE_SPLIT,
        json_contract=args.json_contract))
    live: dict = {"item": str(item.id), "type": str(item.type), "model": harness.model_facts(
        model_path), "backend": args.backend, "gpu_layers": spec.n_gpu_layers,
        "json_contract": args.json_contract if args.cue == "json_instructed" else None}
    started = time.perf_counter()
    with harness.LiveModel(spec) as model:
        live["load_ms"] = round(model.load(), 1)
        live["placement"] = dict(model.placement)
        # the live render: the handle's own template chain + tokenizer (what the engine will decode)
        view = prompt.role_split_context(
            request, template_module.resolve_for_handle(
                model.handle, messages=prompt.chat_messages(
                    request.state,
                    framing=prompt.framing_for(request.options.cue,
                                               request.options.json_contract)),
                think_mode="auto"))
        assert view is not None
        live["prefix_tokens"] = len(model.tokenize(view.prefix))
        live["tail_tokens"] = [len(model.tokenize(tail)) for tail in view.tails]
        live["prefix_last_tokens"] = model.tokenize(view.prefix)[-4:]
        live["tail_first_tokens"] = model.tokenize(view.tails[0])[:6]
        result = model.decide(request)
    answer = dict(result.answers[str(item.id)])
    live["wall_s"] = round(time.perf_counter() - started, 1)
    live["engine_chat_format"] = result.engine.get("chat_format")
    live["engine_template"] = result.engine.get("template")
    live["prefix_tokens_reported"] = result.engine.get("prefix_tokens")
    live["warnings"] = list(result.warnings)
    live["answer"] = {key: value for key, value in answer.items()
                      if key in ("type", "choice", "score", "noul", "confidence", "coverage",
                                 "reliability", "cue", "advance", "decode_steps")}
    if "probabilities" in answer:
        live["answer"]["top_candidates"] = sorted(
            ((key, round(float(value), 6)) for key, value in answer["probabilities"].items()),
            key=lambda pair: -pair[1])[:3]
    live["expected"] = devset_module.gold_key(item)
    live["agrees"] = str(item.gold or "") in (live["answer"].get("choice") or "",
                                              str(live["answer"].get("score") or ""),
                                              str(live["answer"].get("noul") or ""))
    return live


def render_report(record: dict) -> str:
    lines = [f"## E3e — the role split per family ({record['chat_format']} + {record['cue']})", "",
             f"- generated: {record['generated_at']}",
             f"- models: `{record['models_dir']}` · dev items rendered: {record['dev_items']}",
             "- status: " + " · ".join(f"{key} {value}" for key, value in
                                        sorted(record["counts"].items())), ""]
    header = ["file", "family", "arch", "status", "prefix", "tail(s)", "dropped", "checks"]
    rows = []
    for row in record["families"]:
        checks = row.get("checks") or {}
        ok = sum(1 for value in checks.values() if value)
        rows.append(f"| {row['file']} | {row.get('family')} | {row.get('arch')} | "
                    f"{row['status']} | {row.get('prefix_chars', '')} | "
                    f"{','.join(str(n) for n in row.get('tail_chars', []))} | "
                    f"{json.dumps(row.get('dropped', ''), ensure_ascii=False)} | "
                    f"{ok}/{len(checks) if checks else 0} |")
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *rows, ""]
    for row in record["families"]:
        lines.append(f"### {row.get('family') or row['file']}")
        lines.append("")
        if row["status"] != "rendered":
            lines.append(f"- **{row['status']}**: {row.get('reason')}")
            if row.get("fallback"):
                lines.append(f"- fallback: {row['fallback']}")
            lines.append("")
            continue
        lines.append(f"- generation prompt + opener: `{json.dumps(row['tail_tail_excerpt'])}`")
        lines.append(f"- tail opens: `{json.dumps(row['tail_head_excerpt'])}`")
        lines.append(f"- prefix ends: `{json.dumps(row['prefix_tail_excerpt'])}` "
                     f"({row['prefix_chars']} chars, `prefix + tail == render + opener`)")
        lines.append(f"- state-only render drops {len(row['dropped'])} char(s) from the shared "
                     f"prefix: {json.dumps(row['dropped'])} (the template's own trailing text)")
        lines.append("")
    live = record.get("live")
    if live:
        lines += ["### live check", "",
                  f"- model: {live['model']['name']} · {live.get('backend')} · "
                  f"gpu_layers {live.get('gpu_layers')} · load {live.get('load_ms')} ms · "
                  f"wall {live.get('wall_s')} s",
                  f"- split: prefix {live['prefix_tokens']} tokens, tail "
                  f"{live['tail_tokens']} tokens (live tokenizer; reported "
                  f"{live.get('prefix_tokens_reported')})",
                  f"- answer: {json.dumps(live['answer'], ensure_ascii=False)}",
                  f"- warnings: {', '.join(live['warnings']) or 'none'} · "
                  f"expected {live.get('expected')} · agrees {live.get('agrees')}", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="~/.hermes/models",
                        help="directory of .gguf files (default: ~/.hermes/models)")
    parser.add_argument("--cue", default="json_instructed", choices=list(schema.CUE_SHAPES))
    parser.add_argument("--json-contract", dest="json_contract", default=schema.JSON_CONTRACT,
                        choices=list(schema.JSON_CONTRACTS),
                        help="question | system — where the json_instructed contract is stated")
    parser.add_argument("--json", default=None, help="write the record here")
    parser.add_argument("--report", default=None, help="write the markdown report here")
    parser.add_argument("--live", action="store_true", help="also run one live decision")
    parser.add_argument("--model", default=None, help="the live model (a .gguf path)")
    parser.add_argument("--runtime", default=None, help="TYPED_GGUF_RUNTIME_DIR (default: the env)")
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=None)
    parser.add_argument("--kv-type", dest="kv_type", default="auto")
    parser.add_argument("--id", default=None, help="the dev item id for the live check")
    parser.add_argument("--hide-devices", dest="hide_devices", action="store_true",
                        help="hide the Vulkan ICD (`TYPED_GGUF_HIDDEN_ICD`) for a CPU-only "
                             "live check")
    args = parser.parse_args(argv)
    record = offline_record(args)
    if args.live:
        if not args.model:
            raise SystemExit("--live needs --model")
        record["live"] = live_check(args, record)
    text = render_report(record)
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"record: {args.json}")
    if args.report:
        pathlib.Path(args.report).write_text(text, encoding="utf-8")
        print(f"report: {args.report}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
