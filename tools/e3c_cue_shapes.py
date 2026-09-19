#!/usr/bin/env python3
"""E3c: cue shapes that put the readout mid-answer (card t_6c119626).

E3b (`t_6952f0dd`) left a clean negative: no rendering of the candidate label lifts the coverage
floor for `qwen35moe`, because the *cue row* itself is dominated by `<|im_end|>` (p = 0.9976 …
1.00000) — the model closes the assistant turn instead of answering. The hypothesis this probe
tests is the other half: **the readout sits at the start of the assistant turn**, so move it
*inside an answer*:

* `answer_is` / `answer_colon` / `wybieram_pl` — a forced partial answer after the shipped cue, so
  the decision point is mid-sentence. English and Polish on purpose: the model is
  co-work-post-trained and the language of the opener is not assumed;
* `json_field` — the field is opened *for* the model (`{"severity": "`) instead of asking it to
  start an answer;
* `two_step_shipped` / `two_step_answer_is` — the engine's own protocol generalised: decode the
  model's first **content** token (the argmax among tokens that are not turn-closers) and read the
  label mass at the **second** decision point. `newline` (E3b) approximated this with a label
  variant; here the model chooses the token.

Every shape is the shipped suffix plus an opener — byte-identical up to its last token — so the
question, the criteria and the instructions can never differ between shapes. Per shape the probe
reports, on the row the shape proposes to read: the top token (piece + full-vocab mass), the
turn-closer mass (the EOT side of the comparison), the coverage of every label variant, how many
items clear the engine's floor, the engine's own `cue` verdict (`engine/cue.py` — the same function
the serving path calls), and — for the shapes asked for with `--rank` — the ranked readout
(`labels.score_paths` + restricted softmax) against the probe's coverage ranking.

Cost design (one model load, no sampling): every shape of one item is a sequence forked from the
item's prefix inside ONE decode batch, exactly E3b's protocol; the `two_step_*` shapes pay one
extra batched step. Coverage needs no further decode — it is read from the row.

    # the sweep (Occamy 1.0), the 6-item E3b subset for a directly comparable table
    GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> python3 tools/e3c_cue_shapes.py run \\
        --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \\
        --devset docs/evidence/e3_chunks/devset_001.jsonl --ids c01 c02 s01 s02 n01 n02 \\
        --rank shipped=bare --rank two_step_shipped=bare \\
        --out .e3c/occamy.json --report docs/evidence/e3c_cue_shapes_occamy.md

    # offline: the markdown tables from a stored run
    python3 tools/e3c_cue_shapes.py report --run .e3c/occamy.json
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import e3b_label_policy as e3b  # noqa: E402  (the E3b probe: batching, ranking, formatting)

from ggufone.bench import devset, harness, labels  # noqa: E402
from ggufone.engine import cue as cue_module  # noqa: E402
from ggufone.engine import decide, prompt, readout  # noqa: E402
from ggufone.engine import session as session_module  # noqa: E402

SCHEMA = "ggufone.e3c.cue-shapes/v1"
DEFAULT_THREADS = 4
DEFAULT_VK_DRIVER_FILES = "/work/e3scratch/nvidia_egl_icd.json"
DEFAULT_STATES_HOME = "/work/e3c/states"
DEFAULT_TOP_TOKENS = 6
MAX_SEQUENCES = 64
#: the field each question type opens in the `json_field` shape
JSON_FIELDS = {"choice": "choice", "score": "severity", "noul": "answer"}
GREEDY = "greedy"
CONTENT = "content"


@dataclasses.dataclass(frozen=True, slots=True)
class Shape:
    """One cue shape: what follows the shipped cue line, and where the readout sits.

    `advance` is how many model-chosen tokens are decoded after the suffix before the row is read
    (`0` = the row at the cue, the shipped position). `rule` says how those tokens are chosen:
    `greedy` (the row's own argmax) or `content` (the argmax among tokens that are not
    turn-closers — "if the model cannot close the turn, what does it start to say?").
    """

    name: str
    opener: str = ""
    advance: int = 0
    rule: str = GREEDY

    def __post_init__(self) -> None:
        if self.advance < 0:
            raise ValueError(f"shape {self.name!r}: advance must be >= 0, got {self.advance}")
        if self.rule not in (GREEDY, CONTENT):
            raise ValueError(f"shape {self.name!r}: rule must be {GREEDY} or {CONTENT}")

    def suffix(self, shipped: str, question: Any) -> str:
        """The shipped suffix plus this shape's opener — never a rewrite of what precedes it."""
        return shipped + self.opener.format(field=JSON_FIELDS.get(question.type, "answer"))


#: The measured shapes. `shipped` is the E3/E3b control (the row every published `low_mass` was
#: read at); the rest move the decision point into an answer, in the three ways the card names.
SHAPES: tuple[Shape, ...] = (
    Shape("shipped"),
    Shape("answer_is", "The answer is "),
    Shape("answer_colon", "Answer: "),
    Shape("wybieram_pl", "Zgodnie z opisem, wybieram: "),
    Shape("json_field", '{{"{field}": "'),
    Shape("two_step_shipped", "", advance=1, rule=CONTENT),
    Shape("two_step_answer_is", "The answer is ", advance=1, rule=CONTENT),
)

SHAPE_NAMES: tuple[str, ...] = tuple(shape.name for shape in SHAPES)


def shape_by_name(name: str) -> Shape:
    """The catalogue entry for `name` (a typo is a caller error, never a silent default)."""
    for shape in SHAPES:
        if shape.name == name:
            return shape
    raise SystemExit(f"unknown cue shape {name!r}; known: {', '.join(SHAPE_NAMES)}")


def advance_token(row: Sequence[float], closers: Mapping[int, str], rule: str) -> int:
    """The token a `two_step_*` shape decodes before it reads (the engine's tie-break kept).

    `content` never returns a closer, so the second decision point really is inside an answer;
    `greedy` is what the engine itself would read first, which is exactly the refusal this card
    measures.
    """
    if rule == GREEDY:
        return readout.argmax_first(row)
    excluded = set(closers)
    best, best_value = -1, float("-inf")
    for token, value in enumerate(row):
        if token in excluded:
            continue
        if float(value) > best_value:
            best, best_value = int(token), float(value)
    if best < 0:
        raise SystemExit("the row carries no token outside the turn-closer catalogue")
    return best


def closer_mass(row: Sequence[float], scale: float, closers: Mapping[int, str]) -> dict[str, float]:
    """`{closer: full-vocab mass}` at one row — the EOT side of the E3b comparison, per shape."""
    return {text: readout.coverage_from_scale(row, (token,), scale)
            for token, text in sorted(closers.items())}


def label_block(texts: Sequence[str], tokenize: Any, row: Sequence[float], scale: float,
                ) -> dict[str, Any]:
    """One label variant's coverage at one row (the engine's own function, never re-derived)."""
    first = [int(tokenize(text)[0]) for text in texts]
    coverage = labels.label_coverage(list(texts), tokenize, row, scale)
    return {"texts": list(texts), "first_tokens": first, "coverage": coverage,
            "reliability": labels.reliability_of(coverage),
            # per-candidate first-token mass: what the probe's own ranking reads (the aggregate
            # `coverage` above is their sum and cannot separate the candidates by itself)
            "first_mass": [readout.coverage_from_scale(row, (token,), scale) for token in first]}


def row_block(row: Sequence[float], scale: float, handle: Any, *, which: str, position: int,
              closers: Mapping[int, str], sets: Mapping[str, Sequence[str]],
              top_tokens_count: int) -> dict[str, Any]:
    """Everything the probe records about ONE decision row (the cue row or the advanced one)."""
    return {
        "row": which,
        "position": int(position),
        "top_tokens": e3b.top_tokens(row, scale, handle, top_tokens_count),
        "closer_mass": closer_mass(row, scale, closers),
        "cue": cue_module.cue_verdict(row, scale, closers),
        "labels": {variant: label_block(texts, handle.tokenize, row, scale)
                   for variant, texts in sets.items()},
    }


def measure_item(handle: session_module.ModelHandle, item: devset.DevItem, *, shapes: Sequence[Shape],
                 variants: Sequence[str], rank: Sequence[tuple[str, str]], threads: int,
                 states_home: pathlib.Path, length_norm: float, temperature: float,
                 max_sequences: int, top_tokens_count: int) -> dict[str, Any]:
    """One dev item: prefill once, one batched decode for every shape, one step for the two-steps."""
    n_seq_max = max(1 + len(shapes) + 4 * len(item.criteria) + 4, 8)
    request = e3b.request_for_item(item, threads=threads, n_seq_max=n_seq_max)
    question = request.questions[0]
    plan = decide.plan_context(request, handle)
    rendered = e3b.prefix_text(request, handle, variant=e3b.SHIPPED_PREFIX)
    fixed = dataclasses.replace(plan, prefix_tokens=tuple(handle.tokenize(rendered)),
                                n_ctx=plan.n_ctx + 4096)
    shipped = prompt.build_question(question).suffix
    tokenized = {shape.name: handle.tokenize(shape.suffix(shipped, question)) for shape in shapes}
    sets = e3b.label_sets(question, variants)
    closers = cue_module.single_token_closers(handle.tokenize)
    record: dict[str, Any] = {
        "id": item.id, "type": item.type, "expected": devset.gold_key(item),
        "prefix_tokens": fixed.n_prefix, "n_ctx": fixed.n_ctx, "n_seq_max": fixed.n_seq_max,
        "prefix_tail": rendered[-60:], "opener": {shape.name: shape.opener for shape in shapes},
        "shapes": {}, "ranked": {}}
    with session_module.ModelSession(handle, fixed, backend="vulkan", states_home=states_home,
                                     log=[]) as live:
        info = live.prefill(list(fixed.prefix_tokens), state_cache=False)
        record["prefill_ms"] = round(info.prefill_ms, 1)
        wanted: list[tuple[int, Sequence[int]]] = []
        heads: dict[str, int] = {}
        for index, shape in enumerate(shapes, start=1):
            heads[shape.name] = index
            live.fork(0, index, fixed.n_prefix)
            wanted.append((index, tokenized[shape.name]))
        positions = [list(range(fixed.n_prefix, fixed.n_prefix + len(tokenized[shape.name])))
                     for shape in shapes]
        started = time.perf_counter()
        rows = live.decode(e3b.batch_for(wanted, positions))
        record["cue_decode_s"] = round(time.perf_counter() - started, 1)
        readouts: dict[str, dict[str, Any]] = {}
        for shape, tokens, row in zip(shapes, tokenized.values(), rows, strict=True):
            scale = readout.logsumexp(row)
            ended = fixed.n_prefix + len(tokens) - 1
            entry: dict[str, Any] = {"suffix_tokens": len(tokens),
                                     "readout": row_block(row, scale, handle, which="cue",
                                                          position=ended, closers=closers,
                                                          sets=sets,
                                                          top_tokens_count=top_tokens_count)}
            if shape.advance:
                choice = advance_token(row, closers, shape.rule)
                entry["advance"] = {"rule": shape.rule, "token": int(choice),
                                    "piece": session_module.ctypes_binding.token_piece(
                                        handle.runtime, handle.vocab, choice)}
                entry["cue_row"] = dict(entry["readout"])          # what the shipped row held
            record["shapes"][shape.name] = entry
            readouts[shape.name] = {"row": row, "scale": scale, "ended": ended,
                                    "choice": entry.get("advance", {}).get("token")}
        # ---- the one extra step: the two-step shapes decode their own token, then read again
        wanted = [(heads[shape.name], [readouts[shape.name]["choice"]])
                  for shape in shapes if shape.advance]
        if wanted:
            positions = [[readouts[shape.name]["ended"] + 1] for shape in shapes if shape.advance]
            started = time.perf_counter()
            advanced = live.decode(e3b.batch_for(wanted, positions))
            record["advance_decode_s"] = round(time.perf_counter() - started, 1)
            for shape, row in zip([shape for shape in shapes if shape.advance], advanced,
                                 strict=True):
                scale = readout.logsumexp(row)
                readouts[shape.name] = {"row": row, "scale": scale,
                                        "ended": readouts[shape.name]["ended"] + 1,
                                        "choice": readouts[shape.name]["choice"]}
                record["shapes"][shape.name]["readout"] = row_block(
                    row, scale, handle, which="advanced", position=readouts[shape.name]["ended"],
                    closers=closers, sets=sets, top_tokens_count=top_tokens_count)
        # ---- the ranked readout for the shapes that were asked for (the engine's own arithmetic)
        cursor = 1 + len(shapes)
        for shape_name, variant in rank:
            shape = shape_by_name(shape_name)
            if shape_name not in record["shapes"]:
                raise SystemExit(f"--rank {shape_name}: this run did not measure that shape")
            sequences = [tuple(tokenized_text)
                         for tokenized_text in (handle.tokenize(text) for text in sets[variant])]
            nodes = labels.trie_nodes(sequences)
            if cursor + nodes > max_sequences:
                raise SystemExit(
                    f"--max-sequences {max_sequences} is too small: the ranked policy "
                    f"{shape_name}={variant} needs {cursor + nodes} sequences")
            pool = list(range(cursor, cursor + nodes))
            cursor += nodes
            state = readouts[shape_name]
            ranked = e3b.rank_policy(live, head_seq=heads[shape_name], base=state["ended"] + 1,
                                     decision_row=state["row"], decision_scale=state["scale"],
                                     pool=pool, sequences=sequences, options=question.options,
                                     length_norm=length_norm, temperature=temperature)
            ranked["shape"] = shape_name
            ranked["label"] = variant
            ranked["coverage"] = record["shapes"][shape_name]["readout"]["labels"][variant][
                "coverage"]
            ranked["reliability"] = labels.reliability_of(ranked["coverage"])
            ranked["correct"] = ranked["got"] == devset.gold_key(item)
            ranked["top_coverage"] = top_coverage(record["shapes"][shape_name]["readout"]["labels"],
                                                  variant, question.options)
            ranked["agrees"] = ranked["got"] == ranked["top_coverage"]
            record["ranked"][f"{shape_name}={variant}"] = ranked
    return record


def top_coverage(labels_block: Mapping[str, Any], variant: str,
                 options: Sequence[str]) -> str | None:
    """The candidate the `variant` coverage favours at a row (the probe's own ranking).

    A variant whose labels collapse onto one first token cannot separate the candidates at step
    one — `labels.shared_first_tokens` says so — and `None` is the honest answer there.
    """
    block = labels_block.get(variant) or {}
    masses = [float(value) for value in block.get("first_mass", [])]
    if not masses or len(options) != len(masses):
        return None
    order = sorted(range(len(masses)), key=lambda index: -masses[index])
    return str(options[order[0]])


def select_items(items: Sequence[devset.DevItem], *, ids: Sequence[str] | None,
                 per_type: int | None, limit: int | None) -> list[devset.DevItem]:
    return e3b.select_items(items, per_type=per_type, ids=ids, limit=limit)


# --------------------------------------------------------------------------- offline report
def item_shape(item: Mapping[str, Any], shape: str) -> Mapping[str, Any]:
    entry = (item.get("shapes") or {}).get(shape)
    if entry is None:
        raise KeyError(f"item {item.get('id')} carries no `{shape}` measurement")
    return entry


def verdict_block(record: Mapping[str, Any]) -> list[str]:
    """The card's verdict table: per item and shape, what the readout row holds and its verdict."""
    lines = ["| item | shape | readout row | top token at the readout | top-token mass "
             "| turn-closer mass | verdict |",
             "|---|---|---|---|---|---|---|"]
    for item in record["items"]:
        for shape in record["shapes"]:
            readout_block = item_shape(item, shape)["readout"]
            top = (readout_block.get("top_tokens") or [{"piece": "—", "p_full": 0.0}])[0]
            closer = readout_block.get("cue") or {}
            masses = readout_block.get("closer_mass") or {}
            nearest = max(masses.values()) if masses else 0.0
            verdict = "W_CUE_REFUSED" if closer.get("refused") else "ok"
            lines.append(f"| {item['id']} | `{shape}` | {readout_block.get('row')} "
                         f"| `{top['piece']}` | {top['p_full']:.4f} | {nearest:.4f} "
                         f"| {verdict} |")
    lines.append("")
    lines.append("`W_CUE_REFUSED` = the row's top token is a turn-closer the model's own "
                 "tokenizer encodes as one token (`engine/cue.py`): the model closes the assistant "
                 "turn instead of answering, so *no* label rendering can reach the floor here. "
                 "`turn-closer mass` is the largest mass any catalogue closer holds at that row.")
    return lines


def closer_table(record: Mapping[str, Any]) -> list[str]:
    """The EOT-vs-candidate numbers: every closer's mass at every shape's readout row."""
    names = sorted({name for item in record["items"] for shape in record["shapes"]
                    for name in (item_shape(item, shape)["readout"].get("closer_mass") or {})})
    lines = ["| item | shape | " + " | ".join(f"`{name}`" for name in names)
             + " | `bare` coverage |", "|---|---|" + "---|" * (len(names) + 1)]
    for item in record["items"]:
        for shape in record["shapes"]:
            readout_block = item_shape(item, shape)["readout"]
            masses = readout_block.get("closer_mass") or {}
            coverage = readout_block["labels"]["bare"]["coverage"]
            cells = " | ".join(f"{masses.get(name, 0.0):.4f}" for name in names)
            lines.append(f"| {item['id']} | `{shape}` | {cells} | {coverage:.3e} |")
    return lines


def coverage_table(record: Mapping[str, Any]) -> list[str]:
    """One row per (shape, label) variant: per-item coverage, the floor verdict, the aggregate."""
    lines = ["| shape | label | " + " | ".join(item["id"] for item in record["items"])
             + " | above floor | median |", "|---|---|" + "---|" * (len(record["items"]) + 2)]
    for shape in record["shapes"]:
        for variant in record["label_variants"]:
            values = [item_shape(item, shape)["readout"]["labels"][variant]["coverage"]
                      for item in record["items"]]
            fmt = e3b.value_format(values)
            above = sum(1 for value in values if value >= record["mass_floor"])
            ordered = sorted(values)
            cells = [f"{value:{fmt}}{'' if value >= record['mass_floor'] else '*'}"
                     for value in values]
            lines.append(f"| `{shape}` | `{variant}` | " + " | ".join(cells)
                         + f" | {above}/{len(values)} | {ordered[len(ordered) // 2]:{fmt}} |")
    lines.append("")
    lines.append(f"`*` = below the engine's floor ({record['mass_floor']:.2f} ⇒ `low_mass`). "
                 f"Coverage is the full-vocabulary mass of the label's first token at the shape's "
                 f"readout row (`readout.coverage_from_scale`) — every label variant of one shape "
                 f"costs no extra forward pass.")
    return lines


def summary_table(record: Mapping[str, Any]) -> list[str]:
    lines = ["| shape | readout row | mean `bare` coverage | median | above floor "
             "| refused items | advance rule |", "|---|---|---|---|---|---|---|"]
    for shape in record["shapes"]:
        entries = [item_shape(item, shape) for item in record["items"]]
        values = [entry["readout"]["labels"]["bare"]["coverage"] for entry in entries]
        refused = sum(1 for entry in entries if entry["readout"]["cue"]["refused"])
        any_advance = next((entry for entry in entries if entry.get("advance")), None)
        rule = any_advance["advance"]["rule"] if any_advance else "—"
        fmt = e3b.value_format(values)
        ordered = sorted(values)
        lines.append(f"| `{shape}` | {entries[0]['readout']['row']} "
                     f"| {sum(values) / len(values):{fmt}} | {ordered[len(ordered) // 2]:{fmt}} "
                     f"| {sum(1 for value in values if value >= record['mass_floor'])}"
                     f"/{len(values)} | {refused}/{len(entries)} | {rule} |")
    return lines


def ranked_block(record: Mapping[str, Any]) -> list[str]:
    keys = list(record.get("ranked_keys", []))
    if not keys:
        return ["No shape was ranked in this run: coverage alone was measured "
                "(`--rank <shape>=<label>` runs the engine's full readout)."]
    lines = ["| policy | n | correct | agreement | 95% CI | low_mass | median coverage "
             "| ranked == coverage |", "|---|---|---|---|---|---|---|---|"]
    for key in keys:
        rows = [item["ranked"][key] for item in record["items"] if key in item.get("ranked", {})]
        if not rows:
            continue
        correct = sum(1 for row in rows if row["correct"])
        low, high = harness.wilson_interval(correct, len(rows))
        low_mass = sum(1 for row in rows if row["reliability"] == "low_mass")
        agrees = sum(1 for row in rows if row.get("agrees"))
        coverages = sorted(row["coverage"] for row in rows)
        lines.append(f"| `{key}` | {len(rows)} | {correct} | {correct / len(rows):.3f} "
                     f"| {low:.3f}–{high:.3f} | {low_mass}/{len(rows)} "
                     f"| {coverages[len(coverages) // 2]:.4f} | {agrees}/{len(rows)} |")
    return lines


def render_report(record: Mapping[str, Any]) -> str:
    model = record.get("model") or {}
    lines = [
        f"# E3c — cue shapes that put the readout mid-answer (`{model.get('arch')}`)",
        "",
        f"- card `t_6c119626` · model `{model.get('name')}` "
        f"({model.get('bytes') or 0:,} bytes, sha256 `{str(record.get('model_sha256'))[:16]}…`)",
        f"- runtime `{record['runtime']}` · backend `vulkan` · threads {record['threads']} "
        f"· `--gpu-layers` requested {record['gpu_layers']}",
        f"- placement used: `{json.dumps(record['placement'])}`",
        f"- dev items: {', '.join(item['id'] for item in record['items'])} "
        f"({', '.join(f'{key} {value}' for key, value in sorted(record['counts'].items()))}) "
        f"· label variants: {', '.join(f'`{name}`' for name in record['label_variants'])} "
        f"· generated {record['generated_at']}",
        f"- engine floor: coverage < **{record['mass_floor']:.2f}** ⇒ `low_mass` "
        f"(`OPTION_DEFAULTS['coverage_floor']`)",
        f"- shapes: " + ", ".join(f"`{name}`" for name in record["shapes"]),
        f"- wall: {record['wall_s']:.1f} s (one model load {record['load_ms']:.0f} ms, one context "
        f"per item, all shapes of one item in one decode batch)",
        "",
        "## 1. The verdict table (the card's fixture: closer ⇒ warning, content token ⇒ none)",
        "",
        *verdict_block(record),
        "",
        "## 2. The EOT side: turn-closer mass vs the candidate's mass",
        "",
        *closer_table(record),
        "",
        "## 3. Coverage per item, shape × label variant",
        "",
        *coverage_table(record),
        "",
        "## 4. Shape summary (the `bare` label the engine ships)",
        "",
        *summary_table(record),
        "",
        "## 5. The ranked readout (the engine's own arithmetic) vs the probe's ranking",
        "",
        *ranked_block(record),
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- live driver
def live_run(args: argparse.Namespace) -> int:
    os.environ.setdefault("VK_DRIVER_FILES", args.vk_driver_files)
    model_path = str(pathlib.Path(os.path.expanduser(args.model)))
    runtime = args.runtime or os.environ.get("GGUFONE_RUNTIME_DIR")
    items = select_items(devset.load(args.devset), ids=args.ids, per_type=args.per_type,
                         limit=args.limit)
    if not items:
        raise SystemExit("no dev item selected")
    shapes = [shape_by_name(name) for name in (args.shapes or SHAPE_NAMES)]
    rank: list[tuple[str, str]] = []
    for pair in args.rank or []:
        name, _, variant = pair.partition("=")
        if not variant:
            raise SystemExit(f"--rank takes shape=label, got {pair!r}")
        shape_by_name(name)                                  # reject an unknown shape early
        labels.label_texts("choice", ("a",), (None,), variant)
        rank.append((name, variant))
    sha = ""
    try:
        from ggufone.runtime import fit
        sha = fit.ModelFacts.read(model_path, want_sha256=True).sha256
    except Exception as exc:                                # noqa: BLE001 - a pin, not a gate
        print(f"warning: could not pin the model sha256 ({exc})", file=sys.stderr)
    log: list[str] = []
    started = time.perf_counter()
    handle = session_module.open_model(model_path, runtime_dir=runtime,
                                       fit_plan=harness.Placement(args.gpu_layers), log=log)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": harness.model_facts(model_path),
        "model_sha256": sha, "runtime": str(handle.runtime.directory),
        "threads": args.threads, "gpu_layers": args.gpu_layers,
        "placement": handle.placement.to_dict(), "load_ms": round(handle.load_ms, 1),
        "shapes": [shape.name for shape in shapes],
        "label_variants": list(args.label_variants),
        "mass_floor": labels.MASS_FLOOR,
        "ranked_keys": [f"{name}={variant}" for name, variant in rank],
        "counts": devset.counts(items), "devset": args.devset or str(devset.devset_path()),
        "items": [],
    }
    states = pathlib.Path(args.states_home)
    states.mkdir(parents=True, exist_ok=True)
    try:
        for item in items:
            piece = measure_item(handle, item, shapes=shapes, variants=args.label_variants,
                                 rank=rank, threads=args.threads, states_home=states,
                                 length_norm=args.length_norm, temperature=args.temperature,
                                 max_sequences=args.max_sequences,
                                 top_tokens_count=args.top_tokens)
            record["items"].append(piece)
            print(json.dumps({
                "item": item.id,
                "shapes": {name: {
                    "verdict": piece["shapes"][name]["readout"]["cue"],
                    "top": [(token["piece"], round(token["p_full"], 4))
                            for token in piece["shapes"][name]["readout"]["top_tokens"][:2]],
                    "coverage": {variant: round(block["coverage"], 10) for variant, block
                                 in piece["shapes"][name]["readout"]["labels"].items()},
                } for name in record["shapes"]},
                "ranked": {key: {"got": row["got"], "correct": row["correct"],
                                 "agrees": row["agrees"]} for key, row in piece["ranked"].items()},
            }, ensure_ascii=False), flush=True)
    finally:
        handle.close()
    record["wall_s"] = round(time.perf_counter() - started, 1)
    record["device_log_tail"] = "\n".join(log[-6:])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"run: {out}", flush=True)
    markdown = render_report(record)
    if args.report:
        pathlib.Path(args.report).write_text(markdown, encoding="utf-8")
        print(f"report: {args.report}")
    print(markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the live sweep (loads the model once)")
    run.add_argument("--model", required=True)
    run.add_argument("--runtime", default=None, help="GGUFONE_RUNTIME_DIR; default: the env")
    run.add_argument("--devset", default=None, help="devset jsonl (default: the committed set)")
    run.add_argument("--ids", nargs="+", default=None, help="exact dev item ids")
    run.add_argument("--per-type", dest="per_type", type=int, default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--shapes", nargs="+", default=None, help=f"default: {', '.join(SHAPE_NAMES)}")
    run.add_argument("--label-variants", dest="label_variants", nargs="+",
                     default=list(labels.LABEL_VARIANTS))
    run.add_argument("--rank", action="append", default=None,
                     help="shape=label policies to score fully (repeatable)")
    run.add_argument("--top-tokens", dest="top_tokens", type=int, default=DEFAULT_TOP_TOKENS)
    run.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    run.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=7)
    run.add_argument("--length-norm", dest="length_norm", type=float, default=1.0)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--max-sequences", dest="max_sequences", type=int, default=MAX_SEQUENCES)
    run.add_argument("--states-home", dest="states_home", default=DEFAULT_STATES_HOME)
    run.add_argument("--vk-driver-files", dest="vk_driver_files",
                     default=os.environ.get("GGUFONE_VK_DRIVER_FILES", DEFAULT_VK_DRIVER_FILES))
    run.add_argument("--out", required=True)
    run.add_argument("--report", default=None)
    report = sub.add_parser("report", help="offline: the markdown tables from a stored run")
    report.add_argument("--run", required=True)
    report.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.command == "report":
        record = json.loads(pathlib.Path(args.run).read_text(encoding="utf-8"))
        markdown = render_report(record)
        if args.out:
            pathlib.Path(args.out).write_text(markdown, encoding="utf-8")
        print(markdown)
        return 0
    return live_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
